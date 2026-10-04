"""End-to-end training pipeline: data -> features -> models -> gate -> artifacts.

Run with ``python -m energy_sourcing.pipelines.train`` (or ``es-train``).
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys

import pandas as pd

from energy_sourcing.config import DataSourceKind, Settings
from energy_sourcing.data.repository import TimeSeriesRepository, build_repository
from energy_sourcing.domain import ALL_TARGETS, FUNDAMENTAL_TARGETS, TSO_LOAD_FORECAST, Target
from energy_sourcing.exceptions import EnergySourcingError, QualityGateError
from energy_sourcing.features.builder import FEATURE_SPECS, CalendarFeatures, FeatureBuilder
from energy_sourcing.logging_config import configure_logging
from energy_sourcing.models.registry import build_candidates, build_quantile_model
from energy_sourcing.models.store import Manifest, ModelStore, TargetReport
from energy_sourcing.training.metrics import mae, quantile_report
from energy_sourcing.training.splitter import TimeSeriesSplitter
from energy_sourcing.training.tracking import ExperimentTracker, NullTracker
from energy_sourcing.training.trainer import ForecastTrainer, TargetTrainingResult

logger = logging.getLogger(__name__)


class TrainingPipeline:
    def __init__(
        self,
        settings: Settings,
        repository: TimeSeriesRepository,
        tracker: ExperimentTracker,
        store: ModelStore,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._tracker = tracker
        self._store = store
        self._calendar = CalendarFeatures(settings.zone.holidays_country, settings.zone.timezone)
        self._splitter = TimeSeriesSplitter(settings.split)
        self._trainer = ForecastTrainer(self._splitter, tracker, settings.gate)

    def run(self, *, refresh_data: bool = False) -> Manifest:
        frame = self._repository.load(refresh=refresh_data)
        results: dict[Target, TargetTrainingResult] = {}

        for target in FUNDAMENTAL_TARGETS:
            results[target] = self._train_target(target, frame)

        # Price is evaluated on *forecast* fundamentals (what is known at gate closure),
        # while it is trained on actuals: the honest way to chain the models.
        exogenous = self._chained_exogenous(frame, results)
        results[Target.PRICE] = self._train_target(Target.PRICE, frame, exogenous)

        quantile_metrics = self._train_quantile_model(frame, exogenous)
        residuals = self._residual_bank(frame, results)

        manifest = self._build_manifest(frame, results, quantile_metrics)
        self._persist(frame, results, residuals, manifest)

        if not manifest.all_gates_passed:
            failed = [t for t, r in manifest.targets.items() if not r.gate_passed]
            raise QualityGateError(f"quality gate failed for: {failed}")
        return manifest

    def _train_target(
        self, target: Target, frame: pd.DataFrame, exogenous: pd.DataFrame | None = None
    ) -> TargetTrainingResult:
        settings = self._settings
        return self._trainer.train(
            target,
            FeatureBuilder(FEATURE_SPECS[target], self._calendar),
            frame,
            candidates=build_candidates(target, settings.training_profile, settings.random_state),
            exogenous=exogenous,
        )

    @staticmethod
    def _chained_exogenous(
        frame: pd.DataFrame, results: dict[Target, TargetTrainingResult]
    ) -> pd.DataFrame:
        exogenous = frame[[t.value for t in FUNDAMENTAL_TARGETS]].copy()
        for target in FUNDAMENTAL_TARGETS:
            predictions = results[target].out_of_sample_predictions
            exogenous.loc[predictions.index, target.value] = predictions
        return exogenous

    def _train_quantile_model(self, frame: pd.DataFrame, exogenous: pd.DataFrame) -> dict[str, float]:
        builder = FeatureBuilder(FEATURE_SPECS[Target.PRICE], self._calendar)
        X, y = builder.build_xy(frame, exogenous)
        masks = self._splitter.masks(X.index)
        model = build_quantile_model(self._settings.training_profile, self._settings.random_state)
        model.fit(X.loc[masks.train], y.loc[masks.train])
        report = quantile_report(y.loc[masks.test], model.predict(X.loc[masks.test]), model.quantiles)
        with self._tracker.start_run("price_forecasting", run_name="quantiles:gradient_boosting") as run:
            run.set_tags({"target": "price", "stage": "quantile"})
            run.log_metrics({f"test_{k}": v for k, v in report.items()})
        logger.info("[price] P10-P90 test coverage %.1f%%", 100 * report["interval_coverage"])
        self._store.save_quantile_model(model)
        return report

    @staticmethod
    def _residual_bank(
        frame: pd.DataFrame, results: dict[Target, TargetTrainingResult]
    ) -> pd.DataFrame:
        """Validation-period errors (actual - forecast) of every selected model."""
        columns = {}
        for target in ALL_TARGETS:
            result = results[target]
            predictions = result.out_of_sample_predictions.loc[result.validation_index]
            columns[target.value] = frame.loc[predictions.index, target.value] - predictions
        return pd.DataFrame(columns).astype("float32")

    def _build_manifest(
        self,
        frame: pd.DataFrame,
        results: dict[Target, TargetTrainingResult],
        quantile_metrics: dict[str, float],
    ) -> Manifest:
        settings = self._settings
        targets = {
            target.value: TargetReport(
                best_model=r.best_model.name,
                feature_names=r.feature_names,
                validation_metrics=r.validation_metrics,
                test_metrics=r.test_metrics,
                naive_test_metrics=r.naive_test_metrics,
                skill_vs_naive=r.skill_vs_naive,
                gate_passed=r.gate_passed,
            )
            for target, r in results.items()
        }
        targets[Target.LOAD.value].benchmarks = self._tso_benchmark(frame, results[Target.LOAD])
        return Manifest(
            git_sha=_git_sha(),
            data_source=str(settings.data_source),
            zone=settings.zone.code,
            timezone=settings.zone.timezone,
            holidays_country=settings.zone.holidays_country,
            data_start=frame.index[0].to_pydatetime(),
            data_end=frame.index[-1].to_pydatetime(),
            train_end=settings.split.train_end,
            validation_end=settings.split.validation_end,
            targets=targets,
            quantile_metrics=quantile_metrics,
        )

    def _tso_benchmark(self, frame: pd.DataFrame, load: TargetTrainingResult) -> dict[str, float]:
        """MAE of the TSO's own day-ahead load forecast on the same test hours."""
        test_index = load.out_of_sample_predictions.index.difference(load.validation_index)
        subset = frame.loc[test_index, ["load", TSO_LOAD_FORECAST]].dropna()
        if subset.empty:
            return {}
        return {"tso_test_mae": mae(subset["load"], subset[TSO_LOAD_FORECAST])}

    def _persist(
        self,
        frame: pd.DataFrame,
        results: dict[Target, TargetTrainingResult],
        residuals: pd.DataFrame,
        manifest: Manifest,
    ) -> None:
        for target, result in results.items():
            self._store.save_model(target, result.best_model)
        self._store.save_frame("dataset", frame)
        self._store.save_frame("residuals", residuals)
        path = self._store.save_manifest(manifest)
        with self._tracker.start_run("pipeline", run_name="manifest") as run:
            run.set_tags({"git_sha": manifest.git_sha, "data_source": manifest.data_source})
            run.log_artifact(path)
        logger.info("Artifacts written to %s", self._store.root)


def _git_sha() -> str:
    if sha := os.environ.get("GITHUB_SHA"):
        return sha
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train forecasting models and export artifacts.")
    parser.add_argument("--source", choices=[k.value for k in DataSourceKind])
    parser.add_argument("--profile", choices=["full", "fast"])
    parser.add_argument("--refresh-data", action="store_true", help="ignore the parquet cache")
    args = parser.parse_args(argv)

    overrides: dict[str, object] = {}
    if args.source:
        overrides["data_source"] = args.source
    if args.profile:
        overrides["training_profile"] = args.profile
    settings = Settings(**overrides)  # type: ignore[arg-type]
    configure_logging(settings.log_level)

    pipeline = TrainingPipeline(
        settings,
        build_repository(settings),
        NullTracker(),
        ModelStore(settings.artifacts_dir),
    )
    try:
        manifest = pipeline.run(refresh_data=args.refresh_data)
    except QualityGateError as exc:
        logger.error("Release blocked: %s", exc)
        return 2
    except EnergySourcingError as exc:
        logger.error("Training failed: %s", exc)
        return 1
    for target, report in manifest.targets.items():
        logger.info("%-6s %-14s skill vs naive %+.1f%%", target, report.best_model,
                    100 * report.skill_vs_naive)
    return 0


if __name__ == "__main__":
    sys.exit(main())
