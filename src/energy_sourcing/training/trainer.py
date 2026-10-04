"""Model selection for one target: fit candidates, select on validation, gate on test."""

from __future__ import annotations

import gc
import logging
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from energy_sourcing.config import QualityGateConfig
from energy_sourcing.domain import Target
from energy_sourcing.features.builder import FeatureBuilder
from energy_sourcing.models.base import Forecaster
from energy_sourcing.models.registry import NAIVE_MODEL_NAME
from energy_sourcing.training.metrics import regression_report, skill_score
from energy_sourcing.training.splitter import TimeSeriesSplitter
from energy_sourcing.training.tracking import ExperimentTracker

logger = logging.getLogger(__name__)

SELECTION_METRIC = "mae"


@dataclass(slots=True)
class TargetTrainingResult:
    target: Target
    best_model: Forecaster
    feature_names: list[str]
    validation_metrics: dict[str, dict[str, float]]
    test_metrics: dict[str, float]
    naive_test_metrics: dict[str, float]
    skill_vs_naive: float
    gate_passed: bool
    #: Best model's predictions on validation + test rows (used for chaining and residuals).
    out_of_sample_predictions: pd.Series
    validation_index: pd.DatetimeIndex


class ForecastTrainer:
    def __init__(
        self,
        splitter: TimeSeriesSplitter,
        tracker: ExperimentTracker,
        gate: QualityGateConfig,
    ) -> None:
        self._splitter = splitter
        self._tracker = tracker
        self._gate = gate

    def train(
        self,
        target: Target,
        builder: FeatureBuilder,
        frame: pd.DataFrame,
        candidates: Sequence[Forecaster],
        exogenous: pd.DataFrame | None = None,
    ) -> TargetTrainingResult:
        X, y = builder.build_xy(frame, exogenous)
        masks = self._splitter.masks(X.index)
        X_train, y_train = X.loc[masks.train], y.loc[masks.train]
        X_val, y_val = X.loc[masks.validation], y.loc[masks.validation]
        X_test, y_test = X.loc[masks.test], y.loc[masks.test]
        logger.info(
            "[%s] rows train=%d validation=%d test=%d features=%d",
            target, len(X_train), len(X_val), len(X_test), X.shape[1],
        )

        experiment = f"{target}_forecasting"
        validation_metrics: dict[str, dict[str, float]] = {}
        fitted: dict[str, Forecaster] = {}
        for model in candidates:
            with self._tracker.start_run(experiment, run_name=model.name) as run:
                model.fit(X_train, y_train)
                metrics = regression_report(y_val, model.predict(X_val))
                run.set_tags({"target": str(target), "stage": "candidate"})
                run.log_params({"model": model.name, **model.params})
                run.log_metrics({f"val_{k}": v for k, v in metrics.items()})
            validation_metrics[model.name] = metrics
            fitted[model.name] = model
            logger.info("[%s] %-16s val MAE=%.2f", target, model.name, metrics[SELECTION_METRIC])

        best_name = min(validation_metrics, key=lambda n: validation_metrics[n][SELECTION_METRIC])
        best = fitted[best_name]
        naive = fitted[NAIVE_MODEL_NAME]

        test_metrics = regression_report(y_test, best.predict(X_test))
        naive_test_metrics = regression_report(y_test, naive.predict(X_test))
        skill = skill_score(test_metrics["mae"], naive_test_metrics["mae"])
        gate_passed = skill >= self._gate.min_skill_vs_naive

        with self._tracker.start_run(experiment, run_name=f"selected:{best_name}") as run:
            run.set_tags({"target": str(target), "stage": "selected", "model": best_name})
            run.log_metrics(
                {
                    **{f"test_{k}": v for k, v in test_metrics.items()},
                    **{f"naive_test_{k}": v for k, v in naive_test_metrics.items()},
                    "test_skill_vs_naive": skill,
                    "gate_passed": float(gate_passed),
                }
            )
        logger.info(
            "[%s] selected %s: test MAE=%.2f (naive %.2f, skill %.1f%%) gate=%s",
            target, best_name, test_metrics["mae"], naive_test_metrics["mae"],
            100 * skill, "PASS" if gate_passed else "FAIL",
        )

        X_oos = X.loc[masks.out_of_sample]
        predictions = pd.Series(best.predict(X_oos), index=X_oos.index, name=str(target))

        # Release the losing candidates (random forests can be large) before the next target.
        fitted.clear()
        gc.collect()

        return TargetTrainingResult(
            target=target,
            best_model=best,
            feature_names=list(X.columns),
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            naive_test_metrics=naive_test_metrics,
            skill_vs_naive=skill,
            gate_passed=gate_passed,
            out_of_sample_predictions=predictions,
            validation_index=X_val.index,
        )
