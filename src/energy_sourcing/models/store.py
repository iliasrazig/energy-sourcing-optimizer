"""Filesystem model store: the contract between the training pipeline and the API.

Layout::

    artifacts/
      manifest.json        # metadata, metrics, feature lists (pydantic-validated)
      dataset.parquet      # clean history needed to build features at serving time
      residuals.parquet    # out-of-sample errors, used to simulate scenarios
      models/<target>.joblib
      models/price_quantiles.joblib

Note: joblib files are pickles; only load artifacts produced by your own pipeline.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from pydantic import BaseModel, Field

from energy_sourcing import __version__
from energy_sourcing.domain import Target
from energy_sourcing.exceptions import ArtifactNotFoundError
from energy_sourcing.models.base import Forecaster
from energy_sourcing.models.ml import QuantileForecaster


class TargetReport(BaseModel):
    best_model: str
    feature_names: list[str]
    validation_metrics: dict[str, dict[str, float]]
    test_metrics: dict[str, float]
    naive_test_metrics: dict[str, float]
    skill_vs_naive: float
    gate_passed: bool
    benchmarks: dict[str, float] = Field(default_factory=dict)


class Manifest(BaseModel):
    package_version: str = __version__
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    git_sha: str = "unknown"
    data_source: str
    zone: str
    timezone: str
    holidays_country: str
    data_start: datetime
    data_end: datetime
    train_end: date
    validation_end: date
    targets: dict[str, TargetReport]
    quantile_metrics: dict[str, float] = Field(default_factory=dict)

    @property
    def all_gates_passed(self) -> bool:
        return all(report.gate_passed for report in self.targets.values())


class ModelStore:
    _QUANTILE_FILE = "price_quantiles.joblib"

    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def save_model(self, target: Target, model: Forecaster) -> Path:
        return self._dump(model, self.models_dir / f"{target}.joblib")

    def load_model(self, target: Target) -> Forecaster:
        model = self._load(self.models_dir / f"{target}.joblib")
        if not isinstance(model, Forecaster):
            raise TypeError(f"artifact for {target} is not a Forecaster")
        return model

    def save_quantile_model(self, model: QuantileForecaster) -> Path:
        return self._dump(model, self.models_dir / self._QUANTILE_FILE)

    def load_quantile_model(self) -> QuantileForecaster:
        model = self._load(self.models_dir / self._QUANTILE_FILE)
        if not isinstance(model, QuantileForecaster):
            raise TypeError("quantile artifact has an unexpected type")
        return model

    def save_frame(self, name: str, frame: pd.DataFrame) -> Path:
        path = self.root / f"{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)
        return path

    def load_frame(self, name: str) -> pd.DataFrame:
        path = self.root / f"{name}.parquet"
        self._require(path)
        return pd.read_parquet(path)

    def save_manifest(self, manifest: Manifest) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(manifest.model_dump_json(indent=2))
        return self.manifest_path

    def load_manifest(self) -> Manifest:
        self._require(self.manifest_path)
        return Manifest.model_validate(json.loads(self.manifest_path.read_text()))

    def _dump(self, obj: Any, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, path, compress=3)
        return path

    def _load(self, path: Path) -> Any:
        self._require(path)
        return joblib.load(path)

    @staticmethod
    def _require(path: Path) -> None:
        if not path.exists():
            raise ArtifactNotFoundError(
                f"{path} not found - run the training pipeline first (`make train`)"
            )
