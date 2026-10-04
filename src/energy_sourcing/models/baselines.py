"""Reference forecasters every ML model has to beat."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from energy_sourcing.models.base import Forecaster


class LagNaiveForecaster(Forecaster):
    """Persistence forecast: the value observed ``lag`` hours earlier (D-1 by default)."""

    def __init__(self, lag_column: str, name: str = "naive_d-1") -> None:
        super().__init__(name)
        self._lag_column = lag_column

    @property
    def params(self) -> dict[str, Any]:
        return {"lag_column": self._lag_column}

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        if self._lag_column not in X.columns:
            raise ValueError(f"feature {self._lag_column!r} is required")

    def _predict(self, X: pd.DataFrame) -> NDArray[np.floating[Any]]:
        return X[self._lag_column].to_numpy()


class SeasonalProfileForecaster(Forecaster):
    """Historical average by (day of week, hour) learnt on the training set."""

    _KEYS = ("dayofweek", "hour")

    def __init__(self, name: str = "seasonal_profile") -> None:
        super().__init__(name)
        self._profile: pd.Series | None = None
        self._fallback = 0.0

    @property
    def params(self) -> dict[str, Any]:
        return {"keys": ",".join(self._KEYS)}

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        keys = [X[k].astype(int) for k in self._KEYS]
        self._profile = y.groupby(keys).mean()
        self._fallback = float(y.mean())

    def _predict(self, X: pd.DataFrame) -> NDArray[np.floating[Any]]:
        assert self._profile is not None  # guaranteed by Forecaster.predict
        lookup = pd.MultiIndex.from_arrays([X[k].astype(int) for k in self._KEYS])
        return self._profile.reindex(lookup).fillna(self._fallback).to_numpy()
