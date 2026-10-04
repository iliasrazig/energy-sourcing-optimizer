"""Day-ahead forecasting service: what the API calls at serving time."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd

from energy_sourcing.domain import ALL_TARGETS, FUNDAMENTAL_TARGETS, HOURS_PER_DAY, Target
from energy_sourcing.exceptions import DayNotAvailableError
from energy_sourcing.features.builder import FeatureBuilder
from energy_sourcing.models.base import Forecaster
from energy_sourcing.models.ml import QuantileForecaster

#: History needed before the delivery day: max lag (168h) + longest rolling window (168h).
_HISTORY_DAYS = 15


class ForecastService:
    """Produces the 24-hour forecast of every target for a delivery day.

    Backtest semantics: the delivery day's actual values are masked before
    features are built, so the forecast only uses information available on D-1.
    Results are memoised in a bounded LRU cache (memory stays capped).
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        models: Mapping[Target, Forecaster],
        quantile_model: QuantileForecaster,
        builders: Mapping[Target, FeatureBuilder],
        first_day: date,
        cache_size: int = 128,
    ) -> None:
        self._dataset = dataset
        self._models = dict(models)
        self._quantile_model = quantile_model
        self._builders = dict(builders)
        self._available = self._compute_available_days(dataset, first_day)
        self._cached_forecast = lru_cache(maxsize=cache_size)(self._forecast)

    @property
    def available_days(self) -> list[date]:
        return sorted(self._available)

    @property
    def quantiles(self) -> tuple[float, ...]:
        return self._quantile_model.quantiles

    def forecast(self, day: date) -> pd.DataFrame:
        """Columns ``<target>_forecast``, ``<target>_actual`` and ``price_q<NN>``."""
        if day not in self._available:
            raise DayNotAvailableError(f"{day} is outside the out-of-sample backtest period")
        return self._cached_forecast(day).copy()  # callers may not mutate the cached frame

    def _forecast(self, day: date) -> pd.DataFrame:
        start = pd.Timestamp(day, tz="UTC")
        end = start + pd.Timedelta(hours=HOURS_PER_DAY)
        window = self._dataset.loc[start - pd.Timedelta(days=_HISTORY_DAYS) : end - pd.Timedelta(hours=1)]
        delivery = window.index >= start

        masked = window.copy()
        masked.loc[delivery, [t.value for t in ALL_TARGETS]] = np.nan

        output = pd.DataFrame(index=window.index[delivery])
        exogenous = masked[[t.value for t in FUNDAMENTAL_TARGETS]].copy()
        for target in FUNDAMENTAL_TARGETS:
            prediction = self._predict(target, masked, delivery)
            exogenous.loc[delivery, target.value] = prediction
            output[f"{target}_forecast"] = prediction

        X_price = self._builders[Target.PRICE].build(masked, exogenous).loc[delivery]
        output[f"{Target.PRICE}_forecast"] = self._models[Target.PRICE].predict(X_price)
        quantile_values = self._quantile_model.predict(X_price)
        for i, q in enumerate(self._quantile_model.quantiles):
            output[f"price_q{int(q * 100):02d}"] = quantile_values[:, i]

        for target in ALL_TARGETS:
            output[f"{target}_actual"] = window.loc[delivery, target.value].to_numpy()
        return output

    def _predict(self, target: Target, masked: pd.DataFrame, delivery: np.ndarray) -> np.ndarray:
        X = self._builders[target].build(masked).loc[delivery]
        return self._models[target].predict(X)

    @staticmethod
    def _compute_available_days(dataset: pd.DataFrame, first_day: date) -> set[date]:
        """Days with complete actuals, enough history, and inside the hold-out period."""
        earliest = (dataset.index[0] + timedelta(days=_HISTORY_DAYS + 1)).date()
        complete = dataset[[t.value for t in ALL_TARGETS]].notna().all(axis=1)
        counts = complete.groupby(dataset.index.floor("D")).sum()
        return {
            ts.date()
            for ts, n in counts.items()
            if n == HOURS_PER_DAY and ts.date() >= max(first_day, earliest)
        }
