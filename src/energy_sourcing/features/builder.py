"""Leakage-safe feature engineering for day-ahead forecasting.

Day-ahead constraint: the forecast for every hour of day D is issued on D-1,
so every autoregressive feature uses a lag of at least 24 hours. This lets us
predict all 24 hours in one shot (no recursive forecasting) without leaking
information from the delivery day.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import holidays
import numpy as np
import pandas as pd

from energy_sourcing.domain import HOURS_PER_DAY, Target

CALENDAR_FEATURES: tuple[str, ...] = (
    "hour",
    "dayofweek",
    "month",
    "is_weekend",
    "is_holiday",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
)


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Declarative description of a target's feature set."""

    target: Target
    lags: tuple[int, ...] = (24, 48, 72, 168)
    rolling_windows: tuple[int, ...] = (24, 168)
    exogenous: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if min(self.lags) < HOURS_PER_DAY:
            raise ValueError("day-ahead features need lags >= 24h to avoid leakage")

    @property
    def lag_columns(self) -> tuple[str, ...]:
        return tuple(f"{self.target}_lag_{lag}" for lag in self.lags)

    @property
    def rolling_columns(self) -> tuple[str, ...]:
        return tuple(
            f"{self.target}_roll_{stat}_{w}" for w in self.rolling_windows for stat in ("mean", "std")
        )

    @property
    def exogenous_columns(self) -> tuple[str, ...]:
        if not self.exogenous:
            return ()
        return (*self.exogenous, "residual_load", "renewable_share")


FEATURE_SPECS: Mapping[Target, FeatureSpec] = {
    Target.LOAD: FeatureSpec(Target.LOAD),
    Target.WIND: FeatureSpec(Target.WIND, lags=(24, 48, 168), rolling_windows=(24, 72, 168)),
    Target.SOLAR: FeatureSpec(Target.SOLAR, lags=(24, 48, 168), rolling_windows=(24, 168)),
    # The price model also sees the (forecast) fundamentals of the delivery hour.
    Target.PRICE: FeatureSpec(
        Target.PRICE,
        lags=(24, 48, 168),
        rolling_windows=(24, 168),
        exogenous=(Target.LOAD.value, Target.WIND.value, Target.SOLAR.value),
    ),
}


class CalendarFeatures:
    """Calendar features in local market time, with public holidays."""

    def __init__(self, country: str = "DE", timezone: str = "Europe/Berlin") -> None:
        self._country = country
        self._timezone = timezone

    def transform(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        local = index.tz_convert(self._timezone)
        hour = local.hour.to_numpy()
        doy = local.dayofyear.to_numpy()
        years = range(int(local.year.min()), int(local.year.max()) + 1)
        holiday_dates = pd.to_datetime(list(holidays.country_holidays(self._country, years=years)))
        is_holiday = local.tz_localize(None).normalize().isin(holiday_dates)
        dayofweek = local.dayofweek.to_numpy()
        return pd.DataFrame(
            {
                "hour": hour,
                "dayofweek": dayofweek,
                "month": local.month.to_numpy(),
                "is_weekend": dayofweek >= 5,
                "is_holiday": is_holiday,
                "hour_sin": np.sin(2 * np.pi * hour / 24),
                "hour_cos": np.cos(2 * np.pi * hour / 24),
                "doy_sin": np.sin(2 * np.pi * doy / 365.25),
                "doy_cos": np.cos(2 * np.pi * doy / 365.25),
            },
            index=index,
        ).astype(np.float32)


class FeatureBuilder:
    """Turns an hourly frame into the feature matrix of one target.

    The input frame must sit on a regular hourly grid (the repository enforces
    this), so positional ``shift`` equals a shift in hours.
    """

    def __init__(self, spec: FeatureSpec, calendar: CalendarFeatures) -> None:
        self.spec = spec
        self._calendar = calendar

    @property
    def feature_names(self) -> list[str]:
        spec = self.spec
        return [
            *spec.lag_columns,
            *spec.rolling_columns,
            *CALENDAR_FEATURES,
            *spec.exogenous_columns,
        ]

    def build(self, frame: pd.DataFrame, exogenous: pd.DataFrame | None = None) -> pd.DataFrame:
        """Feature matrix aligned on ``frame.index``.

        ``exogenous`` supplies delivery-hour fundamentals for the price model:
        actuals when training, forecasts when evaluating or serving. When
        omitted the actual values in ``frame`` are used.
        """
        spec = self.spec
        series = frame[spec.target.value]
        features: dict[str, pd.Series] = {
            name: series.shift(lag) for name, lag in zip(spec.lag_columns, spec.lags, strict=True)
        }
        base = series.shift(min(spec.lags))
        for window in spec.rolling_windows:
            rolling = base.rolling(window, min_periods=window // 2)
            features[f"{spec.target}_roll_mean_{window}"] = rolling.mean()
            features[f"{spec.target}_roll_std_{window}"] = rolling.std()

        parts = [pd.DataFrame(features, index=frame.index), self._calendar.transform(frame.index)]
        if spec.exogenous:
            parts.append(self._exogenous_features(exogenous if exogenous is not None else frame,
                                                  frame.index))
        matrix = pd.concat(parts, axis=1)
        return matrix[self.feature_names].astype(np.float32)

    def build_xy(
        self, frame: pd.DataFrame, exogenous: pd.DataFrame | None = None
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Features and target with incomplete rows removed (warm-up period, gaps)."""
        X = self.build(frame, exogenous)
        y = frame[self.spec.target.value]
        mask = X.notna().all(axis=1) & y.notna()
        return X.loc[mask], y.loc[mask].astype(np.float32)

    def _exogenous_features(self, source: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
        exo = source.loc[:, list(self.spec.exogenous)].reindex(index)
        load = exo[Target.LOAD.value]
        renewables = exo[Target.WIND.value] + exo[Target.SOLAR.value]
        exo["residual_load"] = load - renewables
        exo["renewable_share"] = renewables / load.where(load > 0)
        return exo
