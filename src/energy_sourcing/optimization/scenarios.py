"""Probabilistic scenarios from point forecasts via a joint residual bootstrap.

Each scenario adds the *whole day* of out-of-sample forecast errors observed on
one historical day to today's forecasts, for all targets at once. Sampling
whole days for all variables jointly preserves:

* intra-day error autocorrelation (a windy morning error persists all day);
* cross-variable dependence (wind under-forecast <-> price over-forecast),
  which is exactly what makes hedging decisions non-trivial.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from energy_sourcing.domain import ALL_TARGETS, HOURS_PER_DAY, Target

_NON_NEGATIVE = frozenset({Target.LOAD, Target.WIND, Target.SOLAR})


@dataclass(frozen=True, slots=True)
class ScenarioSet:
    """``values[target]`` has shape ``(n_scenarios, 24)``."""

    values: Mapping[Target, NDArray[np.float64]]

    @property
    def n_scenarios(self) -> int:
        return next(iter(self.values.values())).shape[0]

    def __getitem__(self, target: Target) -> NDArray[np.float64]:
        return self.values[target]


class ResidualBootstrapGenerator:
    def __init__(self, residuals: pd.DataFrame, targets: Sequence[Target] = ALL_TARGETS) -> None:
        self._targets = tuple(targets)
        self._bank = self._to_day_cube(residuals, self._targets)

    @property
    def n_days(self) -> int:
        return int(self._bank.shape[0])

    def generate(
        self,
        point_forecast: Mapping[Target, NDArray[np.floating]],
        n_scenarios: int,
        seed: int | None = None,
    ) -> ScenarioSet:
        if n_scenarios <= 0:
            raise ValueError("n_scenarios must be positive")
        rng = np.random.default_rng(seed)
        days = rng.integers(0, self.n_days, size=n_scenarios)
        sampled = self._bank[days]  # (S, 24, n_targets), fancy indexing copies only S days

        values: dict[Target, NDArray[np.float64]] = {}
        for k, target in enumerate(self._targets):
            base = np.asarray(point_forecast[target], dtype=np.float64)
            if base.shape != (HOURS_PER_DAY,):
                raise ValueError(f"{target} forecast must have {HOURS_PER_DAY} values")
            scenario = base[np.newaxis, :] + sampled[:, :, k]
            if target in _NON_NEGATIVE:
                np.maximum(scenario, 0.0, out=scenario)  # in place, no extra allocation
            values[target] = scenario
        return ScenarioSet(values)

    @staticmethod
    def _to_day_cube(residuals: pd.DataFrame, targets: Sequence[Target]) -> NDArray[np.float64]:
        """Keep only complete days and reshape to ``(n_days, 24, n_targets)``."""
        columns = [t.value for t in targets]
        frame = residuals[columns].dropna()
        day = frame.index.floor("D")
        complete = day.value_counts()
        complete_days = complete[complete == HOURS_PER_DAY].index
        frame = frame[day.isin(complete_days)].sort_index()
        if frame.empty:
            raise ValueError("residual bank has no complete day")
        n_days = len(frame) // HOURS_PER_DAY
        return frame.to_numpy(dtype=np.float64).reshape(n_days, HOURS_PER_DAY, len(columns))
