"""Chronological splitting: never shuffle a time series."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from energy_sourcing.config import SplitConfig


@dataclass(frozen=True, slots=True)
class SplitMasks:
    train: NDArray[np.bool_]
    validation: NDArray[np.bool_]
    test: NDArray[np.bool_]

    @property
    def out_of_sample(self) -> NDArray[np.bool_]:
        return self.validation | self.test


def _end_of_day(day: date) -> pd.Timestamp:
    """First instant *after* ``day`` in UTC (delivery days are UTC days)."""
    return pd.Timestamp(day + timedelta(days=1), tz="UTC")


class TimeSeriesSplitter:
    def __init__(self, config: SplitConfig) -> None:
        self._config = config

    def masks(self, index: pd.DatetimeIndex) -> SplitMasks:
        train_end = _end_of_day(self._config.train_end)
        validation_end = _end_of_day(self._config.validation_end)
        train = np.asarray(index < train_end)
        validation = np.asarray((index >= train_end) & (index < validation_end))
        test = np.asarray(index >= validation_end)
        for name, mask in (("train", train), ("validation", validation), ("test", test)):
            if not mask.any():
                raise ValueError(f"{name} split is empty - check the split dates against the data")
        return SplitMasks(train=train, validation=validation, test=test)
