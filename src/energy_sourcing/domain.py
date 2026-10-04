"""Core domain vocabulary shared by every layer (data, ML, optimisation, API)."""

from __future__ import annotations

from enum import StrEnum

HOURS_PER_DAY = 24


class Target(StrEnum):
    """Hourly quantities the platform forecasts."""

    LOAD = "load"
    WIND = "wind"
    SOLAR = "solar"
    PRICE = "price"


#: Physical drivers, forecast first; the price model consumes their forecasts.
FUNDAMENTAL_TARGETS: tuple[Target, ...] = (Target.LOAD, Target.WIND, Target.SOLAR)
ALL_TARGETS: tuple[Target, ...] = (*FUNDAMENTAL_TARGETS, Target.PRICE)

#: Day-ahead load forecast published by the TSO, used as an external benchmark.
TSO_LOAD_FORECAST = "load_tso_forecast"

#: Canonical column layout of the cleaned hourly dataset (MW and EUR/MWh).
DATASET_COLUMNS: tuple[str, ...] = (*(t.value for t in ALL_TARGETS), TSO_LOAD_FORECAST)
