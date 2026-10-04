"""Typed, environment-driven configuration (12-factor style).

Every setting can be overridden with an ``ES_``-prefixed environment variable,
nested fields use ``__`` (e.g. ``ES_SPLIT__TRAIN_END=2019-12-31``).
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

OPSD_URL = (
    "https://data.open-power-system-data.org/time_series/2020-10-06/"
    "time_series_60min_singleindex.csv"
)


class DataSourceKind(StrEnum):
    OPSD = "opsd"
    SYNTHETIC = "synthetic"


class ZoneConfig(BaseModel):
    """Bidding zone definition: where to find each canonical column in OPSD."""

    model_config = ConfigDict(frozen=True)

    code: str = "DE_LU"
    timezone: str = "Europe/Berlin"
    holidays_country: str = "DE"
    columns: dict[str, str] = Field(
        default_factory=lambda: {
            "load": "DE_LU_load_actual_entsoe_transparency",
            "wind": "DE_LU_wind_generation_actual",
            "solar": "DE_LU_solar_generation_actual",
            "price": "DE_LU_price_day_ahead",
            "load_tso_forecast": "DE_LU_load_forecast_entsoe_transparency",
        }
    )


class SplitConfig(BaseModel):
    """Chronological train / validation / test split (inclusive end dates).

    * train      -> fit candidate models
    * validation -> select the best model and build the residual bank used for scenarios
    * test       -> untouched hold-out, used for the quality gate and the backtest UI
    """

    model_config = ConfigDict(frozen=True)

    train_end: date = date(2019, 12, 31)
    validation_end: date = date(2020, 3, 31)

    @model_validator(mode="after")
    def _check_order(self) -> SplitConfig:
        if self.validation_end <= self.train_end:
            raise ValueError("validation_end must be after train_end")
        return self


class QualityGateConfig(BaseModel):
    """Minimum test-set skill vs. the naive D-1 forecast, i.e. ``1 - MAE / MAE_naive``."""

    model_config = ConfigDict(frozen=True)

    min_skill_vs_naive: float = Field(default=0.0, ge=-1.0, le=1.0)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ES_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    data_source: DataSourceKind = DataSourceKind.OPSD
    opsd_url: str = OPSD_URL
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")

    zone: ZoneConfig = Field(default_factory=ZoneConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    gate: QualityGateConfig = Field(default_factory=QualityGateConfig)

    training_profile: Literal["full", "fast"] = "full"
    random_state: int = 42
    log_level: str = "INFO"

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_data_path(self) -> Path:
        return self.data_dir / "processed" / f"{self.data_source}_{self.zone.code}.parquet"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton (cheap to call, parsed once)."""
    return Settings()
