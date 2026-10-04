"""Composition root: builds the object graph from the artifacts once at start-up."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta

from energy_sourcing.domain import ALL_TARGETS
from energy_sourcing.exceptions import EnergySourcingError
from energy_sourcing.features.builder import FEATURE_SPECS, CalendarFeatures, FeatureBuilder
from energy_sourcing.models.store import Manifest, ModelStore
from energy_sourcing.optimization.scenarios import ResidualBootstrapGenerator
from energy_sourcing.services.forecasting import ForecastService
from energy_sourcing.services.sourcing import SourcingService

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ServiceContainer:
    manifest: Manifest
    forecasts: ForecastService
    sourcing: SourcingService


def build_container(store: ModelStore) -> ServiceContainer:
    manifest = store.load_manifest()
    calendar = CalendarFeatures(manifest.holidays_country, manifest.timezone)
    builders = {t: FeatureBuilder(FEATURE_SPECS[t], calendar) for t in ALL_TARGETS}

    # Fail fast if the code's feature definitions drifted from what models were trained on.
    for target, builder in builders.items():
        expected = manifest.targets[target.value].feature_names
        if builder.feature_names != expected:
            raise EnergySourcingError(f"feature mismatch for {target}: retrain the models")

    forecasts = ForecastService(
        dataset=store.load_frame("dataset"),
        models={t: store.load_model(t) for t in ALL_TARGETS},
        quantile_model=store.load_quantile_model(),
        builders=builders,
        first_day=manifest.validation_end + timedelta(days=1),
    )
    scenarios = ResidualBootstrapGenerator(store.load_frame("residuals"))
    logger.info(
        "Loaded models (git %s), %d backtest days, residual bank of %d days",
        manifest.git_sha[:8], len(forecasts.available_days), scenarios.n_days,
    )
    return ServiceContainer(manifest, forecasts, SourcingService(forecasts, scenarios))
