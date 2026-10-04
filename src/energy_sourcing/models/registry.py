"""Factory producing the candidate models compared for each target."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from energy_sourcing.domain import Target
from energy_sourcing.features.builder import FEATURE_SPECS
from energy_sourcing.models.base import Forecaster
from energy_sourcing.models.baselines import LagNaiveForecaster, SeasonalProfileForecaster
from energy_sourcing.models.ml import EstimatorForecaster, QuantileForecaster

Profile = Literal["full", "fast"]
NAIVE_MODEL_NAME = "naive_d-1"


@dataclass(frozen=True, slots=True)
class _Budget:
    """Model capacity per training profile (``fast`` is used by tests and CI smoke runs)."""

    rf_trees: int
    boost_rounds: int
    learning_rate: float


_BUDGETS: dict[Profile, _Budget] = {
    "full": _Budget(rf_trees=200, boost_rounds=600, learning_rate=0.05),
    "fast": _Budget(rf_trees=20, boost_rounds=60, learning_rate=0.2),
}


def build_candidates(target: Target, profile: Profile = "full", random_state: int = 42) -> list[Forecaster]:
    """Fresh, unfitted candidates: two baselines and two tree ensembles."""
    budget = _BUDGETS[profile]
    lag_24 = FEATURE_SPECS[target].lag_columns[0]
    return [
        LagNaiveForecaster(lag_24, name=NAIVE_MODEL_NAME),
        SeasonalProfileForecaster(),
        EstimatorForecaster(
            "random_forest",
            RandomForestRegressor(
                n_estimators=budget.rf_trees,
                max_depth=14,
                min_samples_leaf=3,
                max_features=0.5,
                n_jobs=-1,
                random_state=random_state,
            ),
        ),
        EstimatorForecaster(
            "gradient_boosting",
            HistGradientBoostingRegressor(
                max_iter=budget.boost_rounds,
                learning_rate=budget.learning_rate,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=1.0,
                random_state=random_state,
            ),
        ),
    ]


def build_quantile_model(profile: Profile = "full", random_state: int = 42) -> QuantileForecaster:
    budget = _BUDGETS[profile]
    return QuantileForecaster(
        quantiles=(0.1, 0.5, 0.9),
        n_estimators=budget.boost_rounds,
        learning_rate=budget.learning_rate,
        random_state=random_state,
    )
