"""Adapters for scikit-learn compatible regressors (Adapter pattern)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from energy_sourcing.models.base import Forecaster

_SCALAR = (int, float, str, bool, type(None))


class Regressor(Protocol):
    """Structural type satisfied by any scikit-learn compatible regressor."""

    def fit(self, X: Any, y: Any) -> Any: ...
    def predict(self, X: Any) -> Any: ...
    def get_params(self, deep: bool = ...) -> dict[str, Any]: ...


class EstimatorForecaster(Forecaster):
    """Wraps any :class:`Regressor`; feeds it contiguous float32 arrays.

    float32 halves the memory footprint vs. float64 and is what tree boosters
    use internally anyway, avoiding an extra copy.
    """

    def __init__(self, name: str, estimator: Regressor) -> None:
        super().__init__(name)
        self.estimator = estimator

    @property
    def params(self) -> dict[str, Any]:
        return {k: v for k, v in self.estimator.get_params().items() if isinstance(v, _SCALAR)}

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.estimator.fit(_as_float32(X), y.to_numpy(dtype=np.float32))

    def _predict(self, X: pd.DataFrame) -> NDArray[np.floating[Any]]:
        return np.asarray(self.estimator.predict(_as_float32(X)))


class QuantileForecaster:
    """One gradient-boosting model per quantile; outputs are sorted to prevent quantile crossing."""

    def __init__(
        self,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
        n_estimators: int = 400,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ) -> None:
        from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: PLC0415

        self.quantiles = tuple(sorted(quantiles))
        self._models = [
            EstimatorForecaster(
                f"gbm_q{int(q * 100):02d}",
                HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=q,
                    max_iter=n_estimators,
                    learning_rate=learning_rate,
                    max_leaf_nodes=31,
                    min_samples_leaf=20,
                    random_state=random_state,
                ),
            )
            for q in self.quantiles
        ]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> QuantileForecaster:
        for model in self._models:
            model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> NDArray[np.float32]:
        """Array of shape ``(n_rows, n_quantiles)``."""
        stacked = np.column_stack([m.predict(X) for m in self._models])
        return np.sort(stacked, axis=1)


def _as_float32(X: pd.DataFrame) -> NDArray[np.float32]:
    return np.ascontiguousarray(X.to_numpy(dtype=np.float32))
