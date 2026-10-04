"""Forecaster abstraction (Template Method pattern).

``fit``/``predict`` handle the cross-cutting concerns once (state checks,
column alignment, dtype) and delegate the algorithm to ``_fit``/``_predict``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from energy_sourcing.exceptions import NotFittedError


class Forecaster(ABC):
    """A point forecaster mapping a feature matrix to one value per row."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.feature_names_: list[str] | None = None

    @property
    def is_fitted(self) -> bool:
        return self.feature_names_ is not None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} rows but y has {len(y)}")
        if len(X) == 0:
            raise ValueError("cannot fit on an empty dataset")
        self._fit(X, y)
        self.feature_names_ = list(X.columns)
        return self

    def predict(self, X: pd.DataFrame) -> NDArray[np.float32]:
        if self.feature_names_ is None:
            raise NotFittedError(f"{self.name} must be fitted before predict")
        missing = set(self.feature_names_) - set(X.columns)
        if missing:
            raise ValueError(f"{self.name}: missing features {sorted(missing)}")
        # Re-order columns so callers cannot silently swap features.
        return np.asarray(self._predict(X[self.feature_names_]), dtype=np.float32)

    @property
    @abstractmethod
    def params(self) -> dict[str, Any]:
        """Hyper-parameters, for experiment tracking."""

    @abstractmethod
    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...

    @abstractmethod
    def _predict(self, X: pd.DataFrame) -> NDArray[np.floating[Any]]: ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, fitted={self.is_fitted})"
