"""Forecast evaluation metrics (pure functions on numpy arrays)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _pair(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if true.shape != pred.shape:
        raise ValueError(f"shape mismatch {true.shape} vs {pred.shape}")
    if true.size == 0:
        raise ValueError("metrics need at least one observation")
    return true, pred


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    true, pred = _pair(y_true, y_pred)
    return float(np.mean(np.abs(true - pred)))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    true, pred = _pair(y_true, y_pred)
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def bias(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean error (forecast - actual): positive means over-forecasting."""
    true, pred = _pair(y_true, y_pred)
    return float(np.mean(pred - true))


def skill_score(model_error: float, reference_error: float) -> float:
    """``1 - error/reference``: 0 = as good as the reference, 1 = perfect."""
    if reference_error <= 0:
        return 0.0
    return 1.0 - model_error / reference_error


def regression_report(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    return {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred), "bias": bias(y_true, y_pred)}


def pinball_loss(y_true: ArrayLike, y_quantile: ArrayLike, quantile: float) -> float:
    true, pred = _pair(y_true, y_quantile)
    diff = true - pred
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


def interval_coverage(y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> float:
    true = np.asarray(y_true)
    return float(np.mean((true >= np.asarray(lower)) & (true <= np.asarray(upper))))


def quantile_report(
    y_true: ArrayLike, quantile_preds: NDArray[Any], quantiles: Sequence[float]
) -> dict[str, float]:
    """Pinball loss per quantile and the empirical coverage of the outer interval."""
    report = {
        f"pinball_q{int(q * 100):02d}": pinball_loss(y_true, quantile_preds[:, i], q)
        for i, q in enumerate(quantiles)
    }
    report["interval_coverage"] = interval_coverage(
        y_true, quantile_preds[:, 0], quantile_preds[:, -1]
    )
    report["interval_nominal"] = float(quantiles[-1] - quantiles[0])
    return report
