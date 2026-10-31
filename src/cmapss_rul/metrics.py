"""Evaluation metrics for RUL prediction."""

from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sqrt(np.mean(d**2)))


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """PHM08 asymmetric scoring function (lower is better).

    Late predictions (over-estimating remaining life) are penalised harder than early
    ones, because a missed maintenance is more expensive than an early one:
    ``s = sum(exp(-d/13) - 1)`` for ``d < 0`` and ``sum(exp(d/10) - 1)`` for ``d >= 0``,
    with ``d = predicted - true``.
    """
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


def summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return {
        "rmse": rmse(y_true, y_pred),
        "nasa_score": nasa_score(y_true, y_pred),
        "mae": float(np.mean(np.abs(d))),
        "late_pct": float(100 * np.mean(d > 0)),
    }
