"""Model interface: fit on normalised run-to-failure data, predict RUL per test unit."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class RULModel(ABC):
    """``fit`` receives normalised training rows (with ``rul``) and the sensors to use.

    ``predict_last`` returns one prediction per unit, for the last observed cycle,
    ordered by unit id.
    """

    name = "base"

    def __init__(self, sensors: list[str], window: int = 30, cap: float = 125.0, seed: int = 42):
        self.sensors = sensors
        self.window = window
        self.cap = cap
        self.seed = seed

    @abstractmethod
    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> RULModel: ...

    @abstractmethod
    def predict_last(self, df: pd.DataFrame) -> np.ndarray: ...

    def _clip(self, pred: np.ndarray) -> np.ndarray:
        return np.clip(pred, 0.0, self.cap)
