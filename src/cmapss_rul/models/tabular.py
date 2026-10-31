"""Feature-based models on rolling statistics."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..features import feature_columns, last_cycle, temporal_features
from .base import RULModel


class _FeatureModel(RULModel):
    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        return temporal_features(df, self.sensors, self.window)

    def predict_last(self, df: pd.DataFrame) -> np.ndarray:
        last = last_cycle(self._features(df))
        return self._clip(self.model.predict(last[self.columns]))


class RidgeModel(_FeatureModel):
    """Linear baseline on standardised rolling features."""

    name = "ridge"

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> RidgeModel:
        f = self._features(train)
        self.columns = feature_columns(f)
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        self.model.fit(f[self.columns], f["rul"])
        return self


class LightGBMModel(_FeatureModel):
    """Gradient-boosted trees with early stopping on held-out units."""

    name = "lightgbm"

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> LightGBMModel:
        f = self._features(train)
        self.columns = feature_columns(f)
        self.model = lgb.LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=50,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=self.seed,
            verbose=-1,
        )
        fit_kwargs = {}
        if valid is not None:
            fv = self._features(valid)
            fit_kwargs = {
                "eval_set": [(fv[self.columns], fv["rul"])],
                "callbacks": [lgb.early_stopping(100, verbose=False)],
            }
        self.model.fit(f[self.columns], f["rul"], **fit_kwargs)
        return self
