"""Feature engineering in pandas (the PySpark version lives in ``spark_features``).

Steps
-----
1. **Operating-condition normalisation.** FD002/FD004 mix six flight regimes that
   shift every sensor. Regimes are identified by clustering the three settings
   (rounded settings form exact clusters in C-MAPSS) and each sensor is
   z-scored per regime using training statistics only.
2. **Sensor selection.** Sensors that are (near-)constant after normalisation carry
   no degradation signal and are dropped.
3. **Temporal features.** Per unit: rolling mean, rolling std and the least-squares
   slope over the last ``window`` cycles, plus the cycle counter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import SENSORS, SETTINGS


def regime_keys(df: pd.DataFrame) -> pd.Series:
    """Discrete operating regime from rounded settings (exact for C-MAPSS)."""
    r = df[SETTINGS].round({"setting_1": 0, "setting_2": 1, "setting_3": 0})
    return r.astype(str).agg("|".join, axis=1)


@dataclass
class RegimeNormalizer:
    """Per-regime z-scoring of sensors, fitted on training data."""

    min_std: float = 1e-4
    stats: dict[str, pd.DataFrame] = field(default_factory=dict)
    sensors: list[str] = field(default_factory=list)

    def fit(self, df: pd.DataFrame) -> RegimeNormalizer:
        keys = regime_keys(df)
        grouped = df[SENSORS].groupby(keys)
        self.stats = {"mean": grouped.mean(), "std": grouped.std(ddof=0)}
        normed = self._apply(df, keys)
        spread = normed[SENSORS].std(ddof=0)
        self.sensors = [s for s in SENSORS if spread[s] > self.min_std]
        return self

    def _apply(self, df: pd.DataFrame, keys: pd.Series) -> pd.DataFrame:
        mean = self.stats["mean"].reindex(keys).to_numpy()
        std = self.stats["std"].reindex(keys).to_numpy()
        std = np.where((std < self.min_std) | np.isnan(std), 1.0, std)
        out = df.copy()
        out[SENSORS] = (df[SENSORS].to_numpy() - mean) / std
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        keys = regime_keys(df)
        unknown = set(keys) - set(self.stats["mean"].index)
        if unknown:
            raise ValueError(f"unseen operating regimes: {sorted(unknown)[:3]}")
        return self._apply(df, keys)


def temporal_features(df: pd.DataFrame, sensors: list[str], window: int = 30) -> pd.DataFrame:
    """Rolling mean / std / slope per unit. Output keeps ``unit``, ``cycle`` (and ``rul``)."""
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    t = df["cycle"].astype(float)
    helper = pd.DataFrame({"unit": df["unit"], "t": t, "t2": t * t, "one": 1.0})
    for s in sensors:
        helper[s] = df[s]
        helper[f"{s}__ty"] = t * df[s]
        helper[f"{s}__y2"] = df[s] ** 2

    # trailing-window sums per unit; every statistic below is derived from them
    sums = (
        helper.groupby("unit", sort=False)
        .rolling(window, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    n, st, st2 = sums["one"], sums["t"], sums["t2"]
    denom = n * st2 - st**2
    feats = {"cycle": df["cycle"].to_numpy()}
    for s in sensors:
        sy, sty, sy2 = sums[s], sums[f"{s}__ty"], sums[f"{s}__y2"]
        mean = sy / n
        var = (sy2 / n - mean**2).clip(lower=0.0)
        slope = np.where(denom > 0, (n * sty - st * sy) / denom.where(denom > 0, 1.0), 0.0)
        feats[f"{s}_mean"] = mean.to_numpy()
        feats[f"{s}_std"] = np.sqrt(var).to_numpy()
        feats[f"{s}_slope"] = np.asarray(slope)
    out = pd.DataFrame(feats)
    out.insert(0, "unit", df["unit"].to_numpy())
    if "rul" in df:
        out["rul"] = df["rul"].to_numpy()
    return out


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in ("unit", "rul")]


def last_cycle(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per unit: the most recent cycle (what is scored for test units)."""
    idx = frame.groupby("unit")["cycle"].idxmax()
    return frame.loc[idx].sort_values("unit").reset_index(drop=True)
