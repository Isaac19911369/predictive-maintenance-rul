import numpy as np
import pandas as pd
import pytest

from cmapss_rul.data import add_rul
from cmapss_rul.features import RegimeNormalizer, last_cycle, temporal_features


def test_add_rul_piecewise():
    df = pd.DataFrame({"unit": [1] * 200, "cycle": range(1, 201)})
    out = add_rul(df, cap=125)
    assert out["rul"].iloc[-1] == 0
    assert out["rul"].max() == 125
    assert add_rul(df, cap=None)["rul"].iloc[0] == 199


def test_regime_normalizer_removes_offsets_and_constant_sensors(fleet):
    train, _, _ = fleet
    norm = RegimeNormalizer().fit(train)
    out = norm.transform(train)
    assert {"s_1", "s_5", "s_10"}.isdisjoint(norm.sensors)
    assert abs(out[norm.sensors].mean().mean()) < 1e-6


def test_rolling_features_match_naive_computation():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "unit": [1] * 12 + [2] * 7,
            "cycle": list(range(1, 13)) + list(range(1, 8)),
            "s_2": rng.normal(size=19),
        }
    )
    f = temporal_features(df, ["s_2"], window=5)
    for unit, g in df.groupby("unit"):
        y = g["s_2"].to_numpy()
        fu = f[f["unit"] == unit].reset_index(drop=True)
        for i in range(len(y)):
            w = y[max(0, i - 4) : i + 1]
            assert fu.loc[i, "s_2_mean"] == pytest.approx(w.mean())
            assert fu.loc[i, "s_2_std"] == pytest.approx(w.std(), abs=1e-6)
            exp_slope = np.polyfit(np.arange(len(w)), w, 1)[0] if len(w) > 1 else 0.0
            assert fu.loc[i, "s_2_slope"] == pytest.approx(exp_slope, abs=1e-6)


def test_last_cycle_one_row_per_unit(fleet):
    _, test, _ = fleet
    f = last_cycle(temporal_features(test, ["s_2"], window=10))
    assert len(f) == test["unit"].nunique()
    assert (f["cycle"].to_numpy() == test.groupby("unit")["cycle"].max().to_numpy()).all()
