import numpy as np
import pandas as pd
import pytest

from cmapss_rul.data import SENSORS, SETTINGS, add_rul


def _fleet(n_units: int, seed: int, regimes: int = 1, truncate: bool = False) -> pd.DataFrame:
    """Synthetic C-MAPSS-like fleet: sensors drift exponentially towards failure."""
    rng = np.random.default_rng(seed)
    rows = []
    settings = [(0.0, 0.0, 100.0), (20.0, 0.7, 100.0), (42.0, 0.8, 60.0)][:regimes]
    for unit in range(1, n_units + 1):
        life = int(rng.integers(120, 220))
        length = int(rng.integers(40, life)) if truncate else life
        gain = rng.normal(1.0, 0.1, len(SENSORS))
        for cycle in range(1, length + 1):
            setting = settings[int(rng.integers(len(settings)))]
            offset = 10.0 * settings.index(setting)
            wear = np.exp((cycle - life) / 40.0)
            sensors = offset + gain * wear + rng.normal(0, 0.05, len(SENSORS))
            sensors[[0, 4, 9]] = 1.0 + offset  # constant sensors, like s_1, s_5, s_10
            rows.append([unit, cycle, *setting, *sensors, life - length])
    df = pd.DataFrame(rows, columns=["unit", "cycle", *SETTINGS, *SENSORS, "_rul_end"])
    return df


@pytest.fixture(scope="session")
def fleet():
    train = add_rul(_fleet(30, 0, regimes=3).drop(columns="_rul_end"))
    test_raw = _fleet(12, 1, regimes=3, truncate=True)
    rul = test_raw.groupby("unit")["_rul_end"].first().to_numpy(float)
    return train, test_raw.drop(columns="_rul_end"), rul
