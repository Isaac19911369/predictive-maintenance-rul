"""NASA C-MAPSS turbofan degradation data: download, load, label.

Four sub-datasets with increasing difficulty:

======  ====================  ===========  ===========
subset  operating conditions  fault modes  train units
======  ====================  ===========  ===========
FD001   1                     1 (HPC)      100
FD002   6                     1 (HPC)      260
FD003   1                     2 (HPC, fan) 100
FD004   6                     2 (HPC, fan) 249
======  ====================  ===========  ===========

Training trajectories run to failure; test trajectories stop some time before
failure and the true remaining life of each test unit is given in ``RUL_FDxxx.txt``.

Source: A. Saxena, K. Goebel, D. Simon, N. Eklund, "Damage Propagation Modeling for
Aircraft Engine Run-to-Failure Simulation", PHM 2008. NASA Prognostics Data
Repository (public domain). Downloaded from a GitHub mirror of the original files.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

MIRROR = "https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/master/CMAPSSData"
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
SETTINGS = ["setting_1", "setting_2", "setting_3"]
SENSORS = [f"s_{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle", *SETTINGS, *SENSORS]


@dataclass
class Split:
    train: pd.DataFrame
    test: pd.DataFrame
    test_rul: np.ndarray  # true RUL at the last observed cycle of each test unit


def download(root: str | Path = "data/cmapss", force: bool = False) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for subset in SUBSETS:
        for kind in ("train", "test", "RUL"):
            name = f"{kind}_{subset}.txt"
            dest = root / name
            if dest.exists() and not force:
                continue
            resp = requests.get(f"{MIRROR}/{name}", timeout=120)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            log.info("downloaded %s", name)
    return root


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
    return df.astype({"unit": int, "cycle": int})


def add_rul(df: pd.DataFrame, cap: int | None = 125) -> pd.DataFrame:
    """Target for run-to-failure data: cycles until the last observed cycle.

    ``cap`` applies the common piecewise-linear target: early in life the engine is
    healthy and RUL is not identifiable from sensors, so it is clipped at ``cap``.
    """
    out = df.copy()
    last = out.groupby("unit")["cycle"].transform("max")
    out["rul"] = last - out["cycle"]
    if cap is not None:
        out["rul"] = out["rul"].clip(upper=cap)
    return out


def load(subset: str, root: str | Path = "data/cmapss", cap: int | None = 125) -> Split:
    if subset not in SUBSETS:
        raise ValueError(f"subset must be one of {SUBSETS}")
    root = Path(root)
    if not (root / f"train_{subset}.txt").exists():
        raise FileNotFoundError(f"{root} has no {subset} files. Run `cmapss-rul download` first.")
    train = add_rul(_read(root / f"train_{subset}.txt"), cap)
    test = _read(root / f"test_{subset}.txt")
    rul = pd.read_csv(root / f"RUL_{subset}.txt", header=None).iloc[:, 0].to_numpy(dtype=float)
    return Split(train, test, rul)
