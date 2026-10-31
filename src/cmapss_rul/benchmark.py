"""Train every model on every subset and evaluate on the official test units."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .data import load
from .features import RegimeNormalizer
from .metrics import summary
from .models import MODELS

log = logging.getLogger(__name__)


@dataclass
class Result:
    subset: str
    model: str
    seed: int
    rmse: float
    nasa_score: float
    mae: float
    late_pct: float
    seconds: float


def split_units(train, valid_frac: float, seed: int):
    units = np.array(sorted(train["unit"].unique()))
    rng = np.random.default_rng(seed)
    valid_units = set(rng.choice(units, size=max(1, int(len(units) * valid_frac)), replace=False))
    mask = train["unit"].isin(valid_units)
    return train[~mask], train[mask]


def run(
    subset: str,
    model_names: list[str],
    root: str = "data/cmapss",
    window: int = 30,
    cap: int = 125,
    valid_frac: float = 0.2,
    seed: int = 42,
) -> tuple[list[Result], dict]:
    split = load(subset, root, cap)
    norm = RegimeNormalizer().fit(split.train)
    train = norm.transform(split.train)
    test = norm.transform(split.test)
    fit_part, valid_part = split_units(train, valid_frac, seed)
    y_true = np.minimum(split.test_rul, cap)  # same piecewise target as training
    results, preds = [], {}
    for name in model_names:
        start = time.perf_counter()
        model = MODELS[name](norm.sensors, window=window, cap=cap, seed=seed)
        model.fit(fit_part, valid_part)
        pred = model.predict_last(test)
        m = summary(y_true, pred)
        results.append(Result(subset, name, seed, **m, seconds=time.perf_counter() - start))
        preds[name] = pred
        log.info("%s %-8s RMSE=%.2f score=%.0f", subset, name, m["rmse"], m["nasa_score"])
    preds["true"] = y_true
    return results, preds


def write_report(results: list[Result], out_dir: str | Path) -> str:
    """Markdown table of RMSE and NASA score; mean ± std when several seeds were run."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    subsets = sorted({r.subset for r in results})
    models = list(dict.fromkeys(r.model for r in results))
    n_seeds = len({r.seed for r in results})

    def cell(values: list[float], fmt: str) -> str:
        if not values:
            return "–"
        mean = format(float(np.mean(values)), fmt)
        return f"{mean} ± {format(float(np.std(values)), fmt)}" if n_seeds > 1 else mean

    head = "| Model | " + " | ".join(f"{s} RMSE | {s} Score" for s in subsets) + " |"
    rows = [head, "|---|" + "---|---|" * len(subsets)]
    for m in models:
        cells = []
        for s in subsets:
            rs = [r for r in results if r.subset == s and r.model == m]
            cells += [cell([r.rmse for r in rs], ".2f"), cell([r.nasa_score for r in rs], ",.0f")]
        rows.append(f"| {m} | " + " | ".join(cells) + " |")
    table = "\n".join(rows) + "\n"
    (out_dir / "results.md").write_text(table)
    (out_dir / "results.json").write_text(json.dumps([asdict(r) for r in results], indent=2))
    return table
