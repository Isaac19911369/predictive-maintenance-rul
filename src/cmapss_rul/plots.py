"""Figures for the README."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def prediction_figure(preds: dict[str, np.ndarray], subset: str, path: Path) -> Path:
    """Test units sorted by true RUL, with each model's prediction."""
    path.parent.mkdir(parents=True, exist_ok=True)
    order = np.argsort(preds["true"])
    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=130)
    ax.plot(preds["true"][order], color="black", lw=2, label="true RUL (capped at 125)")
    colors = {"ridge": "#9aa5b1", "lightgbm": "#2f7ed8", "cnn1d": "#e4572e"}
    for name, p in preds.items():
        if name == "true":
            continue
        ax.scatter(np.arange(len(p)), p[order], s=9, alpha=0.75, label=name, color=colors.get(name))
    ax.set_xlabel("test unit (sorted by true RUL)")
    ax.set_ylabel("RUL [cycles]")
    ax.set_title(f"C-MAPSS {subset}: predicted vs. true remaining useful life")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
