"""1-D CNN over the raw (normalised) sensor sequence of the last ``window`` cycles.

Architecture is inspired by the C-MAPSS CNN of Li, Ding & Sun (2018): stacked temporal
convolutions over all sensors and a small dense head. The target is scaled to [0, 1]
(divided by the RUL cap) for stable optimisation.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .base import RULModel


def sequence_windows(
    df: pd.DataFrame, sensors: list[str], window: int, last_only: bool = False
) -> tuple[np.ndarray, np.ndarray | None]:
    """Sliding windows per unit; short units are left-padded with their first row."""
    xs, ys = [], []
    has_y = "rul" in df
    for _, g in df.sort_values(["unit", "cycle"]).groupby("unit", sort=True):
        x = g[sensors].to_numpy(np.float32)
        if len(x) < window:
            x = np.vstack([np.repeat(x[:1], window - len(x), axis=0), x])
        y = g["rul"].to_numpy(np.float32) if has_y else None
        if y is not None and len(y) < window:
            y = np.concatenate([np.full(window - len(y), y[0]), y])
        starts = [len(x) - window] if last_only else range(len(x) - window + 1)
        for s in starts:
            xs.append(x[s : s + window])
            if y is not None:
                ys.append(y[s + window - 1])
    return np.stack(xs), (np.asarray(ys, np.float32) if has_y else None)


class _Net(nn.Module):
    def __init__(self, n_features: int, window: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 32, 9, padding=4),
            nn.ReLU(),
            nn.Conv1d(32, 32, 9, padding=4),
            nn.ReLU(),
            nn.Conv1d(32, 32, 9, padding=4),
            nn.ReLU(),
            nn.Conv1d(32, 8, 3, padding=1),
            nn.Tanh(),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(8 * window, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x.transpose(1, 2))).squeeze(-1)


class CNNModel(RULModel):
    name = "cnn1d"

    def __init__(
        self,
        *args,
        epochs: int = 60,
        batch_size: int = 256,
        lr: float = 1e-3,
        patience: int = 10,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.epochs, self.batch_size, self.lr, self.patience = epochs, batch_size, lr, patience

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> CNNModel:
        torch.manual_seed(self.seed)
        x, y = sequence_windows(train, self.sensors, self.window)
        self.net = _Net(len(self.sensors), self.window)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(x), torch.from_numpy(y / self.cap)),
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )
        if valid is not None:
            xv, yv = sequence_windows(valid, self.sensors, self.window)
            xv, yv = torch.from_numpy(xv), torch.from_numpy(yv)
        best, best_state, bad = np.inf, None, 0
        loss_fn = nn.MSELoss()
        for _ in range(self.epochs):
            self.net.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(self.net(xb), yb)
                loss.backward()
                opt.step()
            if valid is None:
                continue
            self.net.eval()
            with torch.no_grad():
                v = float(torch.sqrt(loss_fn(self.net(xv) * self.cap, yv)))
            sched.step(v)
            if v < best - 1e-3:
                best, best_state, bad = v, copy.deepcopy(self.net.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.valid_rmse = best
        return self

    @torch.no_grad()
    def predict_last(self, df: pd.DataFrame) -> np.ndarray:
        self.net.eval()
        x, _ = sequence_windows(
            df.drop(columns="rul", errors="ignore"), self.sensors, self.window, last_only=True
        )
        return self._clip(self.net(torch.from_numpy(x)).numpy() * self.cap)
