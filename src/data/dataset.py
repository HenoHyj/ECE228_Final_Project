"""Sliding-window PyTorch Dataset over the processed parquet files.

A window is a contiguous (history + horizon) slice. NaNs are preserved as
zeros in the value tensor and the boolean mask records which entries are
real. Time is encoded as a float in hours since the window's start so that
``torchdiffeq`` can integrate the latent ODE on a regular grid.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Hours per step, used to convert step indices to ODE time.
_STEP_HOURS = {"hourly": 1.0, "six_min": 6.0 / 60.0}


@dataclass
class WindowConfig:
    interval: str         # 'hourly' or 'six_min'
    history: int
    horizon: int
    stride: int

    @property
    def step_hours(self) -> float:
        return _STEP_HOURS[self.interval]


class ScrippsWindows(Dataset):
    """Sliding windows over a chunk of the processed parquet."""

    def __init__(
        self,
        parquet_path: Path,
        scaler_path: Path,
        *,
        config: WindowConfig,
        split: str,                            # 'train' | 'val' | 'test'
        splits_path: Path,
        min_obs_fraction: float = 0.5,          # drop windows that are too sparse
    ) -> None:
        super().__init__()
        self.config = config
        self.split = split

        df = pd.read_parquet(parquet_path)
        splits_meta = json.loads(splits_path.read_text(encoding="utf-8"))
        bounds = splits_meta["boundaries"]
        train_start = pd.Timestamp(bounds["train_start"], tz="UTC")
        val_start   = pd.Timestamp(bounds["val_start"],   tz="UTC")
        test_start  = pd.Timestamp(bounds["test_start"],  tz="UTC")

        if split == "train":
            mask = (df.index >= train_start) & (df.index < val_start)
        elif split == "val":
            mask = (df.index >= val_start) & (df.index < test_start)
        elif split == "test":
            mask = df.index >= test_start
        else:
            raise ValueError(split)
        df = df.loc[mask]

        scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
        means = np.array([scaler[c]["mean"] for c in df.columns], dtype=np.float32)
        stds  = np.array([scaler[c]["std"]  for c in df.columns], dtype=np.float32)

        values = df.to_numpy(dtype=np.float32)                       # [T, D]
        valid_mask = ~np.isnan(values)                               # [T, D]
        # Normalize, then zero-fill NaNs. Mask records what was real.
        normed = (values - means) / stds
        normed[~valid_mask] = 0.0

        self.values = normed                                         # [T, D]
        self.mask = valid_mask                                       # [T, D]
        self.timestamps = df.index
        self.channels = list(df.columns)
        self.D = len(self.channels)
        self.window_len = config.history + config.horizon

        n_windows = max(0, (len(df) - self.window_len) // config.stride + 1)
        start_indices = np.arange(n_windows) * config.stride

        # Filter out windows that don't have enough real observations to be useful.
        keep: list[int] = []
        thresh = int(min_obs_fraction * self.window_len * self.D)
        for s in start_indices:
            if valid_mask[s : s + self.window_len].sum() >= thresh:
                keep.append(int(s))
        self._starts = np.asarray(keep, dtype=np.int64)

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return self._starts.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = int(self._starts[idx])
        h = self.config.history
        f = self.config.horizon
        dt = self.config.step_hours

        x_hist = torch.from_numpy(self.values[s : s + h]).float()              # [H, D]
        m_hist = torch.from_numpy(self.mask[s : s + h]).bool()                 # [H, D]
        x_fcst = torch.from_numpy(self.values[s + h : s + h + f]).float()      # [F, D]
        m_fcst = torch.from_numpy(self.mask[s + h : s + h + f]).bool()         # [F, D]

        t_hist = torch.arange(h, dtype=torch.float32) * dt                     # [H]
        t_fcst = (torch.arange(f, dtype=torch.float32) + h) * dt               # [F]

        return {
            "t_hist": t_hist,
            "x_hist": x_hist,
            "mask_hist": m_hist,
            "t_fcst": t_fcst,
            "x_fcst": x_fcst,
            "mask_fcst": m_fcst,
        }


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Batch windows whose time grids are identical (true for fixed-stride windows).

    Time vectors are shared across the batch (taken from item 0) because every
    window in a given dataset uses the same relative grid.
    """
    out = {
        "t_hist": batch[0]["t_hist"],
        "t_fcst": batch[0]["t_fcst"],
        "x_hist":    torch.stack([b["x_hist"]    for b in batch]),
        "mask_hist": torch.stack([b["mask_hist"] for b in batch]),
        "x_fcst":    torch.stack([b["x_fcst"]    for b in batch]),
        "mask_fcst": torch.stack([b["mask_fcst"] for b in batch]),
    }
    return out
