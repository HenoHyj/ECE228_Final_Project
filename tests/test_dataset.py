"""Sanity checks for the preprocessing + windowing pipeline.

These tests fabricate a tiny synthetic NOAA-shaped DataFrame and run it
through the full pipeline (resample → split → scale → window) to verify
shapes, masks, and time vectors.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.dataset import TFEAT_DIM, ScrippsWindows, WindowConfig, collate
from src.data.preprocess import (
    CHANNELS,
    compute_splits,
    fit_scaler,
    save_parquet,
    save_scaler,
    save_splits,
    to_hourly_grid,
    to_six_min_grid,
)


def _synthetic_frame(start: str = "2024-01-01", days: int = 40) -> pd.DataFrame:
    """6-min grid with a couple of NaN gaps for realism."""
    idx = pd.date_range(start, periods=days * 24 * 10, freq="6min", tz="UTC", name="t")
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((len(idx), len(CHANNELS))).astype(np.float32)
    df = pd.DataFrame(arr, index=idx, columns=list(CHANNELS))
    # Punch a 6-hour gap into water_temp on day 5 to exercise mask handling.
    gap = (idx >= idx[0] + pd.Timedelta(days=5)) & (idx < idx[0] + pd.Timedelta(days=5, hours=6))
    df.loc[gap, "water_temp"] = np.nan
    return df


def test_resample_preserves_nans():
    df = _synthetic_frame()
    six = to_six_min_grid(df)
    hourly = to_hourly_grid(six)
    assert hourly["water_temp"].isna().any(), "NaN gap should survive resample"
    assert hourly.shape[1] == len(CHANNELS)


def test_window_shapes_and_time_grid(tmp_path: Path):
    df = _synthetic_frame()
    hourly = to_hourly_grid(to_six_min_grid(df))

    save_parquet(hourly, tmp_path / "hourly.parquet")
    split_cfg = {
        "train_start": str(hourly.index.min().date()),
        "val_start":   str((hourly.index.min() + pd.Timedelta(days=30)).date()),
        "test_start":  str((hourly.index.min() + pd.Timedelta(days=35)).date()),
    }
    splits = compute_splits(hourly.index, split_cfg=split_cfg)
    save_splits(splits, tmp_path / "splits.json", split_cfg=split_cfg)
    save_scaler(fit_scaler(hourly.loc[splits.train]), tmp_path / "scaler.json")

    cfg = WindowConfig(interval="hourly", history=24, horizon=6, stride=4)
    ds = ScrippsWindows(
        parquet_path=tmp_path / "hourly.parquet",
        scaler_path=tmp_path / "scaler.json",
        config=cfg,
        split="train",
        splits_path=tmp_path / "splits.json",
        min_obs_fraction=0.5,
    )
    assert len(ds) > 0
    sample = ds[0]
    assert sample["x_hist"].shape == (24, len(CHANNELS))
    assert sample["x_fcst"].shape == (6, len(CHANNELS))
    assert sample["mask_hist"].dtype == torch.bool
    # Tidal clock features: [H, 2K] and [F, 2K], bounded in [-1, 1].
    assert sample["tfeat_hist"].shape == (24, TFEAT_DIM)
    assert sample["tfeat_fcst"].shape == (6, TFEAT_DIM)
    assert sample["tfeat_hist"].abs().max().item() <= 1.0 + 1e-5
    # Time grid is NORMALIZED by the history span (24h here) so the ODE sees O(1)
    # t: contiguous unit index-steps → spacing 1/24; forecast starts at 1.0.
    assert sample["t_hist"][0].item() == 0.0
    assert sample["t_hist"][1].item() - sample["t_hist"][0].item() == pytest.approx(1.0 / 24)
    assert sample["t_fcst"][0].item() == pytest.approx(1.0)


def test_collate_stacks_correctly(tmp_path: Path):
    df = _synthetic_frame()
    hourly = to_hourly_grid(to_six_min_grid(df))
    save_parquet(hourly, tmp_path / "hourly.parquet")
    split_cfg = {
        "train_start": str(hourly.index.min().date()),
        "val_start":   str((hourly.index.min() + pd.Timedelta(days=30)).date()),
        "test_start":  str((hourly.index.min() + pd.Timedelta(days=35)).date()),
    }
    splits = compute_splits(hourly.index, split_cfg=split_cfg)
    save_splits(splits, tmp_path / "splits.json", split_cfg=split_cfg)
    save_scaler(fit_scaler(hourly.loc[splits.train]), tmp_path / "scaler.json")

    cfg = WindowConfig(interval="hourly", history=24, horizon=6, stride=4)
    ds = ScrippsWindows(
        parquet_path=tmp_path / "hourly.parquet",
        scaler_path=tmp_path / "scaler.json",
        config=cfg,
        split="train",
        splits_path=tmp_path / "splits.json",
        min_obs_fraction=0.5,
    )
    batch = collate([ds[0], ds[1], ds[2]])
    assert batch["x_hist"].shape == (3, 24, len(CHANNELS))
    assert batch["mask_fcst"].shape == (3, 6, len(CHANNELS))
    assert batch["t_hist"].shape == (24,)
    # Tidal features are per-window (absolute-time dependent) → stacked, not shared.
    assert batch["tfeat_hist"].shape == (3, 24, TFEAT_DIM)
    assert batch["tfeat_fcst"].shape == (3, 6, TFEAT_DIM)


def test_mask_marks_gap_as_invalid(tmp_path: Path):
    """The synthetic gap should produce False entries in the mask."""
    df = _synthetic_frame()
    hourly = to_hourly_grid(to_six_min_grid(df))
    save_parquet(hourly, tmp_path / "hourly.parquet")
    split_cfg = {
        "train_start": str(hourly.index.min().date()),
        "val_start":   str((hourly.index.min() + pd.Timedelta(days=30)).date()),
        "test_start":  str((hourly.index.min() + pd.Timedelta(days=35)).date()),
    }
    splits = compute_splits(hourly.index, split_cfg=split_cfg)
    save_splits(splits, tmp_path / "splits.json", split_cfg=split_cfg)
    save_scaler(fit_scaler(hourly.loc[splits.train]), tmp_path / "scaler.json")

    cfg = WindowConfig(interval="hourly", history=24, horizon=6, stride=1)
    ds = ScrippsWindows(
        parquet_path=tmp_path / "hourly.parquet",
        scaler_path=tmp_path / "scaler.json",
        config=cfg,
        split="train",
        splits_path=tmp_path / "splits.json",
        min_obs_fraction=0.0,        # do not filter — we want to see the gap
    )
    any_missing = False
    for i in range(len(ds)):
        if not ds[i]["mask_hist"].all():
            any_missing = True
            break
    assert any_missing, "Expected at least one window with a missing observation"
