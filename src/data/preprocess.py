"""Convert raw NOAA JSON cache into per-resolution parquet files.

Outputs
-------
- ``data/processed/six_min.parquet`` : 6-min grid, NaNs preserved
- ``data/processed/hourly.parquet``  : 1-hour grid, NaNs preserved
- ``data/processed/scaler.json``     : per-channel (mean, std) fit on train split
- ``data/splits/{interval}.json``    : timestamp boundaries for train/val/test

Channels (in order): water_level, air_temp, water_temp, air_pressure, wind_u, wind_v
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CHANNELS = ("water_level", "air_temp", "water_temp", "air_pressure", "wind_u", "wind_v")

# Default chronological split. Override via CLI in scripts/02_preprocess.py if needed.
# CAVEAT: val (Sep–Dec) and test (Jan–) are each a single contiguous season, so
# model selection on val and the test metric both carry a seasonal bias. With
# only ~2.3 yr of data before val, a chronological hold-out is the honest choice;
# a rolling-origin / multi-season CV would remove the bias at a large compute cost.
DEFAULT_SPLIT = {
    "train_start": "2023-01-01",
    "val_start":   "2025-09-01",
    "test_start":  "2026-01-01",
}


@dataclass
class Splits:
    train: pd.DatetimeIndex
    val: pd.DatetimeIndex
    test: pd.DatetimeIndex


# --------------------------------------------------------------------------- #
# Per-product JSON loading
# --------------------------------------------------------------------------- #

def _load_product_jsons(product_dir: Path, value_keys: tuple[str, ...]) -> pd.DataFrame:
    """Load every cached JSON for one product into a long-format DataFrame.

    Returns columns: ['t', *value_keys] indexed by a UTC DatetimeIndex.
    """
    records: list[dict] = []
    for path in sorted(product_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        data = payload.get("data") or []
        for row in data:
            r = {"t": row["t"]}
            for k in value_keys:
                v = row.get(k)
                try:
                    r[k] = float(v) if v not in (None, "") else np.nan
                except (TypeError, ValueError):
                    r[k] = np.nan
            records.append(r)
    if not records:
        return pd.DataFrame(columns=["t", *value_keys]).set_index(
            pd.DatetimeIndex([], tz="UTC", name="t")
        )
    df = pd.DataFrame.from_records(records)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df = df.drop_duplicates(subset="t").set_index("t").sort_index()
    return df


def _load_wind(product_dir: Path) -> pd.DataFrame:
    """Wind comes back as (speed 's', direction-deg 'd'). Decompose to (u, v)."""
    raw = _load_product_jsons(product_dir, value_keys=("s", "d"))
    if raw.empty:
        return pd.DataFrame(columns=["wind_u", "wind_v"], index=raw.index)
    rad = np.deg2rad(raw["d"])
    out = pd.DataFrame(
        {"wind_u": raw["s"] * np.cos(rad), "wind_v": raw["s"] * np.sin(rad)},
        index=raw.index,
    )
    return out


PRODUCT_LOADERS = {
    "water_level":      lambda d: _load_product_jsons(d, ("v",)).rename(columns={"v": "water_level"}),
    "air_temperature":  lambda d: _load_product_jsons(d, ("v",)).rename(columns={"v": "air_temp"}),
    "water_temperature":lambda d: _load_product_jsons(d, ("v",)).rename(columns={"v": "water_temp"}),
    "air_pressure":     lambda d: _load_product_jsons(d, ("v",)).rename(columns={"v": "air_pressure"}),
    "wind":             _load_wind,
}


# --------------------------------------------------------------------------- #
# Merge + resample
# --------------------------------------------------------------------------- #

def load_all_products(raw_root: Path) -> pd.DataFrame:
    """Outer-join every product on UTC timestamp. NaNs preserved."""
    dfs: list[pd.DataFrame] = []
    for product, loader in PRODUCT_LOADERS.items():
        sub_dir = raw_root / product
        if not sub_dir.exists():
            raise FileNotFoundError(f"missing raw dir: {sub_dir}")
        dfs.append(loader(sub_dir))
    merged = pd.concat(dfs, axis=1, join="outer").sort_index()
    return merged[list(CHANNELS)]


def to_six_min_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Snap onto a continuous 6-min UTC grid spanning the data range.

    Observations within ±3 min of a grid point are assigned to that point;
    grid points with no nearby observation become NaN.
    """
    if df.empty:
        return df
    grid_start = df.index.min().floor("6min")
    grid_end = df.index.max().ceil("6min")
    grid = pd.date_range(grid_start, grid_end, freq="6min", tz="UTC", name="t")
    # nearest-merge within 3-minute tolerance, per-column
    snapped = df.reindex(grid, method="nearest", tolerance=pd.Timedelta("3min"))
    return snapped


def to_hourly_grid(six_min_df: pd.DataFrame, min_samples: int = 2) -> pd.DataFrame:
    """Resample the 6-min grid to hourly means. Hours with too few samples → NaN."""
    if six_min_df.empty:
        return six_min_df
    hourly_mean = six_min_df.resample("1h").mean()
    counts = six_min_df.notna().resample("1h").sum()
    hourly = hourly_mean.where(counts >= min_samples, other=np.nan)
    return hourly


# --------------------------------------------------------------------------- #
# Splits + normalization
# --------------------------------------------------------------------------- #

def compute_splits(index: pd.DatetimeIndex, *, split_cfg: dict) -> Splits:
    val_start = pd.Timestamp(split_cfg["val_start"], tz="UTC")
    test_start = pd.Timestamp(split_cfg["test_start"], tz="UTC")
    train_start = pd.Timestamp(split_cfg["train_start"], tz="UTC")
    return Splits(
        train=index[(index >= train_start) & (index < val_start)],
        val=index[(index >= val_start) & (index < test_start)],
        test=index[index >= test_start],
    )


def fit_scaler(df_train: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Per-channel mean and std on the train split, ignoring NaNs."""
    out: dict[str, dict[str, float]] = {}
    for c in df_train.columns:
        x = df_train[c].to_numpy()
        x = x[~np.isnan(x)]
        mean = float(x.mean()) if x.size else 0.0
        std = float(x.std()) if x.size else 1.0
        if std < 1e-8:
            std = 1.0
        out[c] = {"mean": mean, "std": std}
    return out


def apply_scaler(df: pd.DataFrame, scaler: dict[str, dict[str, float]]) -> pd.DataFrame:
    out = df.copy()
    for c, s in scaler.items():
        if c in out.columns:
            out[c] = (out[c] - s["mean"]) / s["std"]
    return out


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #

def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def save_scaler(scaler: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scaler, indent=2), encoding="utf-8")


def save_splits(splits: Splits, path: Path, *, split_cfg: dict) -> None:
    payload = {
        "boundaries": split_cfg,
        "train":  {"start": str(splits.train.min()),  "end": str(splits.train.max()),  "n": len(splits.train)},
        "val":    {"start": str(splits.val.min()),    "end": str(splits.val.max()),    "n": len(splits.val)},
        "test":   {"start": str(splits.test.min()),   "end": str(splits.test.max()),   "n": len(splits.test)},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
