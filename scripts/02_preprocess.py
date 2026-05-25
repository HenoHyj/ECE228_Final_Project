"""Build processed parquet files + scaler + split boundaries.

Run after scripts/01_fetch_data.py completes.

Usage:
    python scripts/02_preprocess.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.preprocess import (  # noqa: E402
    DEFAULT_SPLIT,
    compute_splits,
    fit_scaler,
    load_all_products,
    save_parquet,
    save_scaler,
    save_splits,
    to_hourly_grid,
    to_six_min_grid,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir",       default=str(ROOT / "data" / "raw"))
    p.add_argument("--processed-dir", default=str(ROOT / "data" / "processed"))
    p.add_argument("--splits-dir",    default=str(ROOT / "data" / "splits"))
    p.add_argument("--train-start", default=DEFAULT_SPLIT["train_start"])
    p.add_argument("--val-start",   default=DEFAULT_SPLIT["val_start"])
    p.add_argument("--test-start",  default=DEFAULT_SPLIT["test_start"])
    args = p.parse_args()

    raw_root = Path(args.raw_dir)
    processed_root = Path(args.processed_dir)
    splits_root = Path(args.splits_dir)
    split_cfg = {"train_start": args.train_start, "val_start": args.val_start, "test_start": args.test_start}

    print(f"Loading raw JSONs from {raw_root} ...")
    raw_df = load_all_products(raw_root)
    print(f"  rows={len(raw_df):,}  range={raw_df.index.min()} -> {raw_df.index.max()}")
    print(f"  per-channel non-NaN counts:\n{raw_df.notna().sum().to_string()}")

    print("Building 6-min grid ...")
    six_min = to_six_min_grid(raw_df)
    print(f"  rows={len(six_min):,}")
    save_parquet(six_min, processed_root / "six_min.parquet")

    print("Building hourly grid ...")
    hourly = to_hourly_grid(six_min)
    print(f"  rows={len(hourly):,}")
    save_parquet(hourly, processed_root / "hourly.parquet")

    print("Computing splits ...")
    for interval, df in [("hourly", hourly), ("six_min", six_min)]:
        splits = compute_splits(df.index, split_cfg=split_cfg)
        save_splits(splits, splits_root / f"{interval}.json", split_cfg=split_cfg)
        print(f"  {interval}: train={len(splits.train):,} val={len(splits.val):,} test={len(splits.test):,}")

    print("Fitting scaler on hourly TRAIN split ...")
    train_splits = compute_splits(hourly.index, split_cfg=split_cfg)
    train_df = hourly.loc[train_splits.train]
    scaler = fit_scaler(train_df)
    save_scaler(scaler, processed_root / "scaler.json")
    for c, s in scaler.items():
        print(f"  {c:<14} mean={s['mean']:+.4f}  std={s['std']:.4f}")

    print("Done.")


if __name__ == "__main__":
    main()
