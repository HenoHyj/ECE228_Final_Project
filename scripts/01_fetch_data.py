"""Download NOAA Scripps Pier 6-minute data to data/raw/.

We always fetch at 6-min granularity; hourly is produced by local resampling
in scripts/02_preprocess.py.

Usage:
    python scripts/01_fetch_data.py --start 2023-01-01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.noaa_fetcher import PRODUCTS, fetch_all  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default=None, help="defaults to today UTC")
    p.add_argument("--cache-dir", default=str(ROOT / "data" / "raw"))
    p.add_argument("--products", nargs="*", default=list(PRODUCTS))
    p.add_argument("--sleep", type=float, default=0.5)
    args = p.parse_args()

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC") if args.end else pd.Timestamp.now(tz="UTC")

    print(f"Fetching products={args.products}")
    print(f"Range: {start.date()} -> {end.date()}")
    print(f"Cache: {args.cache_dir}")

    fetch_all(
        Path(args.cache_dir),
        start=start,
        end=end,
        products=tuple(args.products),
        sleep_between=args.sleep,
    )
    print("Done.")


if __name__ == "__main__":
    main()
