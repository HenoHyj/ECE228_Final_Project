"""NOAA CO-OPS data fetcher for Station 9410230 (La Jolla, Scripps Pier).

The CO-OPS API caps each call to one product and a 31-day window for high-
resolution data. We always fetch at 6-minute granularity and resample locally
in preprocessing — this gives us a single raw cache that feeds both the
hourly and 6-min experimental tracks.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

STATION_ID = "9410230"
BASE_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
APPLICATION = "ECE228-Scripps-NeuralODE"

PRODUCTS = ("water_level", "air_temperature", "water_temperature", "air_pressure", "wind")
PRODUCT_EXTRA_PARAMS = {"water_level": {"datum": "MLLW"}}


@dataclass
class FetchSpec:
    product: str
    start: pd.Timestamp
    end: pd.Timestamp


def _monthly_chunks(start: pd.Timestamp, end: pd.Timestamp):
    """Yield (chunk_start, chunk_end, 'YYYY-MM') covering [start, end]."""
    cur = pd.Timestamp(year=start.year, month=start.month, day=1, tz="UTC")
    while cur <= end:
        nxt = cur + pd.offsets.MonthBegin(1)
        chunk_start = max(cur, start)
        chunk_end = min(nxt - pd.Timedelta(seconds=1), end)
        yield chunk_start, chunk_end, cur.strftime("%Y-%m")
        cur = nxt


def _request_with_retry(params: dict, *, max_retries: int = 4, base_sleep: float = 1.0) -> dict:
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            r = requests.get(BASE_URL, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            if 500 <= r.status_code < 600:
                last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            else:
                # 4xx — often "no data for this range"; surface upstream as JSON.
                return r.json()
        except (requests.ConnectionError, requests.Timeout) as e:
            last_err = e
        time.sleep(base_sleep * (2 ** attempt))
    raise RuntimeError(f"CO-OPS request failed after {max_retries} attempts: {last_err}")


def fetch_chunk(spec: FetchSpec, cache_dir: Path, *, sleep_between: float = 0.5) -> None:
    """Fetch every monthly chunk for (spec.product) and cache raw JSON to disk."""
    product_dir = cache_dir / spec.product
    product_dir.mkdir(parents=True, exist_ok=True)

    chunks = list(_monthly_chunks(spec.start, spec.end))
    for idx, (cs, ce, label) in enumerate(chunks, start=1):
        out_path = product_dir / f"{label}.json"
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"[{spec.product:<17}] {idx:>3}/{len(chunks)} {label} cached", flush=True)
            continue

        params = {
            "product": spec.product,
            "station": STATION_ID,
            "begin_date": cs.strftime("%Y%m%d %H:%M"),
            "end_date": ce.strftime("%Y%m%d %H:%M"),
            "units": "metric",
            "time_zone": "gmt",
            "format": "json",
            "application": APPLICATION,
            **PRODUCT_EXTRA_PARAMS.get(spec.product, {}),
        }
        payload = _request_with_retry(params)
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        n_records = len(payload.get("data", []))
        print(f"[{spec.product:<17}] {idx:>3}/{len(chunks)} {label} fetched ({n_records} rows)",
              flush=True)
        time.sleep(sleep_between)


def fetch_all(
    cache_dir: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    products: tuple[str, ...] = PRODUCTS,
    sleep_between: float = 0.5,
) -> None:
    """Download every (product, month) into cache_dir at 6-min resolution."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    for product in products:
        spec = FetchSpec(product=product, start=start, end=end)
        fetch_chunk(spec, cache_dir, sleep_between=sleep_between)
