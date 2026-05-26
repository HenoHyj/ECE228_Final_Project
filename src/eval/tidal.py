"""Tidal-harmonic baseline for water_level.

Least-squares fit of dominant tidal constituents on ABSOLUTE timestamps, using
numpy only (utide / statsmodels are not installed). This is a strong domain
reference: water level is dominated by a handful of deterministic constituents,
so this baseline is dropout-invariant (it never looks at sensor inputs) and
contextualizes whether the learned models add anything on the tidal channel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Principal, well-separated tidal constituents (period in hours): M2, S2
# (semidiurnal) and K1, O1 (diurnal) — the four dominant for the La Jolla mixed
# tide. Adding near-degenerate neighbours (K2≈S2, P1≈K1) makes the least-squares
# design matrix severely ill-conditioned over a ~2.7-yr record, so we omit them.
CONSTITUENTS_H = (
    12.4206012,   # M2
    12.0,         # S2
    23.93447213,  # K1
    25.81933871,  # O1
)


def _abs_hours(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Wall-clock hours since the Unix epoch (phase reference is arbitrary).

    Uses Timedelta rather than `.asi8`: the index is datetime64[us], so `.asi8`
    would be microseconds and scale the tidal phase 1000× wrong.
    """
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    return np.asarray((timestamps - epoch) / pd.Timedelta("1h"), dtype=np.float64)


def _design(abs_hours: np.ndarray, periods) -> np.ndarray:
    cols = [np.ones_like(abs_hours)]
    for period in periods:
        ang = 2.0 * np.pi * abs_hours / period
        cols.append(np.sin(ang))
        cols.append(np.cos(ang))
    return np.stack(cols, axis=1)


def fit_harmonics(timestamps: pd.DatetimeIndex, values, periods=CONSTITUENTS_H) -> np.ndarray:
    """Fit [mean + Σ (a·sin + b·cos)] to (timestamps, values); NaNs ignored."""
    abs_h = _abs_hours(timestamps)
    v = np.asarray(values, dtype=np.float64)
    ok = ~np.isnan(v)
    design = _design(abs_h[ok], periods)
    # rcond drops near-degenerate singular directions so out-of-sample
    # predictions can't blow up from collinear constituents.
    coeffs, *_ = np.linalg.lstsq(design, v[ok], rcond=1e-8)
    return coeffs


def predict_harmonics(coeffs: np.ndarray, timestamps: pd.DatetimeIndex,
                      periods=CONSTITUENTS_H) -> np.ndarray:
    """Predict values at arbitrary timestamps (original/denormalized units)."""
    return _design(_abs_hours(timestamps), periods) @ coeffs
