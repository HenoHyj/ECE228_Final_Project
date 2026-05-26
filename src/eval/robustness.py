"""Artificial input-dropout robustness sweep.

Dropout modes (the targets always stay clean):

- ``iid``    — i.i.d. Bernoulli(p) over (time, channel). Easy for the LSTM:
  linear interpolation across single missing points is nearly lossless.
- ``block``  — drop a CONTIGUOUS run of ``round(p·H)`` timesteps per channel at
  a random start (a sensor outage). Interpolation across a long gap fails, so
  this is where a continuous-time model should show an advantage.
- ``sensor`` — drop entire channels (all timesteps) with prob p (sensor death).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class DroppedBatch:
    x_hist: torch.Tensor
    mask_hist: torch.Tensor
    t_hist: torch.Tensor
    t_fcst: torch.Tensor
    x_fcst: torch.Tensor
    mask_fcst: torch.Tensor
    tfeat_hist: torch.Tensor | None = None
    tfeat_fcst: torch.Tensor | None = None


def _keep_mask(shape, rate, mode, device, generator) -> torch.Tensor:
    """Boolean keep-mask [B,H,D] for the requested dropout mode."""
    B, H, D = shape
    if mode == "iid":
        return torch.bernoulli(
            torch.full(shape, 1.0 - rate, device=device), generator=generator
        ).bool()
    if mode == "sensor":
        # Drop a whole channel (all timesteps) with probability `rate`.
        chan_drop = torch.rand((B, 1, D), device=device, generator=generator) < rate
        return ~chan_drop.expand(B, H, D)
    if mode == "block":
        # Drop a contiguous run of L steps per (sample, channel) at a random start.
        L = int(round(rate * H))
        if L <= 0:
            return torch.ones(shape, dtype=torch.bool, device=device)
        if L >= H:
            return torch.zeros(shape, dtype=torch.bool, device=device)
        starts = torch.randint(0, H - L + 1, (B, 1, D), device=device, generator=generator)
        ar = torch.arange(H, device=device).view(1, H, 1)
        drop = (ar >= starts) & (ar < starts + L)
        return ~drop
    raise ValueError(f"unknown dropout mode: {mode}")


def apply_dropout(batch: dict[str, torch.Tensor], rate: float,
                  generator: torch.Generator | None = None,
                  mode: str = "iid") -> DroppedBatch:
    """Return a new batch with `rate`-fraction of input observations masked off.

    `mask_hist` is the AND of the original mask and the keep-mask: an entry is
    only fed to the model if it was originally observed AND wasn't dropped.
    Tidal clock features are deterministic, so they pass through untouched.
    """
    mask_hist = batch["mask_hist"]
    tfeat_hist = batch.get("tfeat_hist")
    tfeat_fcst = batch.get("tfeat_fcst")
    if rate <= 0.0:
        return DroppedBatch(
            x_hist=batch["x_hist"], mask_hist=mask_hist,
            t_hist=batch["t_hist"], t_fcst=batch["t_fcst"],
            x_fcst=batch["x_fcst"], mask_fcst=batch["mask_fcst"],
            tfeat_hist=tfeat_hist, tfeat_fcst=tfeat_fcst,
        )

    keep = _keep_mask(tuple(mask_hist.shape), rate, mode, mask_hist.device, generator)
    new_mask = mask_hist & keep
    new_x = batch["x_hist"] * new_mask.to(batch["x_hist"].dtype)
    return DroppedBatch(
        x_hist=new_x, mask_hist=new_mask,
        t_hist=batch["t_hist"], t_fcst=batch["t_fcst"],
        x_fcst=batch["x_fcst"], mask_fcst=batch["mask_fcst"],
        tfeat_hist=tfeat_hist, tfeat_fcst=tfeat_fcst,
    )


def select_clean_indices(dataset, fraction_threshold: float = 0.95) -> np.ndarray:
    """Return window indices whose input has at least `fraction_threshold` of obs."""
    keep: list[int] = []
    for i in range(len(dataset)):
        s = dataset[i]
        frac = float(s["mask_hist"].to(torch.float32).mean())
        if frac >= fraction_threshold:
            keep.append(i)
    return np.asarray(keep, dtype=np.int64)
