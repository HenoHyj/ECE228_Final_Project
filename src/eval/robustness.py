"""Artificial input-dropout robustness sweep.

For each dropout rate p, we sample a Bernoulli(p) mask over (time, channel)
in the input history. The targets stay clean, so we measure how well each
model can still forecast when the input becomes sparser.
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


def apply_dropout(batch: dict[str, torch.Tensor], rate: float,
                  generator: torch.Generator | None = None) -> DroppedBatch:
    """Return a new batch with `rate`-fraction of input entries zeroed + masked off.

    `mask_hist` is the AND of the original mask and the dropout mask: an entry
    is only fed to the model if it was originally observed AND wasn't dropped.
    """
    mask_hist = batch["mask_hist"]
    if rate <= 0.0:
        return DroppedBatch(
            x_hist=batch["x_hist"], mask_hist=mask_hist,
            t_hist=batch["t_hist"], t_fcst=batch["t_fcst"],
            x_fcst=batch["x_fcst"], mask_fcst=batch["mask_fcst"],
        )

    keep_prob = 1.0 - rate
    keep = torch.bernoulli(
        torch.full(mask_hist.shape, keep_prob, device=mask_hist.device),
        generator=generator,
    ).bool()
    new_mask = mask_hist & keep
    new_x = batch["x_hist"] * new_mask.to(batch["x_hist"].dtype)
    return DroppedBatch(
        x_hist=new_x, mask_hist=new_mask,
        t_hist=batch["t_hist"], t_fcst=batch["t_fcst"],
        x_fcst=batch["x_fcst"], mask_fcst=batch["mask_fcst"],
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
