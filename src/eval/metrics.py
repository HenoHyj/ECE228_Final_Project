"""Per-channel forecasting metrics in the original (denormalized) units."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def load_scaler(path: Path) -> dict[str, dict[str, float]]:
    return json.loads(path.read_text(encoding="utf-8"))


def denormalize(x: torch.Tensor, scaler: dict, channels: list[str]) -> torch.Tensor:
    """x is [..., D]. Returns same shape in original units."""
    means = torch.tensor([scaler[c]["mean"] for c in channels], dtype=x.dtype, device=x.device)
    stds  = torch.tensor([scaler[c]["std"]  for c in channels], dtype=x.dtype, device=x.device)
    return x * stds + means


def per_channel_errors(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, np.ndarray]:
    """Channel-wise MSE and MAE across all valid (mask=True) entries.

    Inputs are [B, F, D]. Returns numpy arrays of shape [D].
    """
    m = mask.to(pred.dtype)
    diff = pred - target
    sq = (diff ** 2) * m
    ab = diff.abs() * m
    denom = m.sum(dim=(0, 1)).clamp_min(1.0)
    mse = (sq.sum(dim=(0, 1)) / denom).detach().cpu().numpy()
    mae = (ab.sum(dim=(0, 1)) / denom).detach().cpu().numpy()
    return {"mse": mse, "mae": mae}
