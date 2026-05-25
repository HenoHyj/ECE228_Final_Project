"""Mask-aware losses + metrics. Targets at NaN positions are excluded."""

from __future__ import annotations

import torch


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error, averaged over valid (mask=True) entries only.

    Args:
        pred:   [..., D]
        target: [..., D] (entries where mask is False are ignored)
        mask:   bool [..., D]
    """
    m = mask.to(pred.dtype)
    diff2 = (pred - target) ** 2 * m
    denom = m.sum().clamp_min(1.0)
    return diff2.sum() / denom


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(pred.dtype)
    diff = (pred - target).abs() * m
    denom = m.sum().clamp_min(1.0)
    return diff.sum() / denom
