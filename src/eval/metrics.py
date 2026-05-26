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


def persistence_forecast(
    x_hist: torch.Tensor,
    mask_hist: torch.Tensor,
    horizon: int,
) -> torch.Tensor:
    """Naive baseline: repeat the LAST OBSERVED value of each channel.

    Responds to input dropout (when the last obs is dropped, the repeated value
    shifts to an earlier one), so it is an honest robustness floor. Returns
    [B, horizon, D].
    """
    B, H, D = x_hist.shape
    ar = torch.arange(H, device=x_hist.device, dtype=x_hist.dtype).view(1, H, 1).expand(B, H, D)
    last_obs_idx = torch.where(mask_hist, ar, torch.full_like(ar, -1.0)).cummax(dim=1).values
    li = last_obs_idx[:, -1, :].clamp(min=0).long()                # [B, D]
    last_val = torch.gather(x_hist, 1, li.unsqueeze(1)).squeeze(1)  # [B, D]
    return last_val.unsqueeze(1).expand(B, horizon, D).contiguous()


def skill_score(model_mse: np.ndarray, ref_mse: np.ndarray) -> np.ndarray:
    """MSE ratio vs a reference (e.g. persistence). <1 beats the reference.

    Dimensionless, so averaging across channels is legitimate (unlike raw MSE
    in mixed physical units).
    """
    return model_mse / np.maximum(ref_mse, 1e-12)


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
