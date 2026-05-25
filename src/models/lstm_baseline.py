"""LSTM baseline that ingests imputed values + the original mask.

Linear interpolation is the strongest naive imputation for time-series
sensor data; concatenating the mask lets the LSTM distinguish "real low
value" from "imputed unknown" — the fairest possible comparison to a
continuous-time model that does not impute at all.
"""

from __future__ import annotations

import torch
from torch import nn


def linear_interp_impute(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Linearly interpolate missing entries along the time axis, per channel.

    Args:
        x:    [B, T, D]  values, with NaN positions zeroed in the dataset
        mask: [B, T, D]  bool — True where observed
    Returns:
        x_imp [B, T, D]  every entry is finite
    """
    B, T, D = x.shape
    device, dtype = x.device, x.dtype
    t_idx = torch.arange(T, device=device, dtype=dtype)             # [T]

    out = x.clone()
    for b in range(B):
        for d in range(D):
            m = mask[b, :, d]
            if m.all():
                continue
            if not m.any():
                # No observations at all in this channel → leave as zero.
                continue
            obs_t = t_idx[m]                                        # [K]
            obs_v = x[b, :, d][m]                                   # [K]
            # torch lacks a built-in 1-D interp; do it manually via searchsorted.
            interp = _interp1d(t_idx, obs_t, obs_v)                 # [T]
            out[b, :, d] = interp
    return out


def _interp1d(query: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """1-D linear interpolation (numpy.interp semantics, edges clamp to endpoints)."""
    if xp.numel() == 1:
        return fp[0].expand_as(query)
    idx = torch.searchsorted(xp, query)
    idx_lo = (idx - 1).clamp(0, xp.numel() - 1)
    idx_hi = idx.clamp(0, xp.numel() - 1)
    x_lo, x_hi = xp[idx_lo], xp[idx_hi]
    y_lo, y_hi = fp[idx_lo], fp[idx_hi]
    denom = (x_hi - x_lo).clamp_min(1e-8)
    t = ((query - x_lo) / denom).clamp(0.0, 1.0)
    out = y_lo + t * (y_hi - y_lo)
    # Outside the observation range, clamp to the nearest endpoint.
    out = torch.where(query < xp[0], fp[0].expand_as(query), out)
    out = torch.where(query > xp[-1], fp[-1].expand_as(query), out)
    return out


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        horizon: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.horizon = horizon

        # Input: [imputed_value (D), mask (D)] -> 2D
        self.lstm = nn.LSTM(
            input_size=2 * input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, horizon * input_dim)

    def forward(self, x_hist: torch.Tensor, mask_hist: torch.Tensor) -> torch.Tensor:
        x_imp = linear_interp_impute(x_hist, mask_hist)
        x_aug = torch.cat([x_imp, mask_hist.to(x_imp.dtype)], dim=-1)   # [B, H, 2D]
        out, _ = self.lstm(x_aug)
        last = out[:, -1, :]                                            # [B, hidden]
        flat = self.head(last)                                          # [B, F*D]
        return flat.view(-1, self.horizon, self.input_dim)
