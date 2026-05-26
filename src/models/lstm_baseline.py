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
    """Vectorized per-channel linear interpolation along the time axis.

    Missing entries (mask=False) are filled by linear interpolation between the
    nearest observed neighbours; positions before the first / after the last
    observation clamp to that endpoint; channels with no observation stay zero.
    Fully vectorized over (batch, channel) — the old version looped in Python on
    every forward pass.

    Args:
        x:    [B, T, D]  values, with NaN positions zeroed in the dataset
        mask: [B, T, D]  bool — True where observed
    Returns:
        x_imp [B, T, D]  every entry is finite
    """
    B, T, D = x.shape
    device, dtype = x.device, x.dtype
    ar = torch.arange(T, device=device, dtype=dtype).view(1, T, 1).expand(B, T, D)
    sentinel = float(T + 10)

    # Index of the last observed step at/before t (−1 if none) via a running max.
    left_pos = torch.where(mask, ar, torch.full_like(ar, -1.0))
    left_idx = torch.cummax(left_pos, dim=1).values
    # Index of the first observed step at/after t (sentinel if none) via reverse running min.
    right_pos = torch.where(mask, ar, torch.full_like(ar, sentinel))
    right_idx = torch.flip(torch.cummin(torch.flip(right_pos, [1]), dim=1).values, [1])

    left_valid = left_idx >= 0
    right_valid = right_idx < sentinel
    li = left_idx.clamp(min=0).long()
    ri = right_idx.clamp(max=T - 1).long()
    x_lo = torch.gather(x, 1, li)
    x_hi = torch.gather(x, 1, ri)

    denom = (right_idx - left_idx).clamp(min=1.0)
    frac = (ar - left_idx) / denom
    out = x_lo + frac * (x_hi - x_lo)
    out = torch.where(~right_valid, x_lo, out)                      # no obs after → clamp left
    out = torch.where(~left_valid, x_hi, out)                       # no obs before → clamp right
    out = torch.where(~left_valid & ~right_valid, torch.zeros_like(out), out)
    return out


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        horizon: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_tidal_features: bool = False,
        tfeat_dim: int = 0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.horizon = horizon
        self.use_tidal = bool(use_tidal_features)
        self.tfeat_dim = tfeat_dim if self.use_tidal else 0

        # Input: [imputed_value (D), mask (D), optional tidal clock (2K)].
        # The tidal features are fed to BOTH models so the comparison isolates
        # architecture, not who has the clock.
        self.lstm = nn.LSTM(
            input_size=2 * input_dim + self.tfeat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, horizon * input_dim)

    def forward(
        self,
        x_hist: torch.Tensor,
        mask_hist: torch.Tensor,
        tfeat_hist: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_imp = linear_interp_impute(x_hist, mask_hist)
        parts = [x_imp, mask_hist.to(x_imp.dtype)]
        if self.tfeat_dim > 0 and tfeat_hist is not None:
            parts.append(tfeat_hist.to(x_imp.dtype))
        x_aug = torch.cat(parts, dim=-1)                                # [B, H, 2D(+2K)]
        out, _ = self.lstm(x_aug)
        last = out[:, -1, :]                                            # [B, hidden]
        flat = self.head(last)                                          # [B, F*D]
        return flat.view(-1, self.horizon, self.input_dim)
