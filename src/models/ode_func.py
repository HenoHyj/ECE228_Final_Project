"""Drift network f_theta(t, z) parameterizing dz/dt for the Latent ODE."""

from __future__ import annotations

import torch
from torch import nn


class ODEFunc(nn.Module):
    def __init__(self, z_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + 1, hidden_dim),    # +1 for explicit time
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, z_dim),
        )
        # Small init keeps early dynamics tame and stops the integrator from
        # producing huge derivatives before training has done any work.
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # torchdiffeq passes scalar t; broadcast it to match z's batch shape.
        t_col = t.expand(z.shape[:-1]).unsqueeze(-1).to(z.dtype)
        return self.net(torch.cat([z, t_col], dim=-1))
