"""Drift network f_theta(t, z) parameterizing dz/dt for the Latent ODE."""

from __future__ import annotations

import torch
from torch import nn


class ODEFunc(nn.Module):
    def __init__(
        self,
        z_dim: int,
        hidden_dim: int = 64,
        gain: float = 0.5,
        last_gain: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + 1, hidden_dim),    # +1 for explicit time
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, z_dim),
        )
        # Hidden layers get a normal-ish gain so the drift is expressive, but the
        # FINAL layer keeps a small gain so dz/dt ≈ 0 at init (stable integrator
        # before training). The old uniform gain=0.1 made the whole net so
        # contractive that trajectories decayed to a fixed point → mean collapse.
        linears = [m for m in self.net if isinstance(m, nn.Linear)]
        for i, m in enumerate(linears):
            g = last_gain if i == len(linears) - 1 else gain
            nn.init.xavier_uniform_(m.weight, gain=g)
            nn.init.zeros_(m.bias)

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # torchdiffeq passes scalar t; broadcast it to match z's batch shape.
        t_col = t.expand(z.shape[:-1]).unsqueeze(-1).to(z.dtype)
        return self.net(torch.cat([z, t_col], dim=-1))
