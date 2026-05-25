"""Latent ODE forecaster.

Pipeline
--------
1. **Encoder** — an ODE-RNN run backwards over the history. Between observation
   times the hidden state evolves under a small ODE; at each observation time
   a GRU cell ingests the masked observation. Output: ``z0`` at ``t_hist[0]``.
2. **Forward ODE** — ``z0`` is integrated forward over the union of history +
   forecast timestamps under the same ODE function family but a *different*
   trained network (decoupling encoder and predictor dynamics).
3. **Decoder** — linear map ``z(t) -> x_hat(t)`` over the forecast horizon.

This follows Rubanova et al. 2019 ("Latent ODEs for Irregularly-Sampled Time
Series"), simplified to a deterministic z0 (no VAE) since the headline
experiment is forecast accuracy under input dropout, not density estimation.
"""

from __future__ import annotations

import torch
from torch import nn
from torchdiffeq import odeint, odeint_adjoint

from src.models.ode_func import ODEFunc


class _ODERNNEncoder(nn.Module):
    """ODE-RNN that maps history observations → initial latent z0 (at t=0)."""

    def __init__(self, input_dim: int, hidden_dim: int, z_dim: int,
                 ode_hidden: int, ode_method: str) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        # Decay between observations is itself a small ODE on the hidden state.
        self.evolve = ODEFunc(z_dim=hidden_dim, hidden_dim=ode_hidden)
        # Mask is concatenated to the value so the GRU sees what was real.
        self.gru = nn.GRUCell(input_size=2 * input_dim, hidden_size=hidden_dim)
        self.to_z0 = nn.Linear(hidden_dim, z_dim)
        self.ode_method = ode_method

    def forward(self, x_hist: torch.Tensor, mask_hist: torch.Tensor,
                t_hist: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_hist:    [B, H, D]
            mask_hist: [B, H, D] (bool)
            t_hist:    [H]
        Returns:
            z0: [B, z_dim] — latent state at t_hist[0]
        """
        B, H, D = x_hist.shape
        h = x_hist.new_zeros(B, self.hidden_dim)
        # Iterate backwards: the encoder runs from t_hist[-1] -> t_hist[0].
        x_aug = torch.cat([x_hist, mask_hist.to(x_hist.dtype)], dim=-1)  # [B, H, 2D]
        for i in reversed(range(H)):
            h = self.gru(x_aug[:, i, :], h)
            if i > 0:
                # Decay the hidden state backwards by one step.
                t_pair = torch.stack([t_hist[i], t_hist[i - 1]]).to(x_hist.device)
                # odeint requires strictly monotonic t; flip the sign so it is.
                # Equivalently: integrate evolve(-t, h) from -t_i to -t_{i-1}.
                # Simpler: just call odeint with the (descending) pair — recent
                # torchdiffeq versions accept strictly decreasing time too.
                h_out = odeint(self.evolve, h, t_pair,
                               method=self.ode_method, rtol=1e-3, atol=1e-4)
                h = h_out[-1]
        return self.to_z0(h)


class LatentODE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        z_dim: int,
        enc_hidden: int,
        ode_hidden: int,
        ode_method: str = "rk4",
        rtol: float = 1e-3,
        atol: float = 1e-4,
        use_adjoint: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.ode_method = ode_method
        self.rtol, self.atol = rtol, atol
        self.use_adjoint = use_adjoint

        self.encoder = _ODERNNEncoder(input_dim, enc_hidden, z_dim, ode_hidden, ode_method)
        self.dynamics = ODEFunc(z_dim=z_dim, hidden_dim=ode_hidden)
        self.decoder = nn.Linear(z_dim, input_dim)

    def forward(
        self,
        x_hist: torch.Tensor,
        mask_hist: torch.Tensor,
        t_hist: torch.Tensor,
        t_fcst: torch.Tensor,
    ) -> torch.Tensor:
        """Returns forecast tensor of shape [B, F, D]."""
        z0 = self.encoder(x_hist, mask_hist, t_hist)              # [B, z_dim]

        # Integrate from t_hist[0] to every t_fcst point.
        t0 = t_hist[0:1]
        t_full = torch.cat([t0, t_fcst]).to(z0.device)
        # Ensure strict monotonicity for the solver (forecasts are after history).
        assert (t_full[1:] > t_full[:-1]).all(), "t_full must be strictly increasing"

        solver = odeint_adjoint if self.use_adjoint else odeint
        # torchdiffeq's adjoint needs the parameters of `dynamics` to differentiate through.
        z_traj = solver(
            self.dynamics, z0, t_full,
            method=self.ode_method, rtol=self.rtol, atol=self.atol,
        )                                                          # [1+F, B, z_dim]
        z_fcst = z_traj[1:]                                        # drop the t0 anchor
        x_hat = self.decoder(z_fcst)                               # [F, B, D]
        return x_hat.permute(1, 0, 2).contiguous()                 # [B, F, D]
