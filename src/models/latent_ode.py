"""Latent ODE forecaster.

Pipeline
--------
1. **Encoder** — an ODE-RNN run backwards over the history. A GRU cell ingests
   the masked observation at each step; between steps the hidden state decays in
   closed form (GRU-D style). Output: ``z_last`` at ``t_hist[-1]`` (most recent
   obs) plus the per-step latents ``z_hist`` for reconstruction.
2. **Forward ODE** — ``z_last`` is integrated forward from the most recent
   observation to the forecast times under a trained drift network, with an
   explicit fixed step so the solver never takes a giant macro-step.
3. **Decoder** — a small MLP mapping ``(z(t), tidal-clock features) -> x_hat(t)``.
   The clock features let the decoder render the deterministic tide directly.

This follows Rubanova et al. 2019 ("Latent ODEs for Irregularly-Sampled Time
Series"), simplified to a deterministic latent (no VAE) since the headline
experiment is forecast accuracy under input dropout, not density estimation.
A masked history-reconstruction loss (wired in the trainer) regularizes the
latent so it actually encodes the observations instead of collapsing to the mean.
"""

from __future__ import annotations

import torch
from torch import nn
from torchdiffeq import odeint, odeint_adjoint

from src.models.ode_func import ODEFunc


class _ODERNNEncoder(nn.Module):
    """ODE-RNN encoder with a GRU-D-style closed-form decay between observations.

    The old version called ``odeint`` once per history step (≈47 sequential,
    non-adjoint solves per forward) which dominated runtime and retained the
    whole graph. On a uniform grid the inter-observation evolution is just a
    decay, so we use a learned exponential decay ``h·exp(-softplus(λ)·Δt)`` in
    closed form — no solver. Returns the latent at the MOST RECENT observation
    (used as the forward ODE's initial condition) plus the per-step latents
    (used for history reconstruction).
    """

    def __init__(self, input_dim: int, hidden_dim: int, z_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        # Mask is concatenated to the value so the GRU sees what was real.
        self.gru = nn.GRUCell(input_size=2 * input_dim, hidden_size=hidden_dim)
        self.to_z = nn.Linear(hidden_dim, z_dim)
        # Per-unit decay rate, kept positive via softplus.
        self.decay = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x_hist: torch.Tensor, mask_hist: torch.Tensor,
                t_hist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_hist:    [B, H, D]
            mask_hist: [B, H, D] (bool)
            t_hist:    [H] (normalized, increasing)
        Returns:
            z_last: [B, z_dim]    — latent at t_hist[-1] (most recent obs)
            z_hist: [B, H, z_dim] — latent at every history time (for reconstruction)
        """
        B, H, D = x_hist.shape
        h = x_hist.new_zeros(B, self.hidden_dim)
        x_aug = torch.cat([x_hist, mask_hist.to(x_hist.dtype)], dim=-1)  # [B, H, 2D]
        rate = nn.functional.softplus(self.decay)                       # [hidden]
        h_states: list[torch.Tensor] = [x_hist.new_zeros(B, self.hidden_dim)] * H
        # Iterate backwards (t_hist[-1] -> t_hist[0]); h_states[i] is the state
        # right after ingesting observation i, i.e. the state AT time t_hist[i].
        for i in reversed(range(H)):
            h = self.gru(x_aug[:, i, :], h)
            h_states[i] = h
            if i > 0:
                dt = (t_hist[i] - t_hist[i - 1]).abs().to(x_hist.dtype)
                h = h * torch.exp(-rate * dt)
        z_last = self.to_z(h_states[H - 1])                             # [B, z_dim]
        z_hist = self.to_z(torch.stack(h_states, dim=1))                # [B, H, z_dim]
        return z_last, z_hist


# torchdiffeq fixed-step solvers need an explicit step_size (otherwise they take
# one macro-step between consecutive eval times — the old 48h-step bug).
_FIXED_STEP_METHODS = {"euler", "midpoint", "rk4", "explicit_adams", "implicit_adams"}


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
        dec_hidden: int = 64,
        use_tidal_features: bool = False,
        tfeat_dim: int = 0,
        ode_gain: float = 0.5,
        ode_substeps: int = 1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.ode_method = ode_method
        self.rtol, self.atol = rtol, atol
        self.use_adjoint = use_adjoint
        self.use_tidal = bool(use_tidal_features)
        self.tfeat_dim = tfeat_dim if self.use_tidal else 0
        self.ode_substeps = max(1, int(ode_substeps))

        self.encoder = _ODERNNEncoder(input_dim, enc_hidden, z_dim)
        self.dynamics = ODEFunc(z_dim=z_dim, hidden_dim=ode_hidden, gain=ode_gain)
        # Time-aware MLP decoder: maps (latent, optional tidal clock) -> channels.
        # A linear, time-blind decoder cannot render the deterministic tide.
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + self.tfeat_dim, dec_hidden),
            nn.ELU(),
            nn.Linear(dec_hidden, input_dim),
        )

    def _decode(self, z: torch.Tensor, tfeat: torch.Tensor | None) -> torch.Tensor:
        if self.tfeat_dim > 0 and tfeat is not None:
            z = torch.cat([z, tfeat], dim=-1)
        return self.decoder(z)

    def forward(
        self,
        x_hist: torch.Tensor,
        mask_hist: torch.Tensor,
        t_hist: torch.Tensor,
        t_fcst: torch.Tensor,
        tfeat_hist: torch.Tensor | None = None,
        tfeat_fcst: torch.Tensor | None = None,
        return_recon: bool = False,
    ):
        """Forecast [B, F, D]. If return_recon, also return history reconstruction [B, H, D]."""
        z_last, z_hist = self.encoder(x_hist, mask_hist, t_hist)   # [B,z], [B,H,z]

        # Integrate forward from the MOST RECENT history time (not t_hist[0]):
        # short horizon + unit-spaced grid → no giant macro-step, less contraction.
        t0 = t_hist[-1:].to(z_last.device)
        t_full = torch.cat([t0, t_fcst.to(z_last.device)])
        assert (t_full[1:] > t_full[:-1]).all(), "t_full must be strictly increasing"

        kw = dict(method=self.ode_method, rtol=self.rtol, atol=self.atol)
        if self.ode_method in _FIXED_STEP_METHODS:
            dt = float((t_full[1] - t_full[0]).item()) / self.ode_substeps
            kw["options"] = {"step_size": dt}

        solver = odeint_adjoint if self.use_adjoint else odeint
        z_traj = solver(self.dynamics, z_last, t_full, **kw)      # [1+F, B, z_dim]
        z_fcst = z_traj[1:]                                       # drop the anchor → [F, B, z]

        tff = tfeat_fcst.permute(1, 0, 2) if (self.tfeat_dim > 0 and tfeat_fcst is not None) else None
        x_hat = self._decode(z_fcst, tff)                         # [F, B, D]
        x_hat = x_hat.permute(1, 0, 2).contiguous()               # [B, F, D]

        if not return_recon:
            return x_hat
        # Reconstruct the history from the encoder's per-step latents (cheap, no
        # extra solve). Supervising this breaks the "predict the mean" collapse.
        x_rec = self._decode(z_hist, tfeat_hist if self.tfeat_dim > 0 else None)  # [B, H, D]
        return x_hat, x_rec
