"""Regression guards for the modeling fixes.

The headline bug was the Latent ODE collapsing to a near-constant (mean)
forecast. These tests check the forward passes produce correctly-shaped,
time-varying outputs and that the vectorized imputation matches numpy.
"""

from __future__ import annotations

import numpy as np
import torch

from src.data.dataset import TFEAT_DIM
from src.models.latent_ode import LatentODE
from src.models.lstm_baseline import LSTMForecaster, linear_interp_impute


def _fake_batch(B=4, H=48, F=24, D=6, drop=0.3, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, H, D, generator=g)
    mask = torch.rand(B, H, D, generator=g) > drop
    x = x * mask.float()
    t_hist = torch.arange(H).float() * (1.0 / H)
    t_fcst = (torch.arange(F).float() + H) * (1.0 / H)
    tfh = torch.randn(B, H, TFEAT_DIM, generator=g)
    tff = torch.randn(B, F, TFEAT_DIM, generator=g)
    return x, mask, t_hist, t_fcst, tfh, tff


def test_latent_ode_shapes_and_recon():
    x, mask, t_hist, t_fcst, tfh, tff = _fake_batch()
    m = LatentODE(input_dim=6, z_dim=16, enc_hidden=32, ode_hidden=32, dec_hidden=32,
                  use_tidal_features=True, tfeat_dim=TFEAT_DIM, ode_substeps=2)
    x_hat, x_rec = m(x, mask, t_hist, t_fcst, tfeat_hist=tfh, tfeat_fcst=tff, return_recon=True)
    assert x_hat.shape == (4, 24, 6)
    assert x_rec.shape == (4, 48, 6)
    # eval-style call returns a bare tensor (signature used by 05_evaluate.py)
    assert torch.is_tensor(m(x, mask, t_hist, t_fcst, tfeat_hist=tfh, tfeat_fcst=tff))


def test_latent_ode_forecast_is_not_flat():
    """Guards the mean-collapse regression: the forecast must vary over time."""
    x, mask, t_hist, t_fcst, tfh, tff = _fake_batch()
    m = LatentODE(input_dim=6, z_dim=16, enc_hidden=32, ode_hidden=32, dec_hidden=32,
                  use_tidal_features=True, tfeat_dim=TFEAT_DIM, ode_substeps=2)
    x_hat = m(x, mask, t_hist, t_fcst, tfeat_hist=tfh, tfeat_fcst=tff)
    # std across the horizon, averaged over batch/channels, should be clearly > 0.
    assert x_hat.std(dim=1).mean().item() > 1e-3


def test_lstm_shapes_with_tidal():
    x, mask, _, _, tfh, _ = _fake_batch()
    m = LSTMForecaster(input_dim=6, hidden_dim=32, horizon=24, num_layers=2,
                       use_tidal_features=True, tfeat_dim=TFEAT_DIM)
    assert m.lstm.input_size == 2 * 6 + TFEAT_DIM
    assert m(x, mask, tfeat_hist=tfh).shape == (4, 24, 6)


def test_vectorized_impute_matches_numpy():
    x, mask, *_ = _fake_batch(B=3, H=32, D=4, drop=0.5, seed=1)
    imp = linear_interp_impute(x, mask)
    assert torch.isfinite(imp).all()
    idx = np.arange(x.shape[1])
    for b in range(x.shape[0]):
        for d in range(x.shape[2]):
            mm = mask[b, :, d].numpy().astype(bool)
            if not mm.any():
                continue
            ref = np.interp(idx, idx[mm], x[b, :, d].numpy()[mm])
            assert np.allclose(imp[b, :, d].numpy(), ref, atol=1e-5)
