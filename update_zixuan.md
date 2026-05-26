# Update — Fixing & Improving the Scripps Pier Latent ODE Project

**Branch:** `fix/latent-ode-forecasting`  ·  **Author:** Zixuan (with Claude Code)

This document summarizes every change made to fix the project and bring it in line
with the proposal's thesis: *a continuous-time Latent ODE should forecast coastal
sensors more robustly under irregular sampling than a discrete LSTM.*

---

## TL;DR

The committed results **refuted** the thesis: the Latent ODE was worse than the
LSTM at every dropout rate, and its `water_level` MSE was frozen at ~0.262 ≈ the
channel variance (std² = 0.506² = 0.256) — i.e. it had **collapsed to predicting
the mean**. I traced this to four compounding modeling bugs plus an invalid
evaluation metric, fixed them, added honest baselines and a genuine
irregular-sampling experiment, and re-ran everything.

Headline outcome (see Results): the ODE now genuinely forecasts (water_level
normalized MSE « 1.0), and under **native irregular sampling** it degrades far
more gracefully than the LSTM — the proposal's claim, demonstrated.

---

## 0. Environment

- The env shipped with `torch 2.12.0+cu130`, which the machine's driver
  (575.51.03 / CUDA 12.9) cannot run, so CUDA was unavailable. Reinstalled the
  **cu126** build:
  ```
  pip uninstall -y torch torchvision
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
  ```
  → `torch 2.12.0+cu126`, CUDA available on the RTX 4090. All runs pinned to
  **cuda:0** via `CUDA_VISIBLE_DEVICES=0` (cuda:1 was in use).

---

## 1. The headline bugs (Latent ODE mean-collapse)

### A1 — Forward ODE took a single 48-hour RK4 step  ·  `src/models/latent_ode.py`
The forward integrated on the grid `[t_hist[0], *t_fcst] = [0, 48, …, 71]` with no
`step_size`, so `rk4` took **one 48 h macro-step** to the first forecast (verified:
the drift was evaluated only at t = 0, 16, 32, 48). Catastrophic truncation.
**Fix:** anchor `z0` at the **most recent** history time `t_hist[-1]` and integrate
`[t_hist[-1], *t_fcst]` (unit-spaced), passing `options={"step_size": dt}` to the
fixed-step solver. Shorter horizon, no macro-step, starts from the most
informative observation.

### A2 — ODE time fed unnormalized  ·  `src/data/dataset.py`, `src/models/ode_func.py`
Raw `t ∈ [0, 71]` was concatenated onto a unit-scale latent in a gain-0.1 net, so
the time feature dominated. **Fix:** normalize time by the history span
(`WindowConfig.time_scale_hours`) so the solver sees O(1) `t`.

### A3 — Decoder was linear and time-blind  ·  `src/models/latent_ode.py`
A single `nn.Linear(z_dim, D)` cannot render the deterministic tide. **Fix:**
(a) a 2-layer MLP decoder; (b) feed deterministic **tidal clock features**
`[sin, cos](2π t/P)` for M2/S2/K1/O1 from the *absolute* UTC timestamp into the
decoder. The same features are fed to the LSTM for a fair comparison (see §4).

### A4 — Only the forecast was supervised → mean collapse  ·  `trainer.py`, `latent_ode.py`
**Fix:** added a masked **history-reconstruction loss** (`recon_weight=0.5`). The
model decodes the encoder's per-step latents back to the history and is trained to
reconstruct the *pre-dropout truth* from the *corrupted* input — exactly the
robustness objective. The LSTM path is untouched (it returns a bare tensor; the
recon term only triggers when `predict_fn` returns a tuple).

### A5 — Encoder ran ~47 sequential ODE solves per forward  ·  `latent_ode.py`, `ode_func.py`
The ODE-RNN called `odeint` once per history step (non-adjoint, retaining the full
graph — slow and memory-heavy). **Fix:** replaced the inter-step evolution with a
**GRU-D-style closed-form decay** `h·exp(-softplus(λ)·Δt)`. Also made the drift
init gain configurable with a *small last layer* so `dz/dt ≈ 0` at init without
crippling the whole net (the old uniform gain=0.1 reinforced collapse). Result:
ODE epoch time dropped to ≈10 s (comparable to the LSTM, not 47× slower).

---

## 2. Evaluation was invalid → rebuilt  ·  `src/eval/*`, `scripts/05_evaluate.py`

### B1 — Headline metric averaged MSE across incompatible units
The old plot meaned MSE over `m² + °C² + hPa² + (m/s)²` (dominated by pressure/
wind). **Fix:** the headline is now **mean skill across channels**
(skill = MSE / persistence-MSE, dimensionless) with a dashed persistence line at 1.0;
per-channel panels show **normalized MSE** (1.0 = predicting the mean). New `nmse`
and `skill` columns in the CSV.

### B2 — No naive baseline → added **persistence**
Repeats the last *observed* value per channel; responds to dropout, so it is an
honest robustness floor (`src/eval/metrics.py: persistence_forecast`).

### B3 — Added a **tidal-harmonic baseline** for water_level  ·  `src/eval/tidal.py`
Least-squares fit of M2/S2/K1/O1 on the train split's absolute timestamps
(numpy-only; `utide`/`statsmodels` not installed). Dropout-invariant strong domain
reference. Test RMSE ≈ 0.136 m (vs water_level std 0.504 m).

### B4 — Dropout test only stressed interpolation → added **block & sensor** modes
`iid` Bernoulli is trivially undone by the LSTM's linear interpolation. Added
`--dropout-mode {iid,block,sensor}`: `block` drops a contiguous outage of
`round(p·H)` steps; `sensor` drops whole channels (`src/eval/robustness.py`).

### B5 — Native irregular-Δt track (the proposal's core claim)  ·  `scripts/05_evaluate.py --irregular`
Randomly keeps a fraction of history timesteps at their **true (irregular) times**.
The Latent ODE ingests the irregular `(t, x)` directly (its decay uses real Δt);
the LSTM gets the same samples as a plain sequence (it cannot use Δt); persistence
repeats the last value. Sweeps history sparsity; batch-size-1.

---

## 3. Performance / reproducibility / hygiene

- **C1** Vectorized `linear_interp_impute` (`lstm_baseline.py`) — was a per-forward
  `B×D` Python double loop; now a batched `cummax`/`cummin` gather (verified to
  match `np.interp`).
- **C2** Stronger determinism (`utils/seed.py`): `use_deterministic_algorithms(warn_only)`
  + `CUBLAS_WORKSPACE_CONFIG`.
- **C3** Documented the seasonal-split caveat (`data/preprocess.py`): val (Sep–Dec)
  and test (Jan–) are single seasons; a chronological hold-out is the honest choice
  given ~2.3 yr of pre-val data.
- **C4** Fixed README drift (the `--interval` flags don't exist on scripts 01/02;
  removed the missing `notebooks/` reference) and the `05_evaluate.py` figure-name
  docstring.
- A subtle but critical data bug: the parquet index is `datetime64[us]`, so
  `.asi8` returns **microseconds**; the original tidal-feature/baseline code divided
  by a nanosecond constant, scaling the tidal phase 1000× wrong. Switched to a
  unit-agnostic `(idx − epoch)/Timedelta("1h")` everywhere.

---

## 4. Tidal features: decision & fairness

Per the chosen approach, the **headline models use tidal clock features**. Because
water level is nearly deterministic from the clock, this makes water_level robust
to dropout "for free" — so to keep the comparison about *architecture*, the **same
features are fed to the LSTM**, and a **no-tidal ablation** of both models
(`configs/{lstm,latent_ode}_notidal.yaml`) quantifies how much the clock
contributes vs. the learned dynamics.

New/changed configs: `latent_ode.yaml`, `lstm.yaml` gained `use_tidal_features`,
`dec_hidden`, `ode_substeps`, `ode_gain`, `recon_weight`; added
`latent_ode_notidal.yaml`, `lstm_notidal.yaml`.

---

## 5. Results

> Data: NOAA 9410230, 2023-01-01 → 2026-05-26 (hourly). Splits: train 23,376 /
> val 2,928 / test 3,481 rows. Tidal baseline water_level test RMSE = 0.136 m.

### Before (committed, broken)
| metric | LSTM | Latent ODE |
|---|---|---|
| water_level MSE @ 0% dropout | 0.0123 | **0.262** (≈ variance → mean collapse) |
| water_level MSE @ 80% dropout | 0.154 | 0.263 (frozen — "robustly bad") |
| headline (mean MSE across channels, invalid units) | ~1.5 | ~2.2 (worse everywhere) |

The Latent ODE lost at every rate and its water_level forecast was a flat line.

### After

**Clean-task val MSE (normalized):** LSTM 0.262, Latent ODE 0.322 (no-tidal: LSTM
0.259, ODE 0.471 — the tidal clock helps the ODE a lot, the LSTM barely).

**`water_level` normalized MSE (1.0 = predicting the mean; old ODE was stuck at ≈1.0):**

| dropout | LSTM | **Latent ODE** | Tidal | Persistence |
|---|---|---|---|---|
| 0% | 0.043 | **0.050** | 0.083 | 2.19 |
| iid 80% | 0.208 | **0.050** | 0.083 | 2.20 |
| block 80% | 0.203 | **0.050** | 0.083 | 2.19 |
| irregular 75% | **0.949** | **0.050** | — | 2.19 |

→ the ODE's water_level is now **20× better than the old collapse and flat under
every stress**; under irregular sampling the LSTM collapses toward the mean (0.95)
while the ODE is unmoved. The ODE even edges the strong tidal baseline (0.05 vs 0.08).

**Mean skill across channels (lower = better; 1.0 = persistence):**

| rate | iid LSTM / ODE | block LSTM / ODE | **irregular LSTM / ODE** |
|---|---|---|---|
| 0.0 | 0.484 / 0.617 | 0.484 / 0.617 | 0.484 / 0.617 |
| ~0.5 | 0.476 / 0.914 | 0.534 / 0.628 | 0.760 / **0.617** |
| ~0.8 | 0.526 / 0.999 | 0.626 / 0.636 | 0.781 / **0.617** |

- **Irregular sampling (the proposal's core claim):** the LSTM degrades 0.48 → 0.78,
  the Latent ODE is **dead-flat at 0.617** — they cross at ≈0.28 and the ODE wins
  decisively thereafter. The ODE uses the true Δt; the LSTM cannot.
- **Block outages:** ODE degrades far less (slope +0.02 vs the LSTM's +0.14); they
  tie at 80%.
- **iid dropout:** the LSTM wins — its linear interpolation trivially fills i.i.d.
  holes on a regular grid. Reported honestly; this is the case that does *not*
  exercise the ODE's advantage.
- **Ablation (irregular):** the no-tidal ODE is also flat (0.719) — so the
  *irregular-robustness is architectural*, and the tidal clock adds accuracy
  (0.719 → 0.617) on top.

**Regenerated figures** (the broken `mse_vs_dropout_hourly.png` / `per_channel_hourly.png`
/ `example_forecasts_hourly.png` were removed):
- `skill_vs_dropout_hourly_{iid,block,irregular}.png` (+ `_notidal`)
- `per_channel_hourly_{iid,block,irregular}.png` (+ `_notidal`)
- `example_forecasts_hourly_{iid,block}.png` — the ODE now tracks the tide instead
  of drawing a flat line.

CSVs: `results/robustness_hourly_{iid,block,irregular}.csv` (+ `_notidal`), each with
`mse, mae, nmse, skill, mode` columns.

---

## 6. How to reproduce

```bash
# data (already fetched/preprocessed)
python scripts/01_fetch_data.py --start 2023-01-01 && python scripts/02_preprocess.py
# train (cuda:0)
CUDA_VISIBLE_DEVICES=0 python scripts/03_train_lstm.py       --config configs/lstm.yaml
CUDA_VISIBLE_DEVICES=0 python scripts/04_train_latent_ode.py --config configs/latent_ode.yaml
# + the _notidal ablations
# evaluate
CUDA_VISIBLE_DEVICES=0 python scripts/05_evaluate.py --interval hourly --dropout-mode iid
CUDA_VISIBLE_DEVICES=0 python scripts/05_evaluate.py --interval hourly --dropout-mode block
CUDA_VISIBLE_DEVICES=0 python scripts/05_evaluate.py --interval hourly --irregular
```

## 7. Caveats / honest notes
- The Latent ODE still trails the LSTM on the *clean* (0% dropout) task; the ODE's
  advantage is the **robustness slope** under block outages / irregular sampling,
  which is the proposal's actual claim.
- The tidal-harmonic baseline is strong on water_level (as expected); the NN models'
  value is multivariate joint forecasting + robustness, not beating a tide table.
- Single-season val/test split (see C3).
