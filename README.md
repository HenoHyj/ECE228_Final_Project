# Forecasting Scripps Pier Coastal Dynamics via Continuous-Time Neural ODEs

ECE 228 final project (Team 25). We compare a discrete-time **LSTM** baseline
against a continuous-time **Latent ODE** (Rubanova et al. 2019) on multivariate
environmental sensor data from **NOAA Station 9410230 (La Jolla, Scripps Pier)**.

**Core question:** can a continuous-time model forecast directly from irregular,
gappy sensor data — and stay accurate as the data gets sparse? Real ocean
sensors drop out and sample irregularly; discrete RNNs assume a regular grid and
need imputation, which injects bias. A Latent ODE ingests the true time gaps
(Δt) and needs no imputation.

**Headline experiment:** how each model degrades as input observations are
removed, across three realistic missing-data regimes (i.i.d. dropout, contiguous
block outages, and native irregular sampling).

## Task setup

- **Data:** NOAA 9410230, hourly, 2023-01 → 2026 (~3.4 yr). Chronological split:
  train 2023-01→2025-08, val 2025-09→2025-12, test 2026-01→.
- **Targets:** multivariate forecast of **6 channels** — water level (tide),
  air temperature, water temperature, air pressure, wind u, wind v.
- **Windows:** 48 h history → 24 h forecast (direct multi-step, no autoregression).
- **Metric:** *skill* = model MSE / persistence MSE (dimensionless, <1 beats the
  naive "repeat-the-last-value" baseline), averaged across channels; plus
  per-channel NMSE (1.0 = predicting the climatological mean).

## Models

- **LSTM baseline** (`src/models/lstm_baseline.py`): 2-layer LSTM (hidden 128).
  Input = linearly-imputed values + observation mask + tidal-clock features.
- **Latent ODE** (`src/models/latent_ode.py`): GRU-D-style ODE-RNN encoder
  (closed-form `exp(-softplus(λ)·Δt)` decay) → forward integration of
  `dz/dt = f(t,z)` with a fixed-step RK4 solver and adjoint backprop, anchored at
  the most recent observation → MLP decoder with an astronomical tidal clock.
  Trained with **30% input dropout** + a **history-reconstruction loss** that
  prevents collapse to the mean. The same tidal features are fed to both models
  so the comparison isolates architecture.

## Quickstart

```bash
pip install -r requirements.txt

# 1. Fetch NOAA data (cached locally; safe to re-run). Always 6-min granularity.
python scripts/01_fetch_data.py --start 2023-01-01

# 2. Build processed parquet (hourly + six_min) + scaler + train/val/test splits
python scripts/02_preprocess.py

# 3. Train baseline + neural ODE (use CUDA_VISIBLE_DEVICES=0 to pin a GPU)
python scripts/03_train_lstm.py       --config configs/lstm.yaml
python scripts/04_train_latent_ode.py --config configs/latent_ode.yaml

# 4. Robustness evaluation + figures
python scripts/05_evaluate.py --interval hourly --dropout-mode iid     # i.i.d. dropout
python scripts/05_evaluate.py --interval hourly --dropout-mode block   # block outages
python scripts/05_evaluate.py --interval hourly --irregular            # headline: native irregular Δt
```

**Ablation (no tidal clock)** — quantifies how much the clock contributes vs the
learned dynamics; train with the `_notidal` configs and evaluate with a tag:

```bash
python scripts/03_train_lstm.py       --config configs/lstm_notidal.yaml       --run-name lstm_notidal
python scripts/04_train_latent_ode.py --config configs/latent_ode_notidal.yaml --run-name latent_ode_notidal
python scripts/05_evaluate.py --interval hourly --irregular --tag _notidal \
       --lstm-run-name lstm_notidal --latent-ode-run-name latent_ode_notidal
```

`scripts/01`/`02` operate on both resolutions at once; pass `--interval six_min`
to the train/eval scripts for the high-resolution comparison.

```bash
pytest -q          # data-pipeline + model sanity checks
```

## Results (hourly)

- **Clean grid:** the LSTM is slightly better (val MSE 0.262 vs 0.322) —
  interpolation is strong on a regular grid.
- **Irregular sampling (headline):** LSTM skill degrades 0.48 → 0.78 (toward the
  persistence baseline) as history sparsifies, while the **Latent ODE stays flat
  at ≈0.62** — it ingests the real Δt. They cross at ~28% dropout.
- **Water level:** under irregular sampling the LSTM's NMSE collapses 0.04 → 0.95
  (nearly predicting the mean); the **Latent ODE holds at ≈0.05**.
- **By regime:** LSTM wins on i.i.d. dropout; ≈tie on block outages; **Latent ODE
  wins on irregular sampling.** The advantage is **conditional**, not universal.
- **Ablation:** without the tidal clock the Latent ODE is still flat under
  irregular sampling — so the robustness is **architectural**; the clock only
  adds accuracy.

Figures and CSVs are written to `results/` (`skill_vs_dropout_*`,
`per_channel_*`, `robustness_*.csv`).

## Repository layout

```
configs/   YAML hyperparameters per model (+ _notidal ablations)
data/      raw NOAA JSON cache + processed parquet (gitignored)
src/       Python package
  data/      noaa_fetcher · preprocess · dataset (windows, mask, tidal features)
  models/    lstm_baseline · ode_func · latent_ode
  training/  trainer (warmup, recon loss, NaN guards) · losses
  eval/      metrics (skill, persistence) · robustness · tidal · visualize
  utils/     config · seed
scripts/   numbered entry points (01 fetch → 05 evaluate)
tests/     pytest sanity checks
results/   metrics CSVs + figures
runs/      training checkpoints + logs (gitignored)
```

## Data source

NOAA CO-OPS API: <https://api.tidesandcurrents.noaa.gov/api/prod/datagetter>
Station page: <https://tidesandcurrents.noaa.gov/stationhome.html?id=9410230>

Products pulled: `water_level` (MLLW datum), `air_temperature`,
`water_temperature`, `air_pressure`, `wind` (decomposed to u/v). All fetched at
6-min granularity and resampled locally; timestamps standardized to GMT.

## Team

Yijie He · Yifan Peng · Zihao Yang · Zixuan Chen
