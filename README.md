# Forecasting Scripps Pier Coastal Dynamics via Continuous-Time Neural ODEs

ECE 228 final project. Compares an LSTM baseline against a Latent ODE
(Rubanova et al. 2019) on multivariate environmental sensor data from
NOAA Station 9410230 (La Jolla, Scripps Pier). Headline experiment:
**how each model degrades as input observations are dropped at random**.

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

# 4. Robustness evaluation + figures (sweep i.i.d. and contiguous-outage dropout)
python scripts/05_evaluate.py --interval hourly --dropout-mode iid
python scripts/05_evaluate.py --interval hourly --dropout-mode block
```

`scripts/01`/`02` operate on both resolutions at once; pass `--interval six_min`
to the train/eval scripts for the high-resolution comparison. Ablations without
the tidal clock features live in `configs/{lstm,latent_ode}_notidal.yaml`.

## Repository layout

```
configs/   YAML hyperparameters per model
data/      raw NOAA JSON cache + processed parquet (gitignored)
src/       Python package (data, models, training, eval, utils)
scripts/   numbered entry points
tests/     pytest sanity checks
results/   metrics CSVs + figures
runs/      training checkpoints + logs
```

## Data source

NOAA CO-OPS API: <https://api.tidesandcurrents.noaa.gov/api/prod/datagetter>
Station page: <https://tidesandcurrents.noaa.gov/stationhome.html?id=9410230>

Products pulled: `water_level` (MLLW datum), `air_temperature`,
`water_temperature`, `air_pressure`, `wind` (decomposed to u/v).

## Team

Yijie He · Yifan Peng · Zihao Yang · Zixuan Chen
