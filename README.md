# Forecasting Scripps Pier Coastal Dynamics via Continuous-Time Neural ODEs

ECE 228 final project. Compares an LSTM baseline against a Latent ODE
(Rubanova et al. 2019) on multivariate environmental sensor data from
NOAA Station 9410230 (La Jolla, Scripps Pier). Headline experiment:
**how each model degrades as input observations are dropped at random**.

## Quickstart

```bash
pip install -r requirements.txt

# 1. Fetch NOAA data (cached locally; safe to re-run)
python scripts/01_fetch_data.py --interval hourly --start 2023-01-01

# 2. Build processed parquet + train/val/test splits
python scripts/02_preprocess.py --interval hourly

# 3. Train baseline + neural ODE
python scripts/03_train_lstm.py       --config configs/lstm.yaml
python scripts/04_train_latent_ode.py --config configs/latent_ode.yaml

# 4. Robustness evaluation + figures
python scripts/05_evaluate.py --interval hourly
```

Re-run with `--interval six_min` for the high-resolution comparison.

## Repository layout

```
configs/   YAML hyperparameters per model
data/      raw NOAA JSON cache + processed parquet (gitignored)
src/       Python package (data, models, training, eval, utils)
scripts/   numbered entry points
notebooks/ EDA + final figures
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
