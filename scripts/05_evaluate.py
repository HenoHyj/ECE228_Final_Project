"""End-to-end robustness evaluation: dropout sweep, metrics, figures.

Assumes both models have been trained and have checkpoints under
``runs/{lstm,latent_ode}/best.pt``.

Outputs
-------
- results/robustness.csv
- results/figures/mse_vs_dropout.png
- results/figures/per_channel.png
- results/figures/example_forecasts.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import ScrippsWindows, WindowConfig, collate     # noqa: E402
from src.eval.metrics import denormalize, load_scaler, per_channel_errors  # noqa: E402
from src.eval.robustness import apply_dropout, select_clean_indices    # noqa: E402
from src.eval.visualize import (                                       # noqa: E402
    aggregate_summary,
    plot_example_forecasts,
    plot_mse_vs_dropout,
    plot_per_channel,
)
from src.models.latent_ode import LatentODE                            # noqa: E402
from src.models.lstm_baseline import LSTMForecaster                    # noqa: E402
from src.utils.config import load_config                               # noqa: E402
from src.utils.seed import seed_everything                             # noqa: E402


def build_loader(interval: str, win_cfg: WindowConfig, batch_size: int) -> tuple[DataLoader, ScrippsWindows, np.ndarray]:
    parquet = ROOT / "data" / "processed" / f"{interval}.parquet"
    scaler  = ROOT / "data" / "processed" / "scaler.json"
    splits  = ROOT / "data" / "splits" / f"{interval}.json"
    ds = ScrippsWindows(parquet, scaler, config=win_cfg, split="test",
                        splits_path=splits, min_obs_fraction=0.0)
    clean = select_clean_indices(ds, fraction_threshold=0.95)
    if len(clean) == 0:
        raise RuntimeError("No test windows pass the cleanliness threshold")
    subset = Subset(ds, clean.tolist())
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    return loader, ds, clean


def load_lstm(cfg: dict, ds: ScrippsWindows, ckpt: Path, device: torch.device) -> torch.nn.Module:
    model = LSTMForecaster(
        input_dim=ds.D,
        hidden_dim=cfg["model"]["hidden_dim"],
        horizon=cfg["data"]["horizon"],
        num_layers=cfg["model"]["num_layers"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


def load_latent_ode(cfg: dict, ds: ScrippsWindows, ckpt: Path, device: torch.device) -> torch.nn.Module:
    model = LatentODE(
        input_dim=ds.D,
        z_dim=cfg["model"]["z_dim"],
        enc_hidden=cfg["model"]["enc_hidden"],
        ode_hidden=cfg["model"]["ode_hidden"],
        ode_method=cfg["model"]["ode_method"],
        rtol=cfg["model"]["rtol"],
        atol=cfg["model"]["atol"],
        use_adjoint=cfg["model"]["use_adjoint"],
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def evaluate_model(
    name: str,
    model: torch.nn.Module,
    predict_fn,
    loader: DataLoader,
    *,
    rate: float,
    seed: int,
    device: torch.device,
    scaler: dict,
    channels: list[str],
) -> dict[str, np.ndarray]:
    """Run one (model, rate, seed) pass over the loader; return per-channel MSE/MAE."""
    gen = torch.Generator(device=device).manual_seed(seed)
    totals_sq = np.zeros(len(channels), dtype=np.float64)
    totals_ab = np.zeros(len(channels), dtype=np.float64)
    counts = np.zeros(len(channels), dtype=np.float64)
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        dropped = apply_dropout(batch, rate=rate, generator=gen)
        pred = predict_fn(model, dropped)
        pred_d = denormalize(pred, scaler, channels)
        targ_d = denormalize(batch["x_fcst"], scaler, channels)
        m = batch["mask_fcst"].to(pred.dtype)
        sq = ((pred_d - targ_d) ** 2 * m).sum(dim=(0, 1)).detach().cpu().numpy()
        ab = ((pred_d - targ_d).abs() * m).sum(dim=(0, 1)).detach().cpu().numpy()
        c = m.sum(dim=(0, 1)).detach().cpu().numpy()
        totals_sq += sq
        totals_ab += ab
        counts += c
    counts = np.maximum(counts, 1.0)
    return {"mse": totals_sq / counts, "mae": totals_ab / counts}


def lstm_predict(model, batch):
    return model(batch.x_hist, batch.mask_hist)


def latent_predict(model, batch):
    return model(batch.x_hist, batch.mask_hist, batch.t_hist, batch.t_fcst)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--interval", choices=["hourly", "six_min"], default="hourly")
    p.add_argument("--lstm-cfg",       default=str(ROOT / "configs" / "lstm.yaml"))
    p.add_argument("--latent-ode-cfg", default=str(ROOT / "configs" / "latent_ode.yaml"))
    p.add_argument("--lstm-ckpt",      default=None, help="defaults to runs/{name}/best.pt")
    p.add_argument("--latent-ode-ckpt",default=None)
    p.add_argument("--rates", nargs="+", type=float, default=[0.0, 0.2, 0.4, 0.6, 0.8])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lstm-run-name",       default=None)
    p.add_argument("--latent-ode-run-name", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    lstm_cfg = load_config(args.lstm_cfg)
    lo_cfg   = load_config(args.latent_ode_cfg)
    lstm_name = args.lstm_run_name or lstm_cfg["name"]
    lo_name   = args.latent_ode_run_name or lo_cfg["name"]
    lstm_ckpt = Path(args.lstm_ckpt) if args.lstm_ckpt else ROOT / "runs" / lstm_name / "best.pt"
    lo_ckpt   = Path(args.latent_ode_ckpt) if args.latent_ode_ckpt else ROOT / "runs" / lo_name / "best.pt"

    for c in (lstm_ckpt, lo_ckpt):
        if not c.exists():
            raise FileNotFoundError(c)

    win_cfg = WindowConfig(
        interval=args.interval,
        history=lstm_cfg["data"]["history"],
        horizon=lstm_cfg["data"]["horizon"],
        stride=lstm_cfg["data"]["stride"],
    )
    loader, ds, clean = build_loader(args.interval, win_cfg, args.batch_size)
    print(f"Clean test windows: {len(clean):,} / {len(ds):,}")

    scaler = load_scaler(ROOT / "data" / "processed" / "scaler.json")
    channels = ds.channels

    lstm = load_lstm(lstm_cfg, ds, lstm_ckpt, device)
    lo   = load_latent_ode(lo_cfg, ds, lo_ckpt, device)

    rows: list[dict] = []
    for rate in args.rates:
        for seed in args.seeds:
            print(f"  rate={rate:.2f}  seed={seed} ...", flush=True)
            for model_name, model, fn in [
                ("LSTM", lstm, lstm_predict),
                ("Latent ODE", lo, latent_predict),
            ]:
                m = evaluate_model(model_name, model, fn, loader,
                                   rate=rate, seed=seed, device=device,
                                   scaler=scaler, channels=channels)
                for i, ch in enumerate(channels):
                    rows.append({
                        "model": model_name, "rate": rate, "seed": seed,
                        "channel": ch, "mse": float(m["mse"][i]), "mae": float(m["mae"][i]),
                    })

    df = pd.DataFrame(rows)
    out_csv = ROOT / "results" / "robustness.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    plot_mse_vs_dropout(df, ROOT / "results" / "figures" / f"mse_vs_dropout_{args.interval}.png")
    plot_per_channel(df,    ROOT / "results" / "figures" / f"per_channel_{args.interval}.png")
    summary = aggregate_summary(df)
    print("\nSummary (mean MSE across channels):")
    print(summary.to_string(index=False))

    # Example forecasts at 60% dropout for a few illustrative windows.
    print("\nRendering example forecasts at rate=0.6 ...")
    seed_everything(0)
    chosen = np.random.choice(len(clean), size=min(3, len(clean)), replace=False)
    example_loader = DataLoader(Subset(ds, [int(clean[i]) for i in chosen]),
                                batch_size=1, shuffle=False, collate_fn=collate)
    examples = []
    gen = torch.Generator(device=device).manual_seed(0)
    with torch.no_grad():
        for batch in example_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            dropped = apply_dropout(batch, rate=0.6, generator=gen)
            lstm_p = denormalize(lstm_predict(lstm, dropped), scaler, channels)[0].cpu().numpy()
            lo_p   = denormalize(latent_predict(lo, dropped),  scaler, channels)[0].cpu().numpy()
            tgt    = denormalize(batch["x_fcst"], scaler, channels)[0].cpu().numpy()
            examples.append({
                "t_fcst": batch["t_fcst"].cpu().numpy(),
                "target": tgt, "lstm": lstm_p, "latent_ode": lo_p,
                "title": "60% input dropout",
            })
    plot_example_forecasts(examples, channels,
                           ROOT / "results" / "figures" / f"example_forecasts_{args.interval}.png")
    print(f"Figures written to {ROOT / 'results' / 'figures'}")


if __name__ == "__main__":
    main()
