"""End-to-end robustness evaluation: dropout sweep, metrics, figures.

Assumes both models have been trained and have checkpoints under
``runs/{lstm,latent_ode}/best.pt``.

Outputs (suffixed by --interval and --dropout-mode)
-------
- results/robustness_{interval}_{mode}.csv          (adds nmse + skill columns)
- results/figures/skill_vs_dropout_{interval}_{mode}.png
- results/figures/per_channel_{interval}_{mode}.png
- results/figures/example_forecasts_{interval}_{mode}.png
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

from src.data.dataset import TFEAT_DIM, ScrippsWindows, WindowConfig, collate  # noqa: E402
from src.eval.metrics import (                                         # noqa: E402
    denormalize,
    load_scaler,
    per_channel_errors,
    persistence_forecast,
)
from src.eval.robustness import apply_dropout, select_clean_indices    # noqa: E402
from src.eval.tidal import fit_harmonics, predict_harmonics            # noqa: E402
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
        use_tidal_features=cfg["model"].get("use_tidal_features", False),
        tfeat_dim=TFEAT_DIM,
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
        dec_hidden=cfg["model"].get("dec_hidden", 64),
        use_tidal_features=cfg["model"].get("use_tidal_features", False),
        tfeat_dim=TFEAT_DIM,
        ode_gain=cfg["model"].get("ode_gain", 0.5),
        ode_substeps=cfg["model"].get("ode_substeps", 1),
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
    mode: str,
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
        dropped = apply_dropout(batch, rate=rate, generator=gen, mode=mode)
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
    return model(batch.x_hist, batch.mask_hist, tfeat_hist=batch.tfeat_hist)


def latent_predict(model, batch):
    return model(batch.x_hist, batch.mask_hist, batch.t_hist, batch.t_fcst,
                 tfeat_hist=batch.tfeat_hist, tfeat_fcst=batch.tfeat_fcst)


def persistence_predict(model, batch):
    """Naive baseline (ignores `model`): repeat the last observed value."""
    horizon = batch.x_fcst.shape[1]
    return persistence_forecast(batch.x_hist, batch.mask_hist, horizon)


def eval_tidal_baseline(ds: ScrippsWindows, clean: np.ndarray, win_cfg: WindowConfig,
                        parquet: Path, splits: Path, scaler: dict) -> tuple[float, float]:
    """Fit tidal harmonics on TRAIN water_level, score on the clean test windows.

    Dropout-invariant (uses only the clock), so it is a single (mse, mae) pair
    reported as a flat reference across dropout rates.
    """
    import json

    full = pd.read_parquet(parquet)
    bounds = json.loads(splits.read_text())["boundaries"]
    train_start = pd.Timestamp(bounds["train_start"], tz="UTC")
    val_start = pd.Timestamp(bounds["val_start"], tz="UTC")
    train_wl = full.loc[(full.index >= train_start) & (full.index < val_start), "water_level"]
    coeffs = fit_harmonics(train_wl.index, train_wl.to_numpy())

    wl = ds.channels.index("water_level")
    mean, std = scaler["water_level"]["mean"], scaler["water_level"]["std"]
    H, F = win_cfg.history, win_cfg.horizon
    sq = ab = n = 0.0
    for ci in clean:
        s = int(ds._starts[int(ci)])
        ts = ds.timestamps[s + H : s + H + F]
        m = ds.mask[s + H : s + H + F, wl]
        tgt = ds.values[s + H : s + H + F, wl] * std + mean       # denormalize
        pred = predict_harmonics(coeffs, ts)
        d = (pred - tgt)[m]
        sq += float((d ** 2).sum()); ab += float(np.abs(d).sum()); n += float(m.sum())
    n = max(n, 1.0)
    return sq / n, ab / n


@torch.no_grad()
def run_irregular(ds, clean, win_cfg, lstm, lo, scaler, channels, device, *,
                  fracs, seeds, interval, max_windows, tag="") -> None:
    """Native irregular-Δt evaluation (the proposal's core claim).

    For each window we randomly KEEP a fraction of the history timesteps at their
    true (now irregular) times. The Latent ODE ingests the irregular (t, x)
    directly — its inter-observation decay uses the real Δt. The LSTM gets the
    same kept samples but as a plain sequence (it cannot use Δt), and persistence
    repeats the last observed value. Forecast horizon stays on the regular grid.
    The x-axis `rate` is the REMOVED fraction (1 - keep), so higher = sparser.
    """
    H, F = win_cfg.history, win_cfg.horizon
    dt_over_scale = win_cfg.step_hours / win_cfg.time_scale_hours
    var = {c: scaler[c]["std"] ** 2 for c in channels}
    sub = [int(c) for c in clean[:max_windows]]
    print(f"Irregular eval over {len(sub)} windows × {len(fracs)} keep-fracs × {len(seeds)} seeds")

    rows: list[dict] = []
    for frac in fracs:
        K = max(4, int(round(frac * H)))
        for seed in seeds:
            gen = torch.Generator().manual_seed(seed * 1000 + int(frac * 100))
            acc = {m: {"sq": np.zeros(len(channels)), "ab": np.zeros(len(channels)),
                       "n": np.zeros(len(channels))} for m in ["LSTM", "Latent ODE", "Persistence"]}
            for ci in sub:
                s = int(ds._starts[ci])
                # kept history indices: random subset of size K, always include the
                # most recent step (H-1) so the ODE anchor is informative.
                perm = torch.randperm(H - 1, generator=gen)[: K - 1]
                idx = torch.cat([perm, torch.tensor([H - 1])]).sort().values.numpy()
                x_hist = torch.from_numpy(ds.values[s + idx]).float().unsqueeze(0).to(device)
                mask_hist = torch.from_numpy(ds.mask[s + idx]).bool().unsqueeze(0).to(device)
                tfeat_hist = torch.from_numpy(ds.tfeat[s + idx]).float().unsqueeze(0).to(device)
                t_hist = (torch.from_numpy(idx).float() * dt_over_scale).to(device)
                fsl = slice(s + H, s + H + F)
                x_fcst = torch.from_numpy(ds.values[fsl]).float().unsqueeze(0).to(device)
                mask_fcst = torch.from_numpy(ds.mask[fsl]).bool().unsqueeze(0).to(device)
                tfeat_fcst = torch.from_numpy(ds.tfeat[fsl]).float().unsqueeze(0).to(device)
                t_fcst = ((torch.arange(F).float() + H) * dt_over_scale).to(device)

                preds = {
                    "LSTM": lstm(x_hist, mask_hist, tfeat_hist=tfeat_hist),
                    "Latent ODE": lo(x_hist, mask_hist, t_hist, t_fcst,
                                     tfeat_hist=tfeat_hist, tfeat_fcst=tfeat_fcst),
                    "Persistence": persistence_forecast(x_hist, mask_hist, F),
                }
                tgt = denormalize(x_fcst, scaler, channels)
                m = mask_fcst.to(tgt.dtype)
                for mname, p in preds.items():
                    pd_ = denormalize(p, scaler, channels)
                    acc[mname]["sq"] += ((pd_ - tgt) ** 2 * m).sum(dim=(0, 1)).cpu().numpy()
                    acc[mname]["ab"] += ((pd_ - tgt).abs() * m).sum(dim=(0, 1)).cpu().numpy()
                    acc[mname]["n"] += m.sum(dim=(0, 1)).cpu().numpy()
            for mname, a in acc.items():
                n = np.maximum(a["n"], 1.0)
                for i, ch in enumerate(channels):
                    rows.append({"model": mname, "rate": round(1.0 - frac, 3), "seed": seed,
                                 "channel": ch, "mse": float(a["sq"][i] / n[i]),
                                 "mae": float(a["ab"][i] / n[i])})

    df = pd.DataFrame(rows)
    df["nmse"] = df.apply(lambda r: r["mse"] / max(var[r["channel"]], 1e-12), axis=1)
    persist = (df[df["model"] == "Persistence"][["rate", "seed", "channel", "mse"]]
               .rename(columns={"mse": "pmse"}))
    df = df.merge(persist, on=["rate", "seed", "channel"], how="left")
    df["skill"] = df["mse"] / df["pmse"].clip(lower=1e-12)
    df = df.drop(columns=["pmse"])
    df["mode"] = "irregular"

    out_csv = ROOT / "results" / f"robustness_{interval}_irregular{tag}.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")
    figdir = ROOT / "results" / "figures"
    plot_mse_vs_dropout(df, figdir / f"skill_vs_dropout_{interval}_irregular{tag}.png",
                        mode="irregular (history sparsity)")
    plot_per_channel(df, figdir / f"per_channel_{interval}_irregular{tag}.png", value="nmse")
    print("\nIrregular summary (mean skill across channels; <1 beats persistence):")
    print(aggregate_summary(df, value="skill").to_string(index=False))


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
    p.add_argument("--dropout-mode", choices=["iid", "block", "sensor"], default="iid",
                   help="iid Bernoulli, contiguous block outage, or whole-sensor dropout")
    p.add_argument("--irregular", action="store_true",
                   help="run the native irregular-Δt track instead of the grid dropout sweep")
    p.add_argument("--keep-fracs", nargs="+", type=float, default=[1.0, 0.75, 0.5, 0.33, 0.25])
    p.add_argument("--max-windows", type=int, default=300,
                   help="cap on #windows for the (batch-size-1) irregular track")
    p.add_argument("--tag", default="", help="suffix appended to output filenames (e.g. _notidal)")
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

    if args.irregular:
        run_irregular(ds, clean, win_cfg, lstm, lo, scaler, channels, device,
                      fracs=args.keep_fracs, seeds=args.seeds, interval=args.interval,
                      max_windows=args.max_windows, tag=args.tag)
        return

    mode = args.dropout_mode
    rows: list[dict] = []
    for rate in args.rates:
        for seed in args.seeds:
            print(f"  mode={mode}  rate={rate:.2f}  seed={seed} ...", flush=True)
            for model_name, model, fn in [
                ("LSTM", lstm, lstm_predict),
                ("Latent ODE", lo, latent_predict),
                ("Persistence", None, persistence_predict),
            ]:
                m = evaluate_model(model_name, model, fn, loader,
                                   rate=rate, seed=seed, mode=mode, device=device,
                                   scaler=scaler, channels=channels)
                for i, ch in enumerate(channels):
                    rows.append({
                        "model": model_name, "rate": rate, "seed": seed,
                        "channel": ch, "mse": float(m["mse"][i]), "mae": float(m["mae"][i]),
                    })

    # Tidal-harmonic baseline (water_level only; dropout-invariant → flat line).
    parquet = ROOT / "data" / "processed" / f"{args.interval}.parquet"
    splits  = ROOT / "data" / "splits" / f"{args.interval}.json"
    t_mse, t_mae = eval_tidal_baseline(ds, clean, win_cfg, parquet, splits, scaler)
    print(f"Tidal baseline water_level: MSE={t_mse:.4f}  MAE={t_mae:.4f}")
    for rate in args.rates:
        rows.append({"model": "Tidal", "rate": rate, "seed": 0,
                     "channel": "water_level", "mse": t_mse, "mae": t_mae})

    df = pd.DataFrame(rows)
    # Derived metrics: normalized MSE (÷ train variance) and skill vs persistence.
    var = {c: scaler[c]["std"] ** 2 for c in channels}
    df["nmse"] = df.apply(lambda r: r["mse"] / max(var[r["channel"]], 1e-12), axis=1)
    persist = (df[df["model"] == "Persistence"][["rate", "seed", "channel", "mse"]]
               .rename(columns={"mse": "pmse"}))
    df = df.merge(persist, on=["rate", "seed", "channel"], how="left")
    df["skill"] = df["mse"] / df["pmse"].clip(lower=1e-12)
    df = df.drop(columns=["pmse"])
    df["mode"] = mode

    out_csv = ROOT / "results" / f"robustness_{args.interval}_{mode}{args.tag}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    figdir = ROOT / "results" / "figures"
    plot_mse_vs_dropout(df, figdir / f"skill_vs_dropout_{args.interval}_{mode}{args.tag}.png", mode=mode)
    plot_per_channel(df, figdir / f"per_channel_{args.interval}_{mode}{args.tag}.png", value="nmse")
    summary = aggregate_summary(df, value="skill")
    print("\nSummary (mean skill across channels; <1 beats persistence):")
    print(summary.to_string(index=False))

    # Example forecasts at a representative dropout for a few illustrative windows.
    ex_rate = 0.6
    print(f"\nRendering example forecasts at {mode} rate={ex_rate} ...")
    seed_everything(0)
    chosen = np.random.choice(len(clean), size=min(3, len(clean)), replace=False)
    example_loader = DataLoader(Subset(ds, [int(clean[i]) for i in chosen]),
                                batch_size=1, shuffle=False, collate_fn=collate)
    examples = []
    gen = torch.Generator(device=device).manual_seed(0)
    with torch.no_grad():
        for batch in example_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            dropped = apply_dropout(batch, rate=ex_rate, generator=gen, mode=mode)
            lstm_p = denormalize(lstm_predict(lstm, dropped), scaler, channels)[0].cpu().numpy()
            lo_p   = denormalize(latent_predict(lo, dropped),  scaler, channels)[0].cpu().numpy()
            tgt    = denormalize(batch["x_fcst"], scaler, channels)[0].cpu().numpy()
            examples.append({
                "t_fcst": batch["t_fcst"].cpu().numpy() * win_cfg.time_scale_hours,  # back to hours
                "target": tgt, "lstm": lstm_p, "latent_ode": lo_p,
                "title": f"{int(ex_rate*100)}% {mode} dropout",
            })
    plot_example_forecasts(examples, channels,
                           figdir / f"example_forecasts_{args.interval}_{mode}{args.tag}.png")
    print(f"Figures written to {figdir}")


if __name__ == "__main__":
    main()
