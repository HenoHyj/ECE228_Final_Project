"""Train the LSTM baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import ScrippsWindows, WindowConfig, collate  # noqa: E402
from src.models.lstm_baseline import LSTMForecaster                  # noqa: E402
from src.training.trainer import TrainCfg, Trainer                   # noqa: E402
from src.utils.config import load_config                             # noqa: E402
from src.utils.seed import seed_everything                           # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run-name", default=None)
    p.add_argument("--smoke", action="store_true", help="2-epoch sanity run")
    args = p.parse_args()

    cfg = load_config(args.config)
    name = args.run_name or cfg["name"]
    interval = cfg["data"]["interval"]
    seed_everything(cfg["train"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    win_cfg = WindowConfig(
        interval=interval,
        history=cfg["data"]["history"],
        horizon=cfg["data"]["horizon"],
        stride=cfg["data"]["stride"],
    )
    parquet = ROOT / "data" / "processed" / f"{interval}.parquet"
    scaler  = ROOT / "data" / "processed" / "scaler.json"
    splits  = ROOT / "data" / "splits" / f"{interval}.json"

    train_ds = ScrippsWindows(parquet, scaler, config=win_cfg, split="train", splits_path=splits)
    val_ds   = ScrippsWindows(parquet, scaler, config=win_cfg, split="val",   splits_path=splits)
    print(f"Windows: train={len(train_ds):,}  val={len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True,
                              collate_fn=collate, num_workers=cfg["data"]["num_workers"], drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["data"]["batch_size"], shuffle=False,
                            collate_fn=collate, num_workers=cfg["data"]["num_workers"])

    model = LSTMForecaster(
        input_dim=train_ds.D,
        hidden_dim=cfg["model"]["hidden_dim"],
        horizon=win_cfg.horizon,
        num_layers=cfg["model"]["num_layers"],
        dropout=cfg["model"]["dropout"],
    )
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    def predict(batch):
        return model(batch["x_hist"], batch["mask_hist"])

    train_cfg = TrainCfg(
        epochs=2 if args.smoke else cfg["train"]["epochs"],
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        grad_clip=cfg["train"]["grad_clip"],
        early_stopping_patience=cfg["train"]["early_stopping_patience"],
        scheduler_patience=cfg["train"]["scheduler_patience"],
        scheduler_factor=cfg["train"]["scheduler_factor"],
        amp=cfg["train"]["amp"],
        warmup_epochs=cfg["train"].get("warmup_epochs", 0),
        input_dropout_train=cfg["train"].get("input_dropout_train", 0.0),
    )
    trainer = Trainer(model, predict, train_cfg, device, ROOT / "runs" / name)
    result = trainer.fit(train_loader, val_loader)
    print(f"Best val MSE: {result['best_val_mse']:.5f}")


if __name__ == "__main__":
    main()
