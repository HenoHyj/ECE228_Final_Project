"""Shared training loop used by both LSTM and Latent ODE scripts.

A model is wrapped in a callable that takes a batch dict (from the dataset
collate) and returns a forecast tensor of shape [B, horizon, D]. The trainer
handles optimization, scheduling, early stopping, checkpointing, CSV logging,
linear LR warmup, training-time input dropout augmentation, and NaN-batch
safeguards.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader

from src.training.losses import masked_mse


@dataclass
class TrainCfg:
    epochs: int
    lr: float
    weight_decay: float
    grad_clip: float
    early_stopping_patience: int
    scheduler_patience: int
    scheduler_factor: float
    amp: bool
    warmup_epochs: int = 0          # linear warmup from lr / 10 -> lr
    input_dropout_train: float = 0.0  # training-time random input dropout
    recon_weight: float = 0.0       # weight on the history-reconstruction loss (Latent ODE only)


PredictFn = Callable[[dict[str, torch.Tensor]], torch.Tensor]


def _apply_input_dropout(
    batch: dict[str, torch.Tensor],
    rate: float,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    """Randomly drop input entries during training so both models learn
    to handle missing observations directly."""
    if rate <= 0.0:
        return batch
    mask_hist = batch["mask_hist"]
    keep_prob = 1.0 - rate
    keep = torch.bernoulli(
        torch.full(mask_hist.shape, keep_prob, device=mask_hist.device),
        generator=generator,
    ).bool()
    new_mask = mask_hist & keep
    new_x = batch["x_hist"] * new_mask.to(batch["x_hist"].dtype)
    out = dict(batch)
    out["x_hist"] = new_x
    out["mask_hist"] = new_mask
    return out


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        predict_fn: PredictFn,
        cfg: TrainCfg,
        device: torch.device,
        run_dir: Path,
    ) -> None:
        self.model = model.to(device)
        self.predict_fn = predict_fn
        self.cfg = cfg
        self.device = device
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optim, mode="min", factor=cfg.scheduler_factor, patience=cfg.scheduler_patience
        )
        self.use_amp = cfg.amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        # Generator for reproducible training-time input dropout.
        self.dropout_gen = torch.Generator(device=device).manual_seed(0)

        self.log_path = self.run_dir / "log.csv"
        with self.log_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "lr", "train_mse", "val_mse", "wall_sec"])

    def _batch_to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def _set_lr_warmup(self, epoch: int) -> None:
        """Linearly ramp LR from base_lr / 10 -> base_lr across warmup_epochs."""
        if epoch > self.cfg.warmup_epochs or self.cfg.warmup_epochs <= 0:
            return
        frac = epoch / self.cfg.warmup_epochs              # 0 < frac <= 1
        scale = 0.1 + 0.9 * frac
        for g in self.optim.param_groups:
            g["lr"] = self.cfg.lr * scale

    def _step(self, batch: dict[str, torch.Tensor], train: bool) -> float | None:
        batch = self._batch_to_device(batch)
        # Capture the PRE-dropout history so reconstruction targets the truth even
        # when the input is corrupted — that is exactly the robustness objective.
        orig_x_hist = batch["x_hist"]
        orig_mask_hist = batch["mask_hist"]
        if train and self.cfg.input_dropout_train > 0.0:
            batch = _apply_input_dropout(batch, self.cfg.input_dropout_train, self.dropout_gen)
        if train:
            self.optim.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            out = self.predict_fn(batch)
            if isinstance(out, tuple):
                pred, recon = out
            else:
                pred, recon = out, None
            loss = masked_mse(pred, batch["x_fcst"], batch["mask_fcst"])
            if train and recon is not None and self.cfg.recon_weight > 0.0:
                loss = loss + self.cfg.recon_weight * masked_mse(recon, orig_x_hist, orig_mask_hist)

        if not torch.isfinite(loss):
            # Drop the batch — params untouched. Common when the ODE solver
            # transiently diverges; the next batch usually recovers.
            return None

        if train:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optim)
            # If the unscaled grad has NaN/inf, skip the step entirely so we
            # don't corrupt the parameters.
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.grad_clip
            )
            if not torch.isfinite(total_norm):
                self.optim.zero_grad(set_to_none=True)
                self.scaler.update()
                return None
            self.scaler.step(self.optim)
            self.scaler.update()
        return float(loss.detach().cpu())

    def _epoch(self, loader: DataLoader, train: bool) -> float:
        self.model.train() if train else self.model.eval()
        ctx = torch.enable_grad() if train else torch.no_grad()
        total, n = 0.0, 0
        with ctx:
            for batch in loader:
                loss = self._step(batch, train)
                if loss is None:
                    continue
                bs = batch["x_hist"].size(0)
                total += loss * bs
                n += bs
        return total / max(n, 1)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> dict:
        best_val = float("inf")
        bad_epochs = 0
        epoch = 0
        for epoch in range(1, self.cfg.epochs + 1):
            self._set_lr_warmup(epoch)
            t0 = time.time()
            train_mse = self._epoch(train_loader, train=True)
            val_mse = self._epoch(val_loader, train=False)
            # Scheduler should only react after warmup is done.
            if epoch > self.cfg.warmup_epochs:
                self.scheduler.step(val_mse)
            lr = self.optim.param_groups[0]["lr"]
            wall = time.time() - t0

            with self.log_path.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([epoch, f"{lr:.2e}", f"{train_mse:.5f}",
                                        f"{val_mse:.5f}", f"{wall:.1f}"])
            print(f"  epoch {epoch:>3}  train={train_mse:.5f}  val={val_mse:.5f}  "
                  f"lr={lr:.1e}  ({wall:.1f}s)", flush=True)

            if val_mse < best_val - 1e-6:
                best_val = val_mse
                bad_epochs = 0
                torch.save(self.model.state_dict(), self.run_dir / "best.pt")
            else:
                bad_epochs += 1
                if bad_epochs >= self.cfg.early_stopping_patience:
                    print(f"  early stopping at epoch {epoch}", flush=True)
                    break

        return {"best_val_mse": best_val, "epochs_run": epoch}
