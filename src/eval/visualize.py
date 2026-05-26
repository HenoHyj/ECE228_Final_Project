"""Figure generation for the robustness experiment."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_mse_vs_dropout(results: pd.DataFrame, out_path: Path, *, mode: str = "iid") -> None:
    """Headline robustness plot: mean SKILL across channels vs dropout rate.

    Skill = MSE / persistence-MSE is dimensionless, so averaging across channels
    is valid (unlike raw MSE in mixed m²/°C²/hPa²/(m/s)² units). Lower is better;
    the dashed line at 1.0 is the persistence baseline. Models defined on a single
    channel (e.g. the tidal baseline) are excluded from the cross-channel mean.
    """
    n_ch = results["channel"].nunique()
    fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=130)
    for model, group in results.groupby("model"):
        if group["channel"].nunique() < n_ch:
            continue  # not defined on all channels → not comparable as a mean
        agg = (group.groupby(["rate", "seed"])["skill"].mean()      # mean across channels
               .groupby("rate").agg(["mean", "std"]).reset_index().sort_values("rate"))
        ax.errorbar(agg["rate"], agg["mean"], yerr=agg["std"],
                    marker="o", capsize=3, label=model)
    ax.axhline(1.0, ls="--", color="grey", lw=1, alpha=0.7, label="persistence (skill=1)")
    ax.set_xlabel("Input dropout rate")
    ax.set_ylabel("Forecast skill vs persistence (mean across channels)")
    ax.set_title(f"Robustness to {mode} input dropout — Scripps Pier")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_per_channel(results: pd.DataFrame, out_path: Path, *, value: str = "nmse") -> None:
    """Per-channel panels of normalized MSE (1.0 = predicting the mean)."""
    channels = sorted(results["channel"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), dpi=130, sharex=True)
    axes = axes.flatten()
    for ax, ch in zip(axes, channels):
        sub = results[results["channel"] == ch]
        for model, g in sub.groupby("model"):
            agg = g.groupby("rate")[value].agg(["mean", "std"]).reset_index().sort_values("rate")
            ax.errorbar(agg["rate"], agg["mean"], yerr=agg["std"],
                        marker="o", capsize=2, label=model)
        ax.axhline(1.0, ls="--", color="grey", lw=0.8, alpha=0.6)
        ax.set_title(ch)
        ax.grid(True, alpha=0.3)
    for ax in axes[len(channels):]:
        ax.axis("off")
    axes[0].legend(loc="upper left", fontsize=7)
    fig.supxlabel("Input dropout rate")
    fig.supylabel("Normalized MSE (1.0 = climatological mean)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_example_forecasts(
    examples: list[dict],
    channels: list[str],
    out_path: Path,
) -> None:
    """Each item in `examples` has keys: t_fcst, target, lstm, latent_ode, title.

    Plots a column of (1 panel per channel) per example window.
    """
    n_examples = len(examples)
    n_ch = len(channels)
    fig, axes = plt.subplots(n_ch, n_examples, figsize=(4.0 * n_examples, 1.5 * n_ch),
                             dpi=130, sharex="col")
    if n_examples == 1:
        axes = axes[:, None]
    for j, ex in enumerate(examples):
        for i, ch in enumerate(channels):
            ax = axes[i, j]
            ax.plot(ex["t_fcst"], ex["target"][:, i], color="black", lw=1.2, label="truth")
            ax.plot(ex["t_fcst"], ex["lstm"][:, i],   color="tab:orange", lw=1.0, label="LSTM")
            ax.plot(ex["t_fcst"], ex["latent_ode"][:, i], color="tab:blue",  lw=1.0, label="Latent ODE")
            if i == 0:
                ax.set_title(ex["title"], fontsize=9)
            if j == 0:
                ax.set_ylabel(ch, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2)
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.supxlabel("hours from window start")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def aggregate_summary(results: pd.DataFrame, value: str = "skill") -> pd.DataFrame:
    """Wide table for the report: rate × model, mean `value` across channels.

    Defaults to skill (dimensionless). Models not defined on every channel
    (e.g. the water_level-only tidal baseline) are dropped so the cross-channel
    mean is comparable.
    """
    n_ch = results["channel"].nunique()
    full = results.groupby("model").filter(lambda g: g["channel"].nunique() == n_ch)
    pivot = full.groupby(["model", "rate"])[value].mean().reset_index()
    return pivot.pivot(index="rate", columns="model", values=value).reset_index()
