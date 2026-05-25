"""Figure generation for the robustness experiment."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_mse_vs_dropout(results: pd.DataFrame, out_path: Path) -> None:
    """results columns: model, rate, seed, channel, mse, mae

    Plots one line per model: mean MSE (across channels and seeds) vs dropout rate.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=130)
    for model, group in results.groupby("model"):
        agg = (group.groupby("rate")["mse"]
               .agg(["mean", "std"])
               .reset_index()
               .sort_values("rate"))
        ax.errorbar(agg["rate"], agg["mean"], yerr=agg["std"],
                    marker="o", capsize=3, label=model)
    ax.set_xlabel("Input dropout rate")
    ax.set_ylabel("Forecast MSE (denormalized, mean across channels)")
    ax.set_title("Robustness to input sensor dropout — Scripps Pier")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_per_channel(results: pd.DataFrame, out_path: Path) -> None:
    channels = sorted(results["channel"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), dpi=130, sharex=True)
    axes = axes.flatten()
    for ax, ch in zip(axes, channels):
        sub = results[results["channel"] == ch]
        for model, g in sub.groupby("model"):
            agg = g.groupby("rate")["mse"].agg(["mean", "std"]).reset_index().sort_values("rate")
            ax.errorbar(agg["rate"], agg["mean"], yerr=agg["std"],
                        marker="o", capsize=2, label=model)
        ax.set_title(ch)
        ax.grid(True, alpha=0.3)
    for ax in axes[len(channels):]:
        ax.axis("off")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.supxlabel("Input dropout rate")
    fig.supylabel("Forecast MSE (denormalized)")
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


def aggregate_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Wide-format table for the report: model × rate, average MSE across channels."""
    pivot = (results.groupby(["model", "rate"])["mse"]
             .agg(["mean", "std"])
             .reset_index())
    return pivot.pivot(index="rate", columns="model", values="mean").reset_index()
