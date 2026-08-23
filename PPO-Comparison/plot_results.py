"""Generate comparison plots from training curves and evaluation results.

Produces:
  1. Training curves: mean reward vs timesteps (with smoothing and std bands)
  2. Boxplot: distribution of evaluation rewards per implementation
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CURVES_DIR, IMPLEMENTATIONS, PLOTS_DIR, RESULTS_DIR


# Plot styling
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "grid.alpha": 0.6,
    "font.size": 11,
    "font.family": "sans-serif",
})

COLORS = {
    "sb3": "#58a6ff",
    "cleanrl": "#3fb950",
    "custom_continuous": "#f0883e",
    "custom_discrete": "#f85149",
    "torchrl": "#d2a8ff",
    "rllib_extracted": "#ff7b72",
}

LABELS = {impl["id"]: impl["label"] for impl in IMPLEMENTATIONS}


def smooth(values: np.ndarray, window: int = 20) -> np.ndarray:
    """Apply rolling mean smoothing."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_training_curves() -> Path | None:
    """Plot training reward curves for all implementations."""
    fig, ax = plt.subplots(figsize=(12, 6))
    has_data = False

    for impl in IMPLEMENTATIONS:
        impl_id = impl["id"]
        csv_path = CURVES_DIR / f"{impl_id}_curve.csv"
        if not csv_path.exists():
            print(f"  Curve not found: {csv_path}")
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"  Error reading {csv_path}: {exc}")
            continue

        if "step" not in df.columns or "mean_reward" not in df.columns:
            print(f"  Invalid columns in {csv_path}: {df.columns.tolist()}")
            continue

        steps = df["step"].values
        rewards = df["mean_reward"].values

        if len(steps) == 0:
            continue

        color = COLORS.get(impl_id, "#8b949e")
        label = LABELS.get(impl_id, impl_id)

        # Raw data (faded)
        ax.plot(steps, rewards, color=color, alpha=0.15, linewidth=0.8)

        # Smoothed line
        window = min(20, max(3, len(rewards) // 10))
        smoothed = smooth(rewards, window)
        offset = len(rewards) - len(smoothed)
        smooth_steps = steps[offset:]
        ax.plot(smooth_steps, smoothed, color=color, linewidth=2.5, label=label)

        has_data = True

    if not has_data:
        print("  No training curve data found.")
        plt.close(fig)
        return None

    ax.set_xlabel("Training Timesteps", fontsize=13, fontweight="bold")
    ax.set_ylabel("Mean Episode Reward (rolling 20)", fontsize=13, fontweight="bold")
    ax.set_title("PPO Training Curves - Implementation Comparison", fontsize=15, fontweight="bold", color="#f0f6fc")
    ax.legend(loc="lower right", fontsize=11, fancybox=True, framealpha=0.8,
              facecolor="#161b22", edgecolor="#30363d")
    ax.grid(True, linewidth=0.5)
    ax.axhline(y=200, color="#484f58", linestyle="--", linewidth=1.0, alpha=0.7, label="_nolegend_")
    ax.text(ax.get_xlim()[1] * 0.98, 205, "Solved threshold (200)",
            ha="right", fontsize=9, color="#484f58", style="italic")

    fig.tight_layout()
    output_path = PLOTS_DIR / "training_curves.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Training curves saved to {output_path}")
    return output_path


def plot_boxplot() -> Path | None:
    """Plot boxplot of evaluation rewards per implementation."""
    eval_path = RESULTS_DIR / "eval_rewards.csv"
    if not eval_path.exists():
        print("  Evaluation results not found.")
        return None

    try:
        df = pd.read_csv(eval_path)
    except Exception as exc:
        print(f"  Error reading eval results: {exc}")
        return None

    if df.empty:
        print("  Evaluation results are empty.")
        return None

    fig, ax = plt.subplots(figsize=(10, 7))

    data = []
    labels = []
    colors = []
    for col in df.columns:
        values = df[col].dropna().values.astype(float)
        if len(values) > 0:
            data.append(values)
            labels.append(LABELS.get(col, col))
            colors.append(COLORS.get(col, "#8b949e"))

    if not data:
        print("  No valid evaluation data for boxplot.")
        plt.close(fig)
        return None

    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        notch=True,
        widths=0.6,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="white", markersize=7),
        medianprops=dict(color="#f0f6fc", linewidth=2),
        whiskerprops=dict(color="#8b949e", linewidth=1.5),
        capprops=dict(color="#8b949e", linewidth=1.5),
        flierprops=dict(marker="o", markerfacecolor="#484f58", markeredgecolor="#484f58", markersize=4),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor("#c9d1d9")
        patch.set_linewidth(1.5)

    # Add individual points (jittered)
    rng = np.random.default_rng(42)
    for idx, (values, color) in enumerate(zip(data, colors)):
        jitter = rng.uniform(-0.15, 0.15, size=len(values))
        ax.scatter(
            np.full(len(values), idx + 1) + jitter,
            values,
            color=color,
            alpha=0.25,
            s=12,
            zorder=2,
        )

    # Add mean annotation
    for idx, values in enumerate(data):
        mean_val = np.mean(values)
        ax.annotate(
            f"{mean_val:.1f}",
            xy=(idx + 1, mean_val),
            xytext=(idx + 1.35, mean_val),
            fontsize=9,
            color="#c9d1d9",
            fontweight="bold",
            arrowprops=dict(arrowstyle="-", color="#484f58", lw=0.8),
        )

    ax.set_ylabel("Episode Reward", fontsize=13, fontweight="bold")
    ax.set_title("PPO Evaluation Rewards by Implementation\n(100 episodes each, deterministic policy)",
                 fontsize=14, fontweight="bold", color="#f0f6fc")
    ax.grid(True, axis="y", linewidth=0.5)
    ax.axhline(y=200, color="#484f58", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(len(data) + 0.4, 205, "Solved (200)", fontsize=9, color="#484f58", style="italic")

    fig.tight_layout()
    output_path = PLOTS_DIR / "eval_boxplot.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Boxplot saved to {output_path}")
    return output_path


def main() -> None:
    print("=" * 60)
    print("PPO Implementation Comparison - Plot Generation")
    print("=" * 60)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/2] Training curves...")
    plot_training_curves()

    print("\n[2/2] Evaluation boxplot...")
    plot_boxplot()

    print("\nPlot generation complete.")


if __name__ == "__main__":
    main()
