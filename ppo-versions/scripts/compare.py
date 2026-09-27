"""Post-processing: one chart of mean +/- std training curve per library.
Interpolates every seed run found in results/<ENV_TAG>/*.json onto a common timestep grid and
prints a Markdown summary table (final reward, wall time) sorted by final reward.
"""
import json
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import os
ENV_TAG = os.environ.get("PPO_ENV", "CartPole-v1").replace("/", "_") + os.environ.get("PPO_RUN_TAG", "")
RESULTS_DIR = f"/home/claude/ppo-benchmark/results/{ENV_TAG}"
OUT_PATH = f"/mnt/user-data/outputs/ppo_comparison_{ENV_TAG}.png"

colors = {
    "stable_baselines3": "#4C72B0",
    "cleanrl_style_pytorch": "#DD8452",
    "jax_pure": "#55A868",
    "rllib": "#C44E52",
    "cleanrl_original": "#8172B2",
}
labels = {
    "stable_baselines3": "Stable-Baselines3",
    "cleanrl_style_pytorch": "Pure PyTorch (CleanRL style)",
    "jax_pure": "Pure JAX",
    "rllib": "RLlib (Ray)",
    "cleanrl_original": "CleanRL (official)",
}

# group files by library (ignoring the _seedN suffix)
runs = defaultdict(list)
for path in glob.glob(f"{RESULTS_DIR}/*.json"):
    fname = path.split("/")[-1].replace(".json", "")
    m = re.match(r"(.+)_seed\d+$", fname)
    name = m.group(1) if m else fname
    with open(path) as f:
        runs[name].append(json.load(f))

fig, ax = plt.subplots(figsize=(9.5, 6))

summary_rows = []
for name, seed_runs in runs.items():
    seed_runs = sorted(seed_runs, key=lambda d: d.get("seed", 0))
    n_seeds = len(seed_runs)

    common_max = min(r["rewards"][-1][0] for r in seed_runs if r["rewards"])
    grid = np.linspace(0, common_max, 200)
    interp_curves = []
    for r in seed_runs:
        if not r["rewards"]:
            continue
        xs = np.array([p[0] for p in r["rewards"]])
        ys = np.array([p[1] for p in r["rewards"]])
        interp_curves.append(np.interp(grid, xs, ys))
    interp_curves = np.array(interp_curves)
    mean_curve = interp_curves.mean(axis=0)
    std_curve = interp_curves.std(axis=0)

    color = colors.get(name, None)
    ax.plot(grid, mean_curve, label=f"{labels.get(name, name)} (n={n_seeds})", color=color, linewidth=2)
    ax.fill_between(grid, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.15)

    finals = [r["final_reward"] for r in seed_runs]
    times = [r["elapsed_seconds"] for r in seed_runs]
    summary_rows.append({
        "name": labels.get(name, name),
        "n_seeds": n_seeds,
        "final_mean": float(np.mean(finals)),
        "final_std": float(np.std(finals)),
        "time_mean": float(np.mean(times)),
    })

ax.set_xlabel("Timesteps")
ax.set_ylabel("Mean reward (20-episode window)")
ax.set_title("PPO on CartPole-v1 — mean ± std deviation across seeds")
ax.axhline(500, color="gray", linestyle="--", alpha=0.4, label="Max possible (500)")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150)
print(f"saved chart to {OUT_PATH}")

summary_rows.sort(key=lambda r: -r["final_mean"])
print("\n| Library | Seeds | Final reward (mean ± std) | Mean time (s) |")
print("|---|---|---|---|")
for r in summary_rows:
    print(f"| {r['name']} | {r['n_seeds']} | {r['final_mean']:.1f} ± {r['final_std']:.1f} | {r['time_mean']:.1f} |")
