"""Post-processing: boxplot of the final reward per library over every seed on disk.
Reads the results/<ENV_TAG>/*.json written by the train_*.py scripts, pools them by library
(stripping the _seedN suffix) and reports the Kruskal-Wallis omnibus test in the chart title.
"""
import json
import glob
import re
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_TAG = os.environ.get("PPO_ENV", "CartPole-v1").replace("/", "_") + os.environ.get("PPO_RUN_TAG", "")
RESULTS_DIR = f"{PROJECT_ROOT}/results/{ENV_TAG}"
FIGURES_DIR = f"{PROJECT_ROOT}/figures"
OUT_PATH = f"{FIGURES_DIR}/ppo_boxplot_{ENV_TAG}.png"

labels = {
    "stable_baselines3": "Stable-\nBaselines3",
    "cleanrl_style_pytorch": "Pure PyTorch\n(CleanRL style)",
    "jax_pure": "Pure JAX",
    "rllib": "RLlib\n(Ray)",
    "cleanrl_original": "CleanRL\n(official)",
}
colors = {
    "stable_baselines3": "#4C72B0",
    "cleanrl_style_pytorch": "#DD8452",
    "jax_pure": "#55A868",
    "rllib": "#C44E52",
    "cleanrl_original": "#8172B2",
}

runs = defaultdict(list)
for path in glob.glob(f"{RESULTS_DIR}/*.json"):
    fname = os.path.basename(path).replace(".json", "")
    m = re.match(r"(.+)_seed\d+$", fname)
    name = m.group(1) if m else fname
    with open(path) as f:
        runs[name].append(json.load(f))

# One arm (tianshou on LunarLander) stored a null final_reward because its run
# produced no curve; it cannot be boxed, so drop it here and name it in the
# title instead of letting the figure silently omit it.
usable = {n: [r for r in runs[n] if r.get("final_reward") is not None] for n in runs}
excluded = sorted(n for n in usable if not usable[n])
usable = {n: v for n, v in usable.items() if v}
order = sorted(usable.keys(), key=lambda n: -np.mean([r["final_reward"] for r in usable[n]]))
data = [np.array([r["final_reward"] for r in usable[n]]) for n in order]
seed_counts = [len(d) for d in data]
n_seeds = (f"{min(seed_counts)}-{max(seed_counts)}"
           if min(seed_counts) != max(seed_counts) else str(seed_counts[0]))

h_stat, p_kw = stats.kruskal(*data)

fig, ax = plt.subplots(figsize=(10, 6.5))
bp = ax.boxplot(data, labels=[labels.get(n, n) for n in order], patch_artist=True, showmeans=True,
                 meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=6))
for patch, name in zip(bp["boxes"], order):
    patch.set_facecolor(colors.get(name, "#888888"))
    patch.set_alpha(0.6)

for i, d in enumerate(data):
    x = np.random.normal(i + 1, 0.05, size=len(d))
    ax.scatter(x, d, alpha=0.5, s=18, color="black", zorder=3)

ax.axhline(500, color="gray", linestyle="--", alpha=0.4, label="Max possible (500)")
ax.set_ylabel("Final reward (mean of the last 20 episodes)")
title = (f"Distribution of final reward per library on {ENV_TAG}\n"
         f"(n={n_seeds} seeds per library)   "
         f"Kruskal-Wallis: H={h_stat:.2f}, p={p_kw:.4f}")
if excluded:
    title += f"\nnot shown, no final reward recorded: {', '.join(excluded)}"
ax.set_title(title)
ax.legend(loc="lower right")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
os.makedirs(FIGURES_DIR, exist_ok=True)
fig.savefig(OUT_PATH, dpi=150)
print(f"saved boxplot to {OUT_PATH}")
if excluded:
    print(f"excluded, no final reward recorded: {', '.join(excluded)}")
