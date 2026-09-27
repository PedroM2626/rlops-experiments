"""Post-processing of the frozen-policy EVALUATION results (results_eval/<ENV_TAG>): for a
single training seed per library, reports per-episode return statistics plus Kruskal-Wallis
and Bonferroni-corrected Mann-Whitney U tests, and saves a boxplot. The spread shown here is
evaluation noise only -- no training-seed variance. Defaults to LunarLander-v3.
"""
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from itertools import combinations
import os

ENV_TAG = os.environ.get("PPO_ENV", "LunarLander-v3").replace("/", "_") + os.environ.get("PPO_RUN_TAG", "")
EVAL_SEED = os.environ.get("PPO_SEED", "42")
RESULTS_DIR = f"/home/claude/ppo-benchmark/results_eval/{ENV_TAG}"

labels = {
    "stable_baselines3": "Stable-Baselines3",
    "cleanrl_style_pytorch": "Pure PyTorch\n(CleanRL style)",
    "jax_pure": "Pure JAX",
    "rllib": "RLlib (Ray)",
    "cleanrl_original": "CleanRL\n(official)",
}
colors = {
    "stable_baselines3": "#4C72B0",
    "cleanrl_style_pytorch": "#DD8452",
    "jax_pure": "#55A868",
    "rllib": "#C44E52",
    "cleanrl_original": "#8172B2",
}

data = {}
for path in glob.glob(f"{RESULTS_DIR}/*_seed{EVAL_SEED}.json"):
    d = json.load(open(path))
    data[d["name"]] = d

order = sorted(data.keys(), key=lambda n: -np.mean(data[n]["eval_returns"]))
arrays = {n: np.array(data[n]["eval_returns"]) for n in order}
n_eps = len(next(iter(arrays.values())))

print("=" * 70)
print(f"EVALUATION: 1 training seed (seed={EVAL_SEED}), {n_eps} evaluation episodes per library")
print("=" * 70)
for n in order:
    a = arrays[n]
    print(f"{labels.get(n,n).replace(chr(10),' '):32s} mean ={a.mean():8.1f}  median ={np.median(a):8.1f}  "
          f"std={a.std():7.1f}  min={a.min():7.1f}  max={a.max():7.1f}")

h_stat, p_kw = stats.kruskal(*[arrays[n] for n in order])
print(f"\nKruskal-Wallis: H={h_stat:.3f}, p={p_kw:.4f}")

pairs = list(combinations(order, 2))
alpha_corr = 0.05 / len(pairs)
print(f"Mann-Whitney U pairwise (Bonferroni, corrected alpha={alpha_corr:.4f}):")
rows = []
for a, b in pairs:
    u, p = stats.mannwhitneyu(arrays[a], arrays[b], alternative="two-sided")
    rows.append((labels.get(a,a).replace(chr(10),' '), labels.get(b,b).replace(chr(10),' '), p, "yes" if p < alpha_corr else "no"))
rows.sort(key=lambda r: r[2])
for a, b, p, sig in rows:
    print(f"  {a:28s} vs {b:28s} p={p:8.4f}  {sig}")

# boxplot
fig, ax = plt.subplots(figsize=(10, 6.5))
plot_data = [arrays[n] for n in order]
bp = ax.boxplot(plot_data, tick_labels=[labels.get(n, n) for n in order], patch_artist=True,
                 showmeans=True, meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=6))
for patch, n in zip(bp["boxes"], order):
    patch.set_facecolor(colors.get(n, "#888888"))
    patch.set_alpha(0.6)
for i, a in enumerate(plot_data):
    x = np.random.normal(i + 1, 0.05, size=len(a))
    ax.scatter(x, a, alpha=0.4, s=14, color="black", zorder=3)
ax.set_ylabel("Return per evaluation episode")
ax.set_title(f"1 training run per library (seed={EVAL_SEED}, 1M steps) — variance comes only from evaluation (n={n_eps} episodes)\n"
             f"Kruskal-Wallis: H={h_stat:.2f}, p={p_kw:.4f}")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
out_path = f"/mnt/user-data/outputs/ppo_eval_boxplot_{ENV_TAG}.png"
fig.savefig(out_path, dpi=150)
print(f"\nsaved boxplot to {out_path}")
