"""Post-processing of the TRAINING results, text-only: descriptive statistics on each library's
distribution of final reward over however many seeds are on disk in results/<ENV_TAG>, a
Kruskal-Wallis omnibus test, then Bonferroni-corrected pairwise Mann-Whitney U tests.
"""
import json
import glob
import re
from collections import defaultdict
from itertools import combinations

import numpy as np
from scipy import stats

import os
import sys

# Windows consoles default to cp1252 and cannot encode the arrows printed below
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_TAG = os.environ.get("PPO_ENV", "CartPole-v1").replace("/", "_") + os.environ.get("PPO_RUN_TAG", "")
RESULTS_DIR = f"{PROJECT_ROOT}/results/{ENV_TAG}"

labels = {
    "stable_baselines3": "Stable-Baselines3",
    "cleanrl_style_pytorch": "Pure PyTorch (CleanRL style)",
    "jax_pure": "Pure JAX",
    "rllib": "RLlib (Ray)",
    "cleanrl_original": "CleanRL (official)",
}

runs = defaultdict(list)
for path in glob.glob(f"{RESULTS_DIR}/*.json"):
    fname = os.path.basename(path).replace(".json", "")
    m = re.match(r"(.+)_seed\d+$", fname)
    name = m.group(1) if m else fname
    with open(path) as f:
        runs[name].append(json.load(f))

finals = {name: np.array([r["final_reward"] for r in seed_runs
                          if r["final_reward"] is not None])
          for name, seed_runs in runs.items()}
# an arm whose run stored a null final_reward has no sample left to test
excluded = sorted(n for n, v in finals.items() if v.size == 0)
finals = {n: v for n, v in finals.items() if v.size}

counts = [len(v) for v in finals.values()]
n_label = (f"{min(counts)}-{max(counts)}"
           if min(counts) != max(counts) else str(counts[0]))
print("=" * 70)
print(f"DESCRIPTIVE STATISTICS (n={n_label} seeds per library)")
print("=" * 70)
if excluded:
    print(f"excluded, no final reward recorded: {', '.join(excluded)}")
for name, vals in sorted(finals.items(), key=lambda kv: -kv[1].mean()):
    print(f"{labels.get(name, name):32s} mean ={vals.mean():7.1f}  median ={np.median(vals):7.1f}  "
          f"std={vals.std():6.1f}  min={vals.min():6.1f}  max={vals.max():6.1f}")

print()
print("=" * 70)
print("KRUSKAL-WALLIS (is there ANY difference between the 5 distributions?)")
print("=" * 70)
names_order = list(finals.keys())
h_stat, p_kw = stats.kruskal(*[finals[n] for n in names_order])
print(f"H = {h_stat:.3f}, p = {p_kw:.4f}")
if p_kw < 0.05:
    print("→ p < 0.05: there is evidence that at least one library differs from the others.")
else:
    print("→ p >= 0.05: there is NO statistical evidence of a difference between the libraries.")
    print("  The observed difference in the means is consistent with seed variance.")

print()
print("=" * 70)
print("MANN-WHITNEY U pairwise (with Bonferroni correction)")
print("=" * 70)
pairs = list(combinations(names_order, 2))
alpha = 0.05
alpha_corrected = alpha / len(pairs)
print(f"original alpha = {alpha}, corrected alpha (Bonferroni, {len(pairs)} comparisons) = {alpha_corrected:.4f}")
print()

rows = []
for a, b in pairs:
    u_stat, p_val = stats.mannwhitneyu(finals[a], finals[b], alternative="two-sided")
    sig = "yes" if p_val < alpha_corrected else "no"
    rows.append((labels.get(a, a), labels.get(b, b), p_val, sig))

rows.sort(key=lambda r: r[2])
print(f"{'Library A':32s} {'Library B':32s} {'p-value':>10s}  {'sig?':>5s}")
for a, b, p, sig in rows:
    print(f"{a:32s} {b:32s} {p:10.4f}  {sig:>5s}")

n_sig = sum(1 for r in rows if r[3] == "yes")
print()
print(f"{n_sig} of {len(rows)} pairs showed a statistically significant difference "
      f"(p < {alpha_corrected:.4f} after Bonferroni correction).")
