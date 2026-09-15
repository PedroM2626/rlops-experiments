import json
import glob
import re
from collections import defaultdict
from itertools import combinations

import numpy as np
from scipy import stats

import os
ENV_TAG = os.environ.get("PPO_ENV", "CartPole-v1").replace("/", "_") + os.environ.get("PPO_RUN_TAG", "")
RESULTS_DIR = f"/home/claude/ppo-benchmark/results/{ENV_TAG}"

labels = {
    "stable_baselines3": "Stable-Baselines3",
    "cleanrl_style_pytorch": "PyTorch puro (estilo CleanRL)",
    "jax_pure": "JAX puro",
    "rllib": "RLlib (Ray)",
    "cleanrl_original": "CleanRL (oficial)",
}

runs = defaultdict(list)
for path in glob.glob(f"{RESULTS_DIR}/*.json"):
    fname = path.split("/")[-1].replace(".json", "")
    m = re.match(r"(.+)_seed\d+$", fname)
    name = m.group(1) if m else fname
    with open(path) as f:
        runs[name].append(json.load(f))

finals = {name: np.array([r["final_reward"] for r in seed_runs]) for name, seed_runs in runs.items()}

print("=" * 70)
print("ESTATÍSTICA DESCRITIVA (n=20 seeds cada)")
print("=" * 70)
for name, vals in sorted(finals.items(), key=lambda kv: -kv[1].mean()):
    print(f"{labels.get(name, name):32s} média={vals.mean():7.1f}  mediana={np.median(vals):7.1f}  "
          f"std={vals.std():6.1f}  min={vals.min():6.1f}  max={vals.max():6.1f}")

print()
print("=" * 70)
print("KRUSKAL-WALLIS (existe QUALQUER diferença entre as 5 distribuições?)")
print("=" * 70)
names_order = list(finals.keys())
h_stat, p_kw = stats.kruskal(*[finals[n] for n in names_order])
print(f"H = {h_stat:.3f}, p = {p_kw:.4f}")
if p_kw < 0.05:
    print("→ p < 0.05: há evidência de que pelo menos uma lib difere das outras.")
else:
    print("→ p >= 0.05: NÃO há evidência estatística de diferença entre as libs.")
    print("  A diferença observada nas médias é compatível com variância de seed.")

print()
print("=" * 70)
print("MANN-WHITNEY U pairwise (com correção de Bonferroni)")
print("=" * 70)
pairs = list(combinations(names_order, 2))
alpha = 0.05
alpha_corrected = alpha / len(pairs)
print(f"alpha original = {alpha}, alpha corrigido (Bonferroni, {len(pairs)} comparações) = {alpha_corrected:.4f}")
print()

rows = []
for a, b in pairs:
    u_stat, p_val = stats.mannwhitneyu(finals[a], finals[b], alternative="two-sided")
    sig = "SIM" if p_val < alpha_corrected else "não"
    rows.append((labels.get(a, a), labels.get(b, b), p_val, sig))

rows.sort(key=lambda r: r[2])
print(f"{'Lib A':32s} {'Lib B':32s} {'p-valor':>10s}  {'sig?':>5s}")
for a, b, p, sig in rows:
    print(f"{a:32s} {b:32s} {p:10.4f}  {sig:>5s}")

n_sig = sum(1 for r in rows if r[3] == "SIM")
print()
print(f"{n_sig} de {len(rows)} pares mostraram diferença estatisticamente significativa "
      f"(p < {alpha_corrected:.4f} após correção de Bonferroni).")
