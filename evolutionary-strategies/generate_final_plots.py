"""
Generate the final unified comparative plots with all 3 families across all 4 environments.
Reads Families 1+3 from raw_results.json and Family 2 from gp_results.json.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ART_DIR  = OUT_DIR
os.makedirs(OUT_DIR, exist_ok=True)

# ── Data for Families 1, 2, and 3 ────────────────────────────────────────────
f13_path = os.path.join(OUT_DIR, "raw_results.json")
gp_path  = os.path.join(OUT_DIR, "gp_results.json")

with open(f13_path) as f:
    f13_data = json.load(f)
with open(gp_path) as f:
    gp_data = json.load(f)

fam_of = {"SimpleGA": 1, "DE": 1, "OpenAI-ES": 1, "LinearGP": 2, "CartesianGP": 2, "CMA-ES": 3, "PBIL": 3}
ALL_ENVS = ["CartPole-v1", "Acrobot-v1", "Pendulum-v1", "MountainCarContinuous-v0"]

# Consolidated table across all 4 environments
ROWS = []
for env in ALL_ENVS:
    for r in f13_data.get(env, []):
        fam = fam_of.get(r["algorithm"], r.get("family", 1))
        ROWS.append(dict(
            algo=r["algorithm"], family=fam, env=env,
            best=r["best_fitness"][-1], n_gen=r["generations"][-1] + 1,
            t=r["total_time"], n_eval=r["n_evals"],
        ))
    for r in gp_data.get(env, []):
        # avoid duplicating if already in f13_data
        if not any(x["algo"] == r["algorithm"] and x["env"] == env for x in ROWS):
            ROWS.append(dict(
                algo=r["algorithm"], family=fam_of.get(r["algorithm"], 2), env=env,
                best=r["best_fitness"][-1], n_gen=r["generations"][-1] + 1,
                t=r["total_time"], n_eval=r["n_evals"],
            ))

# ── 1. Unified 2x2 learning curves: 3 families × 4 environments ──────────────
COLORS = {1: "#E63946", 2: "#457B9D", 3: "#52B788"}
MARKERS= {"SimpleGA": "o", "DE": "s", "CMA-ES": "^", "LinearGP": "D", "CartesianGP": "P", "OpenAI-ES": "v", "PBIL": "X"}

fig, axes = plt.subplots(2, 2, figsize=(16, 11), facecolor="#0d1117")
fig.suptitle("Full Evolutionary Benchmark — 3 Families × 4 Environments (JAX Vectorized)",
             color="white", fontsize=15, fontweight="bold", y=0.995)
axes = axes.flatten()

for ax, env in zip(axes, ALL_ENVS):
    ax.set_facecolor("#161b22")
    ax.set_title(env, color="white", fontsize=12, pad=8, fontweight="bold")
    ax.set_xlabel("Generation", color="#8b949e", fontsize=9)
    ax.set_ylabel("Best Fitness", color="#8b949e", fontsize=9)
    ax.tick_params(colors="#8b949e", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.grid(True, color="#21262d", linewidth=0.5, linestyle="--")

    # F1 and F3
    for r in f13_data.get(env, []):
        fam = fam_of.get(r["algorithm"], r.get("family", 1))
        if fam == 2:
            continue
        color = COLORS[fam]
        mk = MARKERS.get(r["algorithm"], "o")
        ls = {"SimpleGA": "-", "DE": "--", "CMA-ES": "-.", "OpenAI-ES": "-", "PBIL": "--"}.get(r["algorithm"], "-")
        x = np.array(r["generations"])
        y = np.array(r["best_fitness"])
        ax.plot(x, y, color=color, linestyle=ls, linewidth=1.8,
                marker=mk, markevery=max(1, len(x) // 6), markersize=4.5,
                label=f"{r['algorithm']} (F{fam})", alpha=0.9)

    # F2 GP
    for r in gp_data.get(env, []):
        fam = 2
        color = COLORS[fam]
        mk = MARKERS.get(r["algorithm"], "D")
        ls = {"LinearGP": "-", "CartesianGP": "--"}.get(r["algorithm"], "-")
        x = np.array(r["generations"])
        y = np.array(r["best_fitness"])
        ax.plot(x, y, color=color, linestyle=ls, linewidth=1.8,
                marker=mk, markevery=max(1, len(x) // 6), markersize=5,
                label=f"{r['algorithm']} (F2)", alpha=0.9)

    ax.legend(fontsize=7.5, framealpha=0.25, labelcolor="white",
              facecolor="#21262d", edgecolor="#30363d", loc="lower right")

plt.tight_layout()
out1 = os.path.join(OUT_DIR, "all_families_curves.png")
out2 = os.path.join(ART_DIR, "all_families_curves.png")
plt.savefig(out1, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.savefig(out2, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {out1}")

# ── 2. Unified 2x2 barplot: 3 families × 4 environments ───────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 11), facecolor="#0d1117")
fig2.suptitle("Final Performance Comparison: All 3 Families × 4 Environments (JAX Vectorized)",
              color="white", fontsize=15, fontweight="bold", y=0.995)
axes2 = axes2.flatten()

all_algos_ordered = ["SimpleGA", "DE", "OpenAI-ES", "LinearGP", "CartesianGP", "CMA-ES", "PBIL"]
algo_colors = {
    "SimpleGA": "#E63946", "DE": "#FF6B6B", "OpenAI-ES": "#FF8FA3",
    "LinearGP": "#457B9D", "CartesianGP": "#A8DADC",
    "CMA-ES": "#2D6A4F", "PBIL": "#52B788"
}

# Assemble final fitness values per env and algo
vals = {}
for env in ALL_ENVS:
    vals[env] = {}
    for r in f13_data.get(env, []):
        vals[env][r["algorithm"]] = r["best_fitness"][-1]
    for r in gp_data.get(env, []):
        vals[env][r["algorithm"]] = r["best_fitness"][-1]

for idx, env in enumerate(ALL_ENVS):
    ax = axes2[idx]
    ax.set_facecolor("#161b22")
    ax.set_title(env, color="white", fontsize=12, pad=8, fontweight="bold")
    ax.set_ylabel("Final Best Fitness", color="#8b949e", fontsize=9.5)
    ax.tick_params(colors="#8b949e", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.grid(True, axis="y", color="#21262d", linewidth=0.5, linestyle="--")

    env_vals = [vals.get(env, {}).get(a, 0.0) for a in all_algos_ordered]
    x_pos = np.arange(len(all_algos_ordered))
    bars = ax.bar(x_pos, env_vals, width=0.72,
                  color=[algo_colors[a] for a in all_algos_ordered], alpha=0.88,
                  edgecolor="#0d1117", linewidth=0.5)

    for bar, v in zip(bars, env_vals):
        offset = abs(v) * 0.03 if v != 0 else (10.0 if "CartPole" in env else 1.0)
        va = "bottom" if v >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, v + (offset if v >= 0 else -offset),
                f"{v:.0f}", ha="center", va=va,
                color="white", fontsize=8, fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{a}\n(F{fam_of[a]})" for a in all_algos_ordered],
                       color="#8b949e", fontsize=8.5)

plt.tight_layout()
out3 = os.path.join(OUT_DIR, "all_families_bar.png")
out4 = os.path.join(ART_DIR, "all_families_bar.png")
out5 = os.path.join(OUT_DIR, "family_comparison.png")
plt.savefig(out3, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
plt.savefig(out4, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
plt.savefig(out5, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
plt.close()
print(f"Saved: {out3}")
print(f"Saved: {out5}")

# ── Complete summary table ────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("COMPLETE TABLE — ALL 3 FAMILIES × 4 ENVIRONMENTS")
print("=" * 80)
print(f"{'Algorithm':<14} {'Family':<8} {'Environment':<28} {'Best':>10} {'Gens':>8} {'t(s)':>7}")
print("-" * 80)
for r in ROWS:
    print(f"{r['algo']:<14} F{r['family']:<7} {r['env']:<28} {r['best']:>10.2f} {r['n_gen']:>8} {r['t']:>7.1f}")
