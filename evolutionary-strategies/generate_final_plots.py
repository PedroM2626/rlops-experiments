"""
Generate the final unified comparative plot with all 3 families.
Reads Families 1+3 from the main results JSON and Family 2 from gp_results.json.
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

# ── Data for Families 1 and 3 ────────────────────────────────────────────────
f13_path = os.path.join(OUT_DIR, "raw_results.json")
gp_path  = os.path.join(OUT_DIR, "gp_results.json")

with open(f13_path) as f:
    f13_data = json.load(f)
with open(gp_path) as f:
    gp_data = json.load(f)

fam_of = {"SimpleGA": 1, "DE": 1, "OpenAI-ES": 1, "LinearGP": 2, "CartesianGP": 2, "CMA-ES": 3, "PBIL": 3}

# Consolidated table for CartPole and Acrobot
ROWS = []

# F1+F3 CartPole
for r in f13_data.get("CartPole-v1", []):
    ROWS.append(dict(
        algo=r["algorithm"], family=fam_of.get(r["algorithm"], r["family"]), env="CartPole-v1",
        best=r["best_fitness"][-1], n_gen=r["generations"][-1]+1,
        t=r["total_time"], n_eval=r["n_evals"],
    ))
# F2 CartPole
for r in gp_data.get("CartPole-v1", []):
    ROWS.append(dict(
        algo=r["algorithm"], family=fam_of.get(r["algorithm"], r["family"]), env="CartPole-v1",
        best=r["best_fitness"][-1], n_gen=r["generations"][-1]+1,
        t=r["total_time"], n_eval=r["n_evals"],
    ))
# F1+F3 Acrobot
for r in f13_data.get("Acrobot-v1", []):
    ROWS.append(dict(
        algo=r["algorithm"], family=fam_of.get(r["algorithm"], r["family"]), env="Acrobot-v1",
        best=r["best_fitness"][-1], n_gen=r["generations"][-1]+1,
        t=r["total_time"], n_eval=r["n_evals"],
    ))
# F2 Acrobot
for r in gp_data.get("Acrobot-v1", []):
    ROWS.append(dict(
        algo=r["algorithm"], family=fam_of.get(r["algorithm"], r["family"]), env="Acrobot-v1",
        best=r["best_fitness"][-1], n_gen=r["generations"][-1]+1,
        t=r["total_time"], n_eval=r["n_evals"],
    ))

# ── Unified comparative plot ────────────────────────────────────────────────
COLORS = {1: "#E63946", 2: "#457B9D", 3: "#52B788"}
MARKERS= {"SimpleGA":"o","DE":"s","CMA-ES":"^","LinearGP":"D","CartesianGP":"P","OpenAI-ES":"v","PBIL":"X"}

envs_plot = ["CartPole-v1","Acrobot-v1"]
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0d1117")
fig.suptitle("Full Benchmark — 3 Families × 2 Environments (CartPole + Acrobot)",
             color="white", fontsize=14, fontweight="bold", y=1.02)

for ax, env in zip(axes, envs_plot):
    ax.set_facecolor("#161b22")
    ax.set_title(env, color="white", fontsize=12, pad=8)
    ax.set_xlabel("Generation", color="#8b949e")
    ax.set_ylabel("Best Fitness", color="#8b949e")
    ax.tick_params(colors="#8b949e", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.grid(True, color="#21262d", linewidth=0.5, linestyle="--")

    # F1+F3
    for r in f13_data.get(env, []):
        fam   = fam_of.get(r["algorithm"], r["family"])
        color = COLORS[fam]
        mk    = MARKERS.get(r["algorithm"], "o")
        ls    = {"SimpleGA":"-","DE":"--","CMA-ES":"-.","OpenAI-ES":"-","PBIL":"--"}.get(r["algorithm"],"-")
        x = np.array(r["generations"])
        y = np.array(r["best_fitness"])
        ax.plot(x, y, color=color, linestyle=ls, linewidth=2,
                marker=mk, markevery=max(1,len(x)//8), markersize=5,
                label=f"{r['algorithm']} (F{fam})", alpha=0.9)

    # F2 GP
    for r in gp_data.get(env, []):
        fam   = r["family"]
        color = COLORS[fam]
        mk    = MARKERS.get(r["algorithm"], "D")
        ls    = {"LinearGP":"-","CartesianGP":"--"}.get(r["algorithm"],"-")
        x = np.array(r["generations"])
        y = np.array(r["best_fitness"])
        ax.plot(x, y, color=color, linestyle=ls, linewidth=2,
                marker=mk, markevery=max(1,len(x)//8), markersize=6,
                label=f"{r['algorithm']} (F2)", alpha=0.9)

    ax.legend(fontsize=8, framealpha=0.2, labelcolor="white",
              facecolor="#21262d", edgecolor="#30363d", loc="lower right")

plt.tight_layout()
out1 = os.path.join(OUT_DIR, "all_families_curves.png")
out2 = os.path.join(ART_DIR, "all_families_curves.png")
plt.savefig(out1, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.savefig(out2, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {out1}")
print(f"Saved: {out2}")

# ── Unified barplot: 3 families × 2 environments ───────────────────────────────
fig2, ax2 = plt.subplots(figsize=(13, 6), facecolor="#0d1117")
ax2.set_facecolor("#161b22")

all_algos_ordered = ["SimpleGA", "DE", "OpenAI-ES", "LinearGP", "CartesianGP", "CMA-ES", "PBIL"]
env_labels = ["CartPole-v1", "Acrobot-v1"]
n_algos = len(all_algos_ordered)
n_envs  = len(env_labels)
x = np.arange(n_envs)
width = 0.8 / n_algos

fam_of = {"SimpleGA": 1, "DE": 1, "OpenAI-ES": 1, "LinearGP": 2, "CartesianGP": 2, "CMA-ES": 3, "PBIL": 3}
cidx   = {1: 0, 2: 0, 3: 0}
algo_colors = {}
for a in all_algos_ordered:
    f = fam_of[a]
    shades = {1: ["#E63946", "#FF6B6B", "#FF8FA3"], 2: ["#457B9D", "#A8DADC"], 3: ["#2D6A4F", "#52B788"]}
    algo_colors[a] = shades[f][cidx[f] % len(shades[f])]
    cidx[f] += 1

# Assemble the bar values
vals = {}
for env in env_labels:
    vals[env] = {}
    for r in f13_data.get(env,[]):
        vals[env][r["algorithm"]] = r["best_fitness"][-1]
    for r in gp_data.get(env,[]):
        vals[env][r["algorithm"]] = r["best_fitness"][-1]

for i, algo in enumerate(all_algos_ordered):
    v = [vals.get(env, {}).get(algo, None) for env in env_labels]
    offset = (i - n_algos/2 + 0.5) * width
    for j, (vv, env) in enumerate(zip(v, env_labels)):
        if vv is None:
            continue
        bar = ax2.bar(x[j]+offset, vv, width*0.88,
                      color=algo_colors[algo], alpha=0.88,
                      label=f"{algo} (F{fam_of[algo]})" if j==0 else "_",
                      edgecolor="#0d1117", linewidth=0.5)
        fv = f"{vv:.0f}"
        ax2.text(x[j]+offset, vv + (abs(vv)*0.03 if vv>0 else -abs(vv)*0.05),
                 fv, ha="center", va="bottom" if vv>=0 else "top",
                 color="white", fontsize=7.5, fontweight="bold")

ax2.set_xticks(x)
ax2.set_xticklabels(env_labels, color="#8b949e", fontsize=11)
ax2.set_ylabel("Final Fitness (best individual)", color="#8b949e", fontsize=10)
ax2.set_title("Final Comparison: All 3 Families (F1=Direct, F2=Programs, F3=EDA)",
              color="white", fontsize=12, fontweight="bold")
ax2.tick_params(colors="#8b949e")
for spine in ax2.spines.values(): spine.set_edgecolor("#30363d")
ax2.legend(fontsize=8.5, framealpha=0.2, labelcolor="white",
           facecolor="#21262d", edgecolor="#30363d",
           bbox_to_anchor=(1.01,1), loc="upper left")
ax2.grid(True, axis="y", color="#21262d", linewidth=0.5, linestyle="--")

plt.tight_layout()
out3 = os.path.join(OUT_DIR, "all_families_bar.png")
out4 = os.path.join(ART_DIR, "all_families_bar.png")
plt.savefig(out3, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
plt.savefig(out4, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
plt.close()
print(f"Saved: {out3}")
print(f"Saved: {out4}")

# ── Complete summary table ────────────────────────────────────────────────────
print("\n" + "="*75)
print("COMPLETE TABLE — 3 FAMILIES × 2 ENVIRONMENTS, INCLUDING GP")
print("="*75)
print(f"{'Algorithm':<14} {'Family':<8} {'Environment':<26} {'Best':>10} {'Gens':>9} {'t(s)':>7}")
print("-"*75)
for r in ROWS:
    print(f"{r['algo']:<14} F{r['family']:<7} {r['env']:<26} {r['best']:>10.2f} {r['n_gen']:>9} {r['t']:>7.1f}")
