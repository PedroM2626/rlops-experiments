"""
Visualização dos resultados do benchmark.
Gera gráficos publicáveis comparando as 3 famílias em cada ambiente.
"""
from __future__ import annotations
import os
from typing import List, Dict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# Paleta e estilos por família
FAMILY_COLORS = {
    1: ["#E63946", "#FF6B6B", "#FF8FA3"],     # Vermelhos — Diretas
    2: ["#457B9D", "#A8DADC"],                # Azuis   — Programas
    3: ["#2D6A4F", "#52B788"],               # Verdes  — EDA/Modelos
}

FAMILY_NAMES = {
    1: "Família 1 — Soluções Diretas",
    2: "Família 2 — Programas",
    3: "Família 3 — EDA/Modelos",
}

LINESTYLES = {
    "SimpleGA":  "-",
    "DE":        "--",
    "CMA-ES":    "-.",
    "LinearGP":  "-",
    "CartesianGP": "--",
    "OpenAI-ES": "-",
    "PBIL":      "--",
}

MARKERS = {
    "SimpleGA":  "o",
    "DE":        "s",
    "CMA-ES":    "^",
    "LinearGP":  "D",
    "CartesianGP": "P",
    "OpenAI-ES": "v",
    "PBIL":      "X",
}


def _smooth(x, w=5):
    """Suavização por média móvel."""
    if len(x) < w:
        return np.array(x)
    kernel = np.ones(w) / w
    pad    = np.pad(x, (w//2, w//2), mode='edge')
    return np.convolve(pad, kernel, mode='valid')[:len(x)]


def plot_learning_curves(results_by_env: Dict, out_dir: str, smooth_w: int = 5):
    """
    Para cada ambiente, plota curvas de aprendizado (best fitness × geração)
    com envelopes de desvio padrão (quando disponível via múltiplas seeds).
    """
    os.makedirs(out_dir, exist_ok=True)
    envs = list(results_by_env.keys())
    n_envs = len(envs)

    fig, axes = plt.subplots(1, n_envs, figsize=(5.5 * n_envs, 5),
                             facecolor="#0d1117")
    if n_envs == 1:
        axes = [axes]

    for ax, env in zip(axes, envs):
        ax.set_facecolor("#161b22")
        ax.set_title(env, color="white", fontsize=11, pad=8)
        ax.set_xlabel("Geração", color="#8b949e", fontsize=9)
        ax.set_ylabel("Melhor Fitness (média)", color="#8b949e", fontsize=9)
        ax.tick_params(colors="#8b949e", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

        fam_color_idx = {1: 0, 2: 0, 3: 0}
        results_env = results_by_env[env]

        for res in results_env:
            fam   = res.family
            cidx  = fam_color_idx[fam] % len(FAMILY_COLORS[fam])
            color = FAMILY_COLORS[fam][cidx]
            fam_color_idx[fam] += 1
            ls    = LINESTYLES.get(res.algo_name, "-")
            mk    = MARKERS.get(res.algo_name, "o")

            y = _smooth(np.array(res.best_fitness), smooth_w)
            x = np.array(res.generations)

            ax.plot(x, y, color=color, linestyle=ls, linewidth=1.8,
                    label=f"{res.algo_name} (F{fam})",
                    marker=mk, markevery=max(1, len(x)//10), markersize=4,
                    alpha=0.9)

        ax.legend(fontsize=7.5, framealpha=0.2, labelcolor="white",
                  facecolor="#21262d", edgecolor="#30363d", loc="lower right")
        ax.grid(True, color="#21262d", linewidth=0.5, linestyle="--")

    plt.suptitle("Benchmark Evolutivo — 3 Famílias × Ambientes RL (JAX)",
                 color="white", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, "learning_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[plot] Salvo: {path}")
    return path


def plot_family_comparison(results_by_env: Dict, out_dir: str):
    """
    Gráfico de barras agrupadas: performance final por algoritmo × ambiente.
    """
    envs    = list(results_by_env.keys())
    all_res = [r for res_list in results_by_env.values() for r in res_list]
    algos   = list(dict.fromkeys(r.algo_name for r in all_res))

    n_envs  = len(envs)
    n_algos = len(algos)
    x       = np.arange(n_envs)
    width   = 0.8 / n_algos

    fig, ax = plt.subplots(figsize=(max(10, 2.5*n_envs), 6), facecolor="#0d1117")
    ax.set_facecolor("#161b22")

    fam_map = {r.algo_name: r.family for res_list in results_by_env.values()
               for r in res_list}
    fam_cidx = {1: 0, 2: 0, 3: 0}
    algo_colors = {}
    for algo in algos:
        fam = fam_map.get(algo, 1)
        cidx = fam_cidx[fam] % len(FAMILY_COLORS[fam])
        algo_colors[algo] = FAMILY_COLORS[fam][cidx]
        fam_cidx[fam] += 1

    for i, algo in enumerate(algos):
        vals = []
        for env in envs:
            res_list = results_by_env[env]
            match = [r for r in res_list if r.algo_name == algo]
            if match:
                vals.append(match[0].best_fitness[-1] if match[0].best_fitness else 0.0)
            else:
                vals.append(0.0)

        offset = (i - n_algos / 2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width * 0.9,
                        color=algo_colors[algo], alpha=0.85,
                        label=f"{algo} (F{fam_map.get(algo,1)})",
                        edgecolor="#0d1117", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if abs(v) > 0.1:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f"{v:.0f}", ha="center", va="bottom",
                        color="white", fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels([e.replace("-v", "\nv") for e in envs],
                       color="#8b949e", fontsize=9)
    ax.set_ylabel("Fitness Final (melhor indivíduo)", color="#8b949e")
    ax.set_title("Comparação de Performance Final — 3 Famílias × Ambientes",
                 color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.legend(fontsize=8, framealpha=0.2, labelcolor="white",
              facecolor="#21262d", edgecolor="#30363d",
              bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(True, axis="y", color="#21262d", linewidth=0.5, linestyle="--")
    ax.set_facecolor("#161b22")

    plt.tight_layout()
    path = os.path.join(out_dir, "family_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[plot] Salvo: {path}")
    return path


def plot_time_profile(results_by_env: Dict, out_dir: str):
    """
    Gráfico de dispersão: tempo total de treino × fitness final.
    """
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#0d1117")
    ax.set_facecolor("#161b22")

    fam_map  = {}
    fam_cidx = {1: 0, 2: 0, 3: 0}
    algo_colors = {}

    for res_list in results_by_env.values():
        for r in res_list:
            if r.algo_name not in fam_map:
                fam = r.family
                fam_map[r.algo_name] = fam
                cidx = fam_cidx[fam] % len(FAMILY_COLORS[fam])
                algo_colors[r.algo_name] = FAMILY_COLORS[fam][cidx]
                fam_cidx[fam] += 1

    seen_labels = set()
    for res_list in results_by_env.values():
        for r in res_list:
            color = algo_colors[r.algo_name]
            mk    = MARKERS.get(r.algo_name, "o")
            label = f"{r.algo_name} (F{r.family})" if r.algo_name not in seen_labels else "_"
            seen_labels.add(r.algo_name)
            final_fit = r.best_fitness[-1] if r.best_fitness else 0.0
            ax.scatter(r.total_time, final_fit,
                       c=color, marker=mk, s=120, alpha=0.85,
                       label=label, edgecolors="#0d1117", linewidth=0.5,
                       zorder=3)
            ax.annotate(r.env_name.split("-")[0],
                        (r.total_time, final_fit),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=7, color=color, alpha=0.8)

    ax.set_xlabel("Tempo Total de Treino (s)", color="#8b949e", fontsize=10)
    ax.set_ylabel("Fitness Final (melhor)", color="#8b949e", fontsize=10)
    ax.set_title("Trade-off: Tempo de Cômputo × Performance",
                 color="white", fontsize=12, fontweight="bold")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.legend(fontsize=8, framealpha=0.2, labelcolor="white",
              facecolor="#21262d", edgecolor="#30363d")
    ax.grid(True, color="#21262d", linewidth=0.5, linestyle="--")

    plt.tight_layout()
    path = os.path.join(out_dir, "time_vs_performance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[plot] Salvo: {path}")
    return path


def save_results_csv(results_by_env: Dict, out_dir: str):
    """Salva resultados em CSV para análise posterior."""
    import csv
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "results_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "family", "environment",
                    "final_best_fitness", "final_mean_fitness",
                    "total_time_s", "n_evaluations", "n_generations"])
        for env, res_list in results_by_env.items():
            for r in res_list:
                w.writerow([
                    r.algo_name, r.family, r.env_name,
                    r.best_fitness[-1] if r.best_fitness else "NA",
                    r.mean_fitness[-1] if r.mean_fitness else "NA",
                    f"{r.total_time:.2f}",
                    r.n_evals,
                    len(r.generations),
                ])
    print(f"[csv]  Salvo: {path}")
    return path
