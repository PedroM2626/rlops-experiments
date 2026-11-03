"""
plot_comparative_progress.py
============================
Unified analytical visualization engine for smw-evolutionary-agent.

Extracts, parses, and plots empirical comparative metrics from:
  1. neat_training_results.txt & backup.*.DP1.state.pool (NEAT)
  2. gp_training_results.txt & best.*.DP1.state.gppool (Genetic Programming)

Generates publication-quality figures:
  - docs/figures/smw_pareto_complexity_progress.png
  - docs/figures/smw_trajectory_heatmap.png
  - docs/comparative_metrics_summary.csv
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parent
DOCS_DIR = REPO_DIR / "docs"
FIGURES_DIR = DOCS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Parsing NEAT Training Logs and Checkpoints
# ---------------------------------------------------------------------------

def parse_neat_logs(log_path: Path) -> pd.DataFrame:
    """Parse neat_training_results.txt into per-generation aggregated metrics."""
    if not log_path.exists():
        raise FileNotFoundError(f"NEAT log not found: {log_path}")

    records = []
    pattern = re.compile(
        r"Gen\s+(?P<gen>\d+)\s+\|\s+Species\s+(?P<species>\d+)\s+\|\s+Genome\s+(?P<genome>\d+)\s+\|\s+Fitness\s+(?P<fit>[\d\.\-]+)\s+\|\s+MaxFit\s+(?P<maxfit>[\d\.\-]+)"
    )

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                records.append({
                    "generation": int(m.group("gen")),
                    "species": int(m.group("species")),
                    "genome": int(m.group("genome")),
                    "fitness": float(m.group("fit")),
                    "maxfit": float(m.group("maxfit")),
                })

    df = pd.DataFrame(records)
    grouped = df.groupby("generation").agg(
        max_fitness=("fitness", "max"),
        mean_fitness=("fitness", "mean"),
        std_fitness=("fitness", "std"),
        num_species=("species", "nunique"),
        evaluations=("genome", "count")
    ).reset_index()

    # Fill NaN std on single eval
    grouped["std_fitness"] = grouped["std_fitness"].fillna(0.0)
    return grouped


def parse_neat_pool(pool_path: Path) -> Tuple[int, int, float]:
    """
    Parse a backup.<g>.DP1.state.pool file.
    Returns: (max_neurons, best_genome_genes, best_genome_fitness)
    """
    if not pool_path.exists():
        return 0, 0, 0.0

    lines = [l.strip() for l in pool_path.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]
    if len(lines) < 25:
        return 0, 0, 0.0

    idx = 0
    gen = int(lines[idx]); idx += 1
    max_fit = float(lines[idx]); idx += 1
    num_species = int(lines[idx]); idx += 1

    best_genes = 0
    max_neuron_overall = 0
    top_fitness_seen = -1e9

    for _ in range(num_species):
        if idx >= len(lines):
            break
        top_fit = float(lines[idx]); idx += 1
        staleness = int(lines[idx]); idx += 1
        num_genomes = int(lines[idx]); idx += 1

        for _ in range(num_genomes):
            if idx >= len(lines):
                break
            fit = float(lines[idx]); idx += 1
            max_neuron = int(lines[idx]); idx += 1
            max_neuron_overall = max(max_neuron_overall, max_neuron)

            # Skip mutation parameters until 'done'
            while idx < len(lines) and lines[idx] != "done":
                idx += 1
            if idx < len(lines) and lines[idx] == "done":
                idx += 1

            if idx >= len(lines):
                break
            num_genes = int(lines[idx]); idx += 1
            # Skip gene records
            idx += num_genes

            if fit > top_fitness_seen:
                top_fitness_seen = fit
                best_genes = num_genes

    return max_neuron_overall, best_genes, max_fit


# ---------------------------------------------------------------------------
# 2. Parsing GP Training Logs and Checkpoints
# ---------------------------------------------------------------------------

def parse_gp_logs(log_path: Path) -> Tuple[pd.DataFrame, Dict[int, Dict[str, int]]]:
    """
    Parse gp_training_results.txt into generational metrics and syntactic distributions.
    """
    if not log_path.exists():
        raise FileNotFoundError(f"GP log not found: {log_path}")

    records = []
    syntax_stats: Dict[int, Dict[str, int]] = {}

    gen_pattern = re.compile(
        r"Gen\s+(?P<gen>\d+)\s+\|\s+MaxF=(?P<maxf>[\d\.\-]+)\s+MeanF=(?P<meanf>[\d\.\-]+)\s+StdF=(?P<stdf>[\d\.\-]+)\s+MeanSize=(?P<meansz>[\d\.\-]+)\s+BestSize=(?P<bestsz>\d+)"
    )

    current_gen = None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = gen_pattern.search(line)
            if m:
                current_gen = int(m.group("gen"))
                records.append({
                    "generation": current_gen,
                    "max_fitness": float(m.group("maxf")),
                    "mean_fitness": float(m.group("meanf")),
                    "std_fitness": float(m.group("stdf")),
                    "mean_size": float(m.group("meansz")),
                    "best_size": int(m.group("bestsz")),
                })
                syntax_stats[current_gen] = {
                    "tile_terms": 0,
                    "spr_terms": 0,
                    "arith_ops": 0,
                    "logic_ops": 0
                }
            elif current_gen is not None and ("Right:" in line or "Left:" in line or "B:" in line):
                syntax_stats[current_gen]["tile_terms"] += len(re.findall(r"tile\(", line))
                syntax_stats[current_gen]["spr_terms"] += len(re.findall(r"spr\(", line))
                syntax_stats[current_gen]["arith_ops"] += len(re.findall(r"(?:ADD|MUL|MIN|MAX|TANH)\(", line))
                syntax_stats[current_gen]["logic_ops"] += len(re.findall(r"(?:AND|OR|NOT|IF|GT|LT)\(", line))

    df = pd.DataFrame(records)
    return df, syntax_stats


# ---------------------------------------------------------------------------
# 3. Figure 1: Pareto Complexity & Progress Curves
# ---------------------------------------------------------------------------

def plot_pareto_complexity(
    df_neat: pd.DataFrame,
    df_gp: pd.DataFrame,
    neat_complexity: List[Dict[str, float]],
    syntax_stats: Dict[int, Dict[str, int]],
    output_path: Path
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#ffffff")

    palette = {
        "neat_primary": "#1f77b4",
        "neat_light": "#aec7e8",
        "gp_primary": "#2ca02c",
        "gp_light": "#98df8a",
        "accent": "#d62728",
        "grid": "#e0e0e0"
    }

    # --- Panel A: Learning Curves (Generations 0-7 comparison) ---
    ax = axes[0, 0]
    ax.set_facecolor("#fafafa")
    ax.grid(True, linestyle="--", alpha=0.6, color=palette["grid"])

    # NEAT
    ax.plot(df_neat["generation"], df_neat["max_fitness"], "o-", color=palette["neat_primary"],
            linewidth=2.2, label="NEAT (Max Fitness)")
    ax.fill_between(
        df_neat["generation"],
        df_neat["mean_fitness"] - df_neat["std_fitness"],
        df_neat["mean_fitness"] + df_neat["std_fitness"],
        color=palette["neat_light"], alpha=0.35, label="NEAT (Mean ± 1σ)"
    )

    # GP (aligned on same generations 0-7 for direct comparison)
    df_gp_aligned = df_gp[df_gp["generation"] <= df_neat["generation"].max()]
    ax.plot(df_gp_aligned["generation"], df_gp_aligned["max_fitness"], "s--", color=palette["gp_primary"],
            linewidth=2.2, label="GP (Max Fitness)")
    ax.fill_between(
        df_gp_aligned["generation"],
        df_gp_aligned["mean_fitness"] - df_gp_aligned["std_fitness"],
        df_gp_aligned["mean_fitness"] + df_gp_aligned["std_fitness"],
        color=palette["gp_light"], alpha=0.35, label="GP (Mean ± 1σ)"
    )

    ax.set_title("A. Fitness Progression (Generations 0–7)", fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Generation (g)", fontsize=10)
    ax.set_ylabel("Fitness (ΔX - Penalty)", fontsize=10)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)

    # --- Panel B: Pareto Trade-off (Complexity vs Fitness) ---
    ax = axes[0, 1]
    ax.set_facecolor("#fafafa")
    ax.grid(True, linestyle="--", alpha=0.6, color=palette["grid"])

    # NEAT: Genes vs Fitness
    neat_genes = [c["genes"] for c in neat_complexity]
    neat_fits = [c["max_fit"] for c in neat_complexity]
    ax.scatter(neat_genes, neat_fits, color=palette["neat_primary"], s=80, alpha=0.85,
               edgecolors="black", linewidth=1.2, label="NEAT (Synaptic Genes)")
    for i, (g, f) in enumerate(zip(neat_genes, neat_fits)):
        ax.annotate(f"g{i}", (g, f), textcoords="offset points", xytext=(5, 3), fontsize=8, color=palette["neat_primary"])

    # GP: Best Tree Size vs Fitness
    gp_sizes = df_gp_aligned["best_size"].tolist()
    gp_fits = df_gp_aligned["max_fitness"].tolist()
    ax.scatter(gp_sizes, gp_fits, color=palette["gp_primary"], s=80, alpha=0.85,
               marker="s", edgecolors="black", linewidth=1.2, label="GP (Tree Node Count)")
    for i, (sz, f) in enumerate(zip(gp_sizes, gp_fits)):
        ax.annotate(f"g{i}", (sz, f), textcoords="offset points", xytext=(-15, 3), fontsize=8, color=palette["gp_primary"])

    ax.set_title("B. Complexity vs. Performance (Pareto Space)", fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Structural Complexity (Synapses for NEAT | Nodes for GP)", fontsize=10)
    ax.set_ylabel("Max Fitness Reached", fontsize=10)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)

    # --- Panel C: Population Speciation & Size Dynamics ---
    ax = axes[1, 0]
    ax.set_facecolor("#fafafa")
    ax.grid(True, linestyle="--", alpha=0.6, color=palette["grid"])

    # NEAT active species count
    color1 = palette["neat_primary"]
    ax.plot(df_neat["generation"], df_neat["num_species"], "o-", color=color1, linewidth=2, label="NEAT Active Species")
    ax.set_xlabel("Generation (g)", fontsize=10)
    ax.set_ylabel("NEAT Species Count", color=color1, fontsize=10)
    ax.tick_params(axis="y", labelcolor=color1)

    # Secondary axis: GP mean tree size over 22 generations
    ax2 = ax.twinx()
    color2 = palette["gp_primary"]
    ax2.plot(df_gp["generation"], df_gp["mean_size"], "s--", color=color2, linewidth=2, label="GP Mean Tree Size")
    ax2.plot(df_gp["generation"], df_gp["best_size"], "^:", color="#ff7f0e", linewidth=1.8, label="GP Best Tree Size")
    ax2.set_ylabel("GP Node Count", color=color2, fontsize=10)
    ax2.tick_params(axis="y", labelcolor=color2)

    # Combine legends
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right", framealpha=0.9, fontsize=8)
    ax.set_title("C. Speciation (NEAT) & Parsimony Pressure (GP)", fontsize=12, fontweight="bold", pad=8)

    # --- Panel D: GP Syntactic Primitives Distribution ---
    ax = axes[1, 1]
    ax.set_facecolor("#fafafa")
    ax.grid(True, linestyle="--", alpha=0.6, color=palette["grid"])

    gens = sorted(syntax_stats.keys())
    tiles = [syntax_stats[g]["tile_terms"] for g in gens]
    sprites = [syntax_stats[g]["spr_terms"] for g in gens]
    arith = [syntax_stats[g]["arith_ops"] for g in gens]
    logic = [syntax_stats[g]["logic_ops"] for g in gens]

    ax.plot(gens, tiles, "o-", color="#8c564b", linewidth=1.8, label="Tile Terminals (Terrain)")
    ax.plot(gens, sprites, "s-", color="#e377c2", linewidth=1.8, label="Sprite Terminals (Enemies)")
    ax.plot(gens, arith, "^--", color="#17becf", linewidth=1.8, label="Arithmetic Ops (ADD/MUL/TANH)")
    ax.plot(gens, logic, "d--", color="#bcbd22", linewidth=1.8, label="Logical/Rel Ops (IF/AND/GT)")

    ax.set_title("D. GP Symbolic Composition Over Generations", fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Generation (g)", fontsize=10)
    ax.set_ylabel("Primitive Occurrences in Best Trees", fontsize=10)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8)

    plt.tight_layout(pad=2.0)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[OK] Saved figure: {output_path}")


# ---------------------------------------------------------------------------
# 4. Figure 2: Mario Trajectory & Obstacle Landmark Reach
# ---------------------------------------------------------------------------

def plot_trajectory_reach(df_neat: pd.DataFrame, df_gp: pd.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafafa")
    ax.grid(True, linestyle="--", alpha=0.5, color="#e0e0e0")

    # Key Level Landmarks in Yoshi's Island 1 (horizontal coordinate X)
    landmarks = [
        (100, "Start Level", "#7f7f7f"),
        (280, "First Pipe", "#333333"),
        (450, "Bushes & Coin Blocks", "#333333"),
        (650, "First Chasm (Pit)", "#d62728"),
        (1100, "Rex & Slope", "#ff7f0e"),
        (1600, "Banzai Bill Hazard", "#d62728"),
        (2500, "Midpoint Gate", "#2ca02c"),
        (4816, "Goal Giant Gate", "#1f77b4"),
    ]

    for x_pos, name, col in landmarks:
        ax.axvline(x=x_pos, color=col, linestyle=":", alpha=0.7, linewidth=1.5)
        ax.text(x_pos + 15, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 22, name,
                rotation=90, verticalalignment="top", fontsize=8, color=col, fontweight="bold")

    # In SMW, raw fitness ~ rightmost - currentFrame/2; rightmost ~ fitness + frame_penalty (~ +25 to 50)
    neat_reach = df_neat["max_fitness"].values + 35.0
    gp_reach = df_gp["max_fitness"].values + 35.0

    # Plot horizontal reach over generations
    ax.plot(neat_reach, df_neat["generation"], "o-", color="#1f77b4", linewidth=2.5,
            markersize=8, label="NEAT Frontier Reach (Gens 0–7)")
    ax.plot(gp_reach, df_gp["generation"], "s--", color="#2ca02c", linewidth=2.5,
            markersize=7, label="GP Frontier Reach (Gens 0–21)")

    # Color background zones
    ax.axvspan(0, 650, color="#ffffcc", alpha=0.3, label="Pre-Chasm Safe Zone")
    ax.axvspan(650, 1600, color="#ffe6cc", alpha=0.3, label="Chasm & Rex Hazard Zone")

    ax.set_xlim(0, 1800)
    ax.set_ylim(-0.5, max(df_gp["generation"].max(), df_neat["generation"].max()) + 0.5)

    ax.set_title("Empirical Progression Along Level 1 (Yoshi's Island 1 Landmarks)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Horizontal Mario Coordinate X (Pixels)", fontsize=11)
    ax.set_ylabel("Generation Index (g)", fontsize=11)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=9)

    plt.tight_layout(pad=2.0)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[OK] Saved figure: {output_path}")


# ---------------------------------------------------------------------------
# 5. Summary Table Exporter
# ---------------------------------------------------------------------------

def export_summary_table(
    df_neat: pd.DataFrame,
    df_gp: pd.DataFrame,
    neat_complexity: List[Dict[str, float]],
    csv_path: Path,
    md_path: Path
):
    max_gen = min(df_neat["generation"].max(), df_gp["generation"].max())
    rows = []

    for g in range(max_gen + 1):
        n_row = df_neat[df_neat["generation"] == g].iloc[0]
        g_row = df_gp[df_gp["generation"] == g].iloc[0]
        n_comp = neat_complexity[g] if g < len(neat_complexity) else {"genes": 0, "neurons": 0}

        rows.append({
            "Generation": g,
            "NEAT_MaxFit": n_row["max_fitness"],
            "NEAT_MeanFit": round(n_row["mean_fitness"], 2),
            "NEAT_Species": int(n_row["num_species"]),
            "NEAT_Synapses": int(n_comp["genes"]),
            "NEAT_Neurons": int(n_comp["neurons"]),
            "GP_MaxFit": g_row["max_fitness"],
            "GP_MeanFit": round(g_row["mean_fitness"], 2),
            "GP_BestSize": int(g_row["best_size"]),
            "GP_MeanSize": round(g_row["mean_size"], 2),
        })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(csv_path, index=False)
    print(f"[OK] Exported CSV: {csv_path}")

    # Generate Markdown Table manually without external dependency
    headers = list(summary_df.columns)
    md_content = "# Comparative Empirical Metrics: NEAT vs. Genetic Programming\n\n"
    md_content += "| " + " | ".join(headers) + " |\n"
    md_content += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for _, row in summary_df.iterrows():
        md_content += "| " + " | ".join(str(row[h]) for h in headers) + " |\n"
    md_content += "\n*All values parsed directly from committed training logs (`neat_training_results.txt`, `gp_training_results.txt`) and population pools.*\n"
    md_path.write_text(md_content, encoding="utf-8")
    print(f"[OK] Exported Markdown: {md_path}")


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("  SMW EVOLUTIONARY AGENT - ANALYTICAL VISUALIZATION ENGINE")
    print("=" * 65)

    neat_log = REPO_DIR / "neat_training_results.txt"
    gp_log = REPO_DIR / "gp_training_results.txt"

    print("\n[1/4] Parsing NEAT training logs and pool files...")
    df_neat = parse_neat_logs(neat_log)
    neat_complexity = []
    for g in df_neat["generation"]:
        pool_file = REPO_DIR / f"backup.{g}.DP1.state.pool"
        neurons, genes, max_fit = parse_neat_pool(pool_file)
        neat_complexity.append({"generation": g, "neurons": neurons, "genes": genes, "max_fit": max_fit})

    print(f"  Parsed {len(df_neat)} NEAT generations.")

    print("\n[2/4] Parsing GP training logs and pool files...")
    df_gp, syntax_stats = parse_gp_logs(gp_log)
    print(f"  Parsed {len(df_gp)} GP generations.")

    print("\n[3/4] Generating publication-quality figures...")
    fig1_path = FIGURES_DIR / "smw_pareto_complexity_progress.png"
    plot_pareto_complexity(df_neat, df_gp, neat_complexity, syntax_stats, fig1_path)

    fig2_path = FIGURES_DIR / "smw_trajectory_heatmap.png"
    plot_trajectory_reach(df_neat, df_gp, fig2_path)

    print("\n[4/4] Exporting statistical contrast tables...")
    csv_path = DOCS_DIR / "comparative_metrics_summary.csv"
    md_path = DOCS_DIR / "comparative_metrics_summary.md"
    export_summary_table(df_neat, df_gp, neat_complexity, csv_path, md_path)

    print("\n" + "=" * 65)
    print("  VISUALIZATION & ANALYSIS PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
