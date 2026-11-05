"""
Multi-Seed Scientific Evaluation Protocol (Agarwal et al., 2021).

Disentangles:
  1. Aleatoric Evaluation Variance (within-policy environment variance across test episodes)
  2. Epistemic Optimization Variance (between-training-run variance across independent training seeds)

Computes formal aggregate statistics:
  - Aggregate Mean across seeds ± SEM
  - Stratified 95% Bootstrap Confidence Intervals across seeds (B=2000)
  - Interquartile Mean (IQM) (Agarwal et al., NeurIPS 2021)
  - Variance ratio: rho = sigma_train / sigma_eval
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import json
import csv
import time
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.environments import ENV_META
from src.family1_direct import SimpleGA, DE, OpenAIES
from src.family2_programs import LinearGP, CartesianGP
from src.family3_eda import CMAES, PBIL
from src.benchmark import run_param_based_v2, run_program_based
from src.evaluation import (make_test_eval_fn, make_program_test_eval_fn,
                            evaluate_program_100, SUCCESS_THRESHOLDS)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT_DIR, exist_ok=True)

DEFAULT_SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
N_TEST_EPISODES = 100
TEST_SEED = 99999


def build_param_algorithms(pop: int = 32):
    return [
        dict(algo=SimpleGA(pop_size=pop, sigma_init=0.5, cx_prob=0.8), name="SimpleGA", family=1),
        dict(algo=DE(pop_size=pop, F=0.8, CR=0.9), name="DE", family=1),
        dict(algo=OpenAIES(pop_size=pop, sigma=0.05, lr=0.01), name="OpenAI-ES", family=1),
        dict(algo=CMAES(sigma0=0.5), name="CMA-ES", family=3),
        dict(algo=PBIL(pop_size=pop, lr=0.1, lr_sigma=0.05, top_k_frac=0.2), name="PBIL", family=3),
    ]


def build_gp_algorithms(obs_dim: int, act_dim: int):
    return [
        dict(algo=LinearGP(obs_dim=obs_dim, act_dim=act_dim, pop_size=16, prog_len=48, n_extra_regs=8, mut_rate=0.15),
             name="LinearGP", family=2),
        dict(algo=CartesianGP(obs_dim=obs_dim, act_dim=act_dim, pop_size=8, n_cols=30, mut_rate=0.05),
             name="CartesianGP", family=2),
    ]


def compute_stratified_bootstrap_ci(seed_returns_matrix: np.ndarray, n_bootstraps: int = 2000, rng_seed: int = 42):
    """
    Stratified bootstrap over seeds (Agarwal et al., 2021).
    seed_returns_matrix: shape [n_seeds, n_test_episodes]
    """
    n_seeds, n_eps = seed_returns_matrix.shape
    rng = np.random.default_rng(rng_seed)
    boot_means = np.empty(n_bootstraps)

    for b in range(n_bootstraps):
        # Resample seeds with replacement
        boot_seed_idx = rng.choice(n_seeds, size=n_seeds, replace=True)
        # Resample episodes within each chosen seed with replacement
        sample_matrix = np.empty((n_seeds, n_eps))
        for i, s_idx in enumerate(boot_seed_idx):
            sample_matrix[i] = rng.choice(seed_returns_matrix[s_idx], size=n_eps, replace=True)
        boot_means[b] = np.mean(sample_matrix)

    ci95_low = float(np.percentile(boot_means, 2.5))
    ci95_high = float(np.percentile(boot_means, 97.5))
    return ci95_low, ci95_high


def compute_iqm(all_returns: np.ndarray) -> float:
    """Interquartile Mean (trimmed mean of middle 50% data points)."""
    flat = np.sort(all_returns.flatten())
    n = len(flat)
    q1_idx = int(0.25 * n)
    q3_idx = int(0.75 * n)
    if q3_idx <= q1_idx:
        return float(np.mean(flat))
    return float(np.mean(flat[q1_idx:q3_idx]))


def run_multiseed_study(seeds=DEFAULT_SEEDS, envs=None):
    if envs is None:
        envs = ["CartPole-v1", "Acrobot-v1", "Pendulum-v1", "MountainCarContinuous-v0"]

    print("=" * 90)
    print("  MULTI-SEED TRAINING STUDY (AGARWAL ET AL., NEURIPS 2021 PROTOCOL)")
    print(f"  Training Seeds (S={len(seeds)}): {seeds}")
    print(f"  Out-of-Sample Test Episodes per Seed: {N_TEST_EPISODES} (Test Seed: {TEST_SEED})")
    print(f"  Environments: {envs}")
    print("=" * 90)

    all_records = []
    test_rng = jax.random.PRNGKey(TEST_SEED)

    for env_name in envs:
        meta = ENV_META[env_name]
        obs_dim = meta["obs_dim"]
        act_dim = meta["act_dim"]
        threshold = SUCCESS_THRESHOLDS.get(env_name, 0.0)

        # Environment-specific budget
        if env_name == "Pendulum-v1":
            n_gen_p = 100
            n_roll_p = 8
            n_gen_gp = 40
            n_roll_gp = 8
        else:
            n_gen_p = 30
            n_roll_p = 2
            n_gen_gp = 15
            n_roll_gp = 3

        print(f"\n{'-'*90}")
        print(f"  ENVIRONMENT: {env_name} | G={n_gen_p} (GP: {n_gen_gp}), N_rollouts={n_roll_p}")
        print(f"{'-'*90}")

        test_eval_jax = make_test_eval_fn(env_name, n_episodes=N_TEST_EPISODES)

        # 1. Families 1 and 3
        for alg_cfg in build_param_algorithms(pop=32):
            algo_name = alg_cfg["name"]
            family = alg_cfg["family"]
            print(f"\n  [F{family}] {algo_name}: Running across {len(seeds)} seeds...")

            seed_test_means = []
            seed_train_bests = []
            seed_returns_list = []
            times = []

            for s in seeds:
                t0 = time.time()
                res = run_param_based_v2(
                    algo=alg_cfg["algo"],
                    algo_name=algo_name,
                    env_name=env_name,
                    family=family,
                    n_generations=n_gen_p,
                    n_rollouts=n_roll_p,
                    seed=s,
                    log_every=999,
                )
                t_train = time.time() - t0
                times.append(t_train)
                f_train = float(res.best_fitness[-1])
                seed_train_bests.append(f_train)

                ret_test = np.array(test_eval_jax(jnp.array(res.best_params), test_rng))
                seed_returns_list.append(ret_test)
                seed_test_means.append(float(np.mean(ret_test)))

            returns_matrix = np.array(seed_returns_list)  # [n_seeds, n_eps]
            mean_across_seeds = float(np.mean(seed_test_means))
            sigma_train = float(np.std(seed_test_means, ddof=1)) if len(seeds) > 1 else 0.0
            sem_train = float(sigma_train / np.sqrt(len(seeds)))
            
            # Intra-seed evaluation std (mean of per-seed stds)
            intra_seed_stds = [float(np.std(r, ddof=1)) for r in returns_matrix]
            sigma_eval = float(np.mean(intra_seed_stds))

            ci95_low, ci95_high = compute_stratified_bootstrap_ci(returns_matrix)
            iqm_val = compute_iqm(returns_matrix)
            overall_success = float(np.mean(returns_matrix >= threshold)) * 100.0
            mean_train_best = float(np.mean(seed_train_bests))
            optimism_gap = float(mean_train_best - mean_across_seeds)
            rho_ratio = float(sigma_train / (sigma_eval + 1e-8))

            record = {
                "algorithm": algo_name,
                "family": family,
                "environment": env_name,
                "n_seeds": len(seeds),
                "seeds": seeds,
                "mean_across_seeds": round(mean_across_seeds, 2),
                "sem_across_seeds": round(sem_train, 2),
                "sigma_train": round(sigma_train, 2),
                "sigma_eval": round(sigma_eval, 2),
                "variance_ratio_rho": round(rho_ratio, 3),
                "stratified_ci95_low": round(ci95_low, 2),
                "stratified_ci95_high": round(ci95_high, 2),
                "iqm": round(iqm_val, 2),
                "success_rate_pct": round(overall_success, 1),
                "mean_train_best": round(mean_train_best, 2),
                "mean_optimism_gap": round(optimism_gap, 2),
                "mean_train_time_s": round(float(np.mean(times)), 2),
                "per_seed_means": [round(x, 2) for x in seed_test_means],
                "per_seed_train_bests": [round(x, 2) for x in seed_train_bests],
            }
            all_records.append(record)

            print(f"     Mean={mean_across_seeds:>8.2f} ± {sem_train:<5.2f} | "
                  f"CI95%=[{ci95_low:>7.2f}, {ci95_high:>7.2f}] | "
                  f"IQM={iqm_val:>7.2f} | σ_train={sigma_train:>6.2f} | σ_eval={sigma_eval:>6.2f} | "
                  f"ρ={rho_ratio:>5.2f} | Success={overall_success:>5.1f}%")

        # 2. Family 2 (GP)
        for alg_cfg in build_gp_algorithms(obs_dim, act_dim):
            algo_name = alg_cfg["name"]
            family = 2
            print(f"\n  [F2] {algo_name}: Running across {len(seeds)} seeds...")

            seed_test_means = []
            seed_train_bests = []
            seed_returns_list = []
            times = []

            for s in seeds:
                t0 = time.time()
                res = run_program_based(
                    algo=alg_cfg["algo"],
                    algo_name=algo_name,
                    env_name=env_name,
                    family=2,
                    n_generations=n_gen_gp,
                    n_rollouts=n_roll_gp,
                    seed=s,
                    log_every=999,
                    pure_jax=True,
                )
                t_train = time.time() - t0
                times.append(t_train)
                f_train = float(res.best_fitness[-1])
                seed_train_bests.append(f_train)

                try:
                    eval_fn_prog = make_program_test_eval_fn(env_name, alg_cfg["algo"], n_episodes=N_TEST_EPISODES)
                    ret_test = np.array(eval_fn_prog(res.best_program, test_rng))
                except Exception:
                    ret_test = evaluate_program_100(res.policy_fn, env_name, n_episodes=N_TEST_EPISODES, seed=TEST_SEED)

                seed_returns_list.append(ret_test)
                seed_test_means.append(float(np.mean(ret_test)))

            returns_matrix = np.array(seed_returns_list)
            mean_across_seeds = float(np.mean(seed_test_means))
            sigma_train = float(np.std(seed_test_means, ddof=1)) if len(seeds) > 1 else 0.0
            sem_train = float(sigma_train / np.sqrt(len(seeds)))
            
            intra_seed_stds = [float(np.std(r, ddof=1)) for r in returns_matrix]
            sigma_eval = float(np.mean(intra_seed_stds))

            ci95_low, ci95_high = compute_stratified_bootstrap_ci(returns_matrix)
            iqm_val = compute_iqm(returns_matrix)
            overall_success = float(np.mean(returns_matrix >= threshold)) * 100.0
            mean_train_best = float(np.mean(seed_train_bests))
            optimism_gap = float(mean_train_best - mean_across_seeds)
            rho_ratio = float(sigma_train / (sigma_eval + 1e-8))

            record = {
                "algorithm": algo_name,
                "family": family,
                "environment": env_name,
                "n_seeds": len(seeds),
                "seeds": seeds,
                "mean_across_seeds": round(mean_across_seeds, 2),
                "sem_across_seeds": round(sem_train, 2),
                "sigma_train": round(sigma_train, 2),
                "sigma_eval": round(sigma_eval, 2),
                "variance_ratio_rho": round(rho_ratio, 3),
                "stratified_ci95_low": round(ci95_low, 2),
                "stratified_ci95_high": round(ci95_high, 2),
                "iqm": round(iqm_val, 2),
                "success_rate_pct": round(overall_success, 1),
                "mean_train_best": round(mean_train_best, 2),
                "mean_optimism_gap": round(optimism_gap, 2),
                "mean_train_time_s": round(float(np.mean(times)), 2),
                "per_seed_means": [round(x, 2) for x in seed_test_means],
                "per_seed_train_bests": [round(x, 2) for x in seed_train_bests],
            }
            all_records.append(record)

            print(f"     Mean={mean_across_seeds:>8.2f} ± {sem_train:<5.2f} | "
                  f"CI95%=[{ci95_low:>7.2f}, {ci95_high:>7.2f}] | "
                  f"IQM={iqm_val:>7.2f} | σ_train={sigma_train:>6.2f} | σ_eval={sigma_eval:>6.2f} | "
                  f"ρ={rho_ratio:>5.2f} | Success={overall_success:>5.1f}%")

    # Save JSON
    json_path = os.path.join(OUT_DIR, "multiseed_study.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2)
    print(f"\n[OK] Saved JSON: {json_path}")

    # Save CSV
    csv_path = os.path.join(OUT_DIR, "multiseed_summary.csv")
    fieldnames = [
        "algorithm", "family", "environment", "n_seeds", "mean_across_seeds",
        "sem_across_seeds", "sigma_train", "sigma_eval", "variance_ratio_rho",
        "stratified_ci95_low", "stratified_ci95_high", "iqm", "success_rate_pct",
        "mean_train_best", "mean_optimism_gap", "mean_train_time_s"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_records)
    print(f"[OK] Saved CSV: {csv_path}")

    # Plot variance decomposition
    _plot_multiseed_analysis(all_records, OUT_DIR)

    return all_records


def _plot_multiseed_analysis(records, out_dir):
    """Generates the multi-seed variance and performance plot."""
    envs = list(dict.fromkeys(r["environment"] for r in records))
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), facecolor="#0d1117")
    n_seeds = records[0]["n_seeds"]
    fig.suptitle(f"Multi-Seed Benchmark Analysis (S={n_seeds} Seeds × 100 Test Eps) — Agarwal et al. (2021)",
                 color="white", fontsize=14, fontweight="bold", y=0.995)
    axes = axes.flatten()

    COLORS = {1: "#E63946", 2: "#457B9D", 3: "#52B788"}

    for idx, env in enumerate(envs):
        ax = axes[idx]
        ax.set_facecolor("#161b22")
        ax.set_title(f"{env} — Aggregate Mean across {n_seeds} Seeds (± 95% Stratified CI)",
                     color="white", fontsize=11, fontweight="bold", pad=8)
        ax.set_ylabel("Evaluation Return", color="#8b949e", fontsize=9.5)
        ax.tick_params(colors="#8b949e", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.grid(True, axis="y", color="#21262d", linewidth=0.5, linestyle="--")

        sub_records = [r for r in records if r["environment"] == env]
        sub_records.sort(key=lambda r: (r["family"], r["algorithm"]))
        x = np.arange(len(sub_records))

        means = [r["mean_across_seeds"] for r in sub_records]
        lows = [r["stratified_ci95_low"] for r in sub_records]
        highs = [r["stratified_ci95_high"] for r in sub_records]
        err_low = [m - l for m, l in zip(means, lows)]
        err_high = [h - m for m, h in zip(means, highs)]
        bar_colors = [COLORS[r["family"]] for r in sub_records]

        bars = ax.bar(x, means, width=0.68, color=bar_colors, alpha=0.88,
                      edgecolor="#0d1117", linewidth=0.5)
        ax.errorbar(x, means, yerr=[err_low, err_high], fmt="none",
                    color="white", elinewidth=1.2, capsize=3.5, capthick=1.0)

        for i, (b, r) in enumerate(zip(bars, sub_records)):
            v = r["mean_across_seeds"]
            offset = abs(v) * 0.03 if v != 0 else (10.0 if "CartPole" in env else 1.0)
            va = "bottom" if v >= 0 else "top"
            ax.text(b.get_x() + b.get_width() / 2, v + (offset if v >= 0 else -offset),
                    (f"{v:.0f}\n(ρ={r['variance_ratio_rho']:.1f})" if r["sigma_eval"] > 1e-6
                     else f"{v:.0f}\n(ρ n/a)"),
                    ha="center", va=va, color="white", fontsize=7.5, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([f"{r['algorithm']}\n(F{r['family']})" for r in sub_records],
                           color="#8b949e", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "multiseed_variance_analysis.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[plot] Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Seed Study Protocol")
    parser.add_argument("--envs", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    args = parser.parse_args()

    run_multiseed_study(seeds=args.seeds, envs=args.envs)
