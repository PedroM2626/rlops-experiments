"""
Main benchmark script.

Runs 3 families of evolutionary algorithms on 5 gymnax/JAX environments:
  Family 1 (Direct):   SimpleGA, DE, OpenAI-ES
  Family 2 (Programs): LinearGP, CartesianGP
  Family 3 (EDA):      CMA-ES, PBIL

Usage:
    C:\\ev\\Scripts\\python run_benchmark.py [--quick] [--env ENV_NAME]

    --quick : quick mode (fewer generations, for testing)
    --env   : run only 1 specific environment
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import argparse
import json
import time
import numpy as np
import jax

# Silence the verbose JAX/flax warnings
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from src.family1_direct  import SimpleGA, DE, OpenAIES
from src.family2_programs import LinearGP, CartesianGP
from src.family3_eda     import CMAES, PBIL
from src.environments    import ENV_META
from src.benchmark       import run_param_based_v2, run_program_based
from src.visualization   import (plot_learning_curves, plot_family_comparison,
                                  plot_time_profile, save_results_csv)

OUT_DIR = os.path.join(os.path.dirname(__file__), "results")


def build_algorithms(obs_dim: int, act_dim: int, quick: bool):
    """Instantiates all algorithms with mode-specific hyper-parameters."""
    pop = 32 if quick else 64

    return [
        # --- Family 1: Direct Solutions ---
        dict(algo=SimpleGA(pop_size=pop, sigma_init=0.5, cx_prob=0.8),
             name="SimpleGA", family=1),
        dict(algo=DE(pop_size=pop, F=0.8, CR=0.9),
             name="DE", family=1),
        dict(algo=OpenAIES(pop_size=pop, sigma=0.05, lr=0.01),
             name="OpenAI-ES", family=1),

        # --- Family 3: EDA/Models ---
        dict(algo=CMAES(sigma0=0.5),
             name="CMA-ES", family=3),
        dict(algo=PBIL(pop_size=pop, lr=0.1, lr_sigma=0.05, top_k_frac=0.2),
             name="PBIL", family=3),
    ]


def build_gp_algorithms(obs_dim: int, act_dim: int, quick: bool):
    """Instantiates the Family 2 algorithms (they need obs_dim/act_dim)."""
    pop = 16 if quick else 32
    return [
        dict(algo=LinearGP(obs_dim=obs_dim, act_dim=act_dim,
                           pop_size=pop, prog_len=48 if quick else 64,
                           n_extra_regs=8, mut_rate=0.15),
             name="LinearGP", family=2),
        dict(algo=CartesianGP(obs_dim=obs_dim, act_dim=act_dim,
                              pop_size=8, n_cols=30 if quick else 50,
                              mut_rate=0.05),
             name="CartesianGP", family=2),
    ]


def main():
    parser = argparse.ArgumentParser(description="Evolutionary RL Benchmark in JAX")
    parser.add_argument("--quick",  action="store_true",
                        help="Quick mode: fewer generations (sanity check)")
    parser.add_argument("--env",    type=str, default=None,
                        help="Run only this environment (e.g. CartPole-v1)")
    parser.add_argument("--no-gp",  action="store_true",
                        help="Skip Family 2 (GP, slower)")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    envs_to_run = list(ENV_META.keys())
    if args.env:
        envs_to_run = [args.env]

    # Number-of-generations settings for each mode
    if args.quick:
        n_gen_direct = 30
        n_gen_gp     = 10
        n_rollouts   = 2
    else:
        n_gen_direct = 150
        n_gen_gp     = 60
        n_rollouts   = 4

    print("=" * 70)
    print("  Benchmark: 3 Families of Evolutionary Algorithms × RL Environments")
    print(f"  Mode: {'quick' if args.quick else 'full'}")
    print(f"  Environments: {envs_to_run}")
    print(f"  Generations (direct): {n_gen_direct}  |  GP: {n_gen_gp}")
    print(f"  Rollouts per evaluation: {n_rollouts}")
    print(f"  Seed: {args.seed}")
    print("=" * 70)

    jax_devices = jax.devices()
    print(f"  JAX devices: {jax_devices}")
    print()

    results_by_env = {}
    t_total = time.time()

    for env_name in envs_to_run:
        meta     = ENV_META[env_name]
        obs_dim  = meta["obs_dim"]
        act_dim  = meta["act_dim"]
        print(f"\n{'-'*70}")
        print(f"  Environment: {env_name}  (obs={obs_dim}, act={act_dim}, discrete={meta['discrete']})")
        print(f"{'-'*70}")

        results_env = []

        # ── Families 1 and 3 ───────────────────────────────────────────────
        for alg_cfg in build_algorithms(obs_dim, act_dim, args.quick):
            print(f"\n[Family {alg_cfg['family']}] {alg_cfg['name']}")
            res = run_param_based_v2(
                algo         = alg_cfg["algo"],
                algo_name    = alg_cfg["name"],
                env_name     = env_name,
                family       = alg_cfg["family"],
                n_generations= n_gen_direct,
                n_rollouts   = n_rollouts,
                seed         = args.seed,
                log_every    = max(1, n_gen_direct // 5),
            )
            results_env.append(res)

        # ── Family 2 (GP) ──────────────────────────────────────────────────
        if not args.no_gp:
            for alg_cfg in build_gp_algorithms(obs_dim, act_dim, args.quick):
                print(f"\n[Family 2] {alg_cfg['name']}")
                res = run_program_based(
                    algo         = alg_cfg["algo"],
                    algo_name    = alg_cfg["name"],
                    env_name     = env_name,
                    family       = 2,
                    n_generations= n_gen_gp,
                    n_rollouts   = n_rollouts,
                    seed         = args.seed,
                    log_every    = max(1, n_gen_gp // 5),
                )
                results_env.append(res)

        results_by_env[env_name] = results_env

    print(f"\n\n{'='*70}")
    print(f"  Training finished in {time.time()-t_total:.1f}s")
    print(f"{'='*70}\n")

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"{'Algorithm':<14} {'Family':<9} {'Environment':<28} "
          f"{'Best Fitness':>14} {'Mean Final':>12} {'Time (s)':>10}")
    print("-" * 90)
    for env, res_list in results_by_env.items():
        for r in res_list:
            bf = r.best_fitness[-1] if r.best_fitness else float("nan")
            mf = r.mean_fitness[-1] if r.mean_fitness else float("nan")
            print(f"{r.algo_name:<14} {'F'+str(r.family):<9} {r.env_name:<28} "
                  f"{bf:>14.2f} {mf:>12.2f} {r.total_time:>10.1f}")
        print()

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\nGenerating visualizations...")
    os.makedirs(OUT_DIR, exist_ok=True)

    plot_learning_curves(results_by_env, OUT_DIR)
    plot_family_comparison(results_by_env, OUT_DIR)
    plot_time_profile(results_by_env, OUT_DIR)
    save_results_csv(results_by_env, OUT_DIR)

    # ── Saves the raw data to JSON ─────────────────────────────────────────
    json_data = {}
    for env, res_list in results_by_env.items():
        json_data[env] = []
        for r in res_list:
            json_data[env].append({
                "algorithm":    r.algo_name,
                "family":       r.family,
                "generations":  r.generations,
                "best_fitness": r.best_fitness,
                "mean_fitness": r.mean_fitness,
                "std_fitness":  r.std_fitness,
                "total_time":   r.total_time,
                "n_evals":      r.n_evals,
            })
    json_path = os.path.join(OUT_DIR, "raw_results.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"[json] Saved: {json_path}")

    print(f"\n[OK] All results in: {OUT_DIR}/")


if __name__ == "__main__":
    main()
