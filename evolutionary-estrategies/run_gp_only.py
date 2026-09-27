"""
Separate script that runs only Family 2 (GP) on CartPole and Acrobot.
Slower because it is interpreted in Python.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import json
import time
import numpy as np
import jax

from src.family2_programs import LinearGP, CartesianGP
from src.environments     import ENV_META
from src.benchmark        import run_program_based
from src.visualization    import plot_learning_curves, save_results_csv

OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
ENVS    = ["CartPole-v1", "Acrobot-v1"]

def main():
    print("=" * 60)
    print("  Family 2 (Programs) — LinearGP + CartesianGP")
    print("=" * 60)

    results_by_env = {}
    t0 = time.time()

    for env_name in ENVS:
        meta    = ENV_META[env_name]
        obs_dim = meta["obs_dim"]
        act_dim = meta["act_dim"]
        print(f"\n--- {env_name} (obs={obs_dim}, act={act_dim}) ---")

        results_env = []
        for alg_cfg in [
            dict(algo=LinearGP(obs_dim=obs_dim, act_dim=act_dim,
                               pop_size=16, prog_len=48, n_extra_regs=8,
                               mut_rate=0.15),
                 name="LinearGP"),
            dict(algo=CartesianGP(obs_dim=obs_dim, act_dim=act_dim,
                                  pop_size=8, n_cols=30, mut_rate=0.05),
                 name="CartesianGP"),
        ]:
            print(f"\n[LinearGP/CartesianGP] {alg_cfg['name']}")
            res = run_program_based(
                algo          = alg_cfg["algo"],
                algo_name     = alg_cfg["name"],
                env_name      = env_name,
                family        = 2,
                n_generations = 15,
                n_rollouts    = 3,
                seed          = 42,
                log_every     = 5,
            )
            results_env.append(res)
        results_by_env[env_name] = results_env

    print(f"\nFinished in {time.time()-t0:.1f}s")

    # Load the F1+F3 results if they exist
    json_path = os.path.join(OUT_DIR, "raw_results.json")
    if os.path.exists(json_path):
        print("Merging with the existing F1+F3 results...")
        # Only the GP results are saved for now

    # Save the GP results
    gp_json = {}
    for env, res_list in results_by_env.items():
        gp_json[env] = []
        for r in res_list:
            gp_json[env].append({
                "algorithm": r.algo_name,
                "family":    r.family,
                "generations": r.generations,
                "best_fitness": r.best_fitness,
                "mean_fitness": r.mean_fitness,
                "std_fitness":  r.std_fitness,
                "total_time": r.total_time,
                "n_evals":    r.n_evals,
            })
    out_path = os.path.join(OUT_DIR, "gp_results.json")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(gp_json, f, indent=2)
    print(f"Saved: {out_path}")

    # Table
    print(f"\n{'Algorithm':<14} {'Environment':<28} {'Best Fitness':>14} {'Time (s)':>10}")
    print("-" * 70)
    for env, res_list in results_by_env.items():
        for r in res_list:
            bf = r.best_fitness[-1] if r.best_fitness else float("nan")
            print(f"{r.algo_name:<14} {r.env_name:<28} {bf:>14.2f} {r.total_time:>10.1f}")

if __name__ == "__main__":
    main()
