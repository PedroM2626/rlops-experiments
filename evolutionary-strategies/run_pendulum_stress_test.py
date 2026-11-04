"""
Formal Long-Horizon Stress Test on Pendulum-v1 (250 Generations).

Investigates the asymptotic plateau of black-box Evolutionary Strategies (CMA-ES)
on continuous under-actuated control under extended optimization budgets (~12.8M steps).
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
import time
import numpy as np
import jax
import jax.numpy as jnp

from src.environments import ENV_META
from src.family3_eda import CMAES
from src.family1_direct import SimpleGA
from src.benchmark import run_param_based_v2
from src.evaluation import make_test_eval_fn, compute_statistical_metrics

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT_DIR, exist_ok=True)


def run_stress_test(n_generations: int = 250, n_rollouts: int = 8, train_seed: int = 42, test_seed: int = 99999):
    env_name = "Pendulum-v1"
    meta = ENV_META[env_name]
    print("=" * 80)
    print("  PENDULUM-V1 ASYMPTOTIC HORIZON STRESS TEST (250 GENERATIONS)")
    print(f"  Generations: {n_generations} | Rollouts/Eval: {n_rollouts} | Train Seed: {train_seed} | Test Seed: {test_seed}")
    print("=" * 80)

    # Instantiate CMA-ES (Hansen 2016)
    algo = CMAES(sigma0=0.5)
    t0 = time.time()
    res = run_param_based_v2(
        algo=algo,
        algo_name="CMA-ES",
        env_name=env_name,
        family=3,
        n_generations=n_generations,
        n_rollouts=n_rollouts,
        seed=train_seed,
        log_every=25,
    )
    train_time = time.time() - t0

    # 100 out-of-sample episodes
    test_eval_fn = make_test_eval_fn(env_name, n_episodes=100)
    test_rng = jax.random.PRNGKey(test_seed)
    returns_jax = test_eval_fn(res.best_params, test_rng)
    returns_np = np.array(returns_jax)

    stats = compute_statistical_metrics(
        returns_np,
        f_train_best=float(res.best_fitness[-1]),
        env_name=env_name,
    )

    pop_size = getattr(algo, "pop_size", getattr(algo, "lambda_", 25))
    out_record = {
        "experiment": "Pendulum-v1 Asymptotic Stress Test",
        "algorithm": "CMA-ES",
        "family": 3,
        "n_generations": n_generations,
        "n_rollouts_per_eval": n_rollouts,
        "train_seed": train_seed,
        "test_seed": test_seed,
        "train_time_s": round(train_time, 2),
        "total_simulated_steps": n_generations * int(pop_size) * n_rollouts * meta["max_steps"],
        "training_trajectory": {
            "generations": res.generations,
            "best_fitness": [float(x) for x in res.best_fitness],
            "mean_fitness": [float(x) for x in res.mean_fitness],
            "std_fitness": [float(x) for x in res.std_fitness],
        },
        "evaluation_metrics": stats,
    }

    out_path = os.path.join(OUT_DIR, "pendulum_stress_test_250.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_record, f, indent=2)

    print(f"\n[OK] Results saved to: {out_path}")
    print(f"     Train Best: {stats['f_train_best']:.2f}")
    print(f"     Test Mean:  {stats['test_mean']:.2f} ± {stats['test_sem']:.2f}")
    print(f"     CI 95%:     [{stats['ci95_low']:.2f}, {stats['ci95_high']:.2f}]")
    print(f"     Median:     {stats['test_median']:.2f} | IQR: {stats['iqr']:.2f}")
    print(f"     Success %:  {stats['success_rate_pct']:.1f}%")
    print(f"     Opt. Gap:   {stats['optimism_gap']:.2f}")
    print(f"     Train Time: {train_time:.1f}s")
    print("=" * 80)
    return out_record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pendulum-v1 250 Generations Stress Test")
    parser.add_argument("--generations", type=int, default=250)
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--test-seed", type=int, default=99999)
    args = parser.parse_args()

    run_stress_test(
        n_generations=args.generations,
        n_rollouts=args.rollouts,
        train_seed=args.train_seed,
        test_seed=args.test_seed,
    )
