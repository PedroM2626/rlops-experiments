"""
Causal ablation on CartPole-v1: is the seed-to-seed variance caused by
(a) noisy fitness estimation (few rollouts per evaluation, N), or
(b) insufficient optimization budget (few generations, G)?

Design (one factor changed at a time relative to the baseline G=30, N=2):
  * N-axis (G fixed = 30):   N in {2, 8, 16}
  * G-axis (N fixed = 2):    G in {30, 100}
  * joint point:             G=100, N=8
Each configuration is run with S=5 training seeds, and each trained policy is
evaluated on the SAME 100 out-of-sample episodes (test seed 99999).

Also runs a determinism check (same seed twice must give identical results),
which verifies that the variance across seeds is not run-to-run nondeterminism.

Reported per (algorithm, G, N): mean test return across seeds, sigma_train
(std of per-seed test means), mean train-best fitness, optimism gap, success rate.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import json
import csv
import time
import numpy as np
import jax
import jax.numpy as jnp

from src.benchmark import run_param_based_v2
from src.evaluation import make_test_eval_fn, SUCCESS_THRESHOLDS
from run_multiseed_study import build_param_algorithms

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ENV = "CartPole-v1"
SEEDS = [42, 101, 202, 303, 404]
CONFIGS = [(30, 2), (30, 8), (30, 16), (100, 2), (100, 8)]  # (G, N)
TEST_SEED = 99999


def main():
    threshold = SUCCESS_THRESHOLDS[ENV]
    test_fn = make_test_eval_fn(ENV, n_episodes=100)
    test_rng = jax.random.PRNGKey(TEST_SEED)
    records = []

    # --- Determinism check -------------------------------------------------
    alg = build_param_algorithms()[0]  # SimpleGA
    outs = []
    for _ in range(2):
        r = run_param_based_v2(alg["algo"], alg["name"], ENV, alg["family"],
                               n_generations=30, n_rollouts=2, seed=42, log_every=999)
        ret = np.array(test_fn(jnp.array(r.best_params), test_rng))
        outs.append((float(r.best_fitness[-1]), float(ret.mean())))
    determinism = {"run_a": outs[0], "run_b": outs[1], "identical": outs[0] == outs[1]}
    print(f"[determinism] SimpleGA seed 42 twice: {outs} identical={determinism['identical']}")

    # --- Ablation ----------------------------------------------------------
    for (G, N) in CONFIGS:
        for alg_cfg in build_param_algorithms():
            name, fam = alg_cfg["name"], alg_cfg["family"]
            test_means, train_bests, rets = [], [], []
            t0 = time.time()
            for s in SEEDS:
                r = run_param_based_v2(alg_cfg["algo"], name, ENV, fam,
                                       n_generations=G, n_rollouts=N, seed=s, log_every=999)
                train_bests.append(float(r.best_fitness[-1]))
                ret = np.array(test_fn(jnp.array(r.best_params), test_rng))
                rets.append(ret)
                test_means.append(float(ret.mean()))
            rets = np.array(rets)
            rec = {
                "algorithm": name, "family": fam, "G": G, "N": N,
                "mean_test": round(float(np.mean(test_means)), 2),
                "sigma_train": round(float(np.std(test_means, ddof=1)), 2),
                "sigma_eval": round(float(np.mean([np.std(x, ddof=1) for x in rets])), 2),
                "mean_train_best": round(float(np.mean(train_bests)), 2),
                "optimism_gap": round(float(np.mean(train_bests) - np.mean(test_means)), 2),
                "success_pct": round(float(np.mean(rets >= threshold)) * 100, 1),
                "per_seed_test_means": [round(x, 2) for x in test_means],
                "per_seed_train_bests": [round(x, 2) for x in train_bests],
                "time_s": round(time.time() - t0, 1),
            }
            records.append(rec)
            print(f"G={G:<3} N={N:<2} {name:<10} test={rec['mean_test']:>7.2f} "
                  f"sig_train={rec['sigma_train']:>7.2f} gap={rec['optimism_gap']:>7.2f} "
                  f"succ={rec['success_pct']:>5.1f}% per_seed={rec['per_seed_test_means']}", flush=True)

    with open(os.path.join(OUT_DIR, "causal_ablation_cartpole.json"), "w", encoding="utf-8") as f:
        json.dump({"determinism_check": determinism, "records": records}, f, indent=2)
    keys = ["algorithm", "family", "G", "N", "mean_test", "sigma_train", "sigma_eval",
            "mean_train_best", "optimism_gap", "success_pct", "time_s"]
    with open(os.path.join(OUT_DIR, "causal_ablation_cartpole.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)
    print("[OK] saved causal_ablation_cartpole.json/csv")


if __name__ == "__main__":
    main()
