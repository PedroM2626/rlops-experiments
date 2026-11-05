"""
Equal environment-step budget comparison on Pendulum-v1: CMA-ES at 30k / 100k / 300k / 1M
simulated environment steps (N_rollouts in {1, 4}), 10 seeds, same 100-episode test protocol,
to be compared with SAC/TD3 at 30k steps (results/sac_pendulum.json, results/td3_pendulum.json).
Steps per generation = lambda * N_rollouts * 200.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import jax, jax.numpy as jnp
from src.family3_eda import CMAES
from src.benchmark import run_param_based_v2
from src.evaluation import make_test_eval_fn, compute_statistical_metrics

ENV = "Pendulum-v1"; H = 200
SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "pendulum_equal_budget.json")
CONFIGS = [(1, 30_000), (1, 100_000), (1, 300_000), (1, 1_000_000), (4, 100_000), (4, 300_000), (4, 1_000_000)]

rec = json.load(open(OUT)) if os.path.exists(OUT) else {}
test_fn = make_test_eval_fn(ENV, n_episodes=100); key = jax.random.PRNGKey(99999)
for N, budget in CONFIGS:
    for s in SEEDS:
        tag = f"N{N}|{budget}|{s}"
        if tag in rec: continue
        algo = CMAES(sigma0=0.5)
        probe = run_param_based_v2(algo=CMAES(sigma0=0.5), algo_name="CMA-ES", env_name=ENV, family=3,
                                   n_generations=1, n_rollouts=N, seed=s, log_every=9999) if False else None
        lam = 25                                   # CMA-ES offspring count for d=1217: 4 + floor(3 ln d)
        G = max(1, round(budget / (lam * N * H)))
        res = run_param_based_v2(algo=algo, algo_name="CMA-ES", env_name=ENV, family=3, n_generations=G,
                                 n_rollouts=N, seed=s, log_every=9999)
        assert algo._hp["lam"] == lam
        rets = np.array(test_fn(jnp.array(res.best_params), key))
        m = compute_statistical_metrics(rets, f_train_best=float(res.best_fitness[-1]), env_name=ENV)
        rec[tag] = {"N": N, "budget": budget, "G": G, "seed": s, "env_steps": G * lam * N * H,
                    "test_mean": m["test_mean"], "success_pct": m["success_rate_pct"], "raw_returns": [float(x) for x in rets]}
        json.dump(rec, open(OUT, "w"))
        print(tag, G, round(m["test_mean"], 1), m["success_rate_pct"], flush=True)
