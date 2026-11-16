"""
Does a larger population close the CMA-ES vs SAC/TD3 gap on Pendulum-v1 (Section 6.5.2 hypothesis 3)?
CMA-ES with population lambda in {25 (default), 100, 400}, N_rollouts=2, G=100 generations, 5 seeds,
same 100-episode test protocol. Simulated environment steps = lambda * N * 200 * G (1M / 4M / 16M).
Results: results/pendulum_cmaes_popsize.json (resumable).
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import jax, jax.numpy as jnp
from src.family3_eda import CMAES
from src.benchmark import run_param_based_v2
from src.evaluation import make_test_eval_fn, compute_statistical_metrics

ENV, G, N = "Pendulum-v1", 100, 2
SEEDS = [42, 101, 202, 303, 404]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "pendulum_cmaes_popsize.json")
rec = json.load(open(OUT)) if os.path.exists(OUT) else {}
test_fn = make_test_eval_fn(ENV, n_episodes=100); key = jax.random.PRNGKey(99999)
for lam in (25, 100, 400):
    for s in SEEDS:
        tag = f"{lam}|{s}"
        if tag in rec:
            continue
        algo = CMAES(pop_size=None if lam == 25 else lam, sigma0=0.5)
        res = run_param_based_v2(algo=algo, algo_name="CMA-ES", env_name=ENV, family=3, n_generations=G, n_rollouts=N, seed=s, log_every=9999)
        assert algo._hp["lam"] == lam
        rets = np.array(test_fn(jnp.array(res.best_params), key))
        m = compute_statistical_metrics(rets, f_train_best=float(res.best_fitness[-1]), env_name=ENV)
        rec[tag] = {"lambda": lam, "seed": s, "env_steps": lam * N * 200 * G, "test_mean": m["test_mean"],
                    "success_pct": m["success_rate_pct"], "raw_returns": [float(x) for x in rets]}
        json.dump(rec, open(OUT, "w"))
        print(tag, round(m["test_mean"], 1), m["success_rate_pct"], flush=True)
