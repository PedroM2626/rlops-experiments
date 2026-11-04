"""
Pendulum-v1: budget test (G=100 vs G=250) for the parametric algorithms other than CMA-ES
(SimpleGA, DE, OpenAI-ES, PBIL), S=10 seeds, N_rollouts=8, same 100-episode test protocol.
G=100 for seeds 42..404 is taken from results/multiseed_study.json (same config); the other
seeds at G=100 and all seeds at G=250 are run here.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import jax, jax.numpy as jnp

from run_multiseed_study import build_param_algorithms
from src.benchmark import run_param_based_v2
from src.evaluation import make_test_eval_fn, compute_statistical_metrics

ENV = "Pendulum-v1"
SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "pendulum_budget_other_algos.json")


def main():
    rec = json.load(open(OUT)) if os.path.exists(OUT) else {}
    test_fn = make_test_eval_fn(ENV, n_episodes=100)
    key = jax.random.PRNGKey(99999)
    for cfg in build_param_algorithms():
        name = cfg["name"]
        if name == "CMA-ES":
            continue
        for G in (100, 250):
            for s in SEEDS:
                tag = f"{name}|G{G}|{s}"
                if tag in rec or (G == 100 and s in SEEDS[:5]):
                    continue
                res = run_param_based_v2(algo=cfg["algo"], algo_name=name, env_name=ENV, family=cfg["family"],
                                         n_generations=G, n_rollouts=8, seed=s, log_every=9999)
                rets = np.array(test_fn(jnp.array(res.best_params), key))
                m = compute_statistical_metrics(rets, f_train_best=float(res.best_fitness[-1]), env_name=ENV)
                rec[tag] = {"algorithm": name, "G": G, "seed": s, "test_mean": m["test_mean"],
                            "success_pct": m["success_rate_pct"], "train_best": m["f_train_best"],
                            "raw_returns": [float(x) for x in rets]}
                json.dump(rec, open(OUT, "w"))
                print(tag, round(m["test_mean"], 1), m["success_rate_pct"], flush=True)


if __name__ == "__main__":
    main()
