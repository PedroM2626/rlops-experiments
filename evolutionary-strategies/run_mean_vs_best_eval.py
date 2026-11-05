"""
Which policy should be evaluated for the model-based algorithms? Every Family 1/3 runner in this
repository returns the best sampled individual (highest single noisy training fitness). For
OpenAI-ES, CMA-ES and PBIL the search distribution also has a mean. This script trains each
algorithm with the standard budget (as in run_multiseed_study.py), 10 seeds, and tests BOTH the
best-so-far individual and the final distribution mean on the same 100 test episodes.
Usage: python run_mean_vs_best_eval.py [--envs ENV ...]
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import jax, jax.numpy as jnp
from run_multiseed_study import build_param_algorithms
from src.benchmark import run_param_based_v2
from src.evaluation import make_test_eval_fn, SUCCESS_THRESHOLDS

SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
ALGOS = ["OpenAI-ES", "CMA-ES", "PBIL"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "mean_vs_best_eval_{}.json")

ap = argparse.ArgumentParser(); ap.add_argument("--envs", nargs="+", default=["CartPole-v1", "Acrobot-v1", "Pendulum-v1", "MountainCarContinuous-v0"])
for env in ap.parse_args().envs:
    G, N = (100, 8) if env == "Pendulum-v1" else (30, 2)
    thr = SUCCESS_THRESHOLDS[env]; fn = make_test_eval_fn(env, n_episodes=100); key = jax.random.PRNGKey(99999)
    path = OUT.format(env)
    rec = json.load(open(path)) if os.path.exists(path) else {}          # resume
    for cfg in build_param_algorithms(pop=32):
        if cfg["name"] not in ALGOS: continue
        per = rec.get(cfg["name"], [])
        done = {x["seed"] for x in per}
        for s in SEEDS:
            if s in done: continue
            r = run_param_based_v2(algo=cfg["algo"], algo_name=cfg["name"], env_name=env, family=cfg["family"],
                                   n_generations=G, n_rollouts=N, seed=s, log_every=9999)
            rb = np.array(fn(jnp.array(r.best_params), key)); rm = np.array(fn(jnp.array(r.final_mean_params), key))
            per.append({"seed": s, "best_mean": float(rb.mean()), "best_succ": float((rb >= thr).mean() * 100),
                        "dist_mean": float(rm.mean()), "dist_succ": float((rm >= thr).mean() * 100)})
            print(env, cfg["name"], s, round(per[-1]["best_mean"], 1), round(per[-1]["dist_mean"], 1), flush=True)
            rec[cfg["name"]] = per
            json.dump(rec, open(path, "w"), indent=1)
