"""
Pendulum-v1 CMA-ES at G=250 across training seeds, to test whether the single-seed
plateau of run_pendulum_stress_test.py (seed 42) holds. Seed 42 is reused from
results/pendulum_stress_test_250.json; the G=100 reference is results/multiseed_study.json.
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax
from src.family3_eda import CMAES
from src.benchmark import run_param_based_v2
from src.evaluation import make_test_eval_fn, compute_statistical_metrics

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "pendulum_stress_multiseed.json")


def main(seeds, generations, rollouts, test_seed, out=OUT):
    global OUT
    OUT = out
    env = "Pendulum-v1"
    records = json.load(open(OUT)) if os.path.exists(OUT) else {}
    test_fn = make_test_eval_fn(env, n_episodes=100)
    for seed in seeds:
        if str(seed) in records:
            continue
        t0 = time.time()
        res = run_param_based_v2(algo=CMAES(sigma0=0.5), algo_name="CMA-ES", env_name=env, family=3,
                                 n_generations=generations, n_rollouts=rollouts, seed=seed, log_every=50)
        rets = np.array(test_fn(res.best_params, jax.random.PRNGKey(test_seed)))
        stats = compute_statistical_metrics(rets, f_train_best=float(res.best_fitness[-1]), env_name=env)
        stats["raw_returns"] = [float(x) for x in rets]
        records[str(seed)] = {"seed": seed, "generations": generations, "rollouts": rollouts,
                              "train_time_s": round(time.time() - t0, 1),
                              "best_fitness_at_100": float(res.best_fitness[min(99, len(res.best_fitness) - 1)]),
                              "evaluation_metrics": stats}
        json.dump(records, open(OUT, "w"), indent=2)
        print(f"seed {seed}: test {stats['test_mean']:.2f} +- {stats['test_sem']:.2f}, success {stats['success_rate_pct']}%", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303, 404])
    ap.add_argument("--generations", type=int, default=250)
    ap.add_argument("--rollouts", type=int, default=8)
    ap.add_argument("--test-seed", type=int, default=99999)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    main(a.seeds, a.generations, a.rollouts, a.test_seed, a.out)
