"""
Equal-effort hyperparameter search on Pendulum-v1 for every algorithm (evolutionary and TD).

Protocol (the same for all algorithms):
  * 12 random configurations per algorithm drawn from a small search space (rng seed 2026);
  * each configuration is trained with 2 tuning seeds (1001, 1002) and scored by the mean test return on
    100 episodes with a TUNING test seed (77777), disjoint from the reporting seed (99999) and from the
    reporting training seeds;
  * the best configuration (highest mean score) is then trained with the reporting seeds and evaluated on
    the reporting protocol (100 episodes, test seed 99999). The default configuration is the reference.
Evolutionary budget: G=100, N_rollouts=8 (the pendulum regime of the README); the population size is part
of the search space, so the number of environment steps differs between configurations. TD budget: 30,000
steps. Results: results/hparam_search_pendulum.json (resumable).

Usage: python run_hparam_search.py [--algos SimpleGA DE ...] [--final-seeds 42 101 ...]
"""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import jax, jax.numpy as jnp

from src.family1_direct import SimpleGA, DE, OpenAIES
from src.family3_eda import CMAES, PBIL
from src.benchmark import run_param_based_v2
from src.evaluation import make_test_eval_fn
import rl_baselines

ENV = "Pendulum-v1"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "hparam_search_pendulum.json")
N_TRIALS, TUNE_SEEDS, TUNE_TEST_SEED, G, N = 12, [1001, 1002], 77777, 100, 8

SPACES = {
    "SimpleGA":  dict(pop=[32, 64, 128], sigma_init=[0.1, 0.3, 0.5, 1.0], cx_prob=[0.5, 0.8, 0.95], sigma_decay=[0.99, 0.995, 0.999]),
    "DE":        dict(pop=[32, 64, 128], F=[0.4, 0.6, 0.8, 1.0], CR=[0.5, 0.7, 0.9, 0.95], init_scale=[0.1, 0.5, 1.0]),
    "OpenAI-ES": dict(pop=[32, 64, 128], sigma=[0.02, 0.05, 0.1, 0.2], lr=[0.003, 0.01, 0.03, 0.1]),
    "PBIL":      dict(pop=[32, 64, 128], lr=[0.05, 0.1, 0.2, 0.4], lr_sigma=[0.02, 0.05, 0.1], top_k_frac=[0.1, 0.2, 0.3], sigma_init=[0.3, 0.5, 1.0]),
    "CMA-ES":    dict(sigma0=[0.1, 0.3, 0.5, 1.0], pop=[None, 50, 100]),
    "SAC":       dict(lr=[1e-4, 3e-4, 1e-3], hidden=[64, 128, 256], gamma=[0.98, 0.99], tau=[0.005, 0.01], batch=[128, 256]),
    "TD3":       dict(lr=[1e-4, 3e-4, 1e-3], hidden=[64, 128, 256], gamma=[0.98, 0.99], tau=[0.005, 0.01], batch=[128, 256]),
}
DEFAULTS = {
    "SimpleGA": dict(pop=32, sigma_init=0.5, cx_prob=0.8, sigma_decay=0.999), "DE": dict(pop=32, F=0.8, CR=0.9, init_scale=0.5),
    "OpenAI-ES": dict(pop=32, sigma=0.05, lr=0.01), "PBIL": dict(pop=32, lr=0.1, lr_sigma=0.05, top_k_frac=0.2, sigma_init=1.0),
    "CMA-ES": dict(sigma0=0.5, pop=None), "SAC": dict(lr=3e-4, hidden=64, gamma=0.99, tau=0.005, batch=256),
    "TD3": dict(lr=3e-4, hidden=64, gamma=0.99, tau=0.005, batch=256),
}


def build(name, c):
    if name == "SimpleGA":
        return SimpleGA(pop_size=c["pop"], sigma_init=c["sigma_init"], cx_prob=c["cx_prob"], sigma_decay=c["sigma_decay"]), 1
    if name == "DE":
        return DE(pop_size=c["pop"], F=c["F"], CR=c["CR"], init_scale=c["init_scale"]), 1
    if name == "OpenAI-ES":
        return OpenAIES(pop_size=c["pop"], sigma=c["sigma"], lr=c["lr"]), 1
    if name == "PBIL":
        return PBIL(pop_size=c["pop"], lr=c["lr"], lr_sigma=c["lr_sigma"], top_k_frac=c["top_k_frac"], sigma_init=c["sigma_init"]), 3
    return CMAES(pop_size=c["pop"], sigma0=c["sigma0"]), 3


def evaluate(name, cfg, seed, test_seed, test_fn):
    """Trains one run and returns (mean test return, success %) on 100 episodes of `test_seed`."""
    if name in ("SAC", "TD3"):
        r = rl_baselines.run(name.lower(), ENV, seed, 30000, hp=cfg, eval_at={30000}, test_seed=test_seed, verbose=False)
        m = r["evals"]["30000"]
        return m["test_mean"], m["success_rate_pct"]
    algo, fam = build(name, cfg)
    res = run_param_based_v2(algo=algo, algo_name=name, env_name=ENV, family=fam, n_generations=G, n_rollouts=N, seed=seed, log_every=9999)
    rets = np.array(test_fn(jnp.array(res.best_params), jax.random.PRNGKey(test_seed)))
    return float(rets.mean()), float((rets >= -200.0).mean() * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", nargs="+", default=list(SPACES))
    ap.add_argument("--final-seeds", type=int, nargs="+", default=[42, 101, 202, 303, 404, 505, 606, 707, 808, 909])
    a = ap.parse_args()
    rec = json.load(open(OUT)) if os.path.exists(OUT) else {}
    test_fn = make_test_eval_fn(ENV, n_episodes=100)
    for name in a.algos:
        r = rec.setdefault(name, {"trials": [], "final": {}})
        rng = np.random.default_rng(2026)
        cfgs = [{k: v[int(rng.integers(len(v)))] for k, v in SPACES[name].items()} for _ in range(N_TRIALS)]
        for i, cfg in enumerate(cfgs):
            if i < len(r["trials"]):
                continue
            t0 = time.time()
            sc = [evaluate(name, cfg, s, TUNE_TEST_SEED, test_fn) for s in TUNE_SEEDS]
            r["trials"].append({"config": cfg, "tune_score": float(np.mean([x[0] for x in sc])), "tune_success": float(np.mean([x[1] for x in sc]))})
            json.dump(rec, open(OUT, "w"), indent=1)
            print(name, i, cfg, round(r["trials"][-1]["tune_score"], 1), f"{time.time()-t0:.0f}s", flush=True)
        best = max(r["trials"], key=lambda t: t["tune_score"])["config"]
        r["best_config"] = best; r["default_config"] = DEFAULTS[name]
        final_seeds = a.final_seeds if name not in ("SAC", "TD3") else a.final_seeds[:5]
        for label, cfg in (("best", best), ("default", DEFAULTS[name])):
            done = r["final"].setdefault(label, {})
            for s in final_seeds:
                if str(s) in done:
                    continue
                m, sp = evaluate(name, cfg, s, 99999, test_fn)
                done[str(s)] = {"test_mean": m, "success_pct": sp}
                json.dump(rec, open(OUT, "w"), indent=1)
                print(name, label, s, round(m, 1), sp, flush=True)


if __name__ == "__main__":
    main()
