"""
Diagnostic for the degenerate LinearGP policy on Pendulum-v1 (Section 6.6.3 item 6).

For each training seed, reports whether the selected program can produce a non-zero action
(output register written, raw output on sampled observations) and, for a random initial
population, the fraction of programs with a non-zero output. Also runs LinearGP with a larger
budget to check whether non-trivial policies emerge.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import jax, jax.numpy as jnp

from src.family2_programs import LinearGP, forward_lgp_jax
from src.environments import ENV_META, make_program_pop_eval_fn
from src.benchmark import run_program_based
from src.evaluation import make_program_test_eval_fn

ENV = "Pendulum-v1"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "linear_gp_pendulum_diagnosis.json")


def probe(algo, prog, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    th = rng.uniform(-np.pi, np.pi, n)
    obs = np.stack([np.cos(th), np.sin(th), rng.uniform(-8, 8, n)], axis=1).astype(np.float32)
    f = jax.jit(jax.vmap(lambda o: forward_lgp_jax(jnp.asarray(prog, jnp.int32), o, 3, 1, algo.n_regs, algo.out_start)))
    raw = np.asarray(f(jnp.asarray(obs)))[:, 0]
    out_reg = algo.n_regs - 1
    written = bool((np.asarray(prog)[:, 1] == out_reg).any())
    return {"out_reg_written": written, "raw_abs_mean": float(np.abs(raw).mean()), "raw_std": float(raw.std()),
            "frac_nonzero": float((np.abs(raw) > 1e-6).mean())}


def main(seeds, generations, rollouts, pop, prog_len, extra, reeval):
    meta = ENV_META[ENV]
    out = {"config": dict(generations=generations, rollouts=rollouts, pop=pop, prog_len=prog_len, extra_regs=extra, reeval_elite=reeval), "seeds": {}}
    for seed in seeds:
        algo = LinearGP(obs_dim=3, act_dim=1, pop_size=pop, prog_len=prog_len, n_extra_regs=extra, mut_rate=0.15,
                        best_so_far=not reeval)
        state = algo.init(jax.random.PRNGKey(seed))
        init_progs = np.asarray(state.programs)
        init_nonzero = np.mean([probe(algo, p, n=200)["frac_nonzero"] > 0 for p in init_progs])
        res = run_program_based(algo=algo, algo_name="LinearGP", env_name=ENV, family=2,
                                n_generations=generations, n_rollouts=rollouts, seed=seed, log_every=max(1, generations))
        best = np.asarray(res.best_program)
        test_fn = make_program_test_eval_fn(ENV, algo, n_episodes=100)
        test = np.asarray(test_fn(jnp.asarray(best, jnp.int32), jax.random.PRNGKey(99999)))
        rec = probe(algo, best)
        rec.update(init_pop_frac_programs_nonzero=float(init_nonzero), train_best=float(res.best_fitness[-1]),
                   test_mean=float(test.mean()), test_std=float(test.std()),
                   success_pct=float((test >= -200).mean() * 100))
        out["seeds"][str(seed)] = rec
        print(seed, rec, flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(OUT), f"linear_gp_pendulum_diagnosis_G{generations}_N{rollouts}_P{pop}{'_reeval' if reeval else ''}.json"), "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 202, 303, 404])
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--rollouts", type=int, default=8)
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--prog-len", type=int, default=48)
    ap.add_argument("--extra", type=int, default=8)
    ap.add_argument("--best-so-far", action="store_true",
                    help="legacy elite rule (best fitness ever seen); default is the current-generation best")
    a = ap.parse_args()
    main(a.seeds, a.generations, a.rollouts, a.pop, a.prog_len, a.extra, not a.best_so_far)
