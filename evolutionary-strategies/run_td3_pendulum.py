"""
TD3 (Fujimoto, van Hoof & Meger, 2018) baseline on Pendulum-v1 in JAX, evaluated with the same
out-of-sample protocol as the other methods (100 test episodes, test seed 99999, deterministic
policy, H=200, success = return >= -200). Standard defaults, not tuned. Networks and helpers are
shared with run_sac_pendulum.py.
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import jax, jax.numpy as jnp
import optax
import gymnax

from run_sac_pendulum import _init_mlp, _mlp, ENV, ACT_SCALE, H
from src.evaluation import compute_statistical_metrics

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "td3_pendulum.json")


def pi(p, o):
    return jnp.tanh(_mlp(p, o))                      # in [-1, 1]; scaled by ACT_SCALE in the env


def q(p, o, a):
    return _mlp(p, jnp.concatenate([o, a], -1))[..., 0]


def run(seed, N, start, batch, test_seed, eval_at):
    env, ep = gymnax.make(ENV)
    key = jax.random.PRNGKey(seed)
    key, ka, k1, k2 = jax.random.split(key, 4)
    ap = _init_mlp(ka, [3, 64, 64, 1]); cp = (_init_mlp(k1, [4, 64, 64, 1]), _init_mlp(k2, [4, 64, 64, 1]))
    tap, tcp = ap, cp
    opt = optax.adam(3e-4); sa, sc = opt.init(ap), opt.init(cp)
    gamma, tau, pn, nc, delay = 0.99, 0.005, 0.2, 0.5, 2

    @jax.jit
    def upd_critic(ap, cp, tap, tcp, sc, b, key):
        o, a, r, n = b
        noise = jnp.clip(pn * jax.random.normal(key, a.shape), -nc, nc)
        na = jnp.clip(pi(tap, n) + noise, -1.0, 1.0)
        y = jax.lax.stop_gradient(r + gamma * jnp.minimum(q(tcp[0], n, na), q(tcp[1], n, na)))
        loss = lambda cp: sum(((q(c, o, a) - y) ** 2).mean() for c in cp)
        g = jax.grad(loss)(cp)
        u, sc = opt.update(g, sc); return optax.apply_updates(cp, u), sc

    @jax.jit
    def upd_actor(ap, cp, tap, tcp, sa, b):
        o = b[0]
        g = jax.grad(lambda ap: -q(cp[0], o, pi(ap, o)).mean())(ap)
        u, sa = opt.update(g, sa); ap = optax.apply_updates(ap, u)
        soft = lambda t, p: jax.tree_util.tree_map(lambda x, y: (1 - tau) * x + tau * y, t, p)
        return ap, soft(tap, ap), soft(tcp, cp), sa

    @jax.jit
    def evaluate(ap, key):
        def episode(k):
            kr, kl = jax.random.split(k)
            obs, s = env.reset(kr, ep)
            def body(c, _):
                obs, s, tot, kk = c
                kk, sub = jax.random.split(kk)
                obs, s, r, *_ = env.step(sub, s, pi(ap, obs) * ACT_SCALE, ep)
                return (obs, s, tot + r, kk), None
            (_, _, tot, _), _ = jax.lax.scan(body, (obs, s, 0.0, kl), None, length=H)
            return tot
        return jax.vmap(episode)(jax.random.split(key, 100))

    step = jax.jit(lambda k, s, a: env.step(k, s, a, ep))
    reset = jax.jit(lambda k: env.reset(k, ep))
    act = jax.jit(lambda ap, o: pi(ap, o))
    buf = dict(o=np.zeros((N, 3), np.float32), a=np.zeros((N, 1), np.float32), r=np.zeros(N, np.float32), n=np.zeros((N, 3), np.float32))
    rng = np.random.default_rng(seed); key, kr = jax.random.split(key); obs, s = reset(kr)
    t0 = time.time(); evals = {}
    for t in range(N):
        key, k1, k2 = jax.random.split(key, 3)
        if t < start:
            a = jnp.asarray(rng.uniform(-1, 1, 1), jnp.float32)
        else:
            a = jnp.clip(act(ap, obs) + 0.1 * jax.random.normal(k1, (1,)), -1.0, 1.0)
        nobs, s, r, _, _, _ = step(k2, s, a * ACT_SCALE)
        buf["o"][t], buf["a"][t], buf["r"][t], buf["n"][t] = np.asarray(obs), np.asarray(a), float(r), np.asarray(nobs)
        obs = nobs
        if (t + 1) % H == 0:
            key, kr = jax.random.split(key); obs, s = reset(kr)
        if t >= start:
            idx = rng.integers(0, t + 1, batch)
            b = tuple(jnp.asarray(buf[k][idx]) for k in ("o", "a", "r", "n"))
            key, ku = jax.random.split(key)
            cp, sc = upd_critic(ap, cp, tap, tcp, sc, b, ku)
            if (t - start) % delay == 0:
                ap, tap, tcp, sa = upd_actor(ap, cp, tap, tcp, sa, b)
        if (t + 1) in eval_at:
            rets = np.asarray(evaluate(ap, jax.random.PRNGKey(test_seed)))
            m = compute_statistical_metrics(rets, f_train_best=float("nan"), env_name=ENV)
            m["raw_returns"] = [float(x) for x in rets]; evals[str(t + 1)] = m
            print(f"seed {seed} step {t+1}: test {m['test_mean']:.2f} +- {m['test_sem']:.2f} success {m['success_rate_pct']}% ({time.time()-t0:.0f}s)", flush=True)
    return {"seed": seed, "total_env_steps": N, "evals": evals}


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 202, 303, 404])
    ap_.add_argument("--steps", type=int, default=30000)
    ap_.add_argument("--start-steps", type=int, default=1000)
    ap_.add_argument("--batch", type=int, default=256)
    ap_.add_argument("--test-seed", type=int, default=99999)
    ap_.add_argument("--eval-at", type=int, nargs="+", default=[5000, 10000, 20000, 30000])
    a = ap_.parse_args()
    recs = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for sd in a.seeds:
        if str(sd) in recs: continue
        recs[str(sd)] = run(sd, a.steps, a.start_steps, a.batch, a.test_seed, set(a.eval_at))
        json.dump(recs, open(OUT, "w"), indent=2)
