"""
Soft Actor-Critic (Haarnoja et al., 2018) baseline on Pendulum-v1 in JAX, evaluated with the
same out-of-sample protocol as the evolutionary methods (100 test episodes, test seed 99999,
deterministic policy, H=200, success = return >= -200).

Pendulum-v1 has no terminal state (only the 200-step time limit), so the TD target always
bootstraps. Hyperparameters are the usual SAC defaults and were not tuned.
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import jax, jax.numpy as jnp
import optax
import gymnax

from src.evaluation import compute_statistical_metrics

ENV = "Pendulum-v1"
ACT_SCALE = 2.0
H = 200
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "sac_pendulum.json")


def _init_mlp(key, sizes):
    ps = []
    for k, (i, o) in zip(jax.random.split(key, len(sizes) - 1), zip(sizes[:-1], sizes[1:])):
        ps.append((jax.random.uniform(k, (i, o), minval=-1.0, maxval=1.0) / jnp.sqrt(i), jnp.zeros(o)))
    return ps


def _mlp(ps, x):
    for w, b in ps[:-1]:
        x = jax.nn.relu(x @ w + b)
    return x @ ps[-1][0] + ps[-1][1]


class Actor:
    def init(self, key, obs):
        return _init_mlp(key, [3, 64, 64, 2])

    def apply(self, params, obs):
        o = _mlp(params, obs)
        return o[..., :1], jnp.clip(o[..., 1:], -5.0, 2.0)


class Critic:
    def init(self, key, obs, act):
        return _init_mlp(key, [4, 64, 64, 1])

    def apply(self, params, obs, act):
        return _mlp(params, jnp.concatenate([obs, act], -1))[..., 0]


def sample_action(actor, params, obs, key):
    mean, log_std = actor.apply(params, obs)
    std = jnp.exp(log_std)
    u = mean + std * jax.random.normal(key, mean.shape)
    a = jnp.tanh(u)
    logp = (-0.5 * ((u - mean) / std) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi)).sum(-1)
    logp -= jnp.log(1 - a ** 2 + 1e-6).sum(-1)
    return a, logp


def run(seed, total_steps, start_steps, batch, test_seed, eval_at):
    env, env_params = gymnax.make(ENV)
    actor, critic = Actor(), Critic()
    key = jax.random.PRNGKey(seed)
    key, ka, kc1, kc2 = jax.random.split(key, 4)
    d_obs = jnp.zeros((1, 3)); d_act = jnp.zeros((1, 1))
    ap = actor.init(ka, d_obs)
    cp = (critic.init(kc1, d_obs, d_act), critic.init(kc2, d_obs, d_act))
    tcp = cp
    log_alpha = jnp.array(0.0)
    opt = optax.adam(3e-4)
    st_a, st_c, st_al = opt.init(ap), opt.init(cp), opt.init(log_alpha)
    gamma, tau, target_ent = 0.99, 0.005, -1.0

    @jax.jit
    def update(ap, cp, tcp, log_alpha, st_a, st_c, st_al, b, key):
        obs, act, rew, nobs = b
        k1, k2 = jax.random.split(key)
        alpha = jnp.exp(log_alpha)
        na, nlogp = sample_action(actor, ap, nobs, k1)
        tq = jnp.minimum(critic.apply(tcp[0], nobs, na), critic.apply(tcp[1], nobs, na)) - alpha * nlogp
        y = jax.lax.stop_gradient(rew + gamma * tq)

        def c_loss(cp):
            return sum(((critic.apply(c, obs, act) - y) ** 2).mean() for c in cp)
        cl, cg = jax.value_and_grad(c_loss)(cp)
        up, st_c = opt.update(cg, st_c); cp = optax.apply_updates(cp, up)

        def a_loss(ap):
            a, logp = sample_action(actor, ap, obs, k2)
            q = jnp.minimum(critic.apply(cp[0], obs, a), critic.apply(cp[1], obs, a))
            return (alpha * logp - q).mean(), logp
        (al, logp), ag = jax.value_and_grad(a_loss, has_aux=True)(ap)
        up, st_a = opt.update(ag, st_a); ap = optax.apply_updates(ap, up)

        def al_loss(la):
            return -(la * jax.lax.stop_gradient(logp + target_ent)).mean()
        alg = jax.grad(al_loss)(log_alpha)
        up, st_al = opt.update(alg, st_al); log_alpha = optax.apply_updates(log_alpha, up)
        tcp = jax.tree_util.tree_map(lambda t, p: (1 - tau) * t + tau * p, tcp, cp)
        return ap, cp, tcp, log_alpha, st_a, st_c, st_al

    @jax.jit
    def act_fn(ap, obs, key):
        a, _ = sample_action(actor, ap, obs[None], key)
        return a[0]

    env_step = jax.jit(lambda k, s, a: env.step(k, s, a, env_params))
    env_reset = jax.jit(lambda k: env.reset(k, env_params))

    @jax.jit
    def evaluate(ap, key):
        def episode(k):
            kr, kl = jax.random.split(k)
            obs, s = env.reset(kr, env_params)
            def body(c, _):
                obs, s, tot, kk = c
                kk, sub = jax.random.split(kk)
                mean, _ = actor.apply(ap, obs[None])
                a = jnp.tanh(mean[0]) * ACT_SCALE
                obs, s, r, *_ = env.step(sub, s, a, env_params)
                return (obs, s, tot + r, kk), None
            (_, _, tot, _), _ = jax.lax.scan(body, (obs, s, 0.0, kl), None, length=H)
            return tot
        return jax.vmap(episode)(jax.random.split(key, 100))

    N = total_steps
    buf = dict(o=np.zeros((N, 3), np.float32), a=np.zeros((N, 1), np.float32),
               r=np.zeros(N, np.float32), n=np.zeros((N, 3), np.float32))
    rng_np = np.random.default_rng(seed)
    key, kr = jax.random.split(key)
    obs, s = env_reset(kr)
    t0 = time.time(); evals = {}
    for t in range(N):
        key, k1, k2 = jax.random.split(key, 3)
        if t < start_steps:
            a = jnp.asarray(rng_np.uniform(-1, 1, 1), jnp.float32)
        else:
            a = act_fn(ap, obs, k1)
        nobs, s, r, _, _, _ = env_step(k2, s, a * ACT_SCALE)
        buf["o"][t], buf["a"][t], buf["r"][t], buf["n"][t] = np.asarray(obs), np.asarray(a), float(r), np.asarray(nobs)
        obs = nobs
        if (t + 1) % H == 0:
            key, kr = jax.random.split(key); obs, s = env_reset(kr)
        if t >= start_steps:
            idx = rng_np.integers(0, t + 1, batch)
            b = tuple(jnp.asarray(buf[k][idx]) for k in ("o", "a", "r", "n"))
            key, ku = jax.random.split(key)
            ap, cp, tcp, log_alpha, st_a, st_c, st_al = update(ap, cp, tcp, log_alpha, st_a, st_c, st_al, b, ku)
        if (t + 1) in eval_at:
            rets = np.asarray(evaluate(ap, jax.random.PRNGKey(test_seed)))
            m = compute_statistical_metrics(rets, f_train_best=float("nan"), env_name=ENV)
            m["raw_returns"] = [float(x) for x in rets]
            evals[str(t + 1)] = m
            print(f"seed {seed} step {t+1}: test {m['test_mean']:.2f} +- {m['test_sem']:.2f} success {m['success_rate_pct']}% ({time.time()-t0:.0f}s)", flush=True)
    return {"seed": seed, "total_env_steps": N, "train_time_s": round(time.time() - t0, 1), "evals": evals}


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
