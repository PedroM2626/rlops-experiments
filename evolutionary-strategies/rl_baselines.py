"""
Temporal-difference baselines in JAX (SAC, TD3 for continuous actions; double DQN for discrete
actions) evaluated with the same out-of-sample protocol as the evolutionary methods: 100 test
episodes, test seed 99999 (same reset keys as src/evaluation.make_test_eval_fn), deterministic
policy, success thresholds from src.evaluation.SUCCESS_THRESHOLDS.

gymnax auto-resets inside `env.step`: when an episode ends the returned observation is already the
reset observation and the true last observation is `info["final_observation"]`. Transitions store the
true next observation and bootstrap unless the episode *terminated* (time-limit truncation does not
stop bootstrapping).

Standard defaults, not tuned unless `hp` overrides are passed (see run_hparam_search.py).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import jax, jax.numpy as jnp
import optax
import gymnax

from src.environments import ENV_META
from src.evaluation import compute_statistical_metrics, SUCCESS_THRESHOLDS

HP_DEFAULT = dict(hidden=64, act="relu", lr=3e-4, gamma=0.99, tau=0.005, batch=256, start_steps=1000,
                  buffer_window=None,            # None: sample from the whole history; int: only the last K transitions
                  expl_noise=0.1, target_entropy=-1.0,                          # TD3 / SAC
                  eps_end=0.05, eps_frac=0.2, target_every=250)                 # DQN


def init_mlp(key, sizes):
    ps = []
    for k, (i, o) in zip(jax.random.split(key, len(sizes) - 1), zip(sizes[:-1], sizes[1:])):
        ps.append((jax.random.uniform(k, (i, o), minval=-1.0, maxval=1.0) / jnp.sqrt(i), jnp.zeros(o)))
    return ps


def mlp(ps, x, act="relu"):
    f = jax.nn.relu if act == "relu" else jnp.tanh
    for w, b in ps[:-1]:
        x = f(x @ w + b)
    return x @ ps[-1][0] + ps[-1][1]


def _make_eval(env, ep, H, policy):
    @jax.jit
    def evaluate(net, key):
        def episode(k):
            kr, kl = jax.random.split(k)
            obs, s = env.reset(kr, ep)

            def body(c, _):
                obs, s, tot, done, kk = c
                kk, sub = jax.random.split(kk)
                obs, s, r, term, trunc, _ = env.step(sub, s, policy(net, obs), ep)
                return (obs, s, tot + jnp.where(done, 0.0, r), done | term | trunc, kk), None
            (_, _, tot, _, _), _ = jax.lax.scan(body, (obs, s, 0.0, False, kl), None, length=H)
            return tot
        return jax.vmap(episode)(jax.random.split(key, 100))
    return evaluate


def _sac_logp(mean, log_std, key):
    std = jnp.exp(log_std)
    u = mean + std * jax.random.normal(key, mean.shape)
    a = jnp.tanh(u)
    logp = (-0.5 * ((u - mean) / std) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi)).sum(-1) - jnp.log(1 - a ** 2 + 1e-6).sum(-1)
    return a, logp


def run(algo, env_name, seed, steps, hp=None, eval_at=(), test_seed=99999, verbose=True):
    """algo in {'sac', 'td3', 'dqn'}. Returns {'seed','total_env_steps','train_time_s','hp','evals': {step: metrics}}."""
    h = dict(HP_DEFAULT); h.update(hp or {})
    env, ep = gymnax.make(env_name)
    meta = ENV_META[env_name]
    od, ad, H, sc = meta["obs_dim"], meta["act_dim"], meta["max_steps"], (meta.get("act_scale") or 1.0)
    thr = SUCCESS_THRESHOLDS[env_name]
    hid, act = int(h["hidden"]), h["act"]
    key = jax.random.PRNGKey(seed)
    key, k0, k1, k2 = jax.random.split(key, 4)
    opt = optax.adam(h["lr"])
    gamma, tau = h["gamma"], h["tau"]

    if algo == "sac":
        net = init_mlp(k0, [od, hid, hid, 2 * ad])
        cp = (init_mlp(k1, [od + ad, hid, hid, 1]), init_mlp(k2, [od + ad, hid, hid, 1]))
        tcp = cp
        log_alpha = jnp.array(0.0)
        state_opt = (opt.init(net), opt.init(cp), opt.init(log_alpha))
        q = lambda p, o, a: mlp(p, jnp.concatenate([o, a], -1), act)[..., 0]

        def actor(p, o, k):
            out = mlp(p, o, act)
            return _sac_logp(out[..., :ad], jnp.clip(out[..., ad:], -5.0, 2.0), k)

        @jax.jit
        def update(net, cp, tcp, log_alpha, so, b, key):
            sa, sc_, sal = so
            o, a, r, n, d = b
            k1_, k2_ = jax.random.split(key)
            alpha = jnp.exp(log_alpha)
            na, nlogp = actor(net, n, k1_)
            tq = jnp.minimum(q(tcp[0], n, na), q(tcp[1], n, na)) - alpha * nlogp
            y = jax.lax.stop_gradient(r + gamma * (1.0 - d) * tq)
            cg = jax.grad(lambda cp: sum(((q(c, o, a) - y) ** 2).mean() for c in cp))(cp)
            u, sc_ = opt.update(cg, sc_); cp = optax.apply_updates(cp, u)

            def a_loss(p):
                aa, lp = actor(p, o, k2_)
                return (alpha * lp - jnp.minimum(q(cp[0], o, aa), q(cp[1], o, aa))).mean(), lp
            (_, lp), ag = jax.value_and_grad(a_loss, has_aux=True)(net)
            u, sa = opt.update(ag, sa); net = optax.apply_updates(net, u)
            alg = jax.grad(lambda la: -(la * jax.lax.stop_gradient(lp + h["target_entropy"])).mean())(log_alpha)
            u, sal = opt.update(alg, sal); log_alpha = optax.apply_updates(log_alpha, u)
            tcp = jax.tree_util.tree_map(lambda t, p: (1 - tau) * t + tau * p, tcp, cp)
            return net, cp, tcp, log_alpha, (sa, sc_, sal)

        @jax.jit
        def act_train(net, o, k):
            return actor(net, o[None], k)[0][0]

        policy = lambda net, o: jnp.tanh(mlp(net, o, act)[:ad]) * sc

    elif algo == "td3":
        net = init_mlp(k0, [od, hid, hid, ad])
        cp = (init_mlp(k1, [od + ad, hid, hid, 1]), init_mlp(k2, [od + ad, hid, hid, 1]))
        tnet, tcp = net, cp
        state_opt = (opt.init(net), opt.init(cp))
        pi = lambda p, o: jnp.tanh(mlp(p, o, act))
        q = lambda p, o, a: mlp(p, jnp.concatenate([o, a], -1), act)[..., 0]

        @jax.jit
        def upd_critic(net, cp, tnet, tcp, sc_, b, key):
            o, a, r, n, d = b
            na = jnp.clip(pi(tnet, n) + jnp.clip(0.2 * jax.random.normal(key, a.shape), -0.5, 0.5), -1.0, 1.0)
            y = jax.lax.stop_gradient(r + gamma * (1.0 - d) * jnp.minimum(q(tcp[0], n, na), q(tcp[1], n, na)))
            g = jax.grad(lambda cp: sum(((q(c, o, a) - y) ** 2).mean() for c in cp))(cp)
            u, sc_ = opt.update(g, sc_); return optax.apply_updates(cp, u), sc_

        @jax.jit
        def upd_actor(net, cp, tnet, tcp, sa, b):
            o = b[0]
            g = jax.grad(lambda p: -q(cp[0], o, pi(p, o)).mean())(net)
            u, sa = opt.update(g, sa); net = optax.apply_updates(net, u)
            soft = lambda t, p: jax.tree_util.tree_map(lambda x, y: (1 - tau) * x + tau * y, t, p)
            return net, soft(tnet, net), soft(tcp, cp), sa

        act_train = jax.jit(lambda net, o, k: jnp.clip(pi(net, o) + h["expl_noise"] * jax.random.normal(k, (ad,)), -1.0, 1.0))
        policy = lambda net, o: pi(net, o) * sc

    elif algo == "dqn":
        na_ = ad
        net = init_mlp(k0, [od, hid, hid, na_]); tnet = net
        state_opt = opt.init(net)
        qf = lambda p, o: mlp(p, o, act)

        @jax.jit
        def update(net, tnet, so, b):
            o, a, r, n, d = b
            best = jnp.argmax(qf(net, n), -1)
            tq = jnp.take_along_axis(qf(tnet, n), best[:, None], 1)[:, 0]
            y = jax.lax.stop_gradient(r + gamma * (1.0 - d) * tq)
            def loss(p):
                qa = jnp.take_along_axis(qf(p, o), a[:, 0].astype(jnp.int32)[:, None], 1)[:, 0]
                return optax.huber_loss(qa, y).mean()
            g = jax.grad(loss)(net)
            u, so = opt.update(g, so); return optax.apply_updates(net, u), so

        policy = lambda net, o: jnp.argmax(qf(net, o))
        greedy = jax.jit(lambda net, o: jnp.argmax(qf(net, o)))
    else:
        raise ValueError(algo)

    evaluate = _make_eval(env, ep, H, policy)
    step = jax.jit(lambda k, s, a: env.step(k, s, a, ep))
    reset = jax.jit(lambda k: env.reset(k, ep))
    A = 1 if algo == "dqn" else ad
    buf = dict(o=np.zeros((steps, od), np.float32), a=np.zeros((steps, A), np.float32), r=np.zeros(steps, np.float32),
               n=np.zeros((steps, od), np.float32), d=np.zeros(steps, np.float32))
    rng = np.random.default_rng(seed)
    key, kr = jax.random.split(key)
    obs, s = reset(kr)
    t0 = time.time(); evals = {}
    for t in range(steps):
        key, ka, ks = jax.random.split(key, 3)
        if t < h["start_steps"]:
            a = jnp.asarray(rng.integers(0, ad) if algo == "dqn" else rng.uniform(-1, 1, ad), jnp.float32)
        elif algo == "dqn":
            eps = max(h["eps_end"], 1.0 - (1.0 - h["eps_end"]) * t / (h["eps_frac"] * steps))
            a = jnp.asarray(rng.integers(0, ad) if rng.random() < eps else int(greedy(net, obs)), jnp.float32)
        else:
            a = act_train(net, obs, ka)
        env_a = int(a) if algo == "dqn" else a * sc
        nobs, s, r, term, trunc, info = step(ks, s, env_a)
        true_next = info["final_observation"] if bool(term | trunc) else nobs
        buf["o"][t], buf["a"][t], buf["r"][t] = np.asarray(obs), np.atleast_1d(np.asarray(a)), float(r)
        buf["n"][t], buf["d"][t] = np.asarray(true_next), float(term)
        obs = nobs
        if t >= h["start_steps"]:
            lo = 0 if h["buffer_window"] is None else max(0, t + 1 - int(h["buffer_window"]))
            idx = rng.integers(lo, t + 1, int(h["batch"]))
            b = tuple(jnp.asarray(buf[k][idx]) for k in ("o", "a", "r", "n", "d"))
            if algo == "sac":
                key, ku = jax.random.split(key)
                net, cp, tcp, log_alpha, state_opt = update(net, cp, tcp, log_alpha, state_opt, b, ku)
            elif algo == "td3":
                key, ku = jax.random.split(key)
                cp, sc_ = upd_critic(net, cp, tnet, tcp, state_opt[1], b, ku); state_opt = (state_opt[0], sc_)
                if (t - h["start_steps"]) % 2 == 0:
                    net, tnet, tcp, sa = upd_actor(net, cp, tnet, tcp, state_opt[0], b); state_opt = (sa, state_opt[1])
            else:
                net, state_opt = update(net, tnet, state_opt, b)
                if (t + 1) % int(h["target_every"]) == 0:
                    tnet = net
        if (t + 1) in eval_at:
            rets = np.asarray(evaluate(net, jax.random.PRNGKey(test_seed)))
            m = compute_statistical_metrics(rets, f_train_best=float("nan"), env_name=env_name)
            m["raw_returns"] = [float(x) for x in rets]
            evals[str(t + 1)] = m
            if verbose:
                print(f"{algo} {env_name} seed {seed} step {t+1}: test {m['test_mean']:.2f} +- {m['test_sem']:.2f} "
                      f"success {m['success_rate_pct']}% ({time.time()-t0:.0f}s)", flush=True)
    return {"seed": seed, "total_env_steps": steps, "train_time_s": round(time.time() - t0, 1), "hp": h, "evals": evals}
