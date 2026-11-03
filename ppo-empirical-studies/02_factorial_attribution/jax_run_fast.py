"""JIT-compiled JAX+Optax PPO training (fast backend for train_jax_ppo.py).

Same math/algorithm as jax_run.py but updates are fully jitted per epoch with
jax.lax.scan, and the PPO forward at rollout is single-jitted.
"""
from __future__ import annotations


def train_main(args) -> None:
    import random as _prng
    import time
    from pathlib import Path as _Path

    import jax
    import jax.numpy as jnp
    import numpy as _np
    import optax

    import config as _c
    import variants as _v
    from common import (RollingMean, compute_gae, make_env, obs_act_dims,
                        write_curve_csv, write_meta_json, package_versions)
    from jax_core import init_params
    from jax_fwd import cat_entropy, cat_logprob, cat_sample, forward
    from specs import get_ablated_spec, get_spec

    if getattr(args, "abl", None):
        spec = get_ablated_spec(args.abl)
        variant = f"jax_abl_{args.abl}"
    else:
        spec = get_spec(args.mode)
        variant = "jax_sb3" if args.mode == "sb3" else "jax_cleanrl"
    _v.ensure_dirs()
    _prng.seed(args.seed)
    _np.random.seed(args.seed)

    total = _v.timesteps_for(args.env, args.timesteps_scale)
    n_updates = total // _c.N_STEPS
    total = n_updates * _c.N_STEPS
    log_every = min(_c.LOG_INTERVAL, max(total // 50, 1))

    n_mb = _c.N_STEPS // _c.BATCH_SIZE

    key = jax.random.PRNGKey(args.seed)
    key, k_init = jax.random.split(key)

    env = make_env(args.env, args.seed)
    obs_dim, act_dim, is_cont = obs_act_dims(env)
    assert not is_cont, "discrete envs only"

    params = init_params(k_init, obs_dim, act_dim, spec)
    params = {k: jnp.asarray(v) for k, v in params.items()}

    if spec["anneal_lr"]:
        sched = optax.linear_schedule(
            _c.LEARNING_RATE, 0.0,
            n_updates * _c.N_EPOCHS * n_mb)
    else:
        sched = optax.constant_schedule(_c.LEARNING_RATE)
    optimizer = optax.chain(
        optax.clip_by_global_norm(_c.MAX_GRAD_NORM),
        optax.adam(learning_rate=sched, eps=_c.ADAM_EPS))
    opt_state = optimizer.init(params)

    @jax.jit
    def act_fn(p, k, o):
        logits, val = forward(p, o)
        act = cat_sample(k, logits)
        lp = cat_logprob(logits, act)
        return act, lp, val

    @jax.jit
    def fwd(p, o):
        """Jitted forward -> (logits, value); used only to value the truncation observation and
        the final state of a rollout, since actions and log-probs come from ``act_fn``."""
        return forward(p, o)

    def loss_fn(p, mo, ma, mlp, mv, madv, mrt):
        """Clipped-surrogate PPO loss for one minibatch; ``spec`` (closed over, so fixed at trace
        time) drives both the value-clip epsilon and the 0.5 MSE factor. No KL term anywhere."""
        logits, nv = forward(p, mo)
        nlp = cat_logprob(logits, ma)
        ent = cat_entropy(logits)
        ratio = jnp.exp(nlp - mlp)
        pg = jnp.maximum(
            -madv * ratio,
            -madv * jnp.clip(ratio, 1 - _c.CLIP_RANGE,
                             1 + _c.CLIP_RANGE)).mean()
        vhalf = 0.5 if spec["value_loss_half"] else 1.0
        if spec["clip_value_loss"]:
            v_un = (nv - mrt) ** 2
            v_c = jnp.clip(nv - mv, -_c.CLIP_RANGE, _c.CLIP_RANGE) + mv
            vf = vhalf * jnp.maximum(v_un, (v_c - mrt) ** 2).mean()
        else:
            vf = vhalf * jnp.mean((nv - mrt) ** 2)
        return pg - _c.ENT_COEF * jnp.mean(ent) + _c.VF_COEF * vf

    @jax.jit
    def epoch(params, opt_state, rollout, perm):
        """One PPO epoch (the caller runs N_EPOCHS of them): a jitted ``lax.scan`` of ``one_mb``
        over ``perm`` reshaped into N_STEPS//BATCH_SIZE contiguous minibatches."""
        jo, ja, jlp, jv, jadv, jret = rollout

        def one_mb(carry, mb):
            """One minibatch SGD step inside the scan: normalises that minibatch's advantages
            (eps 1e-8), differentiates ``loss_fn``, then applies the optax update."""
            params, opt_state = carry
            mba = jadv[mb]
            mba = (mba - mba.mean()) / (mba.std() + 1e-8)
            grad_fn = jax.value_and_grad(loss_fn)
            b = (jo[mb], ja[mb], jlp[mb], jv[mb], mba, jret[mb])
            _, grads = grad_fn(params, *b)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), None
        (params, opt_state), _ = jax.lax.scan(
            one_mb, (params, opt_state), perm.reshape(n_mb, _c.BATCH_SIZE))
        return params, opt_state

    obs = _np.asarray(env.reset(seed=args.seed + 999)[0], dtype=_np.float32)
    roll = RollingMean(_c.ROLLING_WINDOW)
    curve: list[tuple[int, float]] = []
    gs, last_log = 0, 0
    t0 = time.time()

    for update in range(1, n_updates + 1):
        T = _c.N_STEPS
        b_obs = _np.zeros((T, obs_dim), _np.float32)
        b_act = _np.zeros(T, _np.int64)
        b_logp = _np.zeros(T, _np.float32)
        b_rew = _np.zeros(T, _np.float32)
        b_done = _np.zeros(T, _np.float32)
        b_val = _np.zeros(T, _np.float32)
        last_truly_done = False
        obs_j = jnp.asarray(obs)[None, :]
        for t in range(T):
            key, k1 = jax.random.split(key)
            ai_j, lp_j, val_j = act_fn(params, k1, obs_j)
            ai = int(_np.asarray(ai_j)[0])
            lp = float(_np.asarray(lp_j)[0])
            val = float(_np.asarray(val_j)[0])
            nobs, rew, term, trunc, info = env.step(ai)
            done = bool(term or trunc)
            srew, sdone = float(rew), float(done)
            if spec["timeout_bootstrap"] and done and trunc:
                # single-env gymnasium has no "terminal_observation"; on
                # TimeLimit truncation `nobs` IS the final observation.
                tobs = info.get("terminal_observation", nobs)
                _, tv = fwd(params, jnp.asarray(
                    _np.asarray(tobs, dtype=_np.float32))[None, :])
                srew = float(rew) + _c.GAMMA * float(_np.asarray(tv)[0])
                sdone = 0.0
            b_obs[t], b_act[t] = obs, ai
            b_logp[t], b_rew[t] = lp, srew
            b_done[t], b_val[t] = sdone, val
            last_truly_done = done
            if "episode" in info:
                roll.add(float(info["episode"]["r"]))
            obs = _np.asarray(env.reset()[0] if done else nobs, _np.float32)
            obs_j = jnp.asarray(obs)[None, :]
            gs += 1
        _, lv_arr = fwd(params, obs_j)
        lv = float(_np.asarray(lv_arr)[0])
        last_done = bool(last_truly_done and b_done[-1] > 0.5)
        adv, ret = compute_gae(b_rew, b_val, b_done, lv, last_done,
                               _c.GAMMA, _c.GAE_LAMBDA)
        jo = jnp.asarray(b_obs); ja = jnp.asarray(b_act)
        jlp = jnp.asarray(b_logp); jv = jnp.asarray(b_val)
        jret = jnp.asarray(ret)
        jadv = jnp.asarray(adv)
        rollout = (jo, ja, jlp, jv, jadv, jret)
        for _ in range(_c.N_EPOCHS):
            key, kp = jax.random.split(key)
            perm = jax.random.permutation(kp, T)
            params, opt_state = epoch(params, opt_state, rollout, perm)
        jax.block_until_ready(params)
        if gs - last_log >= log_every and len(roll) > 0:
            last_log = gs
            m = roll.mean()
            curve.append((gs, m))
            print(f"[{variant}] {args.env} seed={args.seed} "
                  f"step={gs}/{total} roll20={m:.2f}", flush=True)

    elapsed = time.time() - t0
    mp = _Path(_v.MODELS_DIR) / f"{variant}_{args.env}_seed{args.seed}.npz"
    cp = _Path(_v.CURVES_DIR) / f"{variant}_{args.env}_seed{args.seed}.csv"
    mep = _Path(_v.META_DIR) / f"{variant}_{args.env}_seed{args.seed}.json"
    _np.savez(str(mp), **{k: _np.asarray(v) for k, v in params.items()})
    write_curve_csv(cp, curve)
    write_meta_json(mep, {
        "variant": variant, "library": "jax+optax", "spec": spec,
        "hyperparams": _v.get_hyperparam_dict(),
        "env": args.env, "seed": args.seed, "total_timesteps": total,
        "elapsed_s": elapsed, "device": "cpu",
        "versions": package_versions(
            ["jax", "jaxlib", "optax", "gymnasium", "numpy"]),
    })
    print(f"[{variant}] done env={args.env} seed={args.seed} "
          f"steps={total} time={elapsed:.1f}s model={mp}", flush=True)
    env.close()
