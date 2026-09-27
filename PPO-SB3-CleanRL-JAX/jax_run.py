"""Reference JAX+Optax PPO training loop for discrete envs: same math as
jax_run_fast.py but with per-minibatch gradient steps driven from Python.
Saves the model params, the learning curve and a metadata JSON per run.
"""


def train_main(args) -> None:
    import random as _prng
    import time
    import numpy as _np
    import jax
    import jax.numpy as jnp
    import optax
    import config as _c
    import variants as _v
    from common import (RollingMean, compute_gae, make_env, obs_act_dims,
                        write_curve_csv, write_meta_json, package_versions)
    from jax_core import init_params
    from jax_fwd import cat_entropy, cat_logprob, cat_sample, forward
    from specs import get_spec

    spec = get_spec(args.mode)
    variant = "jax_sb3" if args.mode == "sb3" else "jax_cleanrl"
    _v.ensure_dirs()
    _prng.seed(args.seed)
    _np.random.seed(args.seed)

    total = _v.timesteps_for(args.env, args.timesteps_scale)
    n_updates = total // _c.N_STEPS
    total = n_updates * _c.N_STEPS
    log_every = min(_c.LOG_INTERVAL, max(total // 50, 1))

    key = jax.random.PRNGKey(args.seed)
    key, k_init = jax.random.split(key)

    env = make_env(args.env, args.seed)
    obs_dim, act_dim, is_cont = obs_act_dims(env)
    assert not is_cont, "discrete envs only"

    params = init_params(k_init, obs_dim, act_dim, spec)
    params = {k: jnp.asarray(v) for k, v in params.items()}
    n_minibatches = _c.N_STEPS // _c.BATCH_SIZE
    if spec["anneal_lr"]:
        sched = optax.linear_schedule(
            _c.LEARNING_RATE, 0.0,
            n_updates * _c.N_EPOCHS * n_minibatches)
    else:
        sched = optax.constant_schedule(_c.LEARNING_RATE)
    optimizer = optax.chain(
        optax.clip_by_global_norm(_c.MAX_GRAD_NORM),
        optax.adam(learning_rate=sched, eps=_c.ADAM_EPS))
    opt_state = optimizer.init(params)

    @jax.jit
    def _fwd(p, o):
        return forward(p, o)

    def loss_fn(p, mo, ma, mlp, mv, madv, mrt):
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
            v_c = jnp.clip(nv - mv, -_c.CLIP_RANGE,
                           _c.CLIP_RANGE) + mv
            vf = vhalf * jnp.maximum(v_un, (v_c - mrt) ** 2).mean()
        else:
            vf = vhalf * jnp.mean((nv - mrt) ** 2)
        return pg - _c.ENT_COEF * jnp.mean(ent) + _c.VF_COEF * vf

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
        for t in range(T):
            gs += 1
            logits, val = _fwd(params, jnp.asarray(obs)[None, :])
            lb = _np.asarray(logits)
            key, k1 = jax.random.split(key)
            ai = int(_np.asarray(cat_sample(k1, lb))[0])
            lp = float(_np.asarray(cat_logprob(lb, _np.asarray([ai])))[0])
            nobs, rew, term, trunc, info = env.step(ai)
            done = bool(term or trunc)
            srew, sdone = float(rew), float(done)
            if spec["timeout_bootstrap"] and done and trunc:
                # single-env gymnasium: obs at truncation IS the terminal obs
                tobs = info.get("terminal_observation", nobs)
                _, tv = _fwd(params, jnp.asarray(
                    _np.asarray(tobs, dtype=_np.float32))[None, :])
                srew = float(rew) + _c.GAMMA * float(_np.asarray(tv)[0])
                sdone = 0.0
            b_obs[t], b_act[t] = obs, ai
            b_logp[t], b_rew[t] = lp, srew
            b_done[t], b_val[t] = sdone, float(_np.asarray(val)[0])
            last_truly_done = done
            if "episode" in info:
                roll.add(float(info["episode"]["r"]))
            obs = _np.asarray(env.reset()[0] if done else nobs,
                              dtype=_np.float32)
        _, lv_arr = _fwd(params, jnp.asarray(obs)[None, :])
        lv = float(_np.asarray(lv_arr)[0])
        last_done = bool(last_truly_done and b_done[-1] > 0.5)
        adv, ret = compute_gae(b_rew, b_val, b_done, lv, last_done,
                               _c.GAMMA, _c.GAE_LAMBDA)
        jo, ja = jnp.asarray(b_obs), jnp.asarray(b_act)
        jlp, jv, jret = (jnp.asarray(b_logp), jnp.asarray(b_val),
                         jnp.asarray(ret))
        for _ in range(_c.N_EPOCHS):
            key, kp = jax.random.split(key)
            perm = _np.asarray(jax.random.permutation(kp, T))
            for s in range(0, T, _c.BATCH_SIZE):
                mb = perm[s:s + _c.BATCH_SIZE]
                if len(mb) <= 1:
                    continue
                mba = adv[mb]
                mba = (mba - mba.mean()) / (mba.std() + 1e-8)
                grad_fn = jax.value_and_grad(loss_fn)
                b = (jo[mb], ja[mb], jlp[mb], jv[mb],
                     jnp.asarray(mba.astype(_np.float32)), jret[mb])
                _, grads = grad_fn(params, *b)
                updates, opt_state = optimizer.update(
                    grads, opt_state, params)
                import optax as _ox
                params = _ox.apply_updates(params, updates)
        if gs - last_log >= log_every and len(roll) > 0:
            last_log = gs
            m = roll.mean()
            curve.append((gs, m))
            print(f"[{variant}] {args.env} seed={args.seed} "
                  f"step={gs}/{total} roll20={m:.2f}", flush=True)

    elapsed = time.time() - t0
    from pathlib import Path as _Path
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
          f"steps={total} time={elapsed:.1f}s model={mp}")
    env.close()

