"""Unified deterministic evaluation for all 5 arms."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import config
import variants
from variants import VARIANTS


def eval_sb3_zip(env_id, model_path, n_ep):
    """Rolls out ``n_ep`` greedy episodes of an SB3 ``.zip`` (``predict(deterministic=True)``, so
    the critic head is unused) on fixed seeds ``EVAL_SEED_OFFSET + i``; returns undiscounted returns."""
    import sys as _sys
    _sys.path.insert(0, str(config.BASE_DIR))
    from sb3_shim import ensure_sb3_importable
    ensure_sb3_importable()
    from stable_baselines3 import PPO
    import gymnasium as gym
    model = PPO.load(str(model_path), device="cpu")
    out = []
    for i in range(n_ep):
        env = gym.make(env_id)
        obs, _ = env.reset(seed=config.EVAL_SEED_OFFSET + i)
        tot, done = 0.0, False
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(act)
            tot += float(r)
            done = bool(term or trunc)
        out.append(tot)
        env.close()
    return out



def eval_jax_npz(env_id, model_path, n_ep):
    """Same greedy protocol on saved JAX ``.npz`` params: takes the argmax of the actor logits
    (critic head unused) over the same fixed eval seeds, so it is comparable with ``eval_sb3_zip``."""
    import jax.numpy as jnp
    import gymnasium as gym
    from jax_fwd import forward
    z = np.load(model_path)
    params = {k: jnp.asarray(z[k]) for k in z.files}
    out = []
    for i in range(n_ep):
        env = gym.make(env_id)
        obs, _ = env.reset(seed=config.EVAL_SEED_OFFSET + i)
        tot, done = 0.0, False
        while not done:
            logits, _ = forward(
                params, jnp.asarray(np.asarray(obs, dtype=np.float32))[None, :])
            obs, r, term, trunc, _ = env.step(
                int(np.asarray(logits)[0].argmax()))
            tot += float(r)
            done = bool(term or trunc)
        out.append(tot)
        env.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=None, choices=config.ENVS)
    args = ap.parse_args()
    envs = [args.env] if args.env else list(config.ENVS)
    variants.ensure_dirs()

    import sys
    sys.path.insert(0, str(config.BASE_DIR))
    from eval_torch import eval_torch_pt

    for env_id in envs:
        per_seed: dict[str, list[float]] = {}
        for v in VARIANTS:
            vid = v["id"]
            ext = {"sb3_torch": "zip"}.get(
                vid, "pt" if vid in ("cleanrl_torch", "cleanrl_sb3mode_torch")
                else "npz")
            # discover seeds on disk (allows extra seeds beyond config.SEEDS)
            found = []
            for mp in sorted(variants.MODELS_DIR.glob(
                    f"{vid}_{env_id}_seed*.{ext}")):
                try:
                    seed = int(mp.stem.rsplit("_seed", 1)[1])
                except ValueError:
                    continue
                found.append((seed, mp))
            for seed, mp in sorted(found):
                if ext == "zip":
                    rew = eval_sb3_zip(env_id, mp, config.N_EVAL_EPISODES)
                elif ext == "pt":
                    rew = eval_torch_pt(env_id, mp, config.N_EVAL_EPISODES)
                else:
                    rew = eval_jax_npz(env_id, mp, config.N_EVAL_EPISODES)
                per_seed[f"{vid}__seed{seed}"] = rew
                print(f"  {vid} seed={seed}: mean={np.mean(rew):.2f} "
                      f"std={np.std(rew):.2f}", flush=True)
        if not per_seed:
            print(f"[{env_id}] no models found, skipping")
            continue
        tdir = Path(variants.TABLES_DIR)
        tdir.mkdir(parents=True, exist_ok=True)
        cols = sorted(per_seed)
        with open(tdir / f"eval_rewards_{env_id}.csv", "w",
                  newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for i in range(config.N_EVAL_EPISODES):
                w.writerow([f"{per_seed[c][i]:.4f}" for c in cols])
        with open(tdir / f"eval_summary_{env_id}.csv", "w",
                  newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["variant", "seed", "mean", "std", "median",
                        "min", "max", "n"])
            for c in cols:
                r = np.asarray(per_seed[c])
                vid, seed = c.rsplit("__seed", 1)
                w.writerow([vid, seed, f"{r.mean():.4f}",
                            f"{r.std():.4f}", f"{np.median(r):.4f}",
                            f"{r.min():.4f}", f"{r.max():.4f}", len(r)])
        print(f"[{env_id}] wrote eval tables ({len(cols)} seeds)")


if __name__ == "__main__":
    main()
