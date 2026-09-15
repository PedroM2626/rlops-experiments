"""Arm 1/5 — SB3 library reference (SB3 spec on the SB3+Torch stack).

Uses stable-baselines3 PPO with the shared hyperparameters, n_envs=1,
MlpPolicy [64, 64] + Tanh, constant LR, clip_range_vf=None (SB3 defaults),
normalize_advantage=True (per-minibatch), Adam eps 1e-5, seed=SEED.

CLI:  train_sb3.py --env <EnvId> --seed <int> [--timesteps-scale F]
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import deque

import numpy as np

import config
import variants
from common import set_all_seeds, write_meta_json, package_versions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="CartPole-v1", choices=config.ENVS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timesteps-scale", type=float, default=1.0)
    args = ap.parse_args()

    import gymnasium as gym
    import torch
    torch.set_num_threads(1)
    from sb3_shim import ensure_sb3_importable
    ensure_sb3_importable()
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor

    variants.ensure_dirs()
    set_all_seeds(args.seed, torch=torch)

    total = variants.timesteps_for(args.env, args.timesteps_scale)
    # Align to full rollouts like the manual arms (fair step budget).
    total = max(total // config.N_STEPS, 1) * config.N_STEPS
    log_interval = min(config.LOG_INTERVAL, max(total // 50, 1))

    from config import (ACTIVATION, ADAM_EPS, BATCH_SIZE, CLIP_RANGE, ENT_COEF,
                        GAE_LAMBDA, GAMMA, LEARNING_RATE, MAX_GRAD_NORM,
                        N_EPOCHS, N_STEPS, NET_ARCH, ROLLING_WINDOW, VF_COEF)
    from specs import get_spec
    spec = get_spec("sb3")
    assert spec["clip_value_loss"] is False and spec["anneal_lr"] is False

    activation_fn = torch.nn.Tanh if ACTIVATION == "Tanh" else torch.nn.ReLU

    t0 = time.time()

    env = gym.make(args.env)
    env = Monitor(env)
    env.reset(seed=args.seed)

    rollout: deque[float] = deque(maxlen=ROLLING_WINDOW)
    curve: list[tuple[int, float]] = []

    class CB(BaseCallback):
        def __init__(self):
            super().__init__(0)
            self._last = 0

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                ep = info.get("episode")
                if ep is not None:
                    rollout.append(float(ep["r"]))
            cur = self.num_timesteps
            if cur - self._last >= log_interval:
                self._last = cur
                m = float(np.mean(rollout)) if rollout else 0.0
                curve.append((cur, m))
                print(f"[sb3_torch] {args.env} seed={args.seed} "
                      f"step={cur}/{total} roll20={m:.2f}", flush=True)
            return True

    model = PPO(
        "MlpPolicy", env,
        learning_rate=LEARNING_RATE, gamma=GAMMA, gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE, clip_range_vf=None, normalize_advantage=True,
        n_steps=N_STEPS, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS,
        ent_coef=ENT_COEF, vf_coef=VF_COEF, max_grad_norm=MAX_GRAD_NORM,
        policy_kwargs=dict(net_arch=list(NET_ARCH), activation_fn=activation_fn),
        seed=args.seed, verbose=0, device="cpu",
    )
    # Force SB3's Adam eps to the shared value (already SB3's default).
    for g in model.policy.optimizer.param_groups:
        g["eps"] = ADAM_EPS

    model.learn(total_timesteps=total, callback=CB())
    elapsed = time.time() - t0

    from pathlib import Path
    mp = Path(variants.MODELS_DIR) / f"sb3_torch_{args.env}_seed{args.seed}.zip"
    cp = Path(variants.CURVES_DIR) / f"sb3_torch_{args.env}_seed{args.seed}.csv"
    mep = Path(variants.META_DIR) / f"sb3_torch_{args.env}_seed{args.seed}.json"
    model.save(str(mp))
    with open(cp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "mean_reward"])
        w.writerows(curve)
    write_meta_json(mep, {
        "variant": "sb3_torch", "library": "sb3+torch", "spec": spec,
        "hyperparams": variants.get_hyperparam_dict(),
        "env": args.env, "seed": args.seed, "total_timesteps": total,
        "elapsed_s": elapsed, "device": "cpu",
        "versions": package_versions(["stable-baselines3", "torch",
                                      "gymnasium", "numpy"]),
    })
    print(f"[sb3_torch] done env={args.env} seed={args.seed} "
          f"steps={total} time={elapsed:.1f}s model={mp}")
    env.close()


if __name__ == "__main__":
    main()
