"""Arms 4-5/5: pure JAX+Optax PPO, switchable spec via --mode.

No torch / SB3 / CleanRL imports. Network, GAE semantics, losses and
schedules mirror train_cleanrl_torch.py exactly; only the autodiff
backend changes (jax.value_and_grad + optax Adam + global-norm clip).

CLI: train_jax_ppo.py --env <EnvId> --seed <int> --mode {sb3,cleanrl}
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import config
import variants
from common import (RollingMean, compute_gae, make_env, obs_act_dims,
                    write_curve_csv, write_meta_json, package_versions)
from jax_run_fast import train_main  # noqa: F401  (backend: jitted PPO)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="CartPole-v1", choices=config.ENVS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default="sb3", choices=["sb3", "cleanrl"])
    ap.add_argument("--abl", default=None,
                    choices=["novclip", "noanneal", "fullmse", "tboot"],
                    help="ablation: CleanRL spec with one SB3 delta applied")
    ap.add_argument("--timesteps-scale", type=float, default=1.0)
    train_main(ap.parse_args())


if __name__ == "__main__":
    main()

