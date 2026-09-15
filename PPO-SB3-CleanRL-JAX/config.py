"""Shared config for the SB3-vs-CleanRL PPO attribution experiment.

2 (algorithm spec: SB3 | CleanRL) x 2 (backend: Torch | JAX) + bridge arm:
  1. sb3_torch             - SB3 library, SB3 spec (reference)
  2. cleanrl_torch         - manual PyTorch, CleanRL spec (reference)
  3. cleanrl_sb3mode_torch - manual PyTorch, SB3 spec (bridge)
  4. jax_sb3              - pure JAX+Optax, SB3 spec (replication)
  5. jax_cleanrl          - pure JAX+Optax, CleanRL spec (replication)

Full rationale in README.md. Only `specs.py` deltas + backend may differ.
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = RESULTS_DIR / "models"
CURVES_DIR = RESULTS_DIR / "curves"
PLOTS_DIR = RESULTS_DIR / "plots"
TABLES_DIR = RESULTS_DIR / "tables"
META_DIR = RESULTS_DIR / "meta"

ENVS: list[str] = ["CartPole-v1", "LunarLander-v3"]

TOTAL_TIMESTEPS: dict[str, int] = {
    "CartPole-v1": 500_000,
    "LunarLander-v3": 1_000_000,
}

SOLVED_THRESHOLD: dict[str, float] = {
    "CartPole-v1": 475.0,
    "LunarLander-v3": 200.0,
}

LEARNING_RATE = 2.5e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
N_STEPS = 2048
BATCH_SIZE = 64
N_EPOCHS = 4
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
ADAM_EPS = 1e-5
NET_ARCH: list[int] = [64, 64]
ACTIVATION = "Tanh"
N_ENVS = 1

SEEDS: list[int] = [0, 1, 2, 3, 4]

LOG_INTERVAL = 2048
ROLLING_WINDOW = 20
N_EVAL_EPISODES = 100
EVAL_SEED_OFFSET = 10_000
N_BOOTSTRAP = 5_000
BOOTSTRAP_SEED = 12345
