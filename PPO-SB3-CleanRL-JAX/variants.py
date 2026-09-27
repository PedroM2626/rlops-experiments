"""Variant registry + path helpers (split out to keep config.py small)."""

from __future__ import annotations

import os

from config import (
    BASE_DIR,
    CURVES_DIR,
    META_DIR,
    MODELS_DIR,
    PLOTS_DIR,
    RESULTS_DIR,
    TABLES_DIR,
)

VARIANTS: list[dict[str, str]] = [
    {"id": "sb3_torch", "label": "SB3 (Torch lib)",
     "script": "train_sb3.py", "mode": "", "library": "sb3+torch",
     "spec": "sb3"},
    {"id": "cleanrl_torch", "label": "CleanRL (Torch manual)",
     "script": "train_cleanrl_torch.py", "mode": "cleanrl",
     "library": "torch-manual", "spec": "cleanrl"},
    {"id": "cleanrl_sb3mode_torch", "label": "CleanRL-code / SB3 spec",
     "script": "train_cleanrl_torch.py", "mode": "sb3",
     "library": "torch-manual", "spec": "sb3"},
    {"id": "jax_sb3", "label": "JAX replica / SB3 spec",
     "script": "train_jax_ppo.py", "mode": "sb3",
     "library": "jax+optax", "spec": "sb3"},
    {"id": "jax_cleanrl", "label": "JAX replica / CleanRL spec",
     "script": "train_jax_ppo.py", "mode": "cleanrl",
     "library": "jax+optax", "spec": "cleanrl"},
]

# Ablation arms (README §8): CleanRL spec + one SB3 delta each, JAX stack.
ABLATION_ARMS: list[dict[str, str]] = [
    {"id": f"jax_abl_{a}", "label": f"JAX CleanRL+[{a}]",
     "script": "train_jax_ppo.py", "mode": "cleanrl", "abl": a,
     "library": "jax+optax", "spec": "cleanrl-ablated"}
    for a in ("novclip", "noanneal", "fullmse", "tboot")
]

VARIANTS.extend(ABLATION_ARMS)

VARIANT_IDS = [v["id"] for v in VARIANTS]


def get_hyperparam_dict() -> dict:
    """The hyperparameters shared by every arm, for ``meta.json``. Deliberately excludes the
    SB3/CleanRL spec deltas (those are logged under ``spec``); list values are ``str()``ified."""
    from config import (
        ACTIVATION, ADAM_EPS, BATCH_SIZE, CLIP_RANGE, ENVS, ENT_COEF,
        GAE_LAMBDA, GAMMA, LEARNING_RATE, MAX_GRAD_NORM, N_ENVS, N_EPOCHS,
        N_EVAL_EPISODES, N_STEPS, NET_ARCH, SEEDS, VF_COEF,
    )
    return {
        "learning_rate": LEARNING_RATE, "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA, "clip_range": CLIP_RANGE,
        "n_steps": N_STEPS, "batch_size": BATCH_SIZE,
        "n_epochs": N_EPOCHS, "ent_coef": ENT_COEF,
        "vf_coef": VF_COEF, "max_grad_norm": MAX_GRAD_NORM,
        "adam_eps": ADAM_EPS, "net_arch": str(NET_ARCH),
        "activation": ACTIVATION, "n_envs": N_ENVS,
        "envs": str(ENVS), "seeds": str(SEEDS),
        "n_eval_episodes": N_EVAL_EPISODES,
    }


def timesteps_for(env_id: str, scale: float = 1.0) -> int:
    """Budget for an env after scaling, floored at one whole ``N_STEPS`` rollout; rounding the
    result down to a multiple of ``N_STEPS`` is left to the caller."""
    from config import TOTAL_TIMESTEPS, N_STEPS
    raw = int(TOTAL_TIMESTEPS[env_id] * scale)
    # Always allow at least one full rollout (needed for smoke tests with
    # tiny scales); callers then round down to a multiple of N_STEPS.
    return max(raw, N_STEPS)


def ensure_dirs() -> None:
    for d in (RESULTS_DIR, MODELS_DIR, CURVES_DIR, PLOTS_DIR,
              TABLES_DIR, META_DIR):
        d.mkdir(parents=True, exist_ok=True)
