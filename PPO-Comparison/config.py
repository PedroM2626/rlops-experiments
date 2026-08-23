"""Shared hyperparameters and configuration for the PPO comparison experiment.

All implementations MUST use these exact values to ensure a fair comparison.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Global Environment Settings
# ---------------------------------------------------------------------------
ENV_ID = os.getenv("PPO_ENV_ID", "LunarLander-v3")

# ---------------------------------------------------------------------------
# Directory Structure
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results" / ENV_ID
MODELS_DIR = RESULTS_DIR / "models"
CURVES_DIR = RESULTS_DIR / "curves"
PLOTS_DIR = RESULTS_DIR / "plots"
MLFLOW_URI = f"file:///{BASE_DIR.as_posix()}/mlruns"

# ---------------------------------------------------------------------------
# PPO Hyperparameters (identical across all implementations)
# ---------------------------------------------------------------------------
TOTAL_TIMESTEPS = 1_000_000
LEARNING_RATE = 2.5e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
N_STEPS = 2048          # rollout length per environment
BATCH_SIZE = 64          # minibatch size for gradient updates
N_EPOCHS = 4             # number of SGD passes over each rollout
ENT_COEF = 0.01         # entropy coefficient
VF_COEF = 0.5           # value function loss coefficient
MAX_GRAD_NORM = 0.5     # gradient clipping threshold
NORMALIZE_ADVANTAGE = True

# ---------------------------------------------------------------------------
# Network architecture
# ---------------------------------------------------------------------------
NET_ARCH = [64, 64]      # hidden layer sizes for both actor and critic
ACTIVATION = "Tanh"      # activation function

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
N_ENVS = 1              # single env for fair comparison across frameworks

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
N_EVAL_EPISODES = 100    # episodes per trained model during evaluation

# ---------------------------------------------------------------------------
# Training curve logging
# ---------------------------------------------------------------------------
LOG_INTERVAL = 2048      # log mean reward every N steps

# ---------------------------------------------------------------------------
# MLflow experiment name
# ---------------------------------------------------------------------------
MLFLOW_EXPERIMENT = f"ppo_comparison_{ENV_ID}_v2"

# ---------------------------------------------------------------------------
# Implementations to compare
# ---------------------------------------------------------------------------
IMPLEMENTATIONS = [
    {"id": "sb3", "label": "Stable Baselines3", "script": "train_sb3.py"},
    {"id": "cleanrl", "label": "CleanRL (PyTorch)", "script": "train_cleanrl.py"},
    {"id": "custom_continuous", "label": "Custom PyTorch (Continuous)", "script": "train_custom_continuous.py"},
    {"id": "custom_discrete", "label": "Custom PyTorch (Discrete)", "script": "train_custom_discrete.py"},
    {"id": "torchrl", "label": "TorchRL", "script": "train_torchrl.py"},
    {"id": "rllib_extracted", "label": "RLlib (Extracted)", "script": "train_rllib_extracted.py"},
]


def get_hyperparam_dict() -> dict:
    """Return a flat dictionary of all hyperparameters for MLflow logging."""
    return {
        "env_id": ENV_ID,
        "total_timesteps": TOTAL_TIMESTEPS,
        "learning_rate": LEARNING_RATE,
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "clip_range": CLIP_RANGE,
        "n_steps": N_STEPS,
        "batch_size": BATCH_SIZE,
        "n_epochs": N_EPOCHS,
        "ent_coef": ENT_COEF,
        "vf_coef": VF_COEF,
        "max_grad_norm": MAX_GRAD_NORM,
        "normalize_advantage": NORMALIZE_ADVANTAGE,
        "net_arch": str(NET_ARCH),
        "activation": ACTIVATION,
        "seed": SEED,
        "n_envs": N_ENVS,
        "n_eval_episodes": N_EVAL_EPISODES,
    }
