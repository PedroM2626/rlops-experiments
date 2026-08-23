"""Unified evaluation script for all PPO implementations.

Loads each trained model and runs N_EVAL_EPISODES episodes, recording
individual episode rewards for statistical comparison.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import gymnasium as gym
import mlflow
import numpy as np
import torch
import torch.nn as nn

from config import (
    ENV_ID,
    IMPLEMENTATIONS,
    MLFLOW_EXPERIMENT,
    MLFLOW_URI,
    MODELS_DIR,
    N_EVAL_EPISODES,
    RESULTS_DIR,
    SEED,
)


# ---- SB3 Evaluation -------------------------------------------------------

def evaluate_sb3(n_episodes: int) -> list[float]:
    """Evaluate SB3 trained model."""
    from stable_baselines3 import PPO as SB3_PPO

    model_path = MODELS_DIR / "sb3_ppo.zip"
    if not model_path.exists():
        print(f"  [sb3] Model not found: {model_path}")
        return []

    model = SB3_PPO.load(str(model_path))
    rewards = []
    for ep in range(n_episodes):
        env = gym.make(ENV_ID)
        obs, _ = env.reset(seed=SEED + ep + 1000)
        total_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
        env.close()
    return rewards


# ---- PyTorch-based Evaluation (CleanRL / Custom) ---------------------------

class EvalPolicyNet(nn.Module):
    """Generic 2-layer MLP policy for loading CleanRL/Custom checkpoints."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor(x)


def evaluate_pytorch_model(model_path: Path, n_episodes: int, impl_id: str) -> list[float]:
    """Evaluate a PyTorch model saved as state_dict."""
    if not model_path.exists():
        print(f"  Model not found: {model_path}")
        return []

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    # Try to infer architecture from checkpoint
    env = gym.make(ENV_ID)
    is_continuous = isinstance(env.action_space, gym.spaces.Box)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0] if is_continuous else env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EvalPolicyNet(obs_dim, act_dim).to(device)

    if impl_id == "torchrl":
        # TorchRL nested model_state_dict requires special handling
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    elif isinstance(checkpoint, dict):
        # Try loading directly
        try:
            model.load_state_dict(checkpoint, strict=False)
        except RuntimeError:
            # Keys might have different prefix - try stripping
            cleaned = {}
            for k, v in checkpoint.items():
                new_key = k
                for prefix in ["agent.", "policy.", "network."]:
                    if k.startswith(prefix):
                        new_key = k[len(prefix):]
                        break
                cleaned[new_key] = v
            model.load_state_dict(cleaned, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.eval()
    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=SEED + ep + 1000)
        total_reward = 0.0
        done = False
        while not done:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                logits = model(obs_tensor)
                if is_continuous:
                    action = logits.cpu().numpy()[0]
                else:
                    action = logits.argmax(dim=-1).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
        env.close()
    return rewards


# ---- TorchRL Evaluation ---------------------------------------------------

def evaluate_torchrl(n_episodes: int) -> list[float]:
    """Evaluate TorchRL trained model."""
    model_path = MODELS_DIR / "torchrl_ppo.pt"
    if not model_path.exists():
        print(f"  [torchrl] Model not found: {model_path}")
        return []

    # TorchRL saves actor state_dict; load into a simple network
    return evaluate_pytorch_model(model_path, n_episodes, "torchrl")


# ---- Main ------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("PPO Implementation Comparison - Evaluation")
    print("=" * 60)
    print(f"Environment: {ENV_ID}")
    print(f"Episodes per model: {N_EVAL_EPISODES}")
    print()

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    eval_results: dict[str, list[float]] = {}

    evaluators = {
        "sb3": lambda: evaluate_sb3(N_EVAL_EPISODES),
        "cleanrl": lambda: evaluate_pytorch_model(MODELS_DIR / "cleanrl_ppo.pt", N_EVAL_EPISODES, "cleanrl"),
        "custom_continuous": lambda: evaluate_pytorch_model(MODELS_DIR / "custom_continuous_ppo.pt", N_EVAL_EPISODES, "custom_continuous"),
        "custom_discrete": lambda: evaluate_pytorch_model(MODELS_DIR / "custom_discrete_ppo.pt", N_EVAL_EPISODES, "custom_discrete"),
        "torchrl": lambda: evaluate_torchrl(N_EVAL_EPISODES),
        "rllib_extracted": lambda: evaluate_pytorch_model(MODELS_DIR / "rllib_extracted_ppo.pt", N_EVAL_EPISODES, "rllib_extracted"),
    }

    for impl in IMPLEMENTATIONS:
        impl_id = impl["id"]
        label = impl["label"]
        print(f"Evaluating: {label} ({impl_id})")

        evaluator = evaluators.get(impl_id)
        if evaluator is None:
            print(f"  No evaluator defined for {impl_id}, skipping.")
            continue

        start = time.time()
        try:
            rewards = evaluator()
        except Exception as exc:
            print(f"  ERROR evaluating {impl_id}: {exc}")
            rewards = []

        elapsed = time.time() - start

        if rewards:
            eval_results[impl_id] = rewards
            mean_r = np.mean(rewards)
            std_r = np.std(rewards)
            min_r = np.min(rewards)
            max_r = np.max(rewards)
            median_r = np.median(rewards)
            print(f"  Episodes: {len(rewards)}")
            print(f"  Mean: {mean_r:.2f} +/- {std_r:.2f}")
            print(f"  Median: {median_r:.2f}")
            print(f"  Min: {min_r:.2f}, Max: {max_r:.2f}")
            print(f"  Time: {elapsed:.1f}s")

            with mlflow.start_run(run_name=f"eval_{impl_id}"):
                mlflow.log_param("implementation", impl_id)
                mlflow.log_param("phase", "evaluation")
                mlflow.log_param("n_episodes", len(rewards))
                mlflow.log_metrics({
                    "eval_mean_reward": float(mean_r),
                    "eval_std_reward": float(std_r),
                    "eval_min_reward": float(min_r),
                    "eval_max_reward": float(max_r),
                    "eval_median_reward": float(median_r),
                })
        else:
            print(f"  No results for {impl_id}")

        print()

    # Save all evaluation rewards to a single CSV
    output_path = RESULTS_DIR / "eval_rewards.csv"
    if eval_results:
        max_episodes = max(len(v) for v in eval_results.values())
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            headers = list(eval_results.keys())
            writer.writerow(headers)
            for i in range(max_episodes):
                row = []
                for impl_id in headers:
                    rewards = eval_results[impl_id]
                    row.append(rewards[i] if i < len(rewards) else "")
                writer.writerow(row)
        print(f"Evaluation results saved to {output_path}")

        # Also save per-implementation stats
        stats_path = RESULTS_DIR / "eval_stats.csv"
        with open(stats_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["implementation", "mean", "std", "min", "max", "median", "n_episodes"])
            for impl_id, rewards in eval_results.items():
                writer.writerow([
                    impl_id,
                    f"{np.mean(rewards):.4f}",
                    f"{np.std(rewards):.4f}",
                    f"{np.min(rewards):.4f}",
                    f"{np.max(rewards):.4f}",
                    f"{np.median(rewards):.4f}",
                    len(rewards),
                ])
        print(f"Statistics saved to {stats_path}")
    else:
        print("No evaluation results to save.")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
