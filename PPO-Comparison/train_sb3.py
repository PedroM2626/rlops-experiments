"""Train PPO on LunarLander-v3 using Stable Baselines3.

This script is part of the PPO implementation comparison experiment.
It uses the shared hyperparameters from config.py and logs all metrics,
artifacts, and the trained model to MLflow.
"""

import csv
from collections import deque

import gymnasium as gym
import mlflow
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from config import (
    BATCH_SIZE,
    CLIP_RANGE,
    CURVES_DIR,
    ENT_COEF,
    ENV_ID,
    GAE_LAMBDA,
    GAMMA,
    LEARNING_RATE,
    LOG_INTERVAL,
    MAX_GRAD_NORM,
    MLFLOW_EXPERIMENT,
    MLFLOW_URI,
    MODELS_DIR,
    N_EPOCHS,
    N_STEPS,
    NET_ARCH,
    SEED,
    TOTAL_TIMESTEPS,
    VF_COEF,
    get_hyperparam_dict,
)


class RewardLoggingCallback(BaseCallback):
    """Custom SB3 callback that tracks episode rewards and logs rolling
    mean reward to MLflow at fixed step intervals.

    Attributes:
        log_interval: Number of timesteps between each logging event.
        reward_buffer: Deque holding the most recent 20 episode rewards.
        curve_data: List of (step, mean_reward) tuples for CSV export.
    """

    def __init__(self, log_interval: int, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.reward_buffer: deque = deque(maxlen=20)
        self.curve_data: list[tuple[int, float]] = []
        self._last_log_step = 0

    def _on_step(self) -> bool:
        """Called after each environment step.

        Extracts episode reward from Monitor info dicts and, every
        ``log_interval`` steps, computes the rolling mean reward (last 20
        episodes) and logs it to MLflow.
        """
        # Monitor wrapper injects "episode" key into info when an episode ends.
        infos = self.locals.get("infos", [])
        for info in infos:
            episode_info = info.get("episode")
            if episode_info is not None:
                ep_reward = float(episode_info["r"])
                self.reward_buffer.append(ep_reward)

        # Log at the configured interval.
        current_step = self.num_timesteps
        if current_step - self._last_log_step >= self.log_interval:
            self._last_log_step = current_step
            if len(self.reward_buffer) > 0:
                mean_reward = float(np.mean(self.reward_buffer))
            else:
                mean_reward = 0.0

            self.curve_data.append((current_step, mean_reward))
            mlflow.log_metric("mean_reward", mean_reward, step=current_step)

            print(
                f"[SB3] Step {current_step:>8d} / {TOTAL_TIMESTEPS} | "
                f"Mean reward (last 20 ep): {mean_reward:>8.2f}"
            )

        return True


def make_env(env_id: str, seed: int) -> gym.Env:
    """Create a Gymnasium environment wrapped with SB3's Monitor.

    Args:
        env_id: Gymnasium environment identifier.
        seed: Random seed for reproducibility.

    Returns:
        A Monitor-wrapped environment instance.
    """
    env = gym.make(env_id)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def save_curve_csv(
    curve_data: list[tuple[int, float]], filepath: str
) -> None:
    """Write training curve data to a CSV file.

    Args:
        curve_data: List of (step, mean_reward) tuples.
        filepath: Destination path for the CSV file.
    """
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "mean_reward"])
        writer.writerows(curve_data)
    print(f"[SB3] Training curve saved to {filepath}")


def main() -> None:
    """Train a PPO agent on LunarLander-v3 using Stable Baselines3.

    The function:
      1. Sets up MLflow tracking.
      2. Creates the environment and PPO model with shared hyperparameters.
      3. Trains with a custom callback for rolling reward logging.
      4. Saves the model, training curve CSV, and logs everything to MLflow.
    """
    # Ensure output directories exist.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CURVES_DIR.mkdir(parents=True, exist_ok=True)

    # -- MLflow setup ----------------------------------------------------------
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="sb3"):
        # Log hyperparameters.
        params = get_hyperparam_dict()
        params["implementation"] = "sb3"
        mlflow.log_params(params)

        # -- Environment -------------------------------------------------------
        env = make_env(ENV_ID, SEED)

        # -- PPO model ---------------------------------------------------------
        policy_kwargs = dict(
            net_arch=list(NET_ARCH),
            activation_fn=torch.nn.Tanh,
        )

        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            clip_range=CLIP_RANGE,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            ent_coef=ENT_COEF,
            vf_coef=VF_COEF,
            max_grad_norm=MAX_GRAD_NORM,
            normalize_advantage=True,
            policy_kwargs=policy_kwargs,
            seed=SEED,
            verbose=0,
        )

        print(f"[SB3] Starting training for {TOTAL_TIMESTEPS} timesteps...")

        # -- Training ----------------------------------------------------------
        callback = RewardLoggingCallback(log_interval=LOG_INTERVAL)
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

        print("[SB3] Training complete.")

        # -- Save model --------------------------------------------------------
        model_path = str(MODELS_DIR / "sb3_ppo.zip")
        model.save(model_path)
        print(f"[SB3] Model saved to {model_path}")

        # -- Save training curve CSV -------------------------------------------
        csv_path = str(CURVES_DIR / "sb3_curve.csv")
        save_curve_csv(callback.curve_data, csv_path)

        # -- Log artifacts to MLflow -------------------------------------------
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(csv_path)

        print("[SB3] MLflow run complete. Artifacts logged.")

    env.close()


if __name__ == "__main__":
    main()
