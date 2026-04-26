"""
MLflow callback for the Race environment training.

Logs per-episode reward, episode length, and cumulative timesteps.
Saves model checkpoints as MLflow artifacts every N steps.
"""

import os
import mlflow
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

class MLflowCallback(BaseCallback):
    """
    SB3 callback that logs training metrics to MLflow and saves checkpoints.

    Parameters
    ----------
    checkpoint_freq : int
        Save a model checkpoint every this many env steps.
    model_save_dir : str
        Local directory where checkpoint files are saved.
    verbose : int
        Verbosity level.
    """

    def __init__(self,
                 checkpoint_freq: int = 50_000,
                 model_save_dir: str = "models",
                 verbose: int = 1):
        super().__init__(verbose)
        self.checkpoint_freq   = checkpoint_freq
        self.model_save_dir    = model_save_dir
        self._last_checkpoint  = 0

        # Public stats accessible from interactive training loops
        self.episode_count = 0
        self.best_reward   = float("-inf")
        self.last_reward   = 0.0
        self.all_rewards   = []
        self.all_lengths   = []

    # ------------------------------------------------------------------
    # SB3 hooks
    # ------------------------------------------------------------------

    def _on_training_start(self) -> None:
        os.makedirs(self.model_save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep_info = info.get("episode")
            if ep_info is not None:
                ep_reward = float(ep_info["r"])
                ep_length = int(ep_info["l"])
                self.episode_count += 1
                self.last_reward    = ep_reward
                self.all_rewards.append(ep_reward)
                self.all_lengths.append(ep_length)
                if ep_reward > self.best_reward:
                    self.best_reward = ep_reward

                try:
                    mlflow.log_metrics(
                        {
                            "episode_reward": ep_reward,
                            "episode_length": ep_length,
                            "best_reward":    self.best_reward,
                        },
                        step=self.num_timesteps,
                    )
                except Exception:
                    pass

                if self.verbose >= 1:
                    print(
                        f"  [ep={self.episode_count:04d}] "
                        f"step={self.num_timesteps:,} "
                        f"reward={ep_reward:.2f} "
                        f"length={ep_length} "
                        f"best={self.best_reward:.2f}"
                    )

        # Checkpoint
        if self.num_timesteps - self._last_checkpoint >= self.checkpoint_freq:
            self._save_checkpoint()
            self._last_checkpoint = self.num_timesteps

        return True

    def _on_training_end(self) -> None:
        self._save_checkpoint(tag="final")
        self._generate_and_log_charts()

    # ------------------------------------------------------------------
    # Hooks the chart generation
    # ------------------------------------------------------------------
    def _generate_and_log_charts(self):
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(self.all_rewards, label="Episode Reward")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title("Training Rewards")
            plt.legend()
            chart_path = os.path.join(self.model_save_dir, "training_rewards.png")
            plt.savefig(chart_path)
            plt.close()
            mlflow.log_artifact(chart_path, artifact_path="charts")
        except Exception as e:
            print(f"  [MLflow] Warning: could not generate charts: {e}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_checkpoint(self, tag: str = None) -> None:
        step_str = f"{self.num_timesteps:010d}"
        suffix   = f"_{tag}" if tag else ""
        filename = f"ppo_race_{step_str}{suffix}.zip"
        path     = os.path.join(self.model_save_dir, filename)

        self.model.save(path)

        try:
            mlflow.log_artifact(path, artifact_path="checkpoints")
            if self.verbose >= 1:
                print(f"  [MLflow] Checkpoint -> {path}")
        except Exception as e:
            print(f"  [MLflow] Warning: could not log artifact: {e}")
