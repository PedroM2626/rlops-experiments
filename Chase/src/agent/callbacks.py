"""
Custom Stable Baselines3 callback for MLflow experiment tracking - Chase game.

Logs per-episode reward, length, and cumulative timesteps.
Saves model checkpoints as MLflow artifacts every N steps.
"""

import os
import mlflow
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class MLflowCallback(BaseCallback):
    """
    SB3 callback that logs training metrics to MLflow and saves checkpoints.

    Parameters
    ----------
    checkpoint_freq : int
        Save a model checkpoint every this many env steps.
    model_save_dir : str
        Local directory where checkpoint files are saved before being logged.
    verbose : int
        Verbosity level.
    stats_callback : callable, optional
        Called with (ep_reward, ep_length) after each episode for live UI updates.
    """

    def __init__(self,
                 checkpoint_freq: int = 50_000,
                 model_save_dir: str = "models",
                 verbose: int = 1,
                 stats_callback=None):
        super().__init__(verbose)
        self.checkpoint_freq  = checkpoint_freq
        self.model_save_dir   = model_save_dir
        self._last_checkpoint = 0
        self.stats_callback   = stats_callback

        # Public stats accessible from the training loop
        self.episode_count    = 0
        self.best_reward      = float("-inf")
        self.last_reward      = 0.0

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
                if ep_reward > self.best_reward:
                    self.best_reward = ep_reward

                try:
                    mlflow.log_metrics(
                        {
                            "episode_reward": ep_reward,
                            "episode_length": ep_length,
                        },
                        step=self.num_timesteps,
                    )
                except Exception:
                    pass

                if self.stats_callback is not None:
                    self.stats_callback(ep_reward, ep_length)

                if self.verbose >= 1:
                    print(
                        f"  [ep={self.episode_count}] "
                        f"step={self.num_timesteps:,} "
                        f"reward={ep_reward:.2f} "
                        f"len={ep_length}"
                    )

        # Checkpoint
        if self.num_timesteps - self._last_checkpoint >= self.checkpoint_freq:
            self._save_checkpoint()
            self._last_checkpoint = self.num_timesteps

        return True

    def _on_training_end(self) -> None:
        self._save_checkpoint(tag="final")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_checkpoint(self, tag: str = None) -> None:
        step_str  = f"{self.num_timesteps:010d}"
        suffix    = f"_{tag}" if tag else ""
        filename  = f"ppo_chase_{step_str}{suffix}.zip"
        path      = os.path.join(self.model_save_dir, filename)

        self.model.save(path)

        try:
            mlflow.log_artifact(path, artifact_path="checkpoints")
            if self.verbose >= 1:
                print(f"  [MLflow] Checkpoint saved -> {path}")
        except Exception as e:
            print(f"  [MLflow] Warning: could not log artifact: {e}")
