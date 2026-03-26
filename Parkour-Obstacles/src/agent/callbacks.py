"""
Custom Stable Baselines3 callback for MLflow experiment tracking.

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
    """

    def __init__(self,
                 checkpoint_freq: int = 50_000,
                 model_save_dir: str = "models",
                 verbose: int = 1):
        super().__init__(verbose)
        self.checkpoint_freq  = checkpoint_freq
        self.model_save_dir   = model_save_dir
        self._last_checkpoint = 0

    # ------------------------------------------------------------------
    # SB3 hooks
    # ------------------------------------------------------------------

    def _on_training_start(self) -> None:
        os.makedirs(self.model_save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # SB3 stores episode info in self.locals["infos"]
        infos = self.locals.get("infos", [])
        for info in infos:
            ep_info = info.get("episode")
            if ep_info is not None:
                ep_reward = ep_info["r"]
                ep_length = ep_info["l"]
                mlflow.log_metrics(
                    {
                        "episode_reward": float(ep_reward),
                        "episode_length": int(ep_length),
                    },
                    step=self.num_timesteps,
                )
                if self.verbose >= 1:
                    print(
                        f"  [MLflow] step={self.num_timesteps:,} "
                        f"ep_reward={ep_reward:.2f} "
                        f"ep_length={ep_length}"
                    )

        # Checkpoint
        if self.num_timesteps - self._last_checkpoint >= self.checkpoint_freq:
            self._save_checkpoint()
            self._last_checkpoint = self.num_timesteps

        return True

    def _on_training_end(self) -> None:
        # Save final model
        self._save_checkpoint(tag="final")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_checkpoint(self, tag: str = None) -> None:
        step_str = f"{self.num_timesteps:010d}"
        suffix   = f"_{tag}" if tag else ""
        filename = f"ppo_parkour_{step_str}{suffix}.zip"
        path     = os.path.join(self.model_save_dir, filename)

        self.model.save(path)

        # Log to MLflow
        try:
            mlflow.log_artifact(path, artifact_path="checkpoints")
            if self.verbose >= 1:
                print(f"  [MLflow] Checkpoint saved -> {path}")
        except Exception as e:
            print(f"  [MLflow] Warning: could not log artifact: {e}")
