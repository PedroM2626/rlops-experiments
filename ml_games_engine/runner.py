"""Unified experiment runner used by the dashboard and CLI."""

from __future__ import annotations

import argparse
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict

import mlflow
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ml_games_engine.brains import BrainSelection, PilotBrain, build_pilot_brain
from ml_games_engine.control import RuntimePaths, load_json, utc_now_iso, write_json
from ml_games_engine.scenarios import get_scenario


class EngineCallback(BaseCallback):
    """SB3 callback that syncs control state, metrics, checkpoints, and rendering."""

    def __init__(self, runtime: "EngineRuntime"):
        super().__init__(verbose=0)
        self.runtime = runtime

    def _on_training_start(self) -> None:
        self.runtime.write_state(force=True)

    def _on_step(self) -> bool:
        self.runtime.current_step = self.num_timesteps
        self.runtime.sync_control()
        self.runtime.capture_episode_metrics(self.locals.get("infos", []))
        self.runtime.capture_training_metrics(self.model.logger.name_to_value)
        self.runtime.maybe_save_checkpoint()

        while self.runtime.paused and self.runtime.running:
            self.runtime.sync_control()
            self.runtime.render_preview()
            self.runtime.write_state()
            time.sleep(0.1)

        if not self.runtime.running:
            return False

        if self.runtime.timestep_limit > 0 and self.num_timesteps >= self.runtime.timestep_limit:
            self.runtime.append_event(
                f"Timestep limit reached at {self.num_timesteps:,} steps."
            )
            return False

        self.runtime.render_preview()
        self.runtime.write_state()

        if 0.0 < self.runtime.speed < 1.0:
            time.sleep((1.0 / self.runtime.speed - 1.0) * 0.01)

        return True

    def _on_rollout_end(self) -> None:
        self.runtime.capture_training_metrics(self.model.logger.name_to_value)
        self.runtime.write_state()

    def _on_training_end(self) -> None:
        self.runtime.save_final_model()
        self.runtime.write_state(force=True)


class EngineRuntime:
    """Owns the train loop, hot-reloadable reward config, and live state files."""

    def __init__(self, paths: RuntimePaths):
        self.paths = paths
        self.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.control = load_json(paths.control, default={})
        if not self.control:
            raise RuntimeError(f"Control file is missing or invalid: {paths.control}")

        self.scenario = get_scenario(self.control["scenario_id"])
        self.training_cfg = dict(self.scenario.default_training)
        self.training_cfg.update(self.control.get("training", {}))
        self.reward_cfg = dict(self.scenario.default_reward)
        self.reward_cfg.update(self.control.get("reward", {}).get("values", {}))
        self.reward_version = int(self.control.get("reward", {}).get("version", 0))
        self.pilot_version = int(self.control.get("pilot", {}).get("version", 0))
        self.save_version = int(self.control.get("runtime", {}).get("save_version", 0))
        self.control_mtime = self.paths.control.stat().st_mtime if self.paths.control.exists() else 0.0

        self.current_step = 0
        self.episode_count = 0
        self.last_reward = 0.0
        self.best_reward = float("-inf")
        self.last_episode_length = 0
        self.last_loss = None
        self.last_value_loss = None
        self.last_policy_loss = None
        self.last_entropy_loss = None
        self.last_approx_kl = None
        self.last_train_snapshot = None
        self.running = True
        self.paused = bool(self.control.get("runtime", {}).get("paused", False))
        self.speed = float(self.control.get("runtime", {}).get("speed", 1.0))
        self.timestep_limit = int(self.training_cfg.get("timesteps", 0))
        self.last_checkpoint_step = 0
        self.events = deque(maxlen=16)
        self.metrics = deque(maxlen=600)
        self.state_status = "initializing"
        self.started_at = utc_now_iso()
        self.last_state_write = 0.0

        self.model_dir = self.paths.artifacts_dir / self.scenario.scenario_id
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.vec_env = None
        self.model = None
        self.render_env = None
        self.render_obs = None
        self.pilot_brain: PilotBrain | None = None

        self._setup_mlflow()
        self._setup_envs_and_model()
        self._apply_pilot_selection()
        self.append_event(f"Runner started for {self.scenario.label}.")

    def _setup_mlflow(self) -> None:
        tracking_uri = self.control.get("tracking_uri", self.scenario.tracking_uri)
        experiment_name = self.control.get("experiment_name", self.scenario.experiment_name)
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.mlflow_run = mlflow.start_run(run_name=f"{self.scenario.scenario_id}_{int(time.time())}")
        params = {
            "scenario_id": self.scenario.scenario_id,
            "dimension": self.scenario.dimension,
            "viewport": self.scenario.viewport,
        }
        for key, value in self.training_cfg.items():
            if value is None:
                continue
            params[key] = value
        mlflow.log_params(params)

    def _make_env(self, render_mode: str, *, seed: int | None):
        env = self.scenario.make_env(
            render_mode=render_mode,
            reward_config=self.reward_cfg,
            seed=seed,
        )
        return Monitor(env)

    def _setup_envs_and_model(self) -> None:
        n_envs = int(self.training_cfg["n_envs"])
        seed = self.training_cfg.get("seed")
        env_fns = []
        for offset in range(n_envs):
            env_seed = None if seed is None else int(seed) + offset
            env_fns.append(lambda env_seed=env_seed: self._make_env("direct", seed=env_seed))

        base_vec_env = DummyVecEnv(env_fns)
        load_model_path = self.training_cfg.get("model_path")
        load_vecnorm_path = self.training_cfg.get("vecnorm_path")

        if load_vecnorm_path and os.path.exists(load_vecnorm_path):
            self.vec_env = VecNormalize.load(load_vecnorm_path, base_vec_env)
        else:
            self.vec_env = VecNormalize(base_vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

        if load_model_path:
            self.model = PPO.load(load_model_path, env=self.vec_env, device="auto")
            self.current_step = int(self.model.num_timesteps)
            self.append_event(f"Loaded model from {load_model_path}.")
        else:
            net_arch = [int(part.strip()) for part in str(self.training_cfg["net_arch"]).split(",") if part.strip()]
            policy_kwargs = dict(
                net_arch=dict(pi=net_arch, vf=net_arch),
                activation_fn=torch.nn.Tanh,
            )
            self.model = PPO(
                policy="MlpPolicy",
                env=self.vec_env,
                learning_rate=float(self.training_cfg["lr"]),
                n_steps=int(self.training_cfg["n_steps"]),
                batch_size=int(self.training_cfg["batch_size"]),
                gamma=float(self.training_cfg["gamma"]),
                n_epochs=int(self.training_cfg["n_epochs"]),
                gae_lambda=float(self.training_cfg["gae_lambda"]),
                clip_range=float(self.training_cfg["clip_range"]),
                ent_coef=float(self.training_cfg["ent_coef"]),
                policy_kwargs=policy_kwargs,
                verbose=0,
                device="auto",
            )

        render_mode = "rgb_array" if self.scenario.dimension == "2D" else "human"
        self.render_env = self.scenario.make_env(
            render_mode=render_mode,
            reward_config=self.reward_cfg,
            seed=seed,
        )
        self.render_obs, _ = self.render_env.reset()

    def sync_control(self) -> None:
        if not self.paths.control.exists():
            return

        current_mtime = self.paths.control.stat().st_mtime
        if current_mtime <= self.control_mtime:
            return

        self.control_mtime = current_mtime
        self.control = load_json(self.paths.control, default=self.control)

        runtime_cfg = self.control.get("runtime", {})
        self.paused = bool(runtime_cfg.get("paused", False))
        self.speed = max(0.1, float(runtime_cfg.get("speed", 1.0)))
        self.running = not bool(runtime_cfg.get("stop_requested", False))

        requested_limit = int(self.control.get("training", {}).get("timesteps", self.timestep_limit))
        self.timestep_limit = requested_limit

        reward_meta = self.control.get("reward", {})
        reward_version = int(reward_meta.get("version", self.reward_version))
        if reward_version != self.reward_version:
            self.reward_cfg.clear()
            self.reward_cfg.update(self.scenario.default_reward)
            self.reward_cfg.update(reward_meta.get("values", {}))
            self.reward_version = reward_version
            self.append_event("Reward shaping updated live.")

        pilot_meta = self.control.get("pilot", {})
        pilot_version = int(pilot_meta.get("version", self.pilot_version))
        if pilot_version != self.pilot_version:
            self.pilot_version = pilot_version
            self._apply_pilot_selection()
            self.append_event(f"Viewport brain switched to {pilot_meta.get('name', 'live_policy')}.")

        save_version = int(runtime_cfg.get("save_version", self.save_version))
        if save_version != self.save_version:
            self.save_version = save_version
            self.save_checkpoint(tag="manual")

    def _pilot_selection(self) -> BrainSelection:
        pilot_cfg = self.control.get("pilot", {})
        checkpoint_path = pilot_cfg.get("checkpoint_path") or None
        vecnorm_path = pilot_cfg.get("vecnorm_path") or None
        if checkpoint_path and not vecnorm_path:
            sibling = Path(checkpoint_path).with_name("vec_normalize.pkl")
            if sibling.exists():
                vecnorm_path = str(sibling)
        return BrainSelection(
            name=pilot_cfg.get("name", "live_policy"),
            checkpoint_path=checkpoint_path,
            vecnorm_path=vecnorm_path,
        )

    def _apply_pilot_selection(self) -> None:
        selection = self._pilot_selection()
        try:
            if self.pilot_brain is not None:
                self.pilot_brain.close()
            self.pilot_brain = build_pilot_brain(
                selection,
                action_space=self.render_env.action_space,
                observation_space=self.render_env.observation_space,
                model_getter=lambda: self.model,
                vecnorm_getter=lambda: self.vec_env,
            )
        except Exception as exc:
            self.append_event(f"Brain switch fallback to live_policy: {exc}")
            self.pilot_brain = build_pilot_brain(
                BrainSelection(name="live_policy"),
                action_space=self.render_env.action_space,
                observation_space=self.render_env.observation_space,
                model_getter=lambda: self.model,
                vecnorm_getter=lambda: self.vec_env,
            )

    def capture_episode_metrics(self, infos: list[Dict[str, Any]]) -> None:
        for info in infos or []:
            episode = info.get("episode")
            if episode is None:
                continue

            reward = float(episode["r"])
            length = int(episode["l"])
            self.episode_count += 1
            self.last_reward = reward
            self.last_episode_length = length
            self.best_reward = max(self.best_reward, reward)

            point = {
                "kind": "episode",
                "step": self.current_step,
                "episode": self.episode_count,
                "reward": reward,
                "length": length,
                "timestamp": utc_now_iso(),
            }
            self.metrics.append(point)
            mlflow.log_metrics(
                {
                    "episode_reward": reward,
                    "episode_length": length,
                },
                step=self.current_step,
            )

    def capture_training_metrics(self, logger_values: Dict[str, Any]) -> None:
        if not logger_values:
            return

        self.last_loss = self._maybe_float(logger_values.get("train/loss"))
        self.last_value_loss = self._maybe_float(logger_values.get("train/value_loss"))
        self.last_policy_loss = self._maybe_float(logger_values.get("train/policy_gradient_loss"))
        self.last_entropy_loss = self._maybe_float(logger_values.get("train/entropy_loss"))
        self.last_approx_kl = self._maybe_float(logger_values.get("train/approx_kl"))

        snapshot = (
            self.last_loss,
            self.last_value_loss,
            self.last_policy_loss,
            self.last_entropy_loss,
            self.last_approx_kl,
        )

        if all(
            value is None
            for value in [
                self.last_loss,
                self.last_value_loss,
                self.last_policy_loss,
                self.last_entropy_loss,
                self.last_approx_kl,
            ]
        ):
            return

        if snapshot == self.last_train_snapshot:
            return
        self.last_train_snapshot = snapshot

        point = {
            "kind": "train",
            "step": self.current_step,
            "loss": self.last_loss,
            "value_loss": self.last_value_loss,
            "policy_loss": self.last_policy_loss,
            "entropy_loss": self.last_entropy_loss,
            "approx_kl": self.last_approx_kl,
            "timestamp": utc_now_iso(),
        }
        self.metrics.append(point)

        train_metrics = {key.replace("train/", ""): value for key, value in logger_values.items() if key.startswith("train/")}
        safe_metrics = {k: float(v) for k, v in train_metrics.items() if isinstance(v, (int, float, np.floating))}
        if safe_metrics:
            mlflow.log_metrics(safe_metrics, step=self.current_step)

    def maybe_save_checkpoint(self) -> None:
        checkpoint_freq = int(self.training_cfg["checkpoint_freq"])
        if self.current_step - self.last_checkpoint_step >= checkpoint_freq:
            self.save_checkpoint()

    def save_checkpoint(self, tag: str | None = None) -> None:
        suffix = f"_{tag}" if tag else ""
        filename = f"{self.scenario.model_prefix}_{self.current_step:010d}{suffix}"
        model_path = self.model_dir / filename
        vecnorm_path = self.model_dir / "vec_normalize.pkl"
        self.model.save(str(model_path))
        self.vec_env.save(str(vecnorm_path))
        self.last_checkpoint_step = self.current_step
        self.append_event(f"Checkpoint saved to {model_path}.zip")

    def save_final_model(self) -> None:
        final_path = self.model_dir / f"{self.scenario.model_prefix}_final"
        vecnorm_path = self.model_dir / "vec_normalize.pkl"
        self.model.save(str(final_path))
        self.vec_env.save(str(vecnorm_path))
        self.append_event(f"Final model saved to {final_path}.zip")

    def render_preview(self) -> None:
        if self.pilot_brain is None or self.render_env is None:
            return

        try:
            action = self.pilot_brain.predict(self.render_obs, deterministic=True)
            self.render_obs, _, terminated, truncated, _ = self.render_env.step(action)
            if terminated or truncated:
                self.render_obs, _ = self.render_env.reset()

            if self.scenario.dimension == "2D":
                frame = self.render_env.render()
                np.save(self.paths.frame, frame)
        except Exception as exc:
            self.append_event(f"Viewport preview warning: {exc}")

    def append_event(self, message: str) -> None:
        self.events.appendleft(f"[{utc_now_iso()}] {message}")

    def write_state(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last_state_write < 0.5:
            return

        self.last_state_write = now
        if self.state_status not in {"stopped", "error"}:
            self.state_status = "paused" if self.paused else "running"
            if not self.running:
                self.state_status = "stopping"

        state_payload = {
            "scenario_id": self.scenario.scenario_id,
            "scenario_label": self.scenario.label,
            "dimension": self.scenario.dimension,
            "viewport": self.scenario.viewport,
            "status": self.state_status,
            "started_at": self.started_at,
            "updated_at": utc_now_iso(),
            "pid": os.getpid(),
            "step": self.current_step,
            "timestep_limit": self.timestep_limit,
            "episodes": self.episode_count,
            "last_reward": self.last_reward,
            "best_reward": None if self.best_reward == float("-inf") else self.best_reward,
            "last_episode_length": self.last_episode_length,
            "speed": self.speed,
            "paused": self.paused,
            "reward_version_applied": self.reward_version,
            "pilot_version_applied": self.pilot_version,
            "pilot_name": self.control.get("pilot", {}).get("name", "live_policy"),
            "training": self.training_cfg,
            "reward": self.reward_cfg,
            "loss": self.last_loss,
            "value_loss": self.last_value_loss,
            "policy_loss": self.last_policy_loss,
            "entropy_loss": self.last_entropy_loss,
            "approx_kl": self.last_approx_kl,
            "events": list(self.events),
            "artifacts": {
                "model_dir": str(self.model_dir),
                "frame_path": str(self.paths.frame),
                "metrics_path": str(self.paths.metrics),
            },
        }
        write_json(self.paths.state, state_payload)
        write_json(self.paths.metrics, {"history": list(self.metrics)})

    def close(self) -> None:
        try:
            self.running = False
            self.state_status = "stopped"
            self.write_state(force=True)
        except Exception:
            pass

        try:
            if self.pilot_brain is not None:
                self.pilot_brain.close()
        except Exception:
            pass

        try:
            if self.render_env is not None:
                self.render_env.close()
        except Exception:
            pass

        try:
            if self.vec_env is not None:
                self.vec_env.close()
        except Exception:
            pass

        try:
            mlflow.end_run()
        except Exception:
            pass

    def run(self) -> None:
        callback = EngineCallback(self)
        self.state_status = "running"
        self.write_state(force=True)

        try:
            total_timesteps = max(self.timestep_limit, 10_000_000) if self.timestep_limit > 0 else 10_000_000
            self.model.learn(
                total_timesteps=total_timesteps,
                reset_num_timesteps=False,
                progress_bar=False,
                callback=callback,
            )
        except KeyboardInterrupt:
            self.append_event("Runner interrupted by user.")
        finally:
            self.close()

    @staticmethod
    def _maybe_float(value: Any) -> float | None:
        if isinstance(value, (int, float, np.floating)):
            return float(value)
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a hot-reloadable ML games experiment.")
    parser.add_argument(
        "--control",
        type=str,
        required=True,
        help="Path to the runtime control JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = RuntimePaths.from_control_path(args.control)
    runtime = EngineRuntime(paths)
    runtime.run()


if __name__ == "__main__":
    main()
