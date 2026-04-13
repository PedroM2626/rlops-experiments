"""Unified experiment runner used by the dashboard and CLI."""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable

import mlflow
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ml_games_engine.brains import BrainSelection, PilotBrain, build_pilot_brain
from ml_games_engine.control import (
    RuntimePaths,
    append_jsonl,
    load_json,
    sanitize_slug,
    utc_now_iso,
    write_json,
)
from ml_games_engine.exporters import export_engine_package
from ml_games_engine.scenarios import get_scenario

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency safety
    plt = None


class RunArchive:
    """Persist complete run history, summaries, and chart exports."""

    def __init__(
        self,
        *,
        paths: RuntimePaths,
        run_id: str,
        control: dict,
        scenario,
        training_cfg: dict,
        reward_cfg: dict,
        world_cfg: dict,
    ):
        self.paths = paths
        self.run_id = run_id
        self.run_dir = paths.runs_dir / run_id
        self.models_dir = self.run_dir / "models"
        self.charts_dir = self.run_dir / "charts"
        self.previews_dir = self.run_dir / "previews"
        self.exports_dir = self.run_dir / "engine_export"

        self.manifest_path = self.run_dir / "manifest.json"
        self.summary_path = self.run_dir / "summary.json"
        self.state_latest_path = self.run_dir / "state_latest.json"
        self.metrics_latest_path = self.run_dir / "metrics_latest.json"
        self.control_latest_path = self.run_dir / "control_latest.json"
        self.control_history_path = self.run_dir / "control_history.jsonl"
        self.state_history_path = self.run_dir / "state_history.jsonl"
        self.events_path = self.run_dir / "events.jsonl"
        self.episode_history_path = self.run_dir / "episode_history.jsonl"
        self.train_history_path = self.run_dir / "train_history.jsonl"

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        self.previews_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)

        self.last_chart_export = 0.0
        self.last_export_counts = (-1, -1)

        manifest = {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "scenario": {
                "scenario_id": scenario.scenario_id,
                "label": scenario.label,
                "dimension": scenario.dimension,
                "viewport": scenario.viewport,
                "description": scenario.description,
                "info": scenario.info,
            },
            "training": training_cfg,
            "reward": reward_cfg,
            "world": world_cfg,
            "paths": {
                "run_dir": str(self.run_dir.resolve()),
                "models_dir": str(self.models_dir.resolve()),
                "charts_dir": str(self.charts_dir.resolve()),
                "exports_dir": str(self.exports_dir.resolve()),
            },
            "system": {
                "python": sys.version,
                "platform": platform.platform(),
                "cwd": os.getcwd(),
                "torch": torch.__version__,
                "mlflow": mlflow.__version__,
                "cuda_available": torch.cuda.is_available(),
            },
            "initial_control": control,
        }
        write_json(self.manifest_path, manifest)
        self.snapshot_control(control, reason="run_start")

    def snapshot_control(self, control: dict, *, reason: str) -> None:
        write_json(self.control_latest_path, control)
        append_jsonl(
            self.control_history_path,
            {
                "timestamp": utc_now_iso(),
                "reason": reason,
                "payload": control,
            },
        )

    def update_manifest(self, extra: dict) -> None:
        manifest = load_json(self.manifest_path, default={})
        manifest.update(extra)
        write_json(self.manifest_path, manifest)

    def log_event(self, message: str) -> None:
        append_jsonl(
            self.events_path,
            {
                "timestamp": utc_now_iso(),
                "message": message,
            },
        )

    def log_episode_metric(self, point: dict) -> None:
        append_jsonl(self.episode_history_path, point)

    def log_train_metric(self, point: dict) -> None:
        append_jsonl(self.train_history_path, point)

    def write_state(self, payload: dict) -> None:
        write_json(self.state_latest_path, payload)
        append_jsonl(self.state_history_path, payload)

    def write_metrics_latest(self, history: list[dict]) -> None:
        write_json(self.metrics_latest_path, {"history": history})

    def write_summary(self, payload: dict) -> None:
        write_json(self.summary_path, payload)

    def write_frame(self, frame: np.ndarray) -> None:
        np.save(self.previews_dir / "latest_frame.npy", frame)
        if plt is not None:
            plt.imsave(self.previews_dir / "latest_frame.png", frame)

    def export_charts(self, episode_rows: list[dict], train_rows: list[dict], *, force: bool = False) -> None:
        if plt is None:
            return

        counts = (len(episode_rows), len(train_rows))
        now = time.time()
        if not force and counts == self.last_export_counts and (now - self.last_chart_export) < 2.0:
            return
        if not force and counts == self.last_export_counts:
            return

        self.last_export_counts = counts
        self.last_chart_export = now

        if episode_rows:
            steps = [row["step"] for row in episode_rows]
            rewards = [row["reward"] for row in episode_rows]
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(steps, rewards, color="#2E8B57", linewidth=2)
            ax.set_title("Episode Reward")
            ax.set_xlabel("Step")
            ax.set_ylabel("Reward")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(self.charts_dir / "episode_reward.png", dpi=140)
            plt.close(fig)

        if train_rows:
            fig, ax = plt.subplots(figsize=(9, 4))
            steps = [row["step"] for row in train_rows]
            if any(row.get("loss") is not None for row in train_rows):
                ax.plot(steps, [row.get("loss") for row in train_rows], label="loss", linewidth=2)
            if any(row.get("value_loss") is not None for row in train_rows):
                ax.plot(steps, [row.get("value_loss") for row in train_rows], label="value_loss", linewidth=1.5)
            if any(row.get("approx_kl") is not None for row in train_rows):
                ax.plot(steps, [row.get("approx_kl") for row in train_rows], label="approx_kl", linewidth=1.5)
            ax.set_title("Training Metrics")
            ax.set_xlabel("Step")
            ax.grid(alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(self.charts_dir / "training_metrics.png", dpi=140)
            plt.close(fig)


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
        self.paths.runs_dir.mkdir(parents=True, exist_ok=True)
        self.control = load_json(paths.control, default={})
        if not self.control:
            raise RuntimeError(f"Control file is missing or invalid: {paths.control}")

        self.scenario = get_scenario(self.control["scenario_id"])
        self.training_cfg = dict(self.scenario.default_training)
        self.training_cfg.update(self.control.get("training", {}))
        self.reward_cfg = dict(self.scenario.default_reward)
        self.reward_cfg.update(self.control.get("reward", {}).get("values", {}))
        self.world_cfg = dict(self.scenario.default_world)
        self.world_cfg.update(self.control.get("world", {}).get("values", {}))
        runtime_cfg = self.control.get("runtime", {})
        self.reward_version = int(self.control.get("reward", {}).get("version", 0))
        self.world_version = int(self.control.get("world", {}).get("version", 0))
        self.pilot_version = int(self.control.get("pilot", {}).get("version", 0))
        self.save_version = int(runtime_cfg.get("save_version", 0))
        self.control_mtime = self.paths.control.stat().st_mtime if self.paths.control.exists() else 0.0
        self.run_id = sanitize_slug(runtime_cfg.get("run_id", f"{self.scenario.scenario_id}_{utc_now_iso()}"))
        self.archive = RunArchive(
            paths=self.paths,
            run_id=self.run_id,
            control=self.control,
            scenario=self.scenario,
            training_cfg=self.training_cfg,
            reward_cfg=self.reward_cfg,
            world_cfg=self.world_cfg,
        )

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
        self.paused = bool(runtime_cfg.get("paused", False))
        self.speed = float(runtime_cfg.get("speed", 1.0))
        self.timestep_limit = int(self.training_cfg.get("timesteps", 0))
        self.last_checkpoint_step = 0
        self.events = deque(maxlen=32)
        self.metrics_recent = deque(maxlen=600)
        self.episode_history: list[dict] = []
        self.train_history: list[dict] = []
        self.state_status = "initializing"
        self.started_at = utc_now_iso()
        self.last_state_write = 0.0
        self.last_preview_time = 0.0
        self.preview_interval = 1.0 / (12.0 if self.scenario.dimension == "3D" else 15.0)
        self.final_model_saved = False
        self.last_checkpoint_path = None
        self.last_engine_export_manifest_path = None
        self.last_engine_export_validation_path = None
        self.last_engine_export_status = "pending"
        self._started_ts = time.time()

        self.model_dir = self.archive.models_dir

        self.vec_env = None
        self.model = None
        self.render_env = None
        self.render_obs = None
        self.pilot_brain: PilotBrain | None = None
        self.device_name = "unknown"
        self.mlflow_run = None

        self._setup_mlflow()
        self._setup_envs_and_model()
        self._broadcast_runtime_config()
        self._apply_pilot_selection()
        self.archive.update_manifest(
            {
                "device": self.device_name,
                "mlflow_run_id": self.mlflow_run.info.run_id if self.mlflow_run else None,
            }
        )
        self.append_event(f"Runner started for {self.scenario.label}.")

    def _setup_mlflow(self) -> None:
        tracking_uri = self.control.get("tracking_uri", self.scenario.tracking_uri)
        experiment_name = self.control.get("experiment_name", self.scenario.experiment_name)
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.mlflow_run = mlflow.start_run(run_name=self.run_id)
        params = {
            "scenario_id": self.scenario.scenario_id,
            "dimension": self.scenario.dimension,
            "viewport": self.scenario.viewport,
            "run_id": self.run_id,
        }
        for prefix, values in (
            ("train", self.training_cfg),
            ("reward", self.reward_cfg),
            ("world", self.world_cfg),
        ):
            for key, value in values.items():
                if value is None:
                    continue
                params[f"{prefix}.{key}"] = value
        mlflow.log_params(params)

    def _make_env(self, render_mode: str, *, seed: int | None):
        env = self.scenario.make_env(
            render_mode=render_mode,
            reward_config=self.reward_cfg,
            world_config=self.world_cfg,
            seed=seed,
        )
        return Monitor(env)

    def _infer_vecnorm_path(self, model_path: str | None, explicit_path: str | None) -> str | None:
        if explicit_path and os.path.exists(explicit_path):
            return explicit_path
        if model_path:
            sibling = Path(model_path).with_name("vec_normalize.pkl")
            if sibling.exists():
                return str(sibling.resolve())
        return None

    def _setup_envs_and_model(self) -> None:
        n_envs = int(self.training_cfg["n_envs"])
        seed = self.training_cfg.get("seed")
        env_fns = []
        for offset in range(n_envs):
            env_seed = None if seed is None else int(seed) + offset
            env_fns.append(lambda env_seed=env_seed: self._make_env("direct", seed=env_seed))

        base_vec_env = DummyVecEnv(env_fns)
        load_model_path = self.training_cfg.get("model_path")
        load_vecnorm_path = self._infer_vecnorm_path(load_model_path, self.training_cfg.get("vecnorm_path"))

        if load_vecnorm_path:
            self.vec_env = VecNormalize.load(load_vecnorm_path, base_vec_env)
            self.append_event(f"Loaded VecNormalize from {load_vecnorm_path}.")
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
        self.device_name = str(next(self.model.policy.parameters()).device)
        mlflow.log_param("device", self.device_name)

        render_mode = "rgb_array" if self.scenario.dimension == "2D" else "human"
        self.render_env = self.scenario.make_env(
            render_mode=render_mode,
            reward_config=self.reward_cfg,
            world_config=self.world_cfg,
            seed=seed,
        )
        self.render_obs, _ = self.render_env.reset()

    def _iter_base_envs(self) -> Iterable[object]:
        if self.vec_env is not None:
            base_vec = getattr(self.vec_env, "venv", self.vec_env)
            for env in getattr(base_vec, "envs", []):
                base = env
                while hasattr(base, "env"):
                    base = base.env
                yield base
        if self.render_env is not None:
            base = self.render_env
            while hasattr(base, "env"):
                base = base.env
            yield base

    def _broadcast_runtime_config(self) -> None:
        for env in self._iter_base_envs():
            if hasattr(env, "set_runtime_config"):
                env.set_runtime_config(reward_config=self.reward_cfg, world_config=self.world_cfg)

    def sync_control(self) -> None:
        if not self.paths.control.exists():
            return

        current_mtime = self.paths.control.stat().st_mtime
        if current_mtime <= self.control_mtime:
            return

        self.control_mtime = current_mtime
        self.control = load_json(self.paths.control, default=self.control)
        self.archive.snapshot_control(self.control, reason="control_update")

        runtime_cfg = self.control.get("runtime", {})
        self.paused = bool(runtime_cfg.get("paused", False))
        self.speed = max(0.1, float(runtime_cfg.get("speed", 1.0)))
        self.running = not bool(runtime_cfg.get("stop_requested", False))

        requested_limit = int(self.control.get("training", {}).get("timesteps", self.timestep_limit))
        self.timestep_limit = requested_limit

        reward_meta = self.control.get("reward", {})
        world_meta = self.control.get("world", {})
        reward_version = int(reward_meta.get("version", self.reward_version))
        world_version = int(world_meta.get("version", self.world_version))
        if reward_version != self.reward_version:
            self.reward_cfg.clear()
            self.reward_cfg.update(self.scenario.default_reward)
            self.reward_cfg.update(reward_meta.get("values", {}))
            self.reward_version = reward_version
            self._broadcast_runtime_config()
            self.append_event("Reward shaping updated live.")

        if world_version != self.world_version:
            self.world_cfg.clear()
            self.world_cfg.update(self.scenario.default_world)
            self.world_cfg.update(world_meta.get("values", {}))
            self.world_version = world_version
            self._broadcast_runtime_config()
            self.append_event("World and physics settings updated live.")

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
        vecnorm_path = self._infer_vecnorm_path(checkpoint_path, pilot_cfg.get("vecnorm_path") or None)
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
            self.metrics_recent.append(point)
            self.episode_history.append(point)
            self.archive.log_episode_metric(point)
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
        self.metrics_recent.append(point)
        self.train_history.append(point)
        self.archive.log_train_metric(point)

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
        self.last_checkpoint_path = str(model_path.resolve()) + ".zip"
        self.append_event(f"Checkpoint saved to {model_path}.zip")

    def save_final_model(self) -> None:
        final_path = self.model_dir / f"{self.scenario.model_prefix}_final"
        vecnorm_path = self.model_dir / "vec_normalize.pkl"
        self.model.save(str(final_path))
        self.vec_env.save(str(vecnorm_path))
        self.final_model_saved = True
        self.last_checkpoint_path = str(final_path.resolve()) + ".zip"
        self.append_event(f"Final model saved to {final_path}.zip")
        self.export_engine_bundle(
            model_path=str(final_path.resolve()) + ".zip",
            vecnorm_path=str(vecnorm_path.resolve()),
        )

    def export_engine_bundle(self, *, model_path: str, vecnorm_path: str | None) -> None:
        try:
            manifest = export_engine_package(
                scenario_id=self.scenario.scenario_id,
                model_path=model_path,
                output_dir=self.archive.exports_dir,
                vecnorm_path=vecnorm_path,
                run_id=self.run_id,
                reward_cfg=self.reward_cfg,
                world_cfg=self.world_cfg,
                seed=self.training_cfg.get("seed"),
            )
            self.last_engine_export_manifest_path = str((self.archive.exports_dir / "manifest.json").resolve())
            self.last_engine_export_validation_path = str((self.archive.exports_dir / "validation.json").resolve())
            self.last_engine_export_status = "ready"
            self.archive.update_manifest(
                {
                    "engine_export": {
                        "path": str(self.archive.exports_dir.resolve()),
                        "manifest_path": self.last_engine_export_manifest_path,
                        "validation_path": self.last_engine_export_validation_path,
                        "contract": manifest.get("contract", {}),
                        "compatibility": manifest.get("compatibility", {}),
                    }
                }
            )
            self.append_event(f"Engine export bundle saved to {self.archive.exports_dir}.")
        except Exception as exc:
            self.last_engine_export_status = f"error: {exc}"
            self.append_event(f"Engine export warning: {exc}")

    def render_preview(self) -> None:
        if self.pilot_brain is None or self.render_env is None:
            return
        if self.control.get("pilot", {}).get("name") == "disabled":
            return
        if time.time() - self.last_preview_time < self.preview_interval:
            return

        self.last_preview_time = time.time()

        try:
            action = self.pilot_brain.predict(self.render_obs, deterministic=True)
            self.render_obs, _, terminated, truncated, _ = self.render_env.step(action)
            if terminated or truncated:
                self.render_obs, _ = self.render_env.reset()

            if self.scenario.dimension == "2D":
                frame = self.render_env.render()
                np.save(self.paths.frame, frame)
                self.archive.write_frame(frame)
        except RuntimeError as exc:
            if "disabled" not in str(exc).lower():
                self.append_event(f"Viewport preview warning: {exc}")
        except Exception as exc:
            self.append_event(f"Viewport preview warning: {exc}")

    def append_event(self, message: str) -> None:
        stamped = f"[{utc_now_iso()}] {message}"
        self.events.appendleft(stamped)
        self.archive.log_event(message)

    def _state_payload(self) -> dict:
        wall_seconds = max(0.0, time.time() - self._started_ts)
        return {
            "run_id": self.run_id,
            "scenario_id": self.scenario.scenario_id,
            "scenario_label": self.scenario.label,
            "scenario_info": self.scenario.info,
            "dimension": self.scenario.dimension,
            "viewport": self.scenario.viewport,
            "status": self.state_status,
            "started_at": self.started_at,
            "updated_at": utc_now_iso(),
            "pid": os.getpid(),
            "mlflow_run_id": self.mlflow_run.info.run_id if self.mlflow_run else None,
            "step": self.current_step,
            "timestep_limit": self.timestep_limit,
            "episodes": self.episode_count,
            "last_reward": self.last_reward,
            "best_reward": None if self.best_reward == float("-inf") else self.best_reward,
            "last_episode_length": self.last_episode_length,
            "speed": self.speed,
            "paused": self.paused,
            "reward_version_applied": self.reward_version,
            "world_version_applied": self.world_version,
            "pilot_version_applied": self.pilot_version,
            "pilot_name": self.control.get("pilot", {}).get("name", "live_policy"),
            "training": self.training_cfg,
            "reward": self.reward_cfg,
            "world": self.world_cfg,
            "loss": self.last_loss,
            "value_loss": self.last_value_loss,
            "policy_loss": self.last_policy_loss,
            "entropy_loss": self.last_entropy_loss,
            "approx_kl": self.last_approx_kl,
            "device": self.device_name,
            "wall_time_seconds": wall_seconds,
            "events": list(self.events),
            "artifacts": {
                "run_dir": str(self.archive.run_dir.resolve()),
                "model_dir": str(self.model_dir.resolve()),
                "charts_dir": str(self.archive.charts_dir.resolve()),
                "frame_path": str(self.paths.frame.resolve()),
                "metrics_path": str(self.paths.metrics.resolve()),
                "latest_checkpoint": self.last_checkpoint_path,
                "engine_export_dir": str(self.archive.exports_dir.resolve()),
                "engine_export_manifest": self.last_engine_export_manifest_path,
                "engine_export_validation": self.last_engine_export_validation_path,
                "engine_export_status": self.last_engine_export_status,
            },
        }

    def write_state(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last_state_write < 0.5:
            return

        self.last_state_write = now
        if self.state_status not in {"stopped", "error"}:
            self.state_status = "paused" if self.paused else "running"
            if not self.running:
                self.state_status = "stopping"

        state_payload = self._state_payload()
        write_json(self.paths.state, state_payload)
        write_json(self.paths.metrics, {"history": list(self.metrics_recent)})
        self.archive.write_state(state_payload)
        self.archive.write_metrics_latest(list(self.metrics_recent))
        self.archive.export_charts(self.episode_history, self.train_history)

    def _write_summary(self) -> None:
        summary = {
            "run_id": self.run_id,
            "status": self.state_status,
            "started_at": self.started_at,
            "finished_at": utc_now_iso(),
            "scenario_id": self.scenario.scenario_id,
            "steps": self.current_step,
            "episodes": self.episode_count,
            "best_reward": None if self.best_reward == float("-inf") else self.best_reward,
            "last_reward": self.last_reward,
            "last_episode_length": self.last_episode_length,
            "loss": self.last_loss,
            "value_loss": self.last_value_loss,
            "policy_loss": self.last_policy_loss,
            "approx_kl": self.last_approx_kl,
            "device": self.device_name,
            "training": self.training_cfg,
            "reward": self.reward_cfg,
            "world": self.world_cfg,
            "paths": {
                "run_dir": str(self.archive.run_dir.resolve()),
                "models_dir": str(self.model_dir.resolve()),
                "episode_history": str(self.archive.episode_history_path.resolve()),
                "train_history": str(self.archive.train_history_path.resolve()),
                "state_history": str(self.archive.state_history_path.resolve()),
                "charts_dir": str(self.archive.charts_dir.resolve()),
                "engine_export_dir": str(self.archive.exports_dir.resolve()),
                "engine_export_manifest": self.last_engine_export_manifest_path,
                "engine_export_validation": self.last_engine_export_validation_path,
            },
            "engine_export_status": self.last_engine_export_status,
        }
        self.archive.write_summary(summary)
        self.archive.export_charts(self.episode_history, self.train_history, force=True)

    def close(self) -> None:
        try:
            self.running = False
            if self.model is not None and not self.final_model_saved:
                self.save_final_model()
            self.state_status = "stopped"
            self.write_state(force=True)
            self._write_summary()
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
