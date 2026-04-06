"""Scenario registry for the reusable ML games engine."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ml_games_engine.envs.arena2d_env import Arena2DEnv


ROOT_DIR = Path(__file__).resolve().parents[1]


def _load_module(name: str, file_path: Path, extra_sys_path: Path | None = None):
    if extra_sys_path is not None:
        extra = str(extra_sys_path)
        if extra not in sys.path:
            sys.path.insert(0, extra)

    spec = importlib.util.spec_from_file_location(name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class ScenarioSpec:
    """Static description of an experiment scenario."""

    scenario_id: str
    label: str
    dimension: str
    description: str
    viewport: str
    model_prefix: str
    tracking_uri: str
    experiment_name: str
    default_training: dict
    default_reward: dict
    make_env: Callable[..., object]


def _make_chase_env(*, render_mode: str, reward_config: dict, seed: int | None):
    project_dir = ROOT_DIR / "Chase"
    module = _load_module(
        "ml_games_engine.chase_env_module",
        project_dir / "src" / "env" / "chase_env.py",
        extra_sys_path=project_dir,
    )
    env_seed = 42 if seed is None else seed
    return module.ChaseEnv(render_mode=render_mode, seed=env_seed, reward_config=reward_config)


def _make_parkour_env(*, render_mode: str, reward_config: dict, seed: int | None):
    project_dir = ROOT_DIR / "Parkour-Obstacles"
    module = _load_module(
        "ml_games_engine.parkour_env_module",
        project_dir / "src" / "env" / "parkour_env.py",
        extra_sys_path=project_dir,
    )
    return module.ParkourEnv(render_mode=render_mode, level_seed=seed, reward_config=reward_config)


def _make_arena2d_env(*, render_mode: str, reward_config: dict, seed: int | None):
    return Arena2DEnv(render_mode=render_mode, reward_config=reward_config)


SCENARIOS = {
    "arena2d": ScenarioSpec(
        scenario_id="arena2d",
        label="Arena 2D",
        dimension="2D",
        description="Sandbox rapido para testar PPO, shaping de recompensa e dashboards.",
        viewport="streamlit",
        model_prefix="ppo_arena2d",
        tracking_uri="mlruns",
        experiment_name="arena2d_ppo",
        default_training={
            "timesteps": 50_000,
            "n_envs": 4,
            "lr": 3e-4,
            "n_steps": 512,
            "batch_size": 64,
            "gamma": 0.99,
            "n_epochs": 10,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "net_arch": "128,128",
            "checkpoint_freq": 5_000,
            "seed": 7,
        },
        default_reward={
            "step_penalty": -0.01,
            "alive_bonus": 0.02,
            "goal_reward": 35.0,
            "catch_penalty": -30.0,
            "progress_scale": 3.5,
            "speed_scale": 0.15,
            "energy_scale": 0.02,
            "boundary_penalty": 0.2,
            "chaser_speed": 0.55,
        },
        make_env=_make_arena2d_env,
    ),
    "chase": ScenarioSpec(
        scenario_id="chase",
        label="Chase 3D",
        dimension="3D",
        description="Humanoide foge de um perseguidor em PyBullet com viewport nativo.",
        viewport="pybullet",
        model_prefix="ppo_chase",
        tracking_uri="mlruns",
        experiment_name="chase_ppo_engine",
        default_training={
            "timesteps": 250_000,
            "n_envs": 4,
            "lr": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "gamma": 0.99,
            "n_epochs": 10,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "net_arch": "256,256",
            "checkpoint_freq": 25_000,
            "seed": 42,
        },
        default_reward={
            "survival_bonus": 0.01,
            "close_penalty_scale": 0.5,
            "close_penalty_distance": 3.0,
            "progress_scale": 2.0,
            "speed_scale": 0.2,
            "upright_scale": 0.3,
            "energy_scale": 0.001,
            "caught_penalty": 50.0,
        },
        make_env=_make_chase_env,
    ),
    "parkour": ScenarioSpec(
        scenario_id="parkour",
        label="Parkour 3D",
        dimension="3D",
        description="Humanoide navega plataformas e obstaculos com shaping em tempo real.",
        viewport="pybullet",
        model_prefix="ppo_parkour",
        tracking_uri="mlruns",
        experiment_name="parkour_ppo_engine",
        default_training={
            "timesteps": 250_000,
            "n_envs": 4,
            "lr": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "gamma": 0.99,
            "n_epochs": 10,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "net_arch": "256,256",
            "checkpoint_freq": 25_000,
            "seed": 123,
        },
        default_reward={
            "step_penalty": -0.02,
            "progress_scale": 5.0,
            "speed_scale": 0.3,
            "upright_scale": 0.4,
            "lateral_vel_scale": 0.5,
            "lateral_pos_scale": 0.3,
            "energy_scale": 0.001,
            "goal_reward": 150.0,
            "fall_penalty": -5.0,
        },
        make_env=_make_parkour_env,
    ),
}


def get_scenario(scenario_id: str) -> ScenarioSpec:
    """Return a scenario or raise a readable error."""
    try:
        return SCENARIOS[scenario_id]
    except KeyError as exc:
        raise KeyError(f"Unknown scenario: {scenario_id}") from exc
