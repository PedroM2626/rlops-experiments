"""Viewport brain registry used by the engine runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from gymnasium import Env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


@dataclass
class BrainSelection:
    """Serializable pilot-brain selection."""

    name: str
    checkpoint_path: str | None = None
    vecnorm_path: str | None = None


class ObservationShellEnv(Env):
    """Minimal env used only to load VecNormalize statistics for checkpoints."""

    metadata = {}

    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self, *, seed=None, options=None):
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, 0.0, False, False, {}


class PilotBrain:
    """Policy-like object used for live viewport control."""

    label = "pilot"

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        return None


class DisabledBrain(PilotBrain):
    """Preview-off brain used when the viewport should stay idle."""

    label = "disabled"

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        raise RuntimeError("Viewport preview is disabled.")


class LivePolicyBrain(PilotBrain):
    """Proxy brain that mirrors the actively trained PPO model."""

    label = "live_policy"

    def __init__(self, model_getter: Callable[[], PPO], vecnorm_getter: Callable[[], VecNormalize | None]):
        self._model_getter = model_getter
        self._vecnorm_getter = vecnorm_getter

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        model = self._model_getter()
        vecnorm = self._vecnorm_getter()
        if vecnorm is not None:
            batched = np.expand_dims(np.asarray(obs, dtype=np.float32), axis=0)
            obs = vecnorm.normalize_obs(batched)[0]
        action, _ = model.predict(obs, deterministic=deterministic)
        return np.asarray(action, dtype=np.float32)


class RandomBrain(PilotBrain):
    """Simple baseline pilot for quick comparisons."""

    label = "random"

    def __init__(self, action_space):
        self._action_space = action_space

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        return np.asarray(self._action_space.sample(), dtype=np.float32)


class CheckpointBrain(PilotBrain):
    """Loads a separate PPO checkpoint for the live viewport."""

    label = "checkpoint_ppo"

    def __init__(self, selection: BrainSelection, observation_space, action_space):
        if not selection.checkpoint_path:
            raise ValueError("checkpoint_path is required for checkpoint_ppo")

        self._model = PPO.load(selection.checkpoint_path, device="cpu")
        self._vecnorm = None

        if selection.vecnorm_path:
            shell_env = DummyVecEnv([lambda: ObservationShellEnv(observation_space, action_space)])
            self._vecnorm = VecNormalize.load(selection.vecnorm_path, shell_env)
            self._vecnorm.training = False
            self._vecnorm.norm_reward = False

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        norm_obs = np.asarray(obs, dtype=np.float32)
        if self._vecnorm is not None:
            norm_obs = self._vecnorm.normalize_obs(np.expand_dims(norm_obs, axis=0))[0]
        action, _ = self._model.predict(norm_obs, deterministic=deterministic)
        return np.asarray(action, dtype=np.float32)

    def close(self) -> None:
        self._vecnorm = None


def build_pilot_brain(
    selection: BrainSelection,
    *,
    action_space,
    observation_space,
    model_getter: Callable[[], PPO],
    vecnorm_getter: Callable[[], VecNormalize | None],
) -> PilotBrain:
    """Create the pilot brain requested by the dashboard."""
    if selection.name == "live_policy":
        return LivePolicyBrain(model_getter, vecnorm_getter)
    if selection.name == "disabled":
        return DisabledBrain()
    if selection.name == "random":
        return RandomBrain(action_space)
    if selection.name == "checkpoint_ppo":
        return CheckpointBrain(selection, observation_space, action_space)
    raise ValueError(f"Unsupported pilot brain: {selection.name}")
