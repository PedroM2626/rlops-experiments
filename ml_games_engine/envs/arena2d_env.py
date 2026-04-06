"""Lightweight 2D environment for fast experimentation and engine smoke tests."""

from __future__ import annotations

from typing import Dict, Tuple

import gymnasium as gym
import numpy as np


class Arena2DEnv(gym.Env):
    """Top-down 2D chase-and-goal sandbox with a NumPy renderer."""

    metadata = {"render_modes": ["rgb_array", "direct"], "render_fps": 30}

    WORLD_SIZE = 20.0
    MAX_SPEED = 1.25
    MAX_STEPS = 600
    DT = 0.18
    GOAL_RADIUS = 1.1
    CATCH_RADIUS = 0.9

    STEP_PENALTY = -0.01
    ALIVE_BONUS = 0.02
    GOAL_REWARD = 35.0
    CATCH_PENALTY = -30.0
    PROGRESS_SCALE = 3.5
    SPEED_SCALE = 0.15
    ENERGY_SCALE = 0.02
    BOUNDARY_PENALTY = 0.2
    CHASER_SPEED = 0.55

    def __init__(
        self,
        render_mode: str = "direct",
        reward_config: Dict[str, float] | None = None,
        world_config: Dict[str, float] | None = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.reward_config = reward_config if reward_config is not None else {}
        self.world_config = world_config if world_config is not None else {}
        self.world_size = float(self.world_config.get("world_size", self.WORLD_SIZE))
        self.max_speed = self.MAX_SPEED
        self.max_steps = int(self.world_config.get("max_steps", self.MAX_STEPS))
        self.dt = float(self.world_config.get("dt", self.DT))

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(10,),
            dtype=np.float32,
        )

        self._agent_pos = np.zeros(2, dtype=np.float32)
        self._agent_vel = np.zeros(2, dtype=np.float32)
        self._goal_pos = np.zeros(2, dtype=np.float32)
        self._chaser_pos = np.zeros(2, dtype=np.float32)
        self._step_count = 0
        self._prev_goal_dist = 0.0

    def set_runtime_config(
        self,
        reward_config: Dict[str, float] | None = None,
        world_config: Dict[str, float] | None = None,
    ) -> bool:
        if reward_config is not None:
            self.reward_config = reward_config
        if world_config is not None:
            self.world_config = world_config
            self.world_size = float(self.world_config.get("world_size", self.WORLD_SIZE))
            self.max_steps = int(self.world_config.get("max_steps", self.MAX_STEPS))
            self.dt = float(self.world_config.get("dt", self.DT))
        return True

    def reset(self, seed: int | None = None, options: dict | None = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        rng = self.np_random

        self._agent_pos = np.array([-6.0, 0.0], dtype=np.float32)
        self._agent_vel = np.zeros(2, dtype=np.float32)
        self._goal_pos = np.array([7.5, rng.uniform(-5.0, 5.0)], dtype=np.float32)
        self._chaser_pos = np.array([-8.0, rng.uniform(-6.0, 6.0)], dtype=np.float32)
        self._step_count = 0
        self._prev_goal_dist = float(np.linalg.norm(self._goal_pos - self._agent_pos))
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self._step_count += 1
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        accel = action * 0.32
        self._agent_vel = np.clip(self._agent_vel + accel, -self.max_speed, self.max_speed)
        self._agent_pos = self._agent_pos + self._agent_vel * self.dt

        clipped = np.clip(self._agent_pos, -self.world_size / 2, self.world_size / 2)
        boundary_hits = int(not np.allclose(clipped, self._agent_pos))
        self._agent_pos = clipped.astype(np.float32)

        chase_vec = self._agent_pos - self._chaser_pos
        chase_dist = float(np.linalg.norm(chase_vec))
        if chase_dist > 1e-6:
            chase_dir = chase_vec / chase_dist
        else:
            chase_dir = np.zeros(2, dtype=np.float32)
        self._chaser_pos = self._chaser_pos + chase_dir * self._reward("chaser_speed", self.CHASER_SPEED)

        goal_dist = float(np.linalg.norm(self._goal_pos - self._agent_pos))
        reward = self._reward("step_penalty", self.STEP_PENALTY)
        reward += self._reward("alive_bonus", self.ALIVE_BONUS)

        delta_goal = self._prev_goal_dist - goal_dist
        reward += delta_goal * self._reward("progress_scale", self.PROGRESS_SCALE)
        if delta_goal > 0.0:
            reward += float(np.linalg.norm(self._agent_vel)) * self._reward("speed_scale", self.SPEED_SCALE)

        reward -= float(np.sum(action ** 2)) * self._reward("energy_scale", self.ENERGY_SCALE)
        reward -= boundary_hits * self._reward("boundary_penalty", self.BOUNDARY_PENALTY)

        terminated = False
        truncated = self._step_count >= self.max_steps
        self._prev_goal_dist = goal_dist

        if goal_dist <= self.GOAL_RADIUS:
            reward += self._reward("goal_reward", self.GOAL_REWARD)
            terminated = True

        if float(np.linalg.norm(self._agent_pos - self._chaser_pos)) <= self.CATCH_RADIUS:
            reward += self._reward("catch_penalty", self.CATCH_PENALTY)
            terminated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        return self._draw_frame()

    def close(self):
        return None

    def _reward(self, key: str, fallback: float) -> float:
        return float(self.reward_config.get(key, fallback))

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        return np.clip(vec / (self.world_size / 2), -1.0, 1.0)

    def _get_obs(self) -> np.ndarray:
        goal_delta = self._goal_pos - self._agent_pos
        chaser_delta = self._chaser_pos - self._agent_pos
        time_left = 1.0 - (self._step_count / self.max_steps)
        dist_norm = np.clip(np.linalg.norm(goal_delta) / self.world_size, 0.0, 1.0)

        return np.concatenate(
            [
                self._normalize(self._agent_pos),
                np.clip(self._agent_vel / self.max_speed, -1.0, 1.0),
                self._normalize(goal_delta),
                self._normalize(chaser_delta),
                np.array([time_left, dist_norm], dtype=np.float32),
            ]
        ).astype(np.float32)

    def _draw_frame(self) -> np.ndarray:
        frame = np.zeros((320, 320, 3), dtype=np.uint8)
        frame[:, :] = np.array([14, 20, 30], dtype=np.uint8)

        grid_color = np.array([26, 34, 48], dtype=np.uint8)
        for idx in range(0, frame.shape[0], 32):
            frame[idx : idx + 1, :, :] = grid_color
            frame[:, idx : idx + 1, :] = grid_color

        self._draw_entity(frame, self._goal_pos, radius=10, color=np.array([76, 175, 80], dtype=np.uint8))
        self._draw_entity(frame, self._chaser_pos, radius=10, color=np.array([239, 83, 80], dtype=np.uint8))
        self._draw_entity(frame, self._agent_pos, radius=10, color=np.array([66, 165, 245], dtype=np.uint8))
        return frame

    def _draw_entity(self, frame: np.ndarray, pos: np.ndarray, radius: int, color: np.ndarray) -> None:
        half = self.world_size / 2
        x = int(((float(pos[0]) + half) / self.world_size) * (frame.shape[1] - 1))
        y = int(((half - float(pos[1])) / self.world_size) * (frame.shape[0] - 1))

        yy, xx = np.ogrid[: frame.shape[0], : frame.shape[1]]
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
        frame[mask] = color
