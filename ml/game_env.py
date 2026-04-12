from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

OBS_SIZE = 15
ACTION_SIZE = 8


@dataclass
class EnvConfig:
    episode_seconds: float = 45.0
    dt: float = 1.0 / 30.0
    arena_radius: float = 32.0
    ground_height: float = 1.2
    capture_distance: float = 1.4

    agent_speed: float = 7.5
    agent_accel: float = 8.5
    gravity: float = 18.0
    jump_velocity: float = 6.2
    max_fall_speed: float = 32.0

    pursuer_accel: float = 22.0
    pursuer_drag: float = 0.22
    pursuer_max_speed: float = 10.0
    pursuer_hover_height: float = 1.0
    pursuer_vertical_stabilizer: float = 12.0


class SurvivalTrainingEnv:
    def __init__(self, episode_seconds: float = 45.0, seed: int = 7) -> None:
        self.config = EnvConfig(episode_seconds=episode_seconds)
        self.rng = np.random.default_rng(seed)

        self.time = 0.0
        self.agent_pos = np.zeros(3, dtype=np.float32)
        self.agent_vel = np.zeros(3, dtype=np.float32)
        self.agent_forward = np.array([0.0, 1.0], dtype=np.float32)
        self.agent_on_floor = True

        self.pursuer_pos = np.zeros(3, dtype=np.float32)
        self.pursuer_vel = np.zeros(3, dtype=np.float32)

        self.reset()

    def reset(self) -> np.ndarray:
        cfg = self.config
        self.time = 0.0

        self.agent_pos = np.array([0.0, cfg.ground_height, 0.0], dtype=np.float32)
        self.agent_vel.fill(0.0)
        self.agent_forward = np.array([0.0, 1.0], dtype=np.float32)
        self.agent_on_floor = True

        angle = self.rng.uniform(0.0, 2.0 * math.pi)
        radius = self.rng.uniform(9.0, 14.0)
        self.pursuer_pos = np.array(
            [math.cos(angle) * radius, cfg.ground_height, math.sin(angle) * radius],
            dtype=np.float32,
        )
        self.pursuer_vel.fill(0.0)

        return self._observation()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        cfg = self.config
        action = np.asarray(action, dtype=np.float32)
        if action.shape[0] < ACTION_SIZE:
            padded = np.zeros(ACTION_SIZE, dtype=np.float32)
            padded[: action.shape[0]] = action
            action = padded
        action = np.clip(action, -1.0, 1.0)

        move_input = action[:2]
        desired_planar = move_input * cfg.agent_speed
        planar_vel = self.agent_vel[[0, 2]]
        planar_delta = desired_planar - planar_vel
        planar_vel = planar_vel + planar_delta * (cfg.agent_accel * cfg.dt)

        self.agent_vel[0] = planar_vel[0]
        self.agent_vel[2] = planar_vel[1]

        if self.agent_on_floor and action[2] > 0.35:
            self.agent_vel[1] = cfg.jump_velocity
            self.agent_on_floor = False

        self.agent_vel[1] -= cfg.gravity * cfg.dt
        self.agent_vel[1] = max(self.agent_vel[1], -cfg.max_fall_speed)

        self.agent_pos += self.agent_vel * cfg.dt

        if self.agent_pos[1] <= cfg.ground_height:
            self.agent_pos[1] = cfg.ground_height
            self.agent_vel[1] = 0.0
            self.agent_on_floor = True

        self._clamp_inside_arena(is_agent=True)

        planar_speed = float(np.linalg.norm(self.agent_vel[[0, 2]]))
        if planar_speed > 0.08:
            self.agent_forward = self.agent_vel[[0, 2]] / planar_speed

        target = self.agent_pos + np.array([0.0, cfg.pursuer_hover_height, 0.0], dtype=np.float32)
        chase = target - self.pursuer_pos
        chase_dist = float(np.linalg.norm(chase))

        if chase_dist > 1e-5:
            chase_dir = chase / chase_dist
            self.pursuer_vel += chase_dir * cfg.pursuer_accel * cfg.dt

        self.pursuer_vel *= (1.0 - cfg.pursuer_drag * cfg.dt)

        pursuer_planar_speed = float(np.linalg.norm(self.pursuer_vel[[0, 2]]))
        if pursuer_planar_speed > cfg.pursuer_max_speed:
            scale = cfg.pursuer_max_speed / pursuer_planar_speed
            self.pursuer_vel[0] *= scale
            self.pursuer_vel[2] *= scale

        y_error = target[1] - self.pursuer_pos[1]
        self.pursuer_vel[1] += y_error * cfg.pursuer_vertical_stabilizer * cfg.dt

        self.pursuer_pos += self.pursuer_vel * cfg.dt
        self._clamp_inside_arena(is_agent=False)

        self.time += cfg.dt

        distance = float(np.linalg.norm(self.pursuer_pos - self.agent_pos))
        captured = distance <= cfg.capture_distance
        timeout = self.time >= cfg.episode_seconds
        done = captured or timeout

        reward = cfg.dt
        reward += min(distance, 20.0) * 0.012
        reward -= float(np.mean(np.square(action[3:]))) * 0.01

        if captured:
            reward -= 5.0

        info = {
            "captured": captured,
            "timeout": timeout,
            "distance": distance,
            "time": self.time,
        }

        return self._observation(), float(reward), done, info

    def _observation(self) -> np.ndarray:
        cfg = self.config
        delta = self.pursuer_pos - self.agent_pos
        distance = float(np.linalg.norm(delta))

        threat_planar = np.array([delta[0], delta[2]], dtype=np.float32)
        threat_len = float(np.linalg.norm(threat_planar))
        angle_to_threat = 0.0

        if threat_len > 1e-6:
            threat_dir = threat_planar / threat_len
            dot = float(np.clip(np.dot(self.agent_forward, threat_dir), -1.0, 1.0))
            cross = float(self.agent_forward[0] * threat_dir[1] - self.agent_forward[1] * threat_dir[0])
            angle_to_threat = float(np.arctan2(cross, dot))

        observation = np.array(
            [
                delta[0],
                delta[1],
                delta[2],
                self.agent_vel[0],
                self.agent_vel[1],
                self.agent_vel[2],
                self.pursuer_vel[0],
                self.pursuer_vel[1],
                self.pursuer_vel[2],
                distance,
                1.0 if self.agent_on_floor else 0.0,
                np.clip(self.time / max(cfg.episode_seconds, 1e-6), 0.0, 1.0),
                angle_to_threat,
                np.linalg.norm(self.agent_vel[[0, 2]]),
                np.linalg.norm(self.pursuer_vel[[0, 2]]),
            ],
            dtype=np.float32,
        )

        return observation

    def _clamp_inside_arena(self, is_agent: bool) -> None:
        cfg = self.config
        pos = self.agent_pos if is_agent else self.pursuer_pos
        vel = self.agent_vel if is_agent else self.pursuer_vel

        planar = np.array([pos[0], pos[2]], dtype=np.float32)
        radius = float(np.linalg.norm(planar))
        if radius <= cfg.arena_radius:
            return

        planar = (planar / max(radius, 1e-6)) * cfg.arena_radius
        pos[0] = planar[0]
        pos[2] = planar[1]
        vel[0] *= -0.35
        vel[2] *= -0.35
