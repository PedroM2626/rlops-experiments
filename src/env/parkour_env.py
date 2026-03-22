"""
Parkour Gymnasium environment backed by PyBullet physics.

Coordinate convention: PyBullet Z-up
  X = forward (direction of travel)
  Y = lateral (left/right)
  Z = up

Observation space (20-dim):
  [0:3]   agent position (x, y, z)
  [3:6]   agent linear velocity
  [6:9]   orientation euler (roll, pitch, yaw)
  [9:12]  angular velocity
  [12:17] ground-probe raycasts downward-forward (5 rays, hit fraction)
  [17:20] unit vector toward goal

Action space (3-dim continuous, clipped [-1, 1]):
  [0]  forward force (+X)
  [1]  lateral force (+Y)
  [2]  jump impulse  (+Z, applied only when grounded)
"""

import math
import os
from typing import Optional

import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces

from src.env.level_generator import LevelGenerator


class ParkourEnv(gym.Env):
    metadata = {"render_modes": ["human", "direct"], "render_fps": 60}

    # Physics
    GRAVITY    = -9.81
    TIME_STEP  = 1.0 / 240.0
    FRAME_SKIP = 4
    PHYSICS_HZ = 240.0 / FRAME_SKIP

    # Agent
    AGENT_MASS   = 5.0
    AGENT_RADIUS = 0.3    # capsule radius (m)
    AGENT_HEIGHT = 0.8    # cylindrical section height (m)
    # total capsule height = AGENT_HEIGHT + 2*AGENT_RADIUS = 1.4 m

    # Forces (Aggressive for speed)
    MAX_FORCE       = 1500.0
    JUMP_FORCE_PEAK = 8000.0

    # Episode
    MAX_STEPS = 2000

    # Reward (More punitive for time)
    GOAL_REWARD    = 150.0
    FALL_PENALTY   = -10.0
    STEP_PENALTY   = -0.02
    PROGRESS_SCALE = 5.0

    # Fall threshold: Z below this => fallen off
    FALL_Z = -2.0

    # Raycasts
    RAY_LENGTH  = 6.0
    RAY_PITCHES = [-60, -75, -90, -75, -60]  # degrees from horizontal (negative = down)
    RAY_YAWS    = [-20, -10,   0,  10,  20]  # lateral spread

    def __init__(self, render_mode: str = "direct", level_seed: Optional[int] = None):
        super().__init__()
        self.render_mode = render_mode
        self.level_seed  = level_seed

        # Observation / action spaces
        low  = np.full(20, -np.inf, dtype=np.float32)
        high = np.full(20,  np.inf, dtype=np.float32)
        # Ray fractions are always 0..1
        low[12:17]  = 0.0
        high[12:17] = 1.0
        # Goal direction components -1..1
        low[17:20]  = -1.0
        high[17:20] =  1.0
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space      = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Internal state
        self._client    = None
        self._agent_id  = None
        self._goal_pos  = np.zeros(3, dtype=np.float32)
        self._start_pos = np.zeros(3, dtype=np.float32)
        self._step_count         = 0
        self._prev_dist_to_goal  = 0.0

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self._client is None:
            self._init_physics()
        else:
            p.resetSimulation(physicsClientId=self._client)
            self._setup_world()

        self._step_count = 0
        self._prev_dist_to_goal = float(
            np.linalg.norm(self._get_agent_pos() - self._goal_pos))
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        self._apply_action(action)
        self._update_moving_platforms()

        for _ in range(self.FRAME_SKIP):
            p.stepSimulation(physicsClientId=self._client)
            self._keep_upright()

        self._step_count += 1
        obs = self._get_obs()
        reward, terminated, truncated = self._compute_reward()

        # Update camera to follow agent in GUI mode
        if self.render_mode == "human":
            pos = self._get_agent_pos()
            p.resetDebugVisualizerCamera(
                cameraDistance=10,
                cameraYaw=45,
                cameraPitch=-25,
                cameraTargetPosition=pos.tolist(),
                physicsClientId=self._client)

        return obs, reward, terminated, truncated, {}

    def close(self):
        if self._client is not None:
            try:
                p.disconnect(physicsClientId=self._client)
            except Exception:
                pass
            self._client = None

    def render(self):
        pass  # GUI managed by PyBullet directly

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_physics(self):
        if self.render_mode == "human":
            self._client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0,
                                       physicsClientId=self._client)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1,
                                       physicsClientId=self._client)
        else:
            self._client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self._client)
        self._setup_world()

    def _setup_world(self):
        # Z-up gravity
        p.setGravity(0, 0, self.GRAVITY, physicsClientId=self._client)
        p.setTimeStep(self.TIME_STEP,    physicsClientId=self._client)

        seed = self.level_seed if self.level_seed is not None else int(
            self.np_random.integers(0, 9999))
        gen = LevelGenerator(self._client, seed=seed)
        self._level_ids, self._start_pos, self._goal_pos = gen.build()

        # Spawn agent 0.5 m above the start surface
        half_total = self.AGENT_HEIGHT / 2 + self.AGENT_RADIUS
        spawn = self._start_pos.copy()
        spawn[2] += half_total + 0.2
        self._agent_id = self._create_agent(spawn.tolist())

        # Initial camera
        if self.render_mode == "human":
            p.resetDebugVisualizerCamera(
                cameraDistance=12,
                cameraYaw=30,
                cameraPitch=-25,
                cameraTargetPosition=spawn.tolist(),
                physicsClientId=self._client)

        # Let the agent settle onto the start platform
        for _ in range(80):
            p.stepSimulation(physicsClientId=self._client)

    def _create_agent(self, position):
        col = p.createCollisionShape(
            p.GEOM_CAPSULE,
            radius=self.AGENT_RADIUS,
            height=self.AGENT_HEIGHT,
            physicsClientId=self._client)
        vis = p.createVisualShape(
            p.GEOM_CAPSULE,
            radius=self.AGENT_RADIUS,
            length=self.AGENT_HEIGHT,
            rgbaColor=[0.15, 0.55, 1.0, 1.0],
            physicsClientId=self._client)
        body_id = p.createMultiBody(
            baseMass=self.AGENT_MASS,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=position,
            physicsClientId=self._client)
        # High lateral friction to prevent sliding, low damping for mobility
        p.changeDynamics(body_id, -1,
                         lateralFriction=1.3,
                         spinningFriction=0.1,
                         rollingFriction=0.01,
                         restitution=0.0,
                         linearDamping=0.04,
                         angularDamping=0.4,
                         physicsClientId=self._client)
        # Physics tuning: no bouncing, high friction, low damping for crisp movement
        p.changeDynamics(body_id, -1,
                         lateralFriction=1.2,
                         spinningFriction=0.1,
                         rollingFriction=0.01,
                         restitution=0.0,
                         linearDamping=0.01,
                         angularDamping=0.1,  # Keep low so it can turn
                         physicsClientId=self._client)

        return body_id

    def _update_moving_platforms(self):
        """Finds platforms with 'mover' user data and updates their positions."""
        time_sec = self._step_count * (1.0 / self.PHYSICS_HZ)
        
        for body_id in self._level_ids:
            try:
                # User data key -1 indicates base link
                ud = p.getUserData(body_id, "mover", physicsClientId=self._client)
                if ud is None:
                    continue
                
                # Decode the metadata
                meta = ud.decode('utf-8').split(",")
                axis = meta[0]
                amp = float(meta[1])
                speed = float(meta[2])
                start_x = float(meta[3])
                start_y = float(meta[4])
                start_z = float(meta[5])
                
                # Calculate new position
                offset = amp * math.sin(speed * time_sec)
                
                new_x = start_x
                new_y = start_y
                new_z = start_z
                
                if axis == "Y":
                    new_y += offset
                elif axis == "Z":
                    new_z += offset
                    
                # Update position while keeping orientation
                _, orn = p.getBasePositionAndOrientation(body_id, physicsClientId=self._client)
                
                # When using resetBasePositionAndOrientation, velocities are reset to 0.
                # Because we want friction to carry the player, we must also apply velocity.
                # However, for simplicity in a static-kinematic proxy, we just place it.
                # The agent will slide if it doesn't move with it, which is part of the challenge.
                p.resetBasePositionAndOrientation(
                    body_id,
                    [new_x, new_y, new_z],
                    orn,
                    physicsClientId=self._client)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_agent_pos(self) -> np.ndarray:
        pos, _ = p.getBasePositionAndOrientation(
            self._agent_id, physicsClientId=self._client)
        return np.array(pos, dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        pos, orn = p.getBasePositionAndOrientation(
            self._agent_id, physicsClientId=self._client)
        vel, ang = p.getBaseVelocity(
            self._agent_id, physicsClientId=self._client)
        euler = p.getEulerFromQuaternion(orn)
        rays  = self._cast_rays(pos, euler)

        goal_vec = self._goal_pos - np.array(pos)
        goal_dir = goal_vec / (np.linalg.norm(goal_vec) + 1e-8)

        obs = np.concatenate([
            np.array(pos,   dtype=np.float32),
            np.array(vel,   dtype=np.float32),
            np.array(euler, dtype=np.float32),
            np.array(ang,   dtype=np.float32),
            rays,
            goal_dir.astype(np.float32),
        ]).astype(np.float32)

        return np.clip(obs, self.observation_space.low,
                       self.observation_space.high)

    def _cast_rays(self, pos, euler) -> np.ndarray:
        # Origin slightly above bottom of capsule
        ray_origin = pos + np.array([0, 0, -self.AGENT_HEIGHT/2 + 0.1])
        yaw = euler[2]
        
        distances = []
        pitches = [15, 30, 45, 60, 75]  # Downward angles in degrees
        max_dist = 4.0

        for pitch_deg in pitches:
            pitch = math.radians(pitch_deg)
            # Z-up: forward is X, right is Y, up is Z.
            # To look 'forward and down':
            # -Z component is sin(pitch)
            # XY magnitude is cos(pitch)
            dir_x = math.cos(yaw) * math.cos(pitch)
            dir_y = math.sin(yaw) * math.cos(pitch)
            dir_z = -math.sin(pitch)

            direction = np.array([dir_x, dir_y, dir_z]) * max_dist
            ray_to = ray_origin + direction

            res = p.rayTest(ray_origin.tolist(), ray_to.tolist(), 
                            physicsClientId=self._client)[0]
            hit_fraction = res[2]
            distances.append(hit_fraction)

        return np.array(distances, dtype=np.float32)

    # ------------------------------------------------------------------
    # Actions & Steps
    # ------------------------------------------------------------------

    def _apply_action(self, action: np.ndarray):
        _, orn = p.getBasePositionAndOrientation(
            self._agent_id, physicsClientId=self._client)
        rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)

        # Horizontal direction vectors (projected to XY and normalized)
        forward_h = rot[:, 0].copy(); forward_h[2] = 0
        right_h   = rot[:, 1].copy(); right_h[2]   = 0
        forward_h /= (np.linalg.norm(forward_h) + 1e-8)
        right_h   /= (np.linalg.norm(right_h)   + 1e-8)

        # -- Velocity Control for XY movement --
        # action[0] controls forward/backward speed limit
        # action[1] controls strafe speed limit
        target_v_xy = (action[0] * forward_h + action[1] * right_h) * 8.0 # max 8 m/s
        
        lin_vel, _ = p.getBaseVelocity(self._agent_id, physicsClientId=self._client)
        current_v_xy = np.array([lin_vel[0], lin_vel[1], 0.0])
        
        # We apply force proportional to the difference between target and current speed
        # This acts like a strong P-controller for velocity, making it snappy
        v_diff = target_v_xy - current_v_xy
        force = v_diff * self.MAX_FORCE  # MAX_FORCE acts as the P-gain here
        
        # Cap the maximum correction force so it doesn't explode
        force_mag = np.linalg.norm(force)
        max_allowed_force = self.MAX_FORCE * 2
        if force_mag > max_allowed_force:
            force = (force / force_mag) * max_allowed_force

        p.applyExternalForce(
            self._agent_id, -1,
            force.tolist(), [0, 0, 0], p.WORLD_FRAME,
            physicsClientId=self._client)

        # Jump: impulsive burst only when grounded
        if action[2] > 0.3 and self._is_grounded():
            jump_force = [0, 0, self.JUMP_FORCE_PEAK * float(action[2])]
            p.applyExternalForce(
                self._agent_id, -1,
                jump_force, [0, 0, 0], p.WORLD_FRAME,
                physicsClientId=self._client)

    def _keep_upright(self):
        """
        Force the capsule to stay vertical (preserve yaw only).
        IMPORTANT: After resetBasePositionAndOrientation, we must restore
        the linear and angular velocity or else the agent 'floats' and
        jumps are cancelled immediately.
        """
        pos, orn = p.getBasePositionAndOrientation(
            self._agent_id, physicsClientId=self._client)
        lin_vel, ang_vel = p.getBaseVelocity(
            self._agent_id, physicsClientId=self._client)
        euler = list(p.getEulerFromQuaternion(orn))
        # Keep only yaw (euler[2]), zero roll and pitch
        upright_orn = p.getQuaternionFromEuler([0.0, 0.0, euler[2]])
        p.resetBasePositionAndOrientation(
            self._agent_id, pos, upright_orn,
            physicsClientId=self._client)
        # Restore velocity so gravity and jump forces are not zeroed out
        p.resetBaseVelocity(
            self._agent_id,
            linearVelocity=lin_vel,
            angularVelocity=[0.0, 0.0, ang_vel[2]],  # only yaw spin preserved
            physicsClientId=self._client)

    def _is_grounded(self) -> bool:
        contacts = p.getContactPoints(
            bodyA=self._agent_id, physicsClientId=self._client)
        return len(contacts) > 0

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self):
        pos = self._get_agent_pos()
        dist = float(np.linalg.norm(pos - self._goal_pos))

        terminated = False
        truncated  = False
        reward     = self.STEP_PENALTY

        # Forward progress
        delta = self._prev_dist_to_goal - dist
        reward += delta * self.PROGRESS_SCALE
        self._prev_dist_to_goal = dist

        if dist < 1.5:
            reward += self.GOAL_REWARD
            terminated = True

        if pos[2] < self.FALL_Z:
            reward += self.FALL_PENALTY
            terminated = True

        if self._step_count >= self.MAX_STEPS:
            truncated = True

        return reward, terminated, truncated
