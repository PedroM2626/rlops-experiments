"""
Chase/TAG game environment - "Pega-Pega"
The AI agent must survive as long as possible while being chased by a floating capsule.
"""

import math
import random
import os
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data


class ChaseEnv(gym.Env):
    """
    A chase game where a floating capsule chases the humanoid agent.
    
    The agent (humanoid) must:
    - Stand up
    - Walk/Turn/Run
    - Evade the chaser
    - Survive as long as possible
    
    The chaser:
    - Is a floating capsule (no gravity)
    - Moves directly toward the agent
    - Gets faster over time
    """

    metadata = {"render_modes": ["human", "direct"]}

    # Physics
    # Using 120 Hz for faster real-time simulation while maintaining stability
    PHYSICS_HZ = 120
    TIMESTEP = 1.0 / PHYSICS_HZ

    # Observation space
    NUM_RAY_CHECKS = 16
    RAY_MAX_DIST = 20.0
    
    # Limits
    MAX_STEPS = 3000  # Max steps per episode
    
    # Chaser settings
    CHASER_START_DIST = 20.0  # Start further away
    CHASER_SPEED_BASE = 8.0   # Much faster base speed
    CHASER_SPEED_MAX = 25.0   # Much higher max speed
    CHASER_ACCEL = 0.1        # Faster acceleration
    
    # Agent dimensions
    AGENT_TOTAL_HEIGHT = 0.65
    
    # Rewards
    SURVIVAL_BONUS = 0.01
    CHASER_DISTANCE_PENALTY = 0.005
    CHASER_CLOSE_PENALTY = 0.5
    PROGRESS_SCALE = 2.0
    SPEED_SCALE = 0.2
    UPRIGHT_SCALE = 0.3
    ENERGY_SCALE = 0.001
    
    # Fall threshold
    FALL_Z = 0.25
    
    def __init__(self, render_mode: str = "direct", seed: int = 42):
        super().__init__()
        
        self.render_mode = render_mode
        self.seed = seed
        self.rng = random.Random(seed)
        
        # PyBullet client
        if render_mode == "human":
            self._client = p.connect(p.GUI)
        else:
            self._client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client)
        
        # spaces
        self._setup_spaces()
        
        # State
        self._step_count = 0
        self._grace_steps = 500  # Increased grace period for proper stabilization
        self._chaser_speed = self.CHASER_SPEED_BASE
        self._prev_dist_to_chaser = None
        
        # Agent ID
        self._agent_id = None
        self._chaser_id = None
        
        # Joint info
        self._joint_indices = []
        self._joint_limits = []
        self._joint_torques = []
        self._foot_link_ids = {}
        
        # Spawn tracking
        self._spawn_pos = np.zeros(3, dtype=np.float32)
        
        # For rendering
        self._cam_dist = 8.0
        self._cam_yaw = 50.0
    
    def _setup_spaces(self):
        """Setup gym spaces."""
        # Action space: joint targets [-1, 1]
        # 10 joints: 2 shoulders + 2 elbows + 2 hips + 2 knees + 2 ankles
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(10,), dtype=np.float32
        )
        
        # Observation space:
        # - Chaser info (3): angle to chaser (sin, cos), distance normalized
        # - Agent velocity (3): x, y, z
        # - Agent orientation (3): roll, pitch, yaw  
        # - Joint positions (10)
        # - Joint velocities (10)
        # - Ray distances (NUM_RAY_CHECKS)
        obs_dim = 3 + 3 + 3 + 10 + 10 + self.NUM_RAY_CHECKS
        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        if seed is not None:
            self.seed = seed
            self.rng = random.Random(seed)
        
        p.resetSimulation(physicsClientId=self._client)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)
        p.setTimeStep(self.TIMESTEP, physicsClientId=self._client)
        
        # Create ground
        p.loadURDF("plane.urdf", physicsClientId=self._client)
        
        # Create walls to bound the arena
        self._create_arena()
        
        # Create humanoid agent
        self._create_agent()
        
        # Create chaser capsule
        self._create_chaser()
        
        # Reset state
        self._step_count = 0
        self._grace_steps = 100  # Don't detect fall for first 100 steps
        self._chaser_speed = self.CHASER_SPEED_BASE
        
        pos, _ = p.getBasePositionAndOrientation(self._agent_id, physicsClientId=self._client)
        self._spawn_pos = np.array(pos, dtype=np.float32)
        self._prev_dist_to_chaser = self._get_chaser_distance()
        
        # Setup camera for human mode
        if self.render_mode == "human":
            p.resetDebugVisualizerCamera(
                cameraDistance=self._cam_dist,
                cameraYaw=self._cam_yaw,
                cameraPitch=-20,
                cameraTargetPosition=[0, 0, 1],
                physicsClientId=self._client)
        
        return self._get_obs(), {}
    
    def _create_arena(self):
        """Create a bounded arena."""
        # Ground
        ground_size = 50.0
        col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[ground_size, ground_size, 0.1],
            physicsClientId=self._client)
        vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[ground_size, ground_size, 0.1],
            rgbaColor=[0.3, 0.3, 0.35, 1.0],
            physicsClientId=self._client)
        p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis, basePosition=[0, 0, -0.1],
            physicsClientId=self._client)
        
        # Walls (invisible but present)
        wall_height = 5.0
        wall_thick = 0.5
        wall_dist = 25.0
        
        for (wx, wy) in [(wall_dist, 0), (-wall_dist, 0), (0, wall_dist), (0, -wall_dist)]:
            if wx != 0:
                col = p.createCollisionShape(
                    p.GEOM_BOX, halfExtents=[wall_thick, wall_dist, wall_height],
                    physicsClientId=self._client)
            else:
                col = p.createCollisionShape(
                    p.GEOM_BOX, halfExtents=[wall_dist, wall_thick, wall_height],
                    physicsClientId=self._client)
            p.createMultiBody(
                baseMass=0, baseCollisionShapeIndex=col,
                basePosition=[wx, wy, wall_height],
                physicsClientId=self._client)
    
    def _create_agent(self):
        """Create the humanoid agent."""
        # Get Chase directory
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        urdf_path = os.path.join(base_dir, "assets", "humanoid.urdf")
        
        # Spawn position - higher to give time to land
        spawn_x = self.rng.uniform(-10, 10)
        spawn_y = self.rng.uniform(-10, 10)
        
        self._agent_id = p.loadURDF(
            urdf_path,
            basePosition=[spawn_x, spawn_y, 2.5],
            baseOrientation=[0, 0, 0, 1],
            useFixedBase=False,
            flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS,
            physicsClientId=self._client)
        
        # Get joint info
        num_joints = p.getNumJoints(self._agent_id, physicsClientId=self._client)
        
        # Map joint names to indices
        # We want: shoulders(2), elbows(2), hips(2), knees(2), ankles(2) = 10 joints
        joint_names = [
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow", 
            "left_hip", "right_hip",
            "left_knee", "right_knee",
            "left_ankle", "right_ankle"
        ]
        
        self._joint_indices = []
        self._joint_limits = []
        self._joint_torques = []
        self._foot_link_ids = {}
        
        for i in range(num_joints):
            info = p.getJointInfo(self._agent_id, i, physicsClientId=self._client)
            name = info[1].decode()
            
            if name in joint_names:
                self._joint_indices.append(i)
                self._joint_limits.append((info[8], info[9]))  # (lower, upper)
                self._joint_torques.append(info[10])  # max force
            
            # Track foot links
            if "foot" in name.lower():
                self._foot_link_ids[name] = i
        
        # Let agent fall and settle - more time to stand up
        for _ in range(500):
            p.stepSimulation(physicsClientId=self._client)
        
        # Reset joint positions to standing pose (slightly bent knees)
        for i, ji in enumerate(self._joint_indices):
            # Joint order: left_shoulder, right_shoulder, left_elbow, right_elbow, 
            # left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle
            if 'knee' in p.getJointInfo(self._agent_id, ji, physicsClientId=self._client)[1].decode():
                # Knees slightly bent for stability
                target_pos = -0.2
            else:
                target_pos = 0.0
            p.resetJointState(self._agent_id, ji, target_pos, physicsClientId=self._client)
        
        # Settle again - longer time to stabilize
        for _ in range(500):
            p.stepSimulation(physicsClientId=self._client)
    
    def _create_chaser(self):
        """Create the floating capsule that chases the agent."""
        # Capsule visual
        capsule_len = 1.5
        capsule_radius = 0.5
        
        vis = p.createVisualShape(
            p.GEOM_CAPSULE, radius=capsule_radius, length=capsule_len,
            rgbaColor=[1.0, 0.0, 0.3, 0.8],  # Red/pink
            physicsClientId=self._client)
        
        self._chaser_id = p.createMultiBody(
            baseMass=0,  # Static (we move it manually)
            baseVisualShapeIndex=vis,
            basePosition=[0, 0, 1.0],
            physicsClientId=self._client)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step."""
        self._step_count += 1
        
        # Apply action to agent
        self._apply_action(action)
        
        # Move chaser toward agent
        self._update_chaser()
        
        # Step physics
        p.stepSimulation(physicsClientId=self._client)
        
        # Update camera
        if self.render_mode == "human":
            self._update_camera()
        
        # Check termination
        terminated = False
        truncated = False
        
        # Check if caught
        if self._is_caught():
            terminated = True
        
        # Check if fallen (skip during grace period)
        if self._grace_steps > 0:
            self._grace_steps -= 1
        elif self._is_fallen():
            terminated = True
        
        # Check max steps
        if self._step_count >= self.MAX_STEPS:
            truncated = True
        
        # Compute reward
        reward = self._compute_reward(action)
        
        return self._get_obs(), reward, terminated, truncated, {}
    
    def _apply_action(self, action: np.ndarray):
        """Apply joint targets using position control."""
        for i, ji in enumerate(self._joint_indices):
            lo, hi = self._joint_limits[i]
            target_pos = float(lo + (action[i] + 1.0) * 0.5 * (hi - lo))
            target_pos = float(np.clip(target_pos, lo, hi))
            
            p.setJointMotorControl2(
                self._agent_id, ji,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_pos,
                force=self._joint_torques[i],
                maxVelocity=10.0,
                physicsClientId=self._client)
    
    def _update_chaser(self):
        """Move chaser toward agent."""
        # Increase speed over time
        self._chaser_speed = min(
            self._chaser_speed + self.CHASER_ACCEL,
            self.CHASER_SPEED_MAX
        )
        
        # Get positions
        agent_pos, _ = p.getBasePositionAndOrientation(
            self._agent_id, physicsClientId=self._client)
        chaser_pos, _ = p.getBasePositionAndOrientation(
            self._chaser_id, physicsClientId=self._client)
        
        # Direction to agent
        dx = agent_pos[0] - chaser_pos[0]
        dy = agent_pos[1] - chaser_pos[1]
        dz = (agent_pos[2] + 1.0) - chaser_pos[2]  # Aim slightly above agent
        
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        if dist > 0.01:
            dx, dy, dz = dx/dist, dy/dist, dz/dist
        
        # Move chaser
        speed = self._chaser_speed * self.TIMESTEP
        new_x = chaser_pos[0] + dx * speed
        new_y = chaser_pos[1] + dy * speed
        new_z = chaser_pos[2] + dz * speed
        
        # Keep chaser at a minimum height
        new_z = max(new_z, 0.8)
        
        p.resetBasePositionAndOrientation(
            self._chaser_id, [new_x, new_y, new_z], [0, 0, 0, 1],
            physicsClientId=self._client)
    
    def _is_caught(self) -> bool:
        """Check if chaser caught the agent."""
        agent_pos, _ = p.getBasePositionAndOrientation(
            self._agent_id, physicsClientId=self._client)
        chaser_pos, _ = p.getBasePositionAndOrientation(
            self._chaser_id, physicsClientId=self._client)
        
        dx = agent_pos[0] - chaser_pos[0]
        dy = agent_pos[1] - chaser_pos[1]
        dz = agent_pos[2] - chaser_pos[2]
        
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        return dist < 1.5  # Catch radius
    
    def _is_fallen(self) -> bool:
        """Check if agent fell."""
        # Check void
        pos, _ = p.getBasePositionAndOrientation(self._agent_id, physicsClientId=self._client)
        if pos[2] < -2.0:
            return True
        
        # Check if non-foot links touching ground
        allowed_links = set(self._foot_link_ids.values())
        contacts = p.getContactPoints(bodyA=self._agent_id, physicsClientId=self._client)
        
        for c in (contacts or []):
            link_a = c[3]
            body_b = c[2]
            if body_b == self._agent_id:
                continue
            if link_a not in allowed_links:
                return True
        return False
    
    def _compute_reward(self, action: np.ndarray) -> float:
        """Compute reward."""
        reward = 0.0
        
        # Survival bonus
        reward += self.SURVIVAL_BONUS
        
        # Distance to chaser
        dist = self._get_chaser_distance()
        
        # Penalty for being close
        if dist < 3.0:
            reward -= self.CHASER_CLOSE_PENALTY * (3.0 - dist)
        
        # Reward for increasing distance
        if self._prev_dist_to_chaser is not None:
            delta = dist - self._prev_dist_to_chaser
            reward += delta * self.PROGRESS_SCALE
            
            # Speed bonus when moving away
            if delta > 0.1:
                vel, _ = p.getBaseVelocity(self._agent_id, physicsClientId=self._client)
                forward_speed = max(0, vel[0])
                reward += forward_speed * self.SPEED_SCALE
        
        self._prev_dist_to_chaser = dist
        
        # Upright penalty
        _, orn = p.getBasePositionAndOrientation(self._agent_id, physicsClientId=self._client)
        euler = p.getEulerFromQuaternion(orn)
        roll, pitch = euler[0], euler[1]
        reward -= (abs(roll) + abs(pitch)) * self.UPRIGHT_SCALE
        
        # Energy penalty
        reward -= float(np.sum(action ** 2)) * self.ENERGY_SCALE
        
        # Caught penalty
        if self._is_caught():
            reward -= 50.0
        
        return reward
    
    def _get_chaser_distance(self) -> float:
        """Get distance to chaser."""
        agent_pos, _ = p.getBasePositionAndOrientation(
            self._agent_id, physicsClientId=self._client)
        chaser_pos, _ = p.getBasePositionAndOrientation(
            self._chaser_id, physicsClientId=self._client)
        
        dx = agent_pos[0] - chaser_pos[0]
        dy = agent_pos[1] - chaser_pos[1]
        dz = agent_pos[2] - chaser_pos[2]
        
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    def _get_obs(self) -> np.ndarray:
        """Get observation."""
        obs_parts = []
        
        # Chaser info
        chaser_dist = self._get_chaser_distance()
        agent_pos, _ = p.getBasePositionAndOrientation(self._agent_id, physicsClientId=self._client)
        chaser_pos, _ = p.getBasePositionAndOrientation(self._chaser_id, physicsClientId=self._client)
        
        # Angle to chaser (in X-Y plane)
        dx = chaser_pos[0] - agent_pos[0]
        dy = chaser_pos[1] - agent_pos[1]
        angle_to_chaser = math.atan2(dy, dx)
        
        # Normalize distance
        norm_dist = min(chaser_dist / 25.0, 1.0)
        
        obs_parts.append([math.sin(angle_to_chaser), math.cos(angle_to_chaser)])
        obs_parts.append([norm_dist])
        
        # Agent velocity
        vel, _ = p.getBaseVelocity(self._agent_id, physicsClientId=self._client)
        obs_parts.append([vel[0] / 10.0, vel[1] / 10.0, vel[2] / 10.0])
        
        # Agent orientation
        _, orn = p.getBasePositionAndOrientation(self._agent_id, physicsClientId=self._client)
        euler = p.getEulerFromQuaternion(orn)
        obs_parts.append([euler[0] / 3.14, euler[1] / 3.14, euler[2] / 3.14])
        
        # Joint positions
        joint_pos = []
        joint_vel = []
        for ji in self._joint_indices:
            state = p.getJointState(self._agent_id, ji, physicsClientId=self._client)
            joint_pos.append(state[0] / 3.14)
            joint_vel.append(state[1] / 10.0)
        obs_parts.append(joint_pos)
        obs_parts.append(joint_vel)
        
        # Raycasts for obstacle detection
        ray_dists = self._cast_rays()
        obs_parts.append([ray_dists])
        
        return np.concatenate([np.array(x).flatten() for x in obs_parts]).astype(np.float32)
    
    def _cast_rays(self):
        """Cast rays to detect arena boundaries."""
        # Cast rays in circle around agent
        num_rays = self.NUM_RAY_CHECKS
        ray_dists = []
        
        agent_pos, _ = p.getBasePositionAndOrientation(self._agent_id, physicsClientId=self._client)
        _, _, agent_yaw = p.getEulerFromQuaternion(
            p.getBasePositionAndOrientation(self._agent_id, physicsClientId=self._client)[1])
        
        for i in range(num_rays):
            angle = agent_yaw + (2 * math.pi * i / num_rays)
            ray_dir = [math.cos(angle), math.sin(angle), 0]
            
            ray_from = [agent_pos[0], agent_pos[1], 0.5]
            ray_to = [
                ray_from[0] + ray_dir[0] * self.RAY_MAX_DIST,
                ray_from[1] + ray_dir[1] * self.RAY_MAX_DIST,
                0.5
            ]
            
            result = p.rayTest(ray_from, ray_to, physicsClientId=self._client)
            
            if result[0][0] == 0:  # Hit something (plane has ID 0)
                hit_dist = result[0][2]
            else:
                hit_dist = self.RAY_MAX_DIST
            
            ray_dists.append(hit_dist / self.RAY_MAX_DIST)
        
        return ray_dists
    
    def _update_camera(self):
        """Update camera to follow agent."""
        agent_pos, _ = p.getBasePositionAndOrientation(
            self._agent_id, physicsClientId=self._client)
        
        p.resetDebugVisualizerCamera(
            cameraDistance=self._cam_dist,
            cameraYaw=self._cam_yaw,
            cameraPitch=-20,
            cameraTargetPosition=[agent_pos[0], agent_pos[1], 1.0],
            physicsClientId=self._client)
    
    def render(self):
        """Render (handled in step for human mode)."""
        pass
    
    def close(self):
        """Close environment."""
        if self._client >= 0:
            p.disconnect(physicsClientId=self._client)
            self._client = -1
