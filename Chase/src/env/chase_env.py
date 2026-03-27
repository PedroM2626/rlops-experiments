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
        
        self._client = None
        self._agent_id = None
        self._chaser_id = None
        self._joint_indices = []
        self._joint_limits = []
        self._joint_torques = []
        self._foot_link_ids = {}
        self._chaser_joint_indices = []
        self._chaser_joint_limits = []
        self._chaser_joint_torques = []
        self._chaser_joint_map = {}
        self._spawn_pos = np.zeros(3, dtype=np.float32)
        self._step_count = 0
        self._grace_steps = 500
        self._chaser_speed = self.CHASER_SPEED_BASE
        self._prev_dist_to_chaser = None
        self._cam_dist = 8.0
        self._cam_yaw = 50.0
        
        self._setup_spaces()
    
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
    
    def reset(self, seed: int = None, options: dict = None) -> Tuple[np.ndarray, dict]:
        """Reset the environment."""
        if seed is not None:
            self.seed = seed
            self.rng = random.Random(seed)
        
        setup_arena = False
        if self._client is None:
            if self.render_mode == "human":
                self._client = p.connect(p.GUI)
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=self._client)
            else:
                self._client = p.connect(p.DIRECT)
            setup_arena = True
            p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client)
        
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)
        p.setTimeStep(self.TIMESTEP, physicsClientId=self._client)
        
        if setup_arena:
            p.loadURDF("plane.urdf", physicsClientId=self._client)
            self._create_arena()
            
        if self._agent_id is not None:
            p.removeBody(self._agent_id, physicsClientId=self._client)
        if self._chaser_id is not None:
            p.removeBody(self._chaser_id, physicsClientId=self._client)
            
        self._create_agent()
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
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        urdf_path = os.path.join(base_dir, "assets", "humanoid.urdf")
        
        spawn_x = self.rng.uniform(-10, 10)
        spawn_y = self.rng.uniform(-10, 10)
        
        self._agent_id = p.loadURDF(
            urdf_path,
            basePosition=[spawn_x, spawn_y, 1.4],
            baseOrientation=[0, 0, 0, 1],
            useFixedBase=False,
            flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS,
            physicsClientId=self._client)
        
        num_joints = p.getNumJoints(self._agent_id, physicsClientId=self._client)
        
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
                self._joint_limits.append((info[8], info[9]))
                self._joint_torques.append(info[10])
            
            if "foot" in name.lower():
                self._foot_link_ids[name] = i
        
        self._configure_dynamics()
        
        name_to_idx = {p.getJointInfo(self._agent_id, i, physicsClientId=self._client)[1].decode(): i 
                       for i in range(num_joints)}
        for fname in ("foot_L", "foot_R"):
            if fname not in self._foot_link_ids:
                for jname in ("left_ankle", "right_ankle"):
                    if jname in name_to_idx:
                        self._foot_link_ids[fname] = name_to_idx[jname]
        
        standing_angles = {
            "left_shoulder": 0.0,
            "right_shoulder": 0.0,
            "left_elbow": 0.0,
            "right_elbow": 0.0,
            "left_hip": 0.0,
            "right_hip": 0.0,
            "left_knee": 0.0,
            "right_knee": 0.0,
            "left_ankle": 0.0,
            "right_ankle": 0.0,
        }
        
        for i, ji in enumerate(self._joint_indices):
            joint_name = p.getJointInfo(self._agent_id, ji, physicsClientId=self._client)[1].decode()
            target_angle = standing_angles.get(joint_name, 0.0)
            p.resetJointState(self._agent_id, ji, targetValue=target_angle, physicsClientId=self._client)
        
        for _ in range(100):
            p.stepSimulation(physicsClientId=self._client)
    
    def _configure_dynamics(self):
        """Apply friction and damping to all links and feet."""
        num_joints = p.getNumJoints(self._agent_id, physicsClientId=self._client)
        foot_ids = set(self._foot_link_ids.values())
        
        for ji in range(-1, num_joints):
            friction = 2.0 if ji in foot_ids else 0.9
            p.changeDynamics(self._agent_id, ji,
                             lateralFriction=friction,
                             spinningFriction=0.05,
                             rollingFriction=0.01,
                             restitution=0.0,
                             linearDamping=0.04,
                             angularDamping=0.1,
                             physicsClientId=self._client)
        
        for link_idx in self._foot_link_ids.values():
            p.changeDynamics(self._agent_id, link_idx,
                             lateralFriction=1.4,
                             spinningFriction=0.1,
                             physicsClientId=self._client)
    
    def _create_chaser(self):
        """Create the humanoid chaser."""
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        urdf_path = os.path.join(base_dir, "assets", "humanoid.urdf")
        
        agent_pos, _ = p.getBasePositionAndOrientation(self._agent_id, physicsClientId=self._client)
        spawn_x = agent_pos[0] + self.rng.uniform(5, 8) * self.rng.choice([-1, 1])
        spawn_y = agent_pos[1] + self.rng.uniform(5, 8) * self.rng.choice([-1, 1])
        
        self._chaser_id = p.loadURDF(
            urdf_path,
            basePosition=[spawn_x, spawn_y, 1.4],
            baseOrientation=[0, 0, 0, 1],
            useFixedBase=False,  # Respect gravity
            flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS,
            physicsClientId=self._client)
        
        # Change color to red/pink to distinguish from player
        p.changeVisualShape(self._chaser_id, -1, rgbaColor=[1.0, 0.2, 0.3, 1.0], physicsClientId=self._client)
        num_joints = p.getNumJoints(self._chaser_id, physicsClientId=self._client)
        for j in range(num_joints):
            p.changeVisualShape(self._chaser_id, j, rgbaColor=[1.0, 0.2, 0.3, 1.0], physicsClientId=self._client)

        joint_names = [
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow", 
            "left_hip", "right_hip",
            "left_knee", "right_knee",
            "left_ankle", "right_ankle"
        ]
        
        self._chaser_joint_indices = []
        self._chaser_joint_limits = []
        self._chaser_joint_torques = []
        self._chaser_joint_map = {}
        
        for i in range(num_joints):
            info = p.getJointInfo(self._chaser_id, i, physicsClientId=self._client)
            name = info[1].decode()
            
            if name in joint_names:
                self._chaser_joint_indices.append(i)
                self._chaser_joint_limits.append((info[8], info[9]))
                self._chaser_joint_torques.append(info[10])
                self._chaser_joint_map[name] = i
        
        # Keep joints neutral at spawn
        for ji in self._chaser_joint_indices:
            p.resetJointState(self._chaser_id, ji, targetValue=0.0, physicsClientId=self._client)
            
        # Optional dynamics tune for chaser to not slide endlessly
        for ji in range(-1, num_joints):
            p.changeDynamics(self._chaser_id, ji,
                             lateralFriction=1.0,
                             linearDamping=0.05,
                             angularDamping=0.1,
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
        """Move chaser toward agent using a rule-based walking controller."""
        # Increase speed slightly over time
        self._chaser_speed = min(
            self._chaser_speed + self.CHASER_ACCEL * 0.05, # Slower accel since humanoid
            self.CHASER_SPEED_MAX * 0.5 # Max speed capped for physical humanoid
        )
        
        agent_pos, _ = p.getBasePositionAndOrientation(self._agent_id, physicsClientId=self._client)
        chaser_pos, chaser_orn = p.getBasePositionAndOrientation(self._chaser_id, physicsClientId=self._client)
        chaser_vel, _ = p.getBaseVelocity(self._chaser_id, physicsClientId=self._client)
        
        dx = agent_pos[0] - chaser_pos[0]
        dy = agent_pos[1] - chaser_pos[1]
        
        dist = math.sqrt(dx*dx + dy*dy)
        if dist > 0.01:
            dx, dy = dx/dist, dy/dist
        else:
            dx, dy = 1.0, 0.0
            
        # 1. Face the agent and stay upright (force roll/pitch to 0)
        target_yaw = math.atan2(dy, dx)
        new_orn = p.getQuaternionFromEuler([0, 0, target_yaw])
        p.resetBasePositionAndOrientation(self._chaser_id, chaser_pos, new_orn, physicsClientId=self._client)
        
        # 2. Command velocity (pushing the base horizontally)
        vz = chaser_vel[2] # KEEP GRAVITY
        
        # Don't push if caught to avoid pushing agent endlessly
        chase_speed = self._chaser_speed if dist > 1.0 else 0.0
        p.resetBaseVelocity(self._chaser_id, linearVelocity=[dx * chase_speed, dy * chase_speed, vz], angularVelocity=[0,0,0], physicsClientId=self._client)
        
        # 3. Simulate walking animation via joints
        phase = self._step_count * 0.1 * chase_speed
        
        hip_swing = 0.5  # radians
        knee_bend = 0.5
        
        left_hip_angle = math.sin(phase) * hip_swing
        left_knee_angle = abs(math.sin(phase)) * knee_bend
        
        right_hip_angle = math.sin(phase + math.pi) * hip_swing
        right_knee_angle = abs(math.sin(phase + math.pi)) * knee_bend
        
        left_shoulder_angle = -left_hip_angle * 0.5
        right_shoulder_angle = -right_hip_angle * 0.5
        
        target_angles = {
            "left_hip": left_hip_angle,
            "right_hip": right_hip_angle,
            "left_knee": left_knee_angle,
            "right_knee": right_knee_angle,
            "left_shoulder": left_shoulder_angle,
            "right_shoulder": right_shoulder_angle,
            "left_elbow": 0.0,
            "right_elbow": 0.0,
            "left_ankle": 0.0,
            "right_ankle": 0.0
        }
        
        for j_name, target_angle in target_angles.items():
            if j_name in self._chaser_joint_map:
                ji = self._chaser_joint_map[j_name]
                idx = self._chaser_joint_indices.index(ji)
                max_force = self._chaser_joint_torques[idx]
                p.setJointMotorControl2(
                    self._chaser_id, ji,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target_angle,
                    force=max_force,
                    maxVelocity=10.0,
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
            
            # PyBullet rayTest returns: (body_id, link_index, hit_fraction, hit_position, hit_normal)
            # hit_fraction is in [0, 1], where 0 is ray_from and 1 is ray_to
            if result and result[0][0] != -1:  # -1 means no hit
                hit_fraction = result[0][2]
                hit_dist = hit_fraction * self.RAY_MAX_DIST
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
        if self._client is not None:
            try:
                p.disconnect(physicsClientId=self._client)
            except Exception:
                pass
            self._client = None
