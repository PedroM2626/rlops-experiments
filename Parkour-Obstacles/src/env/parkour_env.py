"""
Parkour Gymnasium environment backed by PyBullet physics.
Agent: Articulated humanoid (URDF) with 10 revolute joints.

Coordinate convention: PyBullet Z-up
  X = forward (direction of travel)
  Y = lateral (left/right)
  Z = up

Observation space (44-dim):
  [0:3]   torso position (x, y, z)
  [3:6]   torso linear velocity
  [6:9]   torso orientation euler (roll, pitch, yaw)
  [9:12]  torso angular velocity
  [12:22] 10 joint angles (normalized -1..1 within joint limits)
  [22:32] 10 joint velocities (clipped to [-20, 20])
  [32:37] 5 downward ground raycasts (hit fraction 0..1)
  [37:40] unit vector toward goal
  [40]    left foot contact (0/1)
  [41]    right foot contact (0/1)
  [42]    torso height above reference ground (useful proxy for "standing")
  [43]    time remaining fraction (steps_left / MAX_STEPS)

Action space (10-dim continuous, clipped [-1, 1]):
  Each dimension maps linearly to a target angle within that joint's limits,
  served by PyBullet's PD position controller; per-joint max torque from
  _JOINT_CFG caps the force the controller may apply.
  Joint order: [l_shoulder, l_elbow, r_shoulder, r_elbow,
                l_hip, l_knee, l_ankle, r_hip, r_knee, r_ankle]
"""

import math
import os
from typing import Optional, List

import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces

from src.env.level_generator import LevelGenerator


# ---------------------------------------------------------------------------
# Joint configuration table
# ---------------------------------------------------------------------------
# Each entry: (joint_name, lower_limit_rad, upper_limit_rad, max_torque_Nm)
_JOINT_CFG = [
    ("left_shoulder",  -1.57,  1.57,  60.0),
    ("left_elbow",      0.00,  2.27,  40.0),
    ("right_shoulder", -1.57,  1.57,  60.0),
    ("right_elbow",     0.00,  2.27,  40.0),
    ("left_hip",       -1.05,  1.05, 120.0),
    ("left_knee",       0.00,  2.09, 100.0),
    ("left_ankle",     -0.70,  0.70,  60.0),
    ("right_hip",      -1.05,  1.05, 120.0),
    ("right_knee",      0.00,  2.09, 100.0),
    ("right_ankle",    -0.70,  0.70,  60.0),
]
N_JOINTS = len(_JOINT_CFG)        # 10
OBS_DIM  = 12 + 2 * N_JOINTS + 5 + 3 + 4  # = 44

# Feet link names for contact detection
_FOOT_LINKS = ("foot_L", "foot_R")


class ParkourEnv(gym.Env):
    metadata = {"render_modes": ["human", "direct"], "render_fps": 60}

    # -----------------------------------------------------------------------
    # Physics constants
    # -----------------------------------------------------------------------
    GRAVITY    = -9.81
    TIME_STEP  = 1.0 / 480.0   # finer step for joint stability
    FRAME_SKIP = 8              # policy runs at 60 Hz (480 / 8)
    PHYSICS_HZ = 480.0 / FRAME_SKIP

    # Episode
    MAX_STEPS = 2000

    # Agent dimensions (used for spawn height)
    AGENT_TOTAL_HEIGHT = 0.65   # metres (from URDF torso height)

    # Reward
    GOAL_REWARD    =  150.0
    FALL_PENALTY   =   -5.0    # reduced constraint to encourage risk-taking
    STEP_PENALTY   =   -0.02
    PROGRESS_SCALE =    5.0
    SPEED_SCALE    =    0.3    # reduced: was 2.0, too high
    UPRIGHT_SCALE  =    0.4
    ENERGY_SCALE   =    0.001
    LATERAL_VEL_SCALE = 0.5
    LATERAL_POS_SCALE = 0.3
    ALIVE_BONUS    =    0.005   # small reward for staying alive
    HEIGHT_SCALE   =    0.1     # reward for standing upright
    STANDING_ABOVE =    0.9    # target: torso this many metres above platform

    # Lateral spring
    LATERAL_SPRING_K   = 80.0
    LATERAL_SPRING_MAX = 200.0

    # Fall threshold: torso Z below this → fallen
    FALL_Z = 0.25   # lowered to allow more walking time

    # Raycasts
    RAY_PITCHES = [15, 30, 45, 60, 75]   # degrees *below* horizontal
    RAY_MAX_DIST = 5.0

    def __init__(
        self,
        render_mode: str = "direct",
        level_seed: Optional[int] = None,
        reward_config: Optional[dict] = None,
        world_config: Optional[dict] = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.level_seed  = level_seed
        self.reward_config = reward_config if reward_config is not None else {}
        self.world_config = world_config if world_config is not None else {}

        # ---- Spaces ----
        obs_low  = np.full(OBS_DIM, -np.inf, dtype=np.float32)
        obs_high = np.full(OBS_DIM,  np.inf, dtype=np.float32)
        # Ray fractions 0..1
        obs_low[32:37]  = 0.0
        obs_high[32:37] = 1.0
        # Goal direction -1..1
        obs_low[37:40]  = -1.0
        obs_high[37:40] =  1.0
        # Foot contacts 0..1
        obs_low[40:42]  = 0.0
        obs_high[40:42] = 1.0
        # Time remaining 0..1
        obs_low[43]     = 0.0
        obs_high[43]    = 1.0

        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, shape=(N_JOINTS,), dtype=np.float32)

        # Internal state
        self._client           = None
        self._agent_id         = None
        self._joint_indices: List[int] = []
        self._foot_link_ids: dict      = {}
        self._goal_pos         = np.zeros(3, dtype=np.float32)
        self._start_pos        = np.zeros(3, dtype=np.float32)
        self._level_ids: List[int]    = []
        self._step_count               = 0
        self._prev_dist_to_goal        = 0.0
        self._spawn_y                  = 0.0
        self._platform_z               = 0.0  # surface Z of the start platform

        # URDF path (relative to project root, resolved at load time)
        self._urdf_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "assets", "humanoid.urdf")
        self._gravity_z = self.GRAVITY
        self._time_step = self.TIME_STEP
        self._frame_skip = self.FRAME_SKIP
        self._physics_hz = self.PHYSICS_HZ
        self._camera_distance = 6.0
        self._camera_yaw = 45.0
        self._camera_pitch = -20.0
        self._show_gui_panels = False
        self._show_shadows = False
        self._apply_world_config()

    def _apply_world_config(self):
        """Refresh runtime-adjustable world parameters."""
        self._gravity_z = float(self.world_config.get("gravity_z", self.GRAVITY))
        self._time_step = float(self.world_config.get("time_step", self.TIME_STEP))
        self._frame_skip = max(1, int(self.world_config.get("frame_skip", self.FRAME_SKIP)))
        self._physics_hz = 1.0 / (self._time_step * self._frame_skip)
        self._camera_distance = float(self.world_config.get("camera_distance", 6.0))
        self._camera_yaw = float(self.world_config.get("camera_yaw", 45.0))
        self._camera_pitch = float(self.world_config.get("camera_pitch", -20.0))
        self._show_gui_panels = bool(self.world_config.get("show_gui_panels", False))
        self._show_shadows = bool(self.world_config.get("show_shadows", False))

    def set_runtime_config(self, reward_config: Optional[dict] = None, world_config: Optional[dict] = None):
        """Apply hot-reloadable reward and world settings."""
        if reward_config is not None:
            self.reward_config = reward_config
        if world_config is not None:
            self.world_config = world_config
            self._apply_world_config()

        if self._client is not None:
            p.setGravity(0, 0, self._gravity_z, physicsClientId=self._client)
            p.setTimeStep(self._time_step, physicsClientId=self._client)
            if self.render_mode == "human":
                self._configure_visualizer()
        return True

    def _configure_visualizer(self):
        """Reduce viewport noise and disable extra preview panes."""
        client = self._client
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, int(self._show_gui_panels), physicsClientId=client)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, int(self._show_shadows), physicsClientId=client)
        for attr in (
            "COV_ENABLE_RGB_BUFFER_PREVIEW",
            "COV_ENABLE_DEPTH_BUFFER_PREVIEW",
            "COV_ENABLE_SEGMENTATION_MARK_PREVIEW",
        ):
            flag = getattr(p, attr, None)
            if flag is not None:
                p.configureDebugVisualizer(flag, 0, physicsClientId=client)

    # -----------------------------------------------------------------------
    # Gym interface
    # -----------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self._client is None:
            if self.render_mode == "human":
                self._client = p.connect(p.GUI)
                self._configure_visualizer()
            else:
                self._client = p.connect(p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client)
        
        if self.render_mode == "human":
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=self._client)
        
        self._setup_world()

        self._step_count = 0
        self._prev_dist_to_goal = float(
            np.linalg.norm(self._get_torso_pos() - self._goal_pos))
        self._spawn_y    = float(self._start_pos[1])
        self._platform_z = float(self._start_pos[2])  # top surface of start platform

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        self._prev_action = action  # saved for energy penalty
        self._apply_action(action)
        self._apply_lateral_spring()   # soft Y-centering force
        self._update_moving_platforms()

        for _ in range(self._frame_skip):
            p.stepSimulation(physicsClientId=self._client)

        self._step_count += 1
        obs    = self._get_obs()
        reward, terminated, truncated = self._compute_reward(action)

        if self.render_mode == "human":
            pos = self._get_torso_pos()
            p.resetDebugVisualizerCamera(
                cameraDistance=self._camera_distance,
                cameraYaw=self._camera_yaw,
                cameraPitch=self._camera_pitch,
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
        pass

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def _init_physics(self):
        if self.render_mode == "human":
            self._client = p.connect(p.GUI)
            self._configure_visualizer()
        else:
            self._client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self._client)
        # NOTE: _setup_world is called from reset(), not here

    def _setup_world(self):
        p.resetSimulation(physicsClientId=self._client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client)
        p.setGravity(0, 0, self._gravity_z, physicsClientId=self._client)
        p.setTimeStep(self._time_step, physicsClientId=self._client)

        seed = self.level_seed if self.level_seed is not None else int(
            self.np_random.integers(0, 9999))
        gen = LevelGenerator(self._client, seed=seed)
        self._level_ids, self._start_pos, self._goal_pos = gen.build()

        spawn_pos = self._start_pos.copy()
        spawn_pos[2] = 1.4  # spawn 1.4m above platform (feet will be above platform)
        spawn_orn = p.getQuaternionFromEuler([0, 0, 0])

        self._agent_id = p.loadURDF(
            self._urdf_path,
            basePosition=spawn_pos.tolist(),
            baseOrientation=spawn_orn,
            useFixedBase=False,
            flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS,
            physicsClientId=self._client)

        self._build_joint_map()
        self._configure_dynamics()

        if self.render_mode == "human":
            p.resetDebugVisualizerCamera(
                cameraDistance=self._camera_distance,
                cameraYaw=self._camera_yaw,
                cameraPitch=self._camera_pitch,
                cameraTargetPosition=spawn_pos.tolist(),
                physicsClientId=self._client)

        initial_pose = np.zeros(N_JOINTS)
        
        for i, ji in enumerate(self._joint_indices):
            lo, hi = self._joint_limits[i]
            mid = (lo + hi) / 2.0
            span = (hi - lo) / 2.0
            target_angle = mid + initial_pose[i] * span
            p.resetJointState(self._agent_id, ji, targetValue=target_angle, physicsClientId=self._client)

        for _ in range(20):
            p.stepSimulation(physicsClientId=self._client)
        if self.render_mode == "human":
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self._client)

    def _build_joint_map(self):
        """Map joint names from _JOINT_CFG to PyBullet joint indices."""
        num_joints = p.getNumJoints(self._agent_id, physicsClientId=self._client)

        name_to_idx = {}
        for ji in range(num_joints):
            info = p.getJointInfo(self._agent_id, ji, physicsClientId=self._client)
            name_to_idx[info[1].decode()] = ji

        self._joint_indices = []
        self._joint_limits  = []
        self._joint_torques = []
        for jname, lo, hi, torq in _JOINT_CFG:
            idx = name_to_idx[jname]
            self._joint_indices.append(idx)
            self._joint_limits.append((lo, hi))
            self._joint_torques.append(torq)

        # Foot links
        self._foot_link_ids = {}
        for fname in _FOOT_LINKS:
            if fname in name_to_idx:
                self._foot_link_ids[fname] = name_to_idx[fname]
            else:
                # foot_L / foot_R are links that come right after their ankle joints
                for jname_check in ("left_ankle", "right_ankle"):
                    key = "foot_L" if "left" in jname_check else "foot_R"
                    if key not in self._foot_link_ids and jname_check in name_to_idx:
                        # The foot link index = the joint index (child link)
                        self._foot_link_ids[key] = name_to_idx[jname_check]

    def _configure_dynamics(self):
        """Apply friction and damping to all links and feet."""
        num_joints = p.getNumJoints(self._agent_id, physicsClientId=self._client)
        foot_ids = set(self._foot_link_ids.values())
        
        for ji in range(-1, num_joints):
            # Give feet much higher friction so agent can push off the ground
            friction = 2.0 if ji in foot_ids else 0.9
            p.changeDynamics(self._agent_id, ji,
                             lateralFriction=friction,
                             spinningFriction=0.05,
                             rollingFriction=0.01,
                             restitution=0.0,
                             linearDamping=0.04,
                             angularDamping=0.1,
                             physicsClientId=self._client)

        # Extra friction on feet for better ground grip
        for link_idx in self._foot_link_ids.values():
            p.changeDynamics(self._agent_id, link_idx,
                             lateralFriction=1.4,
                             spinningFriction=0.1,
                             physicsClientId=self._client)

        # Disable the URDF's default velocity motors so _apply_action's PD
        # position control is the only actuator driving the joints
        for ji in self._joint_indices:
            p.setJointMotorControl2(
                self._agent_id, ji,
                controlMode=p.VELOCITY_CONTROL,
                force=0,
                physicsClientId=self._client)

    # -----------------------------------------------------------------------
    # Moving platforms
    # -----------------------------------------------------------------------

    def _update_moving_platforms(self):
        time_sec = self._step_count * (1.0 / self._physics_hz)
        for body_id in self._level_ids:
            try:
                ud = p.getUserData(body_id, "mover", physicsClientId=self._client)
                if ud is None:
                    continue
                meta  = ud.decode("utf-8").split(",")
                axis  = meta[0]
                amp   = float(meta[1])
                speed = float(meta[2])
                sx    = float(meta[3])
                sy    = float(meta[4])
                sz    = float(meta[5])
                offset = amp * math.sin(speed * time_sec)
                nx, ny, nz = sx, sy, sz
                if axis == "Y": ny += offset
                elif axis == "Z": nz += offset
                _, orn = p.getBasePositionAndOrientation(body_id, physicsClientId=self._client)
                p.resetBasePositionAndOrientation(body_id, [nx, ny, nz], orn,
                                                  physicsClientId=self._client)
            except Exception:
                pass

    # -----------------------------------------------------------------------
    # Observations
    # -----------------------------------------------------------------------

    def _get_torso_pos(self) -> np.ndarray:
        pos, _ = p.getBasePositionAndOrientation(self._agent_id,
                                                  physicsClientId=self._client)
        return np.array(pos, dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        # --- Torso state ---
        pos, orn = p.getBasePositionAndOrientation(self._agent_id,
                                                    physicsClientId=self._client)
        vel, ang = p.getBaseVelocity(self._agent_id, physicsClientId=self._client)
        euler    = p.getEulerFromQuaternion(orn)

        pos_arr  = np.array(pos,   dtype=np.float32)
        vel_arr  = np.array(vel,   dtype=np.float32)
        orn_arr  = np.array(euler, dtype=np.float32)
        ang_arr  = np.array(ang,   dtype=np.float32)

        # --- Joint states ---
        j_angles = []
        j_vels   = []
        for i, ji in enumerate(self._joint_indices):
            jstate = p.getJointState(self._agent_id, ji, physicsClientId=self._client)
            lo, hi = self._joint_limits[i]
            mid    = (lo + hi) / 2.0
            span   = (hi - lo) / 2.0
            norm_angle = (jstate[0] - mid) / (span + 1e-8)   # -1..1
            j_angles.append(np.clip(norm_angle, -1.0, 1.0))
            j_vels.append(np.clip(jstate[1], -20.0, 20.0))

        j_angles_arr = np.array(j_angles, dtype=np.float32)
        j_vels_arr   = np.array(j_vels,   dtype=np.float32)

        # --- Raycasts ---
        rays = self._cast_rays(pos, euler)

        # --- Goal direction ---
        goal_vec = self._goal_pos - pos_arr
        goal_dir = goal_vec / (np.linalg.norm(goal_vec) + 1e-8)

        # --- Foot contacts ---
        foot_contacts = self._get_foot_contacts()
        lf = np.float32(1.0 if foot_contacts[0] else 0.0)
        rf = np.float32(1.0 if foot_contacts[1] else 0.0)

        # --- Height and time ---
        height = np.float32(pos[2])  # torso Z
        time_frac = np.float32(1.0 - self._step_count / self.MAX_STEPS)

        obs = np.concatenate([
            pos_arr, vel_arr, orn_arr, ang_arr,   # 12
            j_angles_arr, j_vels_arr,              # 20
            rays,                                  # 5
            goal_dir.astype(np.float32),           # 3
            [lf, rf, height, time_frac],           # 4
        ]).astype(np.float32)

        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _cast_rays(self, pos, euler) -> np.ndarray:
        """5 rays fanned downward from torso to probe ground ahead."""
        yaw = euler[2]
        ray_origin = np.array([pos[0], pos[1], pos[2] - 0.1], dtype=float)
        distances = []
        for pitch_deg in self.RAY_PITCHES:
            pitch = math.radians(pitch_deg)
            dx = math.cos(yaw) * math.cos(pitch)
            dy = math.sin(yaw) * math.cos(pitch)
            dz = -math.sin(pitch)
            ray_to = ray_origin + np.array([dx, dy, dz]) * self.RAY_MAX_DIST
            res = p.rayTest(ray_origin.tolist(), ray_to.tolist(),
                            physicsClientId=self._client)[0]
            distances.append(float(res[2]))   # hit fraction 0..1
        return np.array(distances, dtype=np.float32)

    def _get_foot_contacts(self):
        """Return (left_foot_contact, right_foot_contact) as booleans."""
        contacts = p.getContactPoints(bodyA=self._agent_id,
                                       physicsClientId=self._client)
        left  = False
        right = False
        lf_idx = self._foot_link_ids.get("foot_L", -99)
        rf_idx = self._foot_link_ids.get("foot_R", -99)
        for c in (contacts or []):
            link = c[3]   # linkIndexA
            if link == lf_idx:
                left  = True
            if link == rf_idx:
                right = True
        return left, right

    def _is_fallen(self) -> bool:
        """
        Return True if the humanoid has fallen.

        Strategy (standard for PyBullet humanoid envs):
        - The torso (base link = -1) or any upper-body link touching ANY surface = fallen.
        - Thighs/shins touching the ground also count as fallen.
        - Only foot links are allowed to touch the ground normally.
        - Also returns True if agent falls into the void (Z < -2).
        """
        # Check if fallen into void
        pos, _ = p.getBasePositionAndOrientation(self._agent_id, physicsClientId=self._client)
        if pos[2] < -2.0:  # fell off the map
            return True
        
        allowed_links = set(self._foot_link_ids.values())  # feet are OK contacts
        contacts = p.getContactPoints(bodyA=self._agent_id,
                                       physicsClientId=self._client)
        for c in (contacts or []):
            link_a = c[3]   # linkIndexA (-1 = base/torso)
            body_b = c[2]   # the OTHER body (ground, platform, etc.)
            # Ignore self-collisions (both bodies are the agent)
            if body_b == self._agent_id:
                continue
            if link_a not in allowed_links:
                return True   # torso, thigh, shin, arm touching ground
        return False

    # -----------------------------------------------------------------------
    # Actions
    # -----------------------------------------------------------------------

    def _apply_action(self, action: np.ndarray):
        """Apply joint targets mapped from action [-1, 1] using Position Control (PD)."""
        for i, ji in enumerate(self._joint_indices):
            lo, hi = self._joint_limits[i]
            # Map action [-1, 1] linearly to [lo, hi]
            target_pos = float(lo + (action[i] + 1.0) * 0.5 * (hi - lo))
            # Clip safely just in case
            target_pos = float(np.clip(target_pos, lo, hi))
            
            p.setJointMotorControl2(
                self._agent_id, ji,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_pos,
                force=self._joint_torques[i],   # Max torque the PD controller can use
                maxVelocity=10.0,
                physicsClientId=self._client)

    def _apply_lateral_spring(self):
        """Invisible soft spring on Y axis, keeps agent from drifting sideways."""
        pos, _ = p.getBasePositionAndOrientation(self._agent_id,
                                                  physicsClientId=self._client)
        y_offset = float(pos[1]) - self._spawn_y
        spring_f = -self.LATERAL_SPRING_K * y_offset
        spring_f = float(np.clip(spring_f, -self.LATERAL_SPRING_MAX,
                                            self.LATERAL_SPRING_MAX))
        p.applyExternalForce(
            self._agent_id, -1,
            [0.0, spring_f, 0.0], [0, 0, 0], p.WORLD_FRAME,
            physicsClientId=self._client)

    # -----------------------------------------------------------------------
    # Reward
    # -----------------------------------------------------------------------

    def _compute_reward(self, action: np.ndarray):
        reward_cfg = self.reward_config
        pos, orn = p.getBasePositionAndOrientation(self._agent_id,
                                                    physicsClientId=self._client)
        vel, _   = p.getBaseVelocity(self._agent_id, physicsClientId=self._client)
        euler    = p.getEulerFromQuaternion(orn)

        pos_arr  = np.array(pos, dtype=np.float32)
        dist     = float(np.linalg.norm(pos_arr - self._goal_pos))

        terminated = False
        truncated  = False

        # --- Base time penalty ---
        reward = float(reward_cfg.get("step_penalty", self.STEP_PENALTY))

        # --- Forward progress ---
        delta = self._prev_dist_to_goal - dist
        reward += delta * float(reward_cfg.get("progress_scale", self.PROGRESS_SCALE))
        self._prev_dist_to_goal = dist

        # --- Forward speed reward (only if making progress) ---
        forward_vel = max(0.0, float(vel[0]))
        if delta > 0.01:  # only reward speed when getting closer to goal
            reward += forward_vel * float(reward_cfg.get("speed_scale", self.SPEED_SCALE))

        # --- Posture reward (penalise leaning) ---
        roll, pitch = euler[0], euler[1]
        reward -= (abs(roll) + abs(pitch)) * float(reward_cfg.get("upright_scale", self.UPRIGHT_SCALE))

        # --- Lateral stability penalties ---
        vy = float(vel[1])
        y_drift = float(pos[1]) - self._spawn_y
        reward -= abs(vy) * float(reward_cfg.get("lateral_vel_scale", self.LATERAL_VEL_SCALE))
        reward -= abs(y_drift) * float(reward_cfg.get("lateral_pos_scale", self.LATERAL_POS_SCALE))

        # --- Energy penalty ---
        reward -= float(np.sum(action ** 2)) * float(reward_cfg.get("energy_scale", self.ENERGY_SCALE))

        # --- Goal reached ---
        if dist < 1.5:
            reward += float(reward_cfg.get("goal_reward", self.GOAL_REWARD))
            terminated = True

        # --- Fallen: torso or non-foot links contact any surface ---
        if self._is_fallen():
            reward += float(reward_cfg.get("fall_penalty", self.FALL_PENALTY))
            terminated = True

        # --- Timeout ---
        if self._step_count >= self.MAX_STEPS:
            truncated = True

        # --- Stagnation Cutoff ---
        # If the agent hasn't moved at least 0.5m forward after 150 steps, end the episode.
        # This prevents the agent from farming standing safe-states.
        if self._step_count > 150:
            dx = float(pos_arr[0] - self._start_pos[0])
            if dx < 0.5:
                truncated = True

        return reward, terminated, truncated
