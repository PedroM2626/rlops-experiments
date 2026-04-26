"""
Race Gymnasium environment - Competitive Multi-Agent Humanoid Race.

Multiple humanoid agents (ragdoll physics) compete to reach a goal as fast
as possible without falling. Agents interact physically with each other and
must navigate procedurally generated terrain with obstacles.

Coordinate convention: PyBullet Z-up
  X = forward (race direction)
  Y = lateral
  Z = up

Observation space per agent (56-dim):
  [0:3]   torso position (x, y, z)
  [3:6]   torso linear velocity
  [6:9]   torso orientation euler (roll, pitch, yaw)
  [9:12]  torso angular velocity
  [12:22] 10 joint angles (normalized -1..1 within joint limits)
  [22:32] 10 joint velocities (clipped +-20)
  [32:37] 5 downward raycasts (hit fraction 0..1)
  [37:40] unit vector toward goal
  [40]    left foot contact (0/1)
  [41]    right foot contact (0/1)
  [42]    torso height (z)
  [43]    time remaining fraction
  [44:47] closest opponent relative position (normalized)
  [47:50] 2nd closest opponent relative position (normalized)
  [50]    race rank normalized (0=1st, 1=last)
  [51]    pushed flag (being contacted by opponent on torso/arms)
  [52:54] padding zeros

Action space (10-dim continuous, clipped [-1, 1]):
  Joint position targets mapped to joint limits.
  Order: [l_shoulder, l_elbow, r_shoulder, r_elbow,
          l_hip, l_knee, l_ankle, r_hip, r_knee, r_ankle]
"""

import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces

from src.env.terrain_generator import TerrainGenerator, CORRIDOR_LENGTH, CORRIDOR_WIDTH


# ---------------------------------------------------------------------------
# Joint configuration
# ---------------------------------------------------------------------------
_JOINT_CFG = [
    ("left_shoulder",  -1.57,  1.57,  15.0),
    ("left_elbow",      0.00,  2.27,  10.0),
    ("right_shoulder", -1.57,  1.57,  15.0),
    ("right_elbow",     0.00,  2.27,  10.0),
    ("left_hip",       -1.05,  1.05,  30.0),
    ("left_knee",       0.00,  2.09,  25.0),
    ("left_ankle",     -0.70,  0.70,  15.0),
    ("right_hip",      -1.05,  1.05,  30.0),
    ("right_knee",      0.00,  2.09,  25.0),
    ("right_ankle",    -0.70,  0.70,  15.0),
]
N_JOINTS = len(_JOINT_CFG)   # 10
OBS_DIM  = 54

_FOOT_LINKS = ("foot_L", "foot_R")

# Per-agent colors: blue, red, green, orange, purple, cyan
_AGENT_COLORS = [
    [0.10, 0.40, 0.95, 1.0],   # blue
    [0.95, 0.15, 0.15, 1.0],   # red
    [0.10, 0.80, 0.20, 1.0],   # green
    [0.95, 0.55, 0.05, 1.0],   # orange
    [0.65, 0.10, 0.90, 1.0],   # purple
    [0.05, 0.85, 0.95, 1.0],   # cyan
]

# Spawn Y offsets (spread agents across corridor width at start)
def _spawn_y_offsets(n: int) -> List[float]:
    if n == 1:
        return [0.0]
    step = (CORRIDOR_WIDTH * 0.8) / (n - 1)
    start = -CORRIDOR_WIDTH * 0.4
    return [start + i * step for i in range(n)]


class RaceEnv(gym.Env):
    """
    Multi-agent competitive race environment.

    When N_AGENTS > 1, this env operates in rotating-agent mode for SB3
    compatibility: each call to step()/reset() controls exactly ONE agent
    (self._active_agent), cycling through 0..N_AGENTS-1.

    For true parallel training, use RaceVecWrapper which creates one
    RaceEnv per parallel instance and assigns each a fixed agent slot.
    """

    metadata = {"render_modes": ["human", "direct"], "render_fps": 60}

    # Physics
    GRAVITY    = -9.81
    TIME_STEP  = 1.0 / 480.0
    FRAME_SKIP = 8              # policy at 60 Hz

    # Episode
    MAX_STEPS = 900  # 15 seconds at 60 Hz

    # Reward weights
    PROGRESS_SCALE   = 6.0
    SPEED_SCALE      = 0.4
    UPRIGHT_SCALE    = 0.3
    ENERGY_SCALE     = 0.001
    STEP_PENALTY     = -0.01
    FALL_PENALTY     = -8.0
    GOAL_BONUS_RANK  = [200.0, 120.0, 60.0, 20.0]  # per finishing rank (0-indexed)
    RANK_BONUS_STEP  = 0.03    # per-step bonus for leading
    COLLISION_PENALTY = 0.05   # per-step penalty for being pushed hard
    ALIVE_BONUS      = 0.005

    # Lateral soft spring (per agent, keeps them in corridor)
    LATERAL_SPRING_K   = 60.0
    LATERAL_SPRING_MAX = 180.0

    # Fall detection
    FALL_VOID_Z = -2.0

    # Raycasts
    RAY_PITCHES  = [15, 30, 45, 60, 75]
    RAY_MAX_DIST = 5.0

    # Goal reach radius
    GOAL_RADIUS = 2.0

    def __init__(
        self,
        render_mode: str = "direct",
        n_agents: int = 4,
        level_seed: Optional[int] = None,
        agent_index: int = 0,         # which agent this env-wrapper controls
        reward_config: Optional[dict] = None,
        world_config: Optional[dict] = None,
    ):
        super().__init__()
        self.render_mode   = render_mode
        self.n_agents      = max(1, n_agents)
        self.level_seed    = level_seed
        self.agent_index   = agent_index   # index of the agent this wrapper drives
        self.reward_config = reward_config or {}
        self.world_config  = world_config  or {}

        # Spaces
        obs_low  = np.full(OBS_DIM, -np.inf, dtype=np.float32)
        obs_high = np.full(OBS_DIM,  np.inf, dtype=np.float32)
        obs_low[32:37]  = 0.0;  obs_high[32:37]  = 1.0   # raycasts
        obs_low[37:40]  = -1.0; obs_high[37:40]  = 1.0   # goal dir
        obs_low[40:42]  = 0.0;  obs_high[40:42]  = 1.0   # foot contacts
        obs_low[43]     = 0.0;  obs_high[43]     = 1.0   # time remaining
        obs_low[50]     = 0.0;  obs_high[50]     = 1.0   # rank
        obs_low[51]     = 0.0;  obs_high[51]     = 1.0   # pushed flag
        # [52:54] padding - unconstrained
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, shape=(N_JOINTS,), dtype=np.float32)

        # Internal state (shared across all agent slots in one physics world)
        self._client: Optional[int]     = None
        self._agent_ids: List[int]      = []     # PyBullet body id per agent
        self._joint_indices: List[List[int]]   = []
        self._joint_limits: List[List[Tuple]]  = []
        self._joint_torques: List[List[float]] = []
        self._foot_link_ids: List[Dict]        = []
        self._terrain_ids: List[int]    = []
        self._goal_pos: np.ndarray      = np.zeros(3, dtype=np.float32)
        self._start_pos: np.ndarray     = np.zeros(3, dtype=np.float32)
        self._spawn_ys: List[float]     = []

        self._step_count: int = 0
        self._prev_dist: List[float]    = []     # dist to goal per agent
        self._finished: List[bool]      = []     # finished the race?
        self._fallen: List[bool]        = []     # fallen and eliminated?
        self._finish_rank: List[int]    = []     # finishing order (rank given when crossing goal)
        self._next_rank: int = 0                 # next rank to assign

        self._urdf_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "assets", "humanoid.urdf"))

        self._apply_world_config()

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _apply_world_config(self):
        wc = self.world_config
        self._gravity_z      = float(wc.get("gravity_z",        self.GRAVITY))
        self._time_step      = float(wc.get("time_step",         self.TIME_STEP))
        self._frame_skip     = max(1, int(wc.get("frame_skip",   self.FRAME_SKIP)))
        self._cam_dist       = float(wc.get("camera_distance",   12.0))
        self._cam_yaw        = float(wc.get("camera_yaw",        30.0))
        self._cam_pitch      = float(wc.get("camera_pitch",      -18.0))
        self._show_gui       = bool(wc.get("show_gui_panels",    False))
        self._show_shadows   = bool(wc.get("show_shadows",       False))

    def set_runtime_config(self, reward_config=None, world_config=None):
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
        c = self._client
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,     int(self._show_gui),     physicsClientId=c)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, int(self._show_shadows), physicsClientId=c)
        for attr in ("COV_ENABLE_RGB_BUFFER_PREVIEW",
                     "COV_ENABLE_DEPTH_BUFFER_PREVIEW",
                     "COV_ENABLE_SEGMENTATION_MARK_PREVIEW"):
            flag = getattr(p, attr, None)
            if flag is not None:
                p.configureDebugVisualizer(flag, 0, physicsClientId=c)

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Connect to physics server (once)
        if self._client is None:
            if self.render_mode == "human":
                self._client = p.connect(p.GUI)
                self._configure_visualizer()
            else:
                self._client = p.connect(p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                      physicsClientId=self._client)

        if self.render_mode == "human":
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0,
                                       physicsClientId=self._client)

        self._setup_world()

        self._step_count  = 0
        self._next_rank   = 0
        self._prev_dist   = [float(np.linalg.norm(self._get_torso_pos(i) - self._goal_pos))
                              for i in range(self.n_agents)]
        self._finished    = [False] * self.n_agents
        self._fallen      = [False] * self.n_agents
        self._finish_rank = [-1] * self.n_agents

        if self.render_mode == "human":
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1,
                                       physicsClientId=self._client)

        obs = self._get_obs(self.agent_index)
        return obs, {}

    def step(self, action: np.ndarray):
        """
        Advance the ENTIRE world by one policy step.
        Action is applied to self.agent_index only.
        All other agents use zero action (they have their own env wrapper in training).
        """
        action = np.clip(action, -1.0, 1.0)

        # Apply action to controlled agent
        self._apply_action(self.agent_index, action)

        # Step physics
        for _ in range(self._frame_skip):
            # Apply continuous forces BEFORE each substep
            for i in range(self.n_agents):
                if self._finished[i]:
                    # Freeze finished agents by counteracting gravity
                    p.applyExternalForce(self._agent_ids[i], -1, [0, 0, 700.0], [0, 0, 0], p.WORLD_FRAME, physicsClientId=self._client)
                    continue
                self._apply_stability_forces(i)
                if not self._fallen[i]:
                    self._apply_lateral_spring(i)
            p.stepSimulation(physicsClientId=self._client)
            # Enforce hard constraints AFTER each substep (ALL agents, always)
            for i in range(self.n_agents):
                if self._finished[i]:
                    # Freeze finished agents' velocity
                    p.resetBaseVelocity(self._agent_ids[i], [0, 0, 0], [0, 0, 0], physicsClientId=self._client)
                    continue
                self._enforce_stability_constraints(i)

        self._step_count += 1

        # Update fall & finish states for all agents
        for i in range(self.n_agents):
            if not self._fallen[i] and not self._finished[i]:
                if self._check_fallen(i):
                    self._fallen[i] = True
                elif self._check_goal(i):
                    self._finished[i] = True
                    self._finish_rank[i] = self._next_rank
                    self._next_rank += 1

        # Camera follows leader and delay to match 60Hz real-time rendering
        if self.render_mode == "human":
            self._update_camera()
            time.sleep(1.0 / 60.0)

        obs        = self._get_obs(self.agent_index)
        reward, terminated, truncated = self._compute_reward(self.agent_index, action)

        info = {
            "race_rank":    self._get_rank(self.agent_index),
            "dist_to_goal": self._prev_dist[self.agent_index],
            "fallen":       self._fallen[self.agent_index],
            "finished":     self._finished[self.agent_index],
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        if self._client is not None:
            try:
                p.disconnect(physicsClientId=self._client)
            except Exception:
                pass
            self._client = None

    def render(self):
        pass

    # ------------------------------------------------------------------
    # World setup
    # ------------------------------------------------------------------

    def _setup_world(self):
        p.resetSimulation(physicsClientId=self._client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client)
        p.setGravity(0, 0, self._gravity_z, physicsClientId=self._client)
        p.setTimeStep(self._time_step, physicsClientId=self._client)

        seed = self.level_seed if self.level_seed is not None else int(
            self.np_random.integers(0, 99999))
        gen = TerrainGenerator(self._client, seed=seed)
        self._terrain_ids, self._start_pos, self._goal_pos = gen.build()

        self._spawn_ys = _spawn_y_offsets(self.n_agents)

        self._agent_ids      = []
        self._joint_indices  = []
        self._joint_limits   = []
        self._joint_torques  = []
        self._foot_link_ids  = []

        for i in range(self.n_agents):
            agent_id = self._spawn_agent(i)
            self._agent_ids.append(agent_id)
            ji, jl, jt, fl = self._build_joint_map(agent_id)
            self._joint_indices.append(ji)
            self._joint_limits.append(jl)
            self._joint_torques.append(jt)
            self._foot_link_ids.append(fl)
            self._configure_dynamics(i)
            self._colorize_agent(i)

        # Settle physics
        for _ in range(30):
            p.stepSimulation(physicsClientId=self._client)

        if self.render_mode == "human":
            p.resetDebugVisualizerCamera(
                cameraDistance=self._cam_dist,
                cameraYaw=self._cam_yaw,
                cameraPitch=self._cam_pitch,
                cameraTargetPosition=self._start_pos.tolist(),
                physicsClientId=self._client)

    def _spawn_agent(self, i: int) -> int:
        spawn_x = float(self._start_pos[0]) + float(self.np_random.uniform(-0.5, 0.5))
        spawn_y = self._spawn_ys[i]
        spawn_z = 1.5   # above ground, feet will settle

        orn = p.getQuaternionFromEuler([0, 0, 0])
        agent_id = p.loadURDF(
            self._urdf_path,
            basePosition=[spawn_x, spawn_y, spawn_z],
            baseOrientation=orn,
            useFixedBase=False,
            flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS,
            physicsClientId=self._client)
        return agent_id

    def _build_joint_map(self, agent_id: int):
        num_joints = p.getNumJoints(agent_id, physicsClientId=self._client)
        name_to_idx = {}
        for ji in range(num_joints):
            info = p.getJointInfo(agent_id, ji, physicsClientId=self._client)
            name_to_idx[info[1].decode()] = ji

        joint_indices = []
        joint_limits  = []
        joint_torques = []
        for jname, lo, hi, torq in _JOINT_CFG:
            idx = name_to_idx[jname]
            joint_indices.append(idx)
            joint_limits.append((lo, hi))
            joint_torques.append(torq)

        foot_link_ids = {}
        for fname in _FOOT_LINKS:
            if fname in name_to_idx:
                foot_link_ids[fname] = name_to_idx[fname]
        for jname_check in ("left_ankle", "right_ankle"):
            key = "foot_L" if "left" in jname_check else "foot_R"
            if key not in foot_link_ids and jname_check in name_to_idx:
                foot_link_ids[key] = name_to_idx[jname_check]

        # Disable default velocity motors
        for ji in joint_indices:
            p.setJointMotorControl2(
                agent_id, ji, controlMode=p.VELOCITY_CONTROL,
                force=0, physicsClientId=self._client)

        return joint_indices, joint_limits, joint_torques, foot_link_ids

    def _configure_dynamics(self, i: int):
        agent_id  = self._agent_ids[i]
        foot_ids  = set(self._foot_link_ids[i].values())
        num_joints = p.getNumJoints(agent_id, physicsClientId=self._client)
        for ji in range(-1, num_joints):
            friction = 2.0 if ji in foot_ids else 0.9
            p.changeDynamics(agent_id, ji,
                             lateralFriction=friction,
                             spinningFriction=0.05,
                             rollingFriction=0.01,
                             restitution=0.0,
                             linearDamping=0.04,
                             angularDamping=0.1,
                             physicsClientId=self._client)
        for link_idx in foot_ids:
            p.changeDynamics(agent_id, link_idx,
                             lateralFriction=1.4,
                             spinningFriction=0.1,
                             physicsClientId=self._client)

    def _colorize_agent(self, i: int):
        color = _AGENT_COLORS[i % len(_AGENT_COLORS)]
        agent_id = self._agent_ids[i]
        num_joints = p.getNumJoints(agent_id, physicsClientId=self._client)
        for link in range(-1, num_joints):
            p.changeVisualShape(agent_id, link,
                                rgbaColor=color,
                                physicsClientId=self._client)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_torso_pos(self, i: int) -> np.ndarray:
        pos, _ = p.getBasePositionAndOrientation(self._agent_ids[i],
                                                  physicsClientId=self._client)
        return np.array(pos, dtype=np.float32)

    def _get_obs(self, i: int) -> np.ndarray:
        agent_id = self._agent_ids[i]
        pos, orn = p.getBasePositionAndOrientation(agent_id, physicsClientId=self._client)
        vel, ang = p.getBaseVelocity(agent_id, physicsClientId=self._client)
        euler    = p.getEulerFromQuaternion(orn)

        pos_arr = np.array(pos,   dtype=np.float32)
        vel_arr = np.array(vel,   dtype=np.float32)
        orn_arr = np.array(euler, dtype=np.float32)
        ang_arr = np.array(ang,   dtype=np.float32)

        # Joint states
        j_angles, j_vels = [], []
        for k, ji in enumerate(self._joint_indices[i]):
            jstate = p.getJointState(agent_id, ji, physicsClientId=self._client)
            lo, hi = self._joint_limits[i][k]
            mid    = (lo + hi) / 2.0
            span   = (hi - lo) / 2.0 + 1e-8
            j_angles.append(float(np.clip((jstate[0] - mid) / span, -1.0, 1.0)))
            j_vels.append(float(np.clip(jstate[1], -20.0, 20.0)))

        # Raycasts
        rays = self._cast_rays(pos, euler)

        # Goal direction
        goal_vec = self._goal_pos - pos_arr
        goal_dir = goal_vec / (np.linalg.norm(goal_vec) + 1e-8)

        # Foot contacts
        lf, rf = self._get_foot_contacts(i)

        # Height
        height = float(pos[2])

        # Time remaining
        time_frac = float(1.0 - self._step_count / self.MAX_STEPS)

        # ---- Competitive observations ----
        # Relative positions of opponents (sorted by distance)
        opp_positions = []
        for j in range(self.n_agents):
            if j != i:
                opp_pos = self._get_torso_pos(j)
                rel     = (opp_pos - pos_arr) / 20.0   # normalize by 20 m
                opp_positions.append((np.linalg.norm(rel), rel))
        opp_positions.sort(key=lambda x: x[0])

        opp1 = opp_positions[0][1] if len(opp_positions) > 0 else np.zeros(3, dtype=np.float32)
        opp2 = opp_positions[1][1] if len(opp_positions) > 1 else np.zeros(3, dtype=np.float32)

        # Race rank
        rank_norm = float(self._get_rank(i)) / max(self.n_agents - 1, 1)

        # Am I being pushed?
        pushed = float(self._is_being_pushed(i))

        obs = np.concatenate([
            pos_arr, vel_arr, orn_arr, ang_arr,          # 12
            np.array(j_angles, dtype=np.float32),         # 10
            np.array(j_vels,   dtype=np.float32),         # 10
            rays,                                          # 5
            goal_dir.astype(np.float32),                  # 3
            [float(lf), float(rf), height, time_frac],   # 4
            opp1.astype(np.float32),                      # 3
            opp2.astype(np.float32),                      # 3
            [rank_norm, pushed, 0.0, 0.0],                # 4
        ]).astype(np.float32)

        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _cast_rays(self, pos, euler) -> np.ndarray:
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
            distances.append(float(res[2]))
        return np.array(distances, dtype=np.float32)

    def _get_foot_contacts(self, i: int) -> Tuple[bool, bool]:
        contacts = p.getContactPoints(bodyA=self._agent_ids[i],
                                       physicsClientId=self._client)
        lf_idx = self._foot_link_ids[i].get("foot_L", -99)
        rf_idx = self._foot_link_ids[i].get("foot_R", -99)
        left = right = False
        for c in (contacts or []):
            link = c[3]
            if link == lf_idx:
                left  = True
            if link == rf_idx:
                right = True
        return left, right

    def _is_being_pushed(self, i: int) -> bool:
        """True if any opponent touches torso/arm/upper-body links."""
        contacts = p.getContactPoints(bodyA=self._agent_ids[i],
                                       physicsClientId=self._client)
        foot_ids = set(self._foot_link_ids[i].values())
        opponent_ids = set(self._agent_ids) - {self._agent_ids[i]}
        for c in (contacts or []):
            body_b = c[2]     # other body
            link_a = c[3]     # our link (-1 = torso)
            if body_b in opponent_ids and link_a not in foot_ids:
                return True
        return False

    # ------------------------------------------------------------------
    # Fall / goal detection
    # ------------------------------------------------------------------

    def _check_fallen(self, i: int) -> bool:
        pos, _ = p.getBasePositionAndOrientation(self._agent_ids[i],
                                                  physicsClientId=self._client)
        if pos[2] < self.FALL_VOID_Z:
            return True
        # Removed the body contact fall check to allow agents to recover/continue
        return False

    def _check_goal(self, i: int) -> bool:
        pos = self._get_torso_pos(i)
        return float(np.linalg.norm(pos - self._goal_pos)) < self.GOAL_RADIUS

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------

    def _apply_action(self, i: int, action: np.ndarray):
        agent_id = self._agent_ids[i]
        if self._finished[i] or self._fallen[i]:
            # Stop applying actions if finished or fallen
            return

        for k, ji in enumerate(self._joint_indices[i]):
            lo, hi = self._joint_limits[i][k]
            target = float(lo + (action[k] + 1.0) * 0.5 * (hi - lo))
            target = float(np.clip(target, lo, hi))
            p.setJointMotorControl2(
                agent_id, ji,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target,
                force=self._joint_torques[i][k],
                maxVelocity=3.0,
                physicsClientId=self._client)

    def _apply_lateral_spring(self, i: int):
        """Soft spring keeping agent within corridor bounds."""
        pos, _ = p.getBasePositionAndOrientation(self._agent_ids[i],
                                                  physicsClientId=self._client)
        y_offset = float(pos[1]) - self._spawn_ys[i]
        spring_f = float(np.clip(
            -self.LATERAL_SPRING_K * y_offset,
            -self.LATERAL_SPRING_MAX, self.LATERAL_SPRING_MAX))
        p.applyExternalForce(
            self._agent_ids[i], -1,
            [0.0, spring_f, 0.0], [0, 0, 0],
            p.WORLD_FRAME, physicsClientId=self._client)

    def _apply_stability_forces(self, i: int):
        """
        Torque-based upright stabilizer inspired by Unity Stabilizer.cs.
        Applied BEFORE each physics substep. Provides organic correction.
        """
        agent_id = self._agent_ids[i]
        pos, orn = p.getBasePositionAndOrientation(agent_id, physicsClientId=self._client)
        vel, ang_vel = p.getBaseVelocity(agent_id, physicsClientId=self._client)

        # --- 1. Upright torque ---
        rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        local_up = rot_matrix[:, 2]
        world_up = np.array([0.0, 0.0, 1.0])

        cross = np.cross(local_up, world_up)
        sin_angle = np.linalg.norm(cross)
        dot = float(np.clip(np.dot(local_up, world_up), -1.0, 1.0))
        angle = math.acos(dot)

        balance_pct = angle / math.pi
        torque_pct = min(1.0, balance_pct)
        upright_torque = 800.0
        torque_magnitude = torque_pct * upright_torque

        if sin_angle > 1e-6:
            torque_axis = cross / sin_angle
            torque_vec = torque_axis * torque_magnitude
            p.applyExternalTorque(
                agent_id, -1,
                torque_vec.tolist(),
                p.WORLD_FRAME, physicsClientId=self._client)

        # --- 2. Angular velocity damping ---
        ang_damp = 150.0
        damping_torque = [
            -ang_damp * ang_vel[0],
            -ang_damp * ang_vel[1],
            -ang_damp * ang_vel[2] * 0.3,
        ]
        p.applyExternalTorque(
            agent_id, -1,
            damping_torque,
            p.WORLD_FRAME, physicsClientId=self._client)

        # The organic upright torque and damping is sufficient for stability.
        # Artificial Z support (anti-gravity) has been removed so agents
        # must use their legs to bear their own weight on the ground.

    def _enforce_stability_constraints(self, i: int):
        """
        Hard constraints applied AFTER each physics substep.
        Clamps orientation and velocity to safe ranges.
        """
        agent_id = self._agent_ids[i]
        pos, orn = p.getBasePositionAndOrientation(agent_id, physicsClientId=self._client)
        vel, ang_vel = p.getBaseVelocity(agent_id, physicsClientId=self._client)

        euler = list(p.getEulerFromQuaternion(orn))
        max_tilt = math.radians(30.0)
        max_yaw = math.radians(20.0)
        needs_orn_reset = False

        # Clamp roll and pitch to max_tilt
        if abs(euler[0]) > max_tilt:
            euler[0] = max(-max_tilt, min(max_tilt, euler[0]))
            needs_orn_reset = True
        if abs(euler[1]) > max_tilt:
            euler[1] = max(-max_tilt, min(max_tilt, euler[1]))
            needs_orn_reset = True
        # Clamp yaw to max_yaw to prevent spinning horizontally
        if abs(euler[2]) > max_yaw:
            euler[2] = max(-max_yaw, min(max_yaw, euler[2]))
            needs_orn_reset = True

        if needs_orn_reset:
            new_orn = p.getQuaternionFromEuler(euler)
            p.resetBasePositionAndOrientation(
                agent_id, list(pos), new_orn, physicsClientId=self._client)
            # Preserve linear velocity but kill roll/pitch angular velocity
            p.resetBaseVelocity(
                agent_id, list(vel),
                [0.0, 0.0, ang_vel[2] * 0.8],
                physicsClientId=self._client)

        # Clamp vertical velocity to prevent excessive launching or meteor drops
        max_up_vel = 1.5
        max_down_vel = -5.0

        # Also clamp horizontal velocity so they don't move too fast
        max_xy_vel = 3.0

        new_v = list(vel)
        clamped = False
        
        # Z clamp
        if new_v[2] > max_up_vel:
            new_v[2] = max_up_vel
            clamped = True
        elif new_v[2] < max_down_vel:
            new_v[2] = max_down_vel
            clamped = True
            
        # XY clamp
        if abs(new_v[0]) > max_xy_vel:
            new_v[0] = max_xy_vel if new_v[0] > 0 else -max_xy_vel
            clamped = True
        if abs(new_v[1]) > max_xy_vel:
            new_v[1] = max_xy_vel if new_v[1] > 0 else -max_xy_vel
            clamped = True

        if clamped:
            cur_ang = list(p.getBaseVelocity(agent_id, physicsClientId=self._client)[1])
            p.resetBaseVelocity(
                agent_id,
                new_v,
                cur_ang,
                physicsClientId=self._client)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _get_rank(self, i: int) -> int:
        """
        Current race rank of agent i (0 = leading, n-1 = last).
        Based on distance to goal (lower = better).
        Finished agents always rank ahead of unfinished.
        Fallen agents rank last.
        """
        if self._finish_rank[i] >= 0:
            return self._finish_rank[i]
        if self._fallen[i]:
            return self.n_agents - 1

        dists = []
        for j in range(self.n_agents):
            if self._finish_rank[j] >= 0:
                dists.append(-1.0)  # finished = best distance
            elif self._fallen[j]:
                dists.append(float("inf"))
            else:
                dists.append(float(np.linalg.norm(
                    self._get_torso_pos(j) - self._goal_pos)))
        sorted_indices = sorted(range(self.n_agents), key=lambda j: dists[j])
        return sorted_indices.index(i)

    def _compute_reward(self, i: int, action: np.ndarray):
        rc  = self.reward_config
        pos = self._get_torso_pos(i)
        vel, _ = p.getBaseVelocity(self._agent_ids[i], physicsClientId=self._client)
        _, orn = p.getBasePositionAndOrientation(self._agent_ids[i],
                                                  physicsClientId=self._client)
        euler = p.getEulerFromQuaternion(orn)

        terminated = False
        truncated  = False

        dist = float(np.linalg.norm(pos - self._goal_pos))

        # Base step penalty
        reward = float(rc.get("step_penalty", self.STEP_PENALTY))

        # Alive bonus
        reward += float(rc.get("alive_bonus", self.ALIVE_BONUS))

        # Progress reward
        delta = self._prev_dist[i] - dist
        reward += delta * float(rc.get("progress_scale", self.PROGRESS_SCALE))
        self._prev_dist[i] = dist

        # Speed reward (when moving toward goal)
        if delta > 0.01:
            fwd_vel = max(0.0, float(vel[0]))
            reward += fwd_vel * float(rc.get("speed_scale", self.SPEED_SCALE))

        # Upright penalty
        roll, pitch = euler[0], euler[1]
        reward -= (abs(roll) + abs(pitch)) * float(rc.get("upright_scale", self.UPRIGHT_SCALE))

        # Energy penalty
        reward -= float(np.sum(action ** 2)) * float(rc.get("energy_scale", self.ENERGY_SCALE))

        # Competitive rank bonus (per step, for leading)
        rank = self._get_rank(i)
        if rank == 0 and not self._finished[i]:
            reward += float(rc.get("rank_bonus_step", self.RANK_BONUS_STEP))

        # Pushed/collision penalty
        if self._is_being_pushed(i):
            reward -= float(rc.get("collision_penalty", self.COLLISION_PENALTY))

        # Goal reached
        if self._finished[i] or self._check_goal(i):
            if not self._finished[i]:
                self._finished[i] = True
                self._finish_rank[i] = self._next_rank
                self._next_rank += 1
            rank_idx = min(self._finish_rank[i], len(self.GOAL_BONUS_RANK) - 1)
            reward += float(rc.get("goal_bonus", self.GOAL_BONUS_RANK[rank_idx]))
            terminated = True

        # Fallen
        if self._fallen[i] or self._check_fallen(i):
            self._fallen[i] = True
            reward += float(rc.get("fall_penalty", self.FALL_PENALTY))
            # Terminated removed: episode will not stop automatically when agents fall

        # Timeout
        if self._step_count >= self.MAX_STEPS:
            truncated = True

        # Stagnation cutoff: no progress after 200 steps
        if self._step_count > 200:
            dx = float(pos[0] - self._start_pos[0])
            if dx < 0.5:
                truncated = True

        return reward, terminated, truncated

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def _update_camera(self):
        """Camera follows the race leader."""
        leader_pos = None
        best_dist  = float("inf")
        for i in range(self.n_agents):
            if not self._fallen[i]:
                d = float(np.linalg.norm(self._get_torso_pos(i) - self._goal_pos))
                if d < best_dist:
                    best_dist  = d
                    leader_pos = self._get_torso_pos(i)
        if leader_pos is None:
            leader_pos = self._start_pos

        p.resetDebugVisualizerCamera(
            cameraDistance=self._cam_dist,
            cameraYaw=self._cam_yaw,
            cameraPitch=self._cam_pitch,
            cameraTargetPosition=leader_pos.tolist(),
            physicsClientId=self._client)
