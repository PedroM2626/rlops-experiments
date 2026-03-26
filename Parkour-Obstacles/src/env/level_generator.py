"""
Procedural parkour level generator - Z-up convention.

Coordinate system (same as PyBullet default):
  X = forward (direction of travel)
  Y = lateral (left/right)
  Z = up

The course runs along the +X axis. Platforms are horizontal boxes
resting on the XY plane at varying heights (Z). Gaps run along X.

Returns:
  platform_ids : list of PyBullet body IDs
  start_pos    : np.ndarray [x, y, z] center of start platform TOP surface
  goal_pos     : np.ndarray [x, y, z] center of goal platform TOP surface
"""

import math
from typing import List, Tuple

import numpy as np
import pybullet as p


# Visual colours  [R, G, B, A]
COL_START    = [0.20, 0.40, 0.90, 1.0]
COL_GOAL     = [0.10, 0.85, 0.25, 1.0]
COL_PLATFORM = [0.50, 0.50, 0.55, 1.0]
COL_NARROW   = [0.70, 0.65, 0.40, 1.0]
COL_RAMP     = [0.75, 0.45, 0.20, 1.0]
COL_MOVING   = [0.85, 0.20, 0.30, 1.0]

# Platform thickness (half-extent in Z)
PLAT_H = 0.25


class LevelGenerator:
    """
    Builds a parkour course made of BOX platforms along the +X axis.

    Section types:
      flat    - same-height platform, modest gap
      raised  - higher platform (step up)
      dropped - lower platform (step down)
      narrow  - very thin ledge
      ramp    - inclined surface connecting two heights
    """

    def __init__(self, client: int, seed: int = 42, num_sections: int = 10):
        self._client     = client
        self.rng         = np.random.default_rng(seed)
        self.num_sections = num_sections

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def build(self) -> Tuple[List[int], np.ndarray, np.ndarray]:
        ids: List[int] = []

        # ---- Starting platform (generous size, no gap) ----
        start_w  = 4.0   # Y width
        start_l  = 5.0   # X length
        start_z  = 0.0   # top surface Z

        ids.append(self._box(
            cx=start_l / 2, cy=0.0, top_z=start_z,
            half_x=start_l / 2, half_y=start_w / 2, half_z=PLAT_H,
            color=COL_START))

        start_pos = np.array([1.0, 0.0, start_z], dtype=np.float32)

        # cursor: X front-edge of last platform, current top Z
        cur_x = start_l
        cur_z = start_z

        # section type weights - easier sections more frequent
        types = self.rng.choice(
            ["flat", "flat", "flat", "raised", "raised", "dropped", "dropped", 
             "narrow", "ramp", "stairs", "stepping_stones", "moving", "moving"],
            size=self.num_sections)

        for stype in types:
            # Gap between platforms - much smaller for beginner agent
            gap = float(self.rng.uniform(0.1, 0.5))
            cur_x += gap

            # 20% chance of an "icy" platform (low friction)
            is_icy = self.rng.random() < 0.20
            color = [0.6, 0.8, 0.9, 1.0] if is_icy else None

            if stype in ("flat", "flat"):
                l = float(self.rng.uniform(3.0, 6.0))
                w = float(self.rng.uniform(2.0, 4.0))
                ids.append(self._box(
                    cx=cur_x + l / 2, cy=0.0, top_z=cur_z,
                    half_x=l / 2, half_y=w / 2, half_z=PLAT_H,
                    color=color, is_icy=is_icy))
                cur_x += l

            elif stype == "raised":
                dz = float(self.rng.uniform(0.2, 0.5))
                cur_z += dz
                l = float(self.rng.uniform(2.0, 4.0))
                w = float(self.rng.uniform(2.0, 3.5))
                ids.append(self._box(
                    cx=cur_x + l / 2, cy=0.0, top_z=cur_z,
                    half_x=l / 2, half_y=w / 2, half_z=PLAT_H,
                    color=color, is_icy=is_icy))
                cur_x += l

            elif stype == "dropped":
                dz = float(self.rng.uniform(0.15, 0.4))
                cur_z = max(0.0, cur_z - dz)
                l = float(self.rng.uniform(3.0, 6.0))
                w = float(self.rng.uniform(2.0, 4.0))
                ids.append(self._box(
                    cx=cur_x + l / 2, cy=0.0, top_z=cur_z,
                    half_x=l / 2, half_y=w / 2, half_z=PLAT_H,
                    color=color, is_icy=is_icy))
                cur_x += l

            elif stype == "narrow":
                l = float(self.rng.uniform(2.5, 4.0))
                w = float(self.rng.uniform(1.0, 1.5))
                cy = float(self.rng.uniform(-0.3, 0.3))
                c_narrow = color if is_icy else COL_NARROW
                ids.append(self._box(
                    cx=cur_x + l / 2, cy=cy, top_z=cur_z,
                    half_x=l / 2, half_y=w / 2, half_z=PLAT_H,
                    color=c_narrow, is_icy=is_icy))
                cur_x += l

            elif stype == "ramp":
                angle = float(self.rng.uniform(8, 15))
                l     = float(self.rng.uniform(2.0, 4.0))
                dz    = l * math.tan(math.radians(angle))
                ids += self._ramp(start_x=cur_x, start_z=cur_z,
                                  length=l, angle_deg=angle, is_icy=is_icy)
                cur_x += l
                cur_z += dz

            elif stype == "stairs":
                # Small steps going up
                n_steps = int(self.rng.integers(3, 6))
                step_height = float(self.rng.uniform(0.1, 0.2))
                step_length = float(self.rng.uniform(0.4, 0.7))
                step_width = float(self.rng.uniform(2.0, 3.0))
                for i in range(n_steps):
                    ids.append(self._box(
                        cx=cur_x + step_length / 2, cy=0.0, top_z=cur_z,
                        half_x=step_length / 2, half_y=step_width / 2, half_z=PLAT_H,
                        color=[0.6, 0.5, 0.4, 1.0], is_icy=False))
                    cur_x += step_length
                    cur_z += step_height

            elif stype == "stepping_stones":
                # Sequence of small blocks - easier gaps
                n_stones = int(self.rng.integers(3, 5))
                for i in range(n_stones):
                    stone_l = float(self.rng.uniform(0.8, 1.2))
                    stone_w = float(self.rng.uniform(1.2, 2.0))
                    # Offset cy slightly to make it zigzag
                    cy = float(self.rng.uniform(-0.5, 0.5))
                    
                    ids.append(self._box(
                        cx=cur_x + stone_l / 2, cy=cy, top_z=cur_z,
                        half_x=stone_l / 2, half_y=stone_w / 2, half_z=PLAT_H,
                        color=[0.9, 0.5, 0.2, 1.0], is_icy=False))
                    
                    cur_x += stone_l
                    if i < n_stones - 1:
                        # tiny gap between stones
                        cur_x += float(self.rng.uniform(0.15, 0.4))
            
            elif stype == "complex_slope":
                # Easier ramp up, small flat top, gap, platform
                angle = float(self.rng.uniform(10, 18))
                l_ramp = float(self.rng.uniform(2.0, 3.5))
                dz = l_ramp * math.tan(math.radians(angle))
                
                ids += self._ramp(start_x=cur_x, start_z=cur_z,
                                  length=l_ramp, angle_deg=angle, is_icy=False)
                cur_x += l_ramp
                cur_z += dz
                
                # small flat top
                l_top = 1.0
                ids.append(self._box(
                    cx=cur_x + l_top / 2, cy=0.0, top_z=cur_z,
                    half_x=l_top / 2, half_y=1.5, half_z=PLAT_H))
                cur_x += l_top + float(self.rng.uniform(0.8, 1.5)) # gap
                
                # landing
                l_land = 4.0
                ids.append(self._box(
                    cx=cur_x + l_land / 2, cy=0.0, top_z=cur_z,
                    half_x=l_land / 2, half_y=2.5, half_z=PLAT_H))
                cur_x += l_land

            elif stype == "moving":
                # Platform that moves left/right or up/down - easier settings
                l = float(self.rng.uniform(2.5, 3.5))
                w = float(self.rng.uniform(2.0, 3.0))
                cx = cur_x + l / 2
                body_id = self._box(
                    cx=cx, cy=0.0, top_z=cur_z,
                    half_x=l / 2, half_y=w / 2, half_z=PLAT_H,
                    color=COL_MOVING, is_icy=is_icy)
                ids.append(body_id)
                # Store movement metadata as user data in PyBullet
                axis = "Y" if self.rng.random() < 0.5 else "Z"
                amp = float(self.rng.uniform(0.5, 1.5))  # smaller movement
                speed = float(self.rng.uniform(0.8, 1.5))  # slower
                # Format: Axis, Amplitude, Speed, StartX, StartY, StartZ
                meta = f"{axis},{amp},{speed},{cx},0.0,{cur_z - PLAT_H}"
                p.addUserData(body_id, "mover", meta, physicsClientId=self._client)
                cur_x += l

        # ---- Goal platform ----
        goal_l = 5.0
        ids.append(self._box(
            cx=cur_x + goal_l / 2, cy=0.0, top_z=cur_z,
            half_x=goal_l / 2, half_y=3.0, half_z=PLAT_H,
            color=COL_GOAL))
        goal_pos = np.array([cur_x + goal_l / 2, 0.0, cur_z],
                             dtype=np.float32)

        # Golden sphere marker above goal
        vis = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.5,
            rgbaColor=[1.0, 0.85, 0.0, 0.9],
            physicsClientId=self._client)
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vis,
            basePosition=[goal_pos[0], goal_pos[1], goal_pos[2] + 1.0],
            physicsClientId=self._client)

        return ids, start_pos, goal_pos

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _box(self, cx: float, cy: float, top_z: float,
             half_x: float, half_y: float, half_z: float,
             color=None, is_icy: bool = False) -> int:
        """Create a static box. top_z is the Z of the TOP surface."""
        if color is None:
            color = COL_PLATFORM
        center_z = top_z - half_z   # center of box in Z

        col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[half_x, half_y, half_z],
            physicsClientId=self._client)
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[half_x, half_y, half_z],
            rgbaColor=color,
            physicsClientId=self._client)
        body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[cx, cy, center_z],
            physicsClientId=self._client)
            
        friction = 0.05 if is_icy else 1.0
        p.changeDynamics(body, -1,
                         lateralFriction=friction,
                         restitution=0.0,
                         physicsClientId=self._client)
        return body

    def _ramp(self, start_x: float, start_z: float,
              length: float, angle_deg: float, is_icy: bool = False) -> List[int]:
        """Inclined ramp rising along +X. Returns list with one body."""
        angle_rad = math.radians(angle_deg)
        half_l    = length / 2
        half_w    = 2.0
        half_h    = 0.25

        # Center of the ramp box in world coordinates
        cx = start_x + half_l * math.cos(angle_rad)
        cz = start_z + half_l * math.sin(angle_rad)

        # Rotation: tilt around Y axis so the top face inclines upward along X
        orn = p.getQuaternionFromEuler([0, -angle_rad, 0])

        col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[half_l, half_w, half_h],
            physicsClientId=self._client)
        
        color = [0.6, 0.8, 0.9, 1.0] if is_icy else COL_RAMP
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[half_l, half_w, half_h],
            rgbaColor=color,
            physicsClientId=self._client)
        body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[cx, 0.0, cz],
            baseOrientation=list(orn),
            physicsClientId=self._client)
            
        friction = 0.05 if is_icy else 0.9
        p.changeDynamics(body, -1,
                         lateralFriction=friction,
                         restitution=0.0,
                         physicsClientId=self._client)
        return [body]


# math import for helper functions
import math
