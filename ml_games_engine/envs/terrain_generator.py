"""
Procedural terrain generator for the Race environment.

Coordinate convention: PyBullet Z-up
  X = forward (direction of travel / race direction)
  Y = lateral (race corridor width)
  Z = up

The race corridor runs along the +X axis.
Generates a flat ground base with procedural obstacles:
  - Rolling hills (heightfield)
  - Rock/box obstacles
  - Wooden pillars
  - Low ramps and ramps
  - Shallow pits (negative bumps)

Returns:
    body_ids  : list of PyBullet body IDs for all terrain objects
    start_pos : np.ndarray [x, y, z] - where agents should spawn
    goal_pos  : np.ndarray [x, y, z] - where the goal is
"""

import math
import os
from typing import List, Tuple

import numpy as np
import pybullet as p


# Visual colours [R, G, B, A]
COL_GROUND    = [0.35, 0.55, 0.28, 1.0]
COL_ROCK      = [0.55, 0.52, 0.48, 1.0]
COL_PILLAR    = [0.65, 0.50, 0.35, 1.0]
COL_RAMP      = [0.75, 0.60, 0.35, 1.0]
COL_GOAL_RING = [0.95, 0.80, 0.10, 1.0]
COL_GOAL_PLAT = [0.15, 0.80, 0.30, 1.0]
COL_START     = [0.20, 0.35, 0.80, 1.0]

# Race corridor parameters
CORRIDOR_WIDTH  = 10.0   # Y half-width  (total width = 20 m)
CORRIDOR_LENGTH = 60.0   # X length of race
GROUND_THICK    = 0.5    # Half-extent in Z for the ground slab
PLAT_H          = 0.15   # Half-extent Z for small platforms


class TerrainGenerator:
    """
    Builds a race corridor with procedural obstacles.

    Obstacle zones:
      - Starting clear zone: first 8 m (no obstacles so agents can begin walking)
      - Obstacle zone: 8 m to (CORRIDOR_LENGTH - 8) m
      - Finishing clear zone: last 8 m (open approach to goal)
    """

    def __init__(self, client: int, seed: int = 42):
        self._client = client
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def build(self) -> Tuple[List[int], np.ndarray, np.ndarray]:
        ids: List[int] = []

        # ---- Ground slab ----
        ids.append(self._build_ground())

        # ---- Heightfield hills (visual + collision) ----
        ids += self._build_hills()

        # ---- Scatter obstacles in middle zone ----
        ids += self._scatter_obstacles()

        # ---- Start platform marker ----
        ids.append(self._box(
            cx=4.0, cy=0.0, top_z=0.0,
            hx=4.0, hy=CORRIDOR_WIDTH * 0.9, hz=PLAT_H,
            color=COL_START))

        # ---- Goal platform ----
        gx = CORRIDOR_LENGTH - 3.0
        ids.append(self._box(
            cx=gx, cy=0.0, top_z=0.0,
            hx=4.0, hy=CORRIDOR_WIDTH * 0.9, hz=PLAT_H,
            color=COL_GOAL_PLAT))

        # Goal visual: a ring of golden spheres above the goal line
        ids += self._build_goal_ring(gx)

        start_pos = np.array([2.0, 0.0, 0.0], dtype=np.float32)
        goal_pos  = np.array([gx,  0.0, 0.0], dtype=np.float32)

        return ids, start_pos, goal_pos

    # ------------------------------------------------------------------
    # Ground
    # ------------------------------------------------------------------

    def _build_ground(self) -> int:
        """Flat ground slab covering the entire corridor."""
        col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[CORRIDOR_LENGTH / 2 + 5, CORRIDOR_WIDTH + 5, GROUND_THICK],
            physicsClientId=self._client)
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[CORRIDOR_LENGTH / 2 + 5, CORRIDOR_WIDTH + 5, GROUND_THICK],
            rgbaColor=COL_GROUND,
            physicsClientId=self._client)
        body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[CORRIDOR_LENGTH / 2, 0.0, -GROUND_THICK],
            physicsClientId=self._client)
        p.changeDynamics(body, -1,
                         lateralFriction=0.9,
                         restitution=0.0,
                         physicsClientId=self._client)
        return body

    # ------------------------------------------------------------------
    # Hills: small raised terrain bumps via box rows
    # ------------------------------------------------------------------

    def _build_hills(self) -> List[int]:
        """
        Scatter gentle hill mounds (rounded box stacks) across the course.
        Uses simple random bumps - no heavy heightfield API to keep it fast.
        """
        ids = []
        x_start = 10.0
        x_end   = CORRIDOR_LENGTH - 10.0
        n_hills = int(self.rng.integers(8, 15))

        for _ in range(n_hills):
            cx  = float(self.rng.uniform(x_start, x_end))
            cy  = float(self.rng.uniform(-CORRIDOR_WIDTH * 0.7, CORRIDOR_WIDTH * 0.7))
            # Terrain bump: a low wide box
            hx  = float(self.rng.uniform(1.0, 3.5))
            hy  = float(self.rng.uniform(1.0, 3.0))
            hz  = float(self.rng.uniform(0.08, 0.25))
            # Color varies slightly per mound
            g   = float(self.rng.uniform(0.42, 0.58))
            color = [0.30 + g * 0.1, g, 0.20, 1.0]
            ids.append(self._box(
                cx=cx, cy=cy, top_z=hz * 2,
                hx=hx, hy=hy, hz=hz,
                color=color))
        return ids

    # ------------------------------------------------------------------
    # Obstacle scatter
    # ------------------------------------------------------------------

    def _scatter_obstacles(self) -> List[int]:
        ids = []
        x_start = 10.0
        x_end   = CORRIDOR_LENGTH - 10.0

        n_rocks   = int(self.rng.integers(12, 20))
        n_pillars = int(self.rng.integers(6, 12))
        n_ramps   = int(self.rng.integers(4, 8))

        # ---- Rocks (irregular box shapes) ----
        for _ in range(n_rocks):
            cx = float(self.rng.uniform(x_start, x_end))
            cy = float(self.rng.uniform(-(CORRIDOR_WIDTH - 1.5), CORRIDOR_WIDTH - 1.5))
            hx = float(self.rng.uniform(0.25, 0.80))
            hy = float(self.rng.uniform(0.25, 0.80))
            hz = float(self.rng.uniform(0.20, 0.55))
            # Slight random tilt for variety
            yaw = float(self.rng.uniform(0, math.pi))
            ids.append(self._box_oriented(
                cx=cx, cy=cy, cz=hz,
                hx=hx, hy=hy, hz=hz,
                yaw=yaw, color=COL_ROCK))

        # ---- Pillars (tall cylinders / capsules the agents must navigate around) ----
        for _ in range(n_pillars):
            cx = float(self.rng.uniform(x_start, x_end))
            cy = float(self.rng.uniform(-(CORRIDOR_WIDTH - 1.0), CORRIDOR_WIDTH - 1.0))
            radius = float(self.rng.uniform(0.15, 0.35))
            height = float(self.rng.uniform(0.8, 2.0))
            ids.append(self._cylinder(cx=cx, cy=cy, radius=radius, height=height))

        # ---- Ramps (inclined boxes angled along X) ----
        for _ in range(n_ramps):
            cx     = float(self.rng.uniform(x_start, x_end))
            cy     = float(self.rng.uniform(-(CORRIDOR_WIDTH - 2.0), CORRIDOR_WIDTH - 2.0))
            angle  = float(self.rng.uniform(8.0, 18.0))  # degrees
            length = float(self.rng.uniform(1.5, 3.5))
            width  = float(self.rng.uniform(1.0, 2.5))
            ids.append(self._ramp(cx=cx, cy=cy, angle_deg=angle, length=length, width=width))

        return ids

    # ------------------------------------------------------------------
    # Goal ring
    # ------------------------------------------------------------------

    def _build_goal_ring(self, gx: float) -> List[int]:
        """Decorative arch of golden spheres above the finish line."""
        ids = []
        n_spheres   = 16
        arch_radius = 3.5
        arch_height = 1.5   # base height of arch bottom

        # Semi-circle arch in Y-Z plane at x = gx
        for i in range(n_spheres + 1):
            theta = math.pi * i / n_spheres      # 0 .. pi
            y = arch_radius * math.cos(theta)
            z = arch_radius * math.sin(theta) + arch_height
            vis = p.createVisualShape(
                p.GEOM_SPHERE, radius=0.25,
                rgbaColor=[1.0, 0.85, 0.0, 0.95],
                physicsClientId=self._client)
            body = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,   # visual only, no collision
                baseVisualShapeIndex=vis,
                basePosition=[gx, y, z],
                physicsClientId=self._client)
            ids.append(body)

        # Central glowing large sphere at top of arch
        vis = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.50,
            rgbaColor=[1.0, 0.95, 0.10, 1.0],
            physicsClientId=self._client)
        body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vis,
            basePosition=[gx, 0.0, arch_height + arch_radius + 0.5],
            physicsClientId=self._client)
        ids.append(body)

        return ids

    # ------------------------------------------------------------------
    # Primitive builders
    # ------------------------------------------------------------------

    def _box(self, cx: float, cy: float, top_z: float,
             hx: float, hy: float, hz: float,
             color=None) -> int:
        if color is None:
            color = COL_ROCK
        cz = top_z - hz
        col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[hx, hy, hz],
            physicsClientId=self._client)
        vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[hx, hy, hz],
            rgbaColor=color, physicsClientId=self._client)
        body = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[cx, cy, cz],
            physicsClientId=self._client)
        p.changeDynamics(body, -1,
                         lateralFriction=0.9,
                         restitution=0.0,
                         physicsClientId=self._client)
        return body

    def _box_oriented(self, cx: float, cy: float, cz: float,
                      hx: float, hy: float, hz: float,
                      yaw: float, color=None) -> int:
        if color is None:
            color = COL_ROCK
        orn = p.getQuaternionFromEuler([0.0, 0.0, yaw])
        col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[hx, hy, hz],
            physicsClientId=self._client)
        vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[hx, hy, hz],
            rgbaColor=color, physicsClientId=self._client)
        body = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[cx, cy, cz],
            baseOrientation=list(orn),
            physicsClientId=self._client)
        p.changeDynamics(body, -1,
                         lateralFriction=0.9,
                         restitution=0.0,
                         physicsClientId=self._client)
        return body

    def _cylinder(self, cx: float, cy: float,
                  radius: float, height: float) -> int:
        col = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=radius, height=height,
            physicsClientId=self._client)
        vis = p.createVisualShape(
            p.GEOM_CYLINDER, radius=radius, length=height,
            rgbaColor=COL_PILLAR, physicsClientId=self._client)
        body = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[cx, cy, height / 2],
            physicsClientId=self._client)
        p.changeDynamics(body, -1,
                         lateralFriction=0.8,
                         restitution=0.0,
                         physicsClientId=self._client)
        return body

    def _ramp(self, cx: float, cy: float,
              angle_deg: float, length: float, width: float) -> int:
        angle_rad = math.radians(angle_deg)
        half_l    = length / 2
        half_w    = width / 2
        half_h    = 0.15
        dz        = half_l * math.sin(angle_rad)
        # tilt around Y axis
        orn = p.getQuaternionFromEuler([0, -angle_rad, 0])
        col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[half_l, half_w, half_h],
            physicsClientId=self._client)
        vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[half_l, half_w, half_h],
            rgbaColor=COL_RAMP, physicsClientId=self._client)
        body = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[cx, cy, dz],
            baseOrientation=list(orn),
            physicsClientId=self._client)
        p.changeDynamics(body, -1,
                         lateralFriction=0.85,
                         restitution=0.0,
                         physicsClientId=self._client)
        return body
