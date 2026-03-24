"""
Debug script to isolate the spawn position bug.
"""
import pybullet as p
import pybullet_data
import numpy as np
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.env.level_generator import LevelGenerator

def test_with_level_generator():
    print("=" * 60)
    print("TEST 1: With LevelGenerator")
    print("=" * 60)
    
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    
    # Setup physics
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setTimeStep(1.0 / 480.0, physicsClientId=client)
    
    # Create level
    gen = LevelGenerator(client, seed=42)
    level_ids, start_pos, goal_pos = gen.build()
    
    print(f"Level created. start_pos = {start_pos}")
    print(f"Goal pos = {goal_pos}")
    print(f"Platform IDs: {level_ids}")
    
    # Check platform positions
    for pid in level_ids[:3]:
        pos, orn = p.getBasePositionAndOrientation(pid, physicsClientId=client)
        print(f"  Platform {pid} at {pos}")
    
    # Load humanoid
    urdf_path = os.path.join(os.path.dirname(__file__), "assets", "humanoid.urdf")
    
    spawn_pos = start_pos.copy()
    spawn_pos[2] = 1.0
    print(f"\nLoading URDF at spawn_pos = {spawn_pos}")
    
    agent_id = p.loadURDF(
        urdf_path,
        basePosition=spawn_pos.tolist(),
        baseOrientation=[0, 0, 0, 1],
        useFixedBase=False,
        physicsClientId=client)
    
    # Check position immediately after load
    pos1, _ = p.getBasePositionAndOrientation(agent_id, physicsClientId=client)
    print(f"Agent position immediately after loadURDF: {pos1}")
    
    # Step simulation a few times
    for step in [1, 10, 50, 100, 200, 240]:
        for _ in range(step if step <= 10 else (step - (step//2))):
            p.stepSimulation(physicsClientId=client)
        pos, _ = p.getBasePositionAndOrientation(agent_id, physicsClientId=client)
        print(f"Agent position after {step} steps: {pos}")
    
    p.disconnect(client)
    print()

def test_without_level_generator():
    print("=" * 60)
    print("TEST 2: Without LevelGenerator (just plane)")
    print("=" * 60)
    
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setTimeStep(1.0 / 480.0, physicsClientId=client)
    
    # Load plane
    plane_id = p.loadURDF("plane.urdf", physicsClientId=client)
    print(f"Plane loaded: {plane_id}")
    
    # Load humanoid
    urdf_path = os.path.join(os.path.dirname(__file__), "assets", "humanoid.urdf")
    spawn_pos = [1.0, 0.0, 1.0]
    print(f"Loading URDF at spawn_pos = {spawn_pos}")
    
    agent_id = p.loadURDF(
        urdf_path,
        basePosition=spawn_pos,
        baseOrientation=[0, 0, 0, 1],
        useFixedBase=False,
        physicsClientId=client)
    
    # Check position immediately after load
    pos1, _ = p.getBasePositionAndOrientation(agent_id, physicsClientId=client)
    print(f"Agent position immediately after loadURDF: {pos1}")
    
    # Step simulation
    for step in [1, 10, 50, 100, 200, 240]:
        for _ in range(step if step <= 10 else (step - (step//2))):
            p.stepSimulation(physicsClientId=client)
        pos, _ = p.getBasePositionAndOrientation(agent_id, physicsClientId=client)
        print(f"Agent position after {step} steps: {pos}")
    
    p.disconnect(client)
    print()

def test_with_boxes_no_levelgen():
    print("=" * 60)
    print("TEST 3: With manual boxes (not LevelGenerator)")
    print("=" * 60)
    
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setTimeStep(1.0 / 480.0, physicsClientId=client)
    
    # Create a simple box platform (same as start platform)
    col = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[2.5, 2.0, 0.25],
        physicsClientId=client)
    vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[2.5, 2.0, 0.25],
        rgbaColor=[0.2, 0.4, 0.9, 1.0],
        physicsClientId=client)
    box_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=[2.5, 0.0, -0.25],  # top surface at Z=0
        physicsClientId=client)
    
    print(f"Box platform created at Z=-0.25 (top surface Z=0)")
    
    # Load humanoid
    urdf_path = os.path.join(os.path.dirname(__file__), "assets", "humanoid.urdf")
    spawn_pos = [1.0, 0.0, 1.0]
    print(f"Loading URDF at spawn_pos = {spawn_pos}")
    
    agent_id = p.loadURDF(
        urdf_path,
        basePosition=spawn_pos,
        baseOrientation=[0, 0, 0, 1],
        useFixedBase=False,
        physicsClientId=client)
    
    # Check position immediately after load
    pos1, _ = p.getBasePositionAndOrientation(agent_id, physicsClientId=client)
    print(f"Agent position immediately after loadURDF: {pos1}")
    
    # Step simulation
    for step in [1, 10, 50, 100, 200, 240]:
        for _ in range(step if step <= 10 else (step - (step//2))):
            p.stepSimulation(physicsClientId=client)
        pos, _ = p.getBasePositionAndOrientation(agent_id, physicsClientId=client)
        print(f"Agent position after {step} steps: {pos}")
    
    p.disconnect(client)
    print()

if __name__ == "__main__":
    test_without_level_generator()
    test_with_boxes_no_levelgen()
    test_with_level_generator()
