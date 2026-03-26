"""
Debug script to test self-collision
"""
import pybullet as p
import pybullet_data
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.env.level_generator import LevelGenerator

def test_with_self_collision_disabled():
    print("=" * 60)
    print("TEST: With self-collision disabled")
    print("=" * 60)
    
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setTimeStep(1.0 / 480.0, physicsClientId=client)
    
    # Create a simple box platform
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
        basePosition=[2.5, 0.0, -0.25],
        physicsClientId=client)
    
    print(f"Box platform created at Z=-0.25 (top surface Z=0)")
    
    # Load humanoid with self-collision disabled
    urdf_path = os.path.join(os.path.dirname(__file__), "assets", "humanoid.urdf")
    spawn_pos = [1.0, 0.0, 0.4]
    print(f"Loading URDF at spawn_pos = {spawn_pos}")
    
    # KEY CHANGE: Use GLOBAL_SAME_BP_CF to disable self-collision
    agent_id = p.loadURDF(
        urdf_path,
        basePosition=spawn_pos,
        baseOrientation=[0, 0, 0, 1],
        useFixedBase=False,
        flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS,  # Disable self-collision
        physicsClientId=client)
    
    # Check position immediately after load
    pos1, _ = p.getBasePositionAndOrientation(agent_id, physicsClientId=client)
    print(f"Agent position immediately after loadURDF: {pos1}")
    
    # Check joint states
    num_joints = p.getNumJoints(agent_id, physicsClientId=client)
    print(f"Number of joints: {num_joints}")
    for i in range(num_joints):
        info = p.getJointInfo(agent_id, i, physicsClientId=client)
        print(f"  Joint {i}: {info[1].decode()}")
    
    # Step simulation
    for step in [1, 10, 50, 100, 200, 240]:
        for _ in range(step if step <= 10 else (step - (step//2))):
            p.stepSimulation(physicsClientId=client)
        pos, _ = p.getBasePositionAndOrientation(agent_id, physicsClientId=client)
        print(f"Agent position after {step} steps: {pos}")
    
    p.disconnect(client)
    print()

def test_initial_pose():
    """Check if the initial joint configuration causes the explosion"""
    print("=" * 60)
    print("TEST: Checking initial joint configuration")
    print("=" * 60)
    
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setTimeStep(1.0 / 480.0, physicsClientId=client)
    
    # Create platform
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
        basePosition=[2.5, 0.0, -0.25],
        physicsClientId=client)
    
    # Load humanoid
    urdf_path = os.path.join(os.path.dirname(__file__), "assets", "humanoid.urdf")
    spawn_pos = [1.0, 0.0, 0.4]
    
    agent_id = p.loadURDF(
        urdf_path,
        basePosition=spawn_pos,
        baseOrientation=[0, 0, 0, 1],
        useFixedBase=False,
        physicsClientId=client)
    
    # Check initial joint states BEFORE any reset
    print("\nInitial joint states (before reset):")
    num_joints = p.getNumJoints(agent_id, physicsClientId=client)
    for i in range(num_joints):
        state = p.getJointState(agent_id, i, physicsClientId=client)
        info = p.getJointInfo(agent_id, i, physicsClientId=client)
        print(f"  Joint {i} ({info[1].decode()}): pos={state[0]:.4f}, vel={state[1]:.4f}")
    
    # Check if there's a collision happening
    contacts = p.getContactPoints(bodyA=agent_id, physicsClientId=client)
    print(f"\nContact points after loadURDF: {len(contacts)}")
    for c in contacts:
        print(f"  Link {c[3]} touching body {c[2]}")
    
    # Take one step
    p.stepSimulation(physicsClientId=client)
    
    contacts = p.getContactPoints(bodyA=agent_id, physicsClientId=client)
    print(f"\nContact points after 1 step: {len(contacts)}")
    for c in contacts:
        print(f"  Link {c[3]} touching body {c[2]}")
    
    pos, _ = p.getBasePositionAndOrientation(agent_id, physicsClientId=client)
    print(f"\nAgent position after 1 step: {pos}")
    
    p.disconnect(client)
    print()

if __name__ == "__main__":
    test_initial_pose()
    test_with_self_collision_disabled()
