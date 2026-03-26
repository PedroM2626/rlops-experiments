"""
Debug script - test with self-collision disabled
"""
import pybullet as p
import pybullet_data
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_no_self_collision():
    print("=" * 60)
    print("TEST: No self-collision")
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
    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=[2.5, 0.0, -0.25],
        physicsClientId=client)
    
    # Load humanoid with self-collision disabled
    urdf_path = os.path.join(os.path.dirname(__file__), "assets", "humanoid.urdf")
    
    agent_id = p.loadURDF(
        urdf_path,
        basePosition=[1.0, 0.0, 1.0],
        baseOrientation=[0, 0, 0, 1],
        useFixedBase=False,
        flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS,  # DISABLE SELF-COLLISION
        physicsClientId=client)
    
    pos, _ = p.getBasePositionAndOrientation(agent_id, physicsClientId=client)
    print(f"Initial: {pos}")
    
    # Step simulation
    for step in [1, 10, 50, 100, 200, 240]:
        p.stepSimulation(physicsClientId=client)
        pos, _ = p.getBasePositionAndOrientation(agent_id, physicsClientId=client)
        print(f"After {step} steps: {pos}")
    
    p.disconnect(client)
    print()

if __name__ == "__main__":
    test_no_self_collision()
