"""
Debug - test spawn at 1.4
"""
import pybullet as p
import pybullet_data
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_spawn_1_4():
    print("=" * 60)
    print("TEST: Spawn at 1.4m")
    print("=" * 60)
    
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setTimeStep(1.0 / 480.0, physicsClientId=client)
    
    # Create platform at Z=0
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
        basePosition=[2.5, 0.0, -0.25],  # Top surface at Z=0
        physicsClientId=client)
    
    # Load humanoid at Z=1.4 (feet should be above platform)
    urdf_path = os.path.join(os.path.dirname(__file__), "assets", "humanoid.urdf")
    
    agent_id = p.loadURDF(
        urdf_path,
        basePosition=[1.0, 0.0, 1.4],
        baseOrientation=[0, 0, 0, 1],
        useFixedBase=False,
        flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS,
        physicsClientId=client)
    
    pos, _ = p.getBasePositionAndOrientation(agent_id, physicsClientId=client)
    print(f"Initial: {pos}")
    # Expected feet: 1.4 - 1.125 = 0.275 (above platform at Z=0)
    
    for step in [1, 10, 50, 100, 200, 240]:
        for _ in range(step if step <= 10 else (step - (step//2))):
            p.stepSimulation(physicsClientId=client)
        pos, _ = p.getBasePositionAndOrientation(agent_id, physicsClientId=client)
        contacts = p.getContactPoints(bodyA=agent_id, physicsClientId=client)
        print(f"After {step} steps: pos={pos}, contacts={len(contacts)}")
    
    p.disconnect(client)
    print()

if __name__ == "__main__":
    test_spawn_1_4()
