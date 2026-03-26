"""
Debug script - test with fixed base
"""
import pybullet as p
import pybullet_data
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_fixed_base():
    print("=" * 60)
    print("TEST: Fixed base (standing)")
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
    
    # Load humanoid with fixed base
    urdf_path = os.path.join(os.path.dirname(__file__), "assets", "humanoid.urdf")
    
    agent_id = p.loadURDF(
        urdf_path,
        basePosition=[1.0, 0.0, 1.0],
        baseOrientation=[0, 0, 0, 1],
        useFixedBase=True,  # FIXED BASE!
        physicsClientId=client)
    
    pos, _ = p.getBasePositionAndOrientation(agent_id, physicsClientId=client)
    print(f"Agent position (fixed base): {pos}")
    
    # Step simulation
    for step in [1, 10, 50, 100, 200, 240]:
        p.stepSimulation(physicsClientId=client)
        pos, _ = p.getBasePositionAndOrientation(agent_id, physicsClientId=client)
        print(f"Agent position after {step} steps: {pos}")
    
    p.disconnect(client)
    print()

def test_check_feet_position():
    """Check where the feet actually are"""
    print("=" * 60)
    print("TEST: Check feet positions")
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
    
    # Load humanoid
    urdf_path = os.path.join(os.path.dirname(__file__), "assets", "humanoid.urdf")
    
    agent_id = p.loadURDF(
        urdf_path,
        basePosition=[1.0, 0.0, 1.0],
        baseOrientation=[0, 0, 0, 1],
        useFixedBase=False,
        physicsClientId=client)
    
    # Get link states
    num_joints = p.getNumJoints(agent_id, physicsClientId=client)
    
    # Find foot links
    foot_links = {}
    for i in range(num_joints):
        info = p.getJointInfo(agent_id, i, physicsClientId=client)
        name = info[1].decode()
        if "foot" in name.lower():
            foot_links[name] = i
            print(f"Foot link: {name} = joint index {i}")
    
    # Check positions
    for step in [0, 1, 5]:
        for _ in range(step):
            p.stepSimulation(physicsClientId=client)
        
        base_pos, _ = p.getBasePositionAndOrientation(agent_id, physicsClientId=client)
        print(f"\nStep {step}: Base at {base_pos}")
        
        # Check each foot
        for fname, fidx in foot_links.items():
            ls = p.getLinkState(agent_id, fidx, physicsClientId=client)
            print(f"  {fname} (link {fidx}): {ls[0]}")
        
        # Check contacts
        contacts = p.getContactPoints(bodyA=agent_id, physicsClientId=client)
        print(f"  Contacts: {len(contacts)}")
        for c in contacts:
            print(f"    Link {c[3]} touches body {c[2]}")
    
    p.disconnect(client)
    print()

if __name__ == "__main__":
    test_fixed_base()
    test_check_feet_position()
