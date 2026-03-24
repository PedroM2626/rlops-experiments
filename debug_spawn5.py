"""
Debug - check contacts
"""
import pybullet as p
import pybullet_data
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_contacts():
    print("=" * 60)
    print("TEST: Check contacts with no self-collision")
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
        flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS,
        physicsClientId=client)
    
    for step in range(10):
        p.stepSimulation(physicsClientId=client)
        
        pos, _ = p.getBasePositionAndOrientation(agent_id, physicsClientId=client)
        contacts = p.getContactPoints(bodyA=agent_id, physicsClientId=client)
        
        print(f"Step {step+1}: pos={pos}, contacts={len(contacts)}")
        
        # Get link positions
        num_joints = p.getNumJoints(agent_id)
        for i in range(num_joints):
            info = p.getJointInfo(agent_id, i, physicsClientId=client)
            ls = p.getLinkState(agent_id, i, physicsClientId=client)
            if "foot" in info[1].decode().lower() or "shin" in info[1].decode().lower():
                print(f"  {info[1].decode()}: z={ls[0][2]:.3f}")
    
    p.disconnect(client)
    print()

if __name__ == "__main__":
    test_contacts()
