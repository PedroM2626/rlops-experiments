"""Simple test to isolate the issue"""
import pybullet as p
import pybullet_data
import os

# Start PyBullet
client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
p.setGravity(0, 0, -9.81, physicsClientId=client)
p.setTimeStep(1./240., physicsClientId=client)

# Load plane
plane = p.loadURDF("plane.urdf", physicsClientId=client)

# Load humanoid from correct path
urdf_path = "C:\\Users\\pedro\\Downloads\\ML-Games\\Chase\\assets\\humanoid.urdf"
print(f"Loading URDF from: {urdf_path}")

humanoid = p.loadURDF(
    urdf_path,
    basePosition=[0, 0, 1.5],
    baseOrientation=[0, 0, 0, 1],
    useFixedBase=False,
    flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS,
    physicsClientId=client)

# Let it settle
for i in range(100):
    p.stepSimulation(physicsClientId=client)
    if i % 20 == 0:
        pos, _ = p.getBasePositionAndOrientation(humanoid, physicsClientId=client)
        print(f"Step {i}: pos={pos}")

print("Done")
p.disconnect()