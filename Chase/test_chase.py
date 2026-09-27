"""Manual smoke test for ChaseEnv: runs the PyBullet chase headless with zero
thrust for 100 steps, printing heights, the gap between them and chaser vel.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.chase_env import ChaseEnv
import pybullet as p
import math

print("Creating ChaseEnv...")
env = ChaseEnv(render_mode="direct")

print("Resetting...")
obs, info = env.reset()

print("\nRunning... 100 steps")
for i in range(100):
    action = env.action_space.sample() * 0.0
    obs, reward, term, trunc, info = env.step(action)
    
    agent_pos, _ = p.getBasePositionAndOrientation(env._agent_id, physicsClientId=env._client)
    chaser_pos, _ = p.getBasePositionAndOrientation(env._chaser_id, physicsClientId=env._client)
    chaser_vel, _ = p.getBaseVelocity(env._chaser_id, physicsClientId=env._client)
    
    dx = agent_pos[0] - chaser_pos[0]
    dy = agent_pos[1] - chaser_pos[1]
    dist = math.sqrt(dx*dx + dy*dy)
    
    # Print every 10 steps
    if i % 10 == 0:
        print(f"Step {i:3d} | Agent Z: {agent_pos[2]:.2f} | Chaser Z: {chaser_pos[2]:.2f} | Dist: {dist:.2f} | Chaser Vel Z: {chaser_vel[2]:.2f}")
    
    if term:
        print("Agent fell/caught")
        break

env.close()
print("Done!")
