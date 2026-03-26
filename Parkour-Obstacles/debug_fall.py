"""
Debug fall detection
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pybullet as p
from src.env.parkour_env import ParkourEnv

# Monkey-patch to add debug
original_is_fallen = ParkourEnv._is_fallen

def debug_is_fallen(self):
    allowed_links = set(self._foot_link_ids.values())
    print(f"  Allowed foot links: {allowed_links}")
    print(f"  Foot link IDs: {self._foot_link_ids}")
    
    contacts = p.getContactPoints(bodyA=self._agent_id, physicsClientId=self._client)
    print(f"  Total contacts: {len(contacts)}")
    
    for c in (contacts or []):
        link_a = c[3]
        body_b = c[2]
        if body_b == self._agent_id:
            continue
        print(f"    Link {link_a} touching body {body_b}, allowed={link_a in allowed_links}")
        if link_a not in allowed_links:
            print(f"    -> FALL DETECTED!")
            return True
    return False

ParkourEnv._is_fallen = debug_is_fallen

env = ParkourEnv(render_mode="direct", level_seed=42)

print("Resetting environment...")
obs, info = env.reset()

print(f"Initial torso position: {obs[:3]}")

# Run a few steps
for i in range(3):
    action = env.action_space.sample() * 0
    obs, reward, term, trunc, info = env.step(action)
    print(f"\nStep {i+1}: pos={obs[:3]}, reward={reward:.3f}, term={term}, trunc={trunc}")
    if term or trunc:
        break

env.close()
