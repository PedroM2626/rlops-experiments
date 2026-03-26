"""
Quick test of the ParkourEnv
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.parkour_env import ParkourEnv

env = ParkourEnv(render_mode="direct", level_seed=42)

print("Resetting environment...")
obs, info = env.reset()

print(f"Initial obs (first 12 values): {obs[:12]}")
print(f"Torso position: {obs[:3]}")
print(f"Torso velocity: {obs[3:6]}")

# Run a few steps
for i in range(10):
    action = env.action_space.sample() * 0  # Zero action first
    obs, reward, term, trunc, info = env.step(action)
    print(f"Step {i+1}: pos={obs[:3]}, reward={reward:.3f}, term={term}, trunc={trunc}")
    if term or trunc:
        break

env.close()
print("Done!")
