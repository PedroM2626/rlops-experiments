"""Quick test of the Chase environment."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.chase_env import ChaseEnv

print("Creating ChaseEnv...")
env = ChaseEnv(render_mode="human")

print("Resetting...")
obs, info = env.reset()

print(f"Observation shape: {obs.shape}")
print(f"Action space: {env.action_space}")
print("\nRunning... Press Ctrl+C to stop")

try:
    while True:
        action = env.action_space.sample() * 0.0  # Zero actions - watch only
        obs, reward, term, trunc, info = env.step(action)
        
        if term:
            print("Agent fell/caught - resetting...")
            obs, info = env.reset()
            
except KeyboardInterrupt:
    print("\nInterrupted!")

env.close()
print("Done!")
