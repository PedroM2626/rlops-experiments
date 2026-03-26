"""
Play script for Chase environment - watch a trained agent play the chase game.
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from stable_baselines3 import PPO
from src.env.chase_env import ChaseEnv


def play_chase(model_path=None, episodes=5, delay=0.01):
    """
    Play the chase game.
    
    Args:
        model_path: Path to trained model (.zip). If None, uses random actions.
        episodes: Number of episodes to run.
        delay: Delay between steps for rendering (seconds).
    """
    
    # Create environment
    env = ChaseEnv(render_mode="human")
    
    # Load model if provided
    model = None
    if model_path and os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        model = PPO.load(model_path)
        print("Model loaded successfully!")
    else:
        print("No model provided - using random actions")
        if model_path:
            print(f"Warning: Model not found at {model_path}")
    
    try:
        for episode in range(episodes):
            print(f"\nStarting Episode {episode + 1}/{episodes}")
            obs, info = env.reset()
            total_reward = 0
            steps = 0
            
            while True:
                # Get action
                if model is not None:
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action = env.action_space.sample() * 0.3  # Small random actions
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                
                # Small delay for visualization
                if delay > 0:
                    time.sleep(delay)
                
                # Check if episode ended
                if terminated or truncated:
                    print(f"Episode {episode + 1} finished:")
                    print(f"  Steps: {steps}")
                    print(f"  Total Reward: {total_reward:.2f}")
                    print(f"  Reason: {'Caught by chaser' if terminated else 'Time limit' if truncated else 'Fell'}")
                    break
                    
    except KeyboardInterrupt:
        print("\n\nPlay interrupted by user.")
    finally:
        env.close()
        print("\nPlay session ended.")


if __name__ == "__main__":
    # Default model path - adjust as needed
    default_model = "models/ppo_chase_final.zip"
    
    # Check if model exists
    if os.path.exists(default_model):
        model_to_use = default_model
        print(f"Found default model: {model_to_use}")
    else:
        model_to_use = None
        print("No trained model found. Will use random actions.")
        print("To train a model first, run: python train.py")
    
    print("\nStarting Chase Play...")
    print("Close the PyBullet window or press Ctrl+C to stop.")
    print("-" * 50)
    
    play_chase(model_path=model_to_use, episodes=3, delay=0.05)