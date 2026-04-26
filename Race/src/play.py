"""
Visualization script for trained Race agents.

Usage:
    # Watch trained model race
    python -m src.play --model models/ppo_race_final.zip

    # Watch with random actions (sanity check)
    python -m src.play --random
"""

import argparse
import os
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from src.env.race_env import RaceEnv


def parse_args():
    p = argparse.ArgumentParser(description="Visualize trained Race agents")
    p.add_argument("--model",    type=str,  default=None,
                   help="Path to trained model (.zip)")
    p.add_argument("--vecnorm",  type=str,  default=None,
                   help="Path to VecNormalize stats (.pkl)")
    p.add_argument("--episodes", type=int,  default=5)
    p.add_argument("--n-agents", type=int,  default=4)
    p.add_argument("--random",   action="store_true",
                   help="Use random actions (no model needed)")
    p.add_argument("--seed",     type=int,  default=None,
                   help="Fix terrain seed for reproducibility")
    return p.parse_args()


def run_episode(env, model, deterministic: bool = True):
    """Run one episode and return total reward + steps."""
    obs, _ = env.reset()
    total_reward = 0.0
    steps = 0
    while True:
        if model is not None:
            action, _ = model.predict(obs, deterministic=deterministic)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps        += 1

        if terminated or truncated:
            break

    return total_reward, steps, info


def main():
    args = parse_args()

    if not args.random and args.model is None:
        # Try to find a model automatically
        model_dir = "models"
        candidates = [
            os.path.join(model_dir, "ppo_race_final.zip"),
        ]
        for c in candidates:
            if os.path.exists(c):
                args.model = c
                break
        if args.model is None:
            print("No trained model found. Use --model PATH or --random to watch random agents.")
            return

    print(f"Race environment - {args.n_agents} competing humanoid agents")
    print(f"Model: {args.model or 'random actions'}")
    print(f"Episodes: {args.episodes}\n")

    env = RaceEnv(
        render_mode="human",
        n_agents=args.n_agents,
        agent_index=0,
        level_seed=args.seed,
    )

    model = None
    if args.model:
        print(f"Loading model from: {args.model}")
        model = PPO.load(args.model)

        if args.vecnorm:
            print(f"Loading VecNormalize from: {args.vecnorm}")
            # For play, wrap in DummyVecEnv to apply normalization
            vec_env = DummyVecEnv([lambda: env])
            vec_env = VecNormalize.load(args.vecnorm, vec_env)
            vec_env.training = False
            vec_env.norm_reward = False

            for ep in range(args.episodes):
                obs = vec_env.reset()
                total_reward = 0.0
                steps = 0
                while True:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, dones, infos = vec_env.step(action)
                    total_reward += float(reward[0])
                    steps += 1
                    if dones[0]:
                        info = infos[0]
                        print(f"  Episode {ep+1}: steps={steps} "
                              f"reward={total_reward:.2f} "
                              f"finished={info.get('finished', False)}")
                        break
            vec_env.close()
            return

    for ep in range(args.episodes):
        total_reward, steps, info = run_episode(env, model)
        print(
            f"Episode {ep+1:02d}: "
            f"steps={steps:4d}  "
            f"reward={total_reward:8.2f}  "
            f"finished={info.get('finished', False)}  "
            f"rank={info.get('race_rank', '?')}"
        )
        time.sleep(0.5)

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
