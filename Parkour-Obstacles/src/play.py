"""
Play script - loads a trained PPO model and renders it in real time.

Usage:
    python play.py                                 # runs until the window closes (uses the default model)
    python play.py --model models/my_model         # a specific model
    python play.py --no-render --n-episodes 5      # headless benchmark

Without --n-episodes it runs indefinitely until the user closes the window.
"""

import argparse
import os
import sys
import time

import numpy as np
import pybullet as p
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.env.parkour_env import ParkourEnv
from stable_baselines3.common.monitor import Monitor

DEFAULT_MODEL  = "models/ppo_parkour_final"
DEFAULT_VECNORM = "models/vec_normalize.pkl"


def parse_args():
    parser = argparse.ArgumentParser(description="Watch the trained parkour agent")
    parser.add_argument("--model",      type=str,  default=DEFAULT_MODEL,
                        help="Path to the .zip model (without extension)")
    parser.add_argument("--vecnorm",    type=str,  default=DEFAULT_VECNORM,
                        help="Path to the VecNormalize .pkl")
    parser.add_argument("--no-render",  action="store_true",
                        help="Run without the GUI window (headless benchmark)")
    parser.add_argument("--level-seed", type=int,  default=None,
                        help="Level seed, for reproducibility")
    parser.add_argument("--n-episodes", type=int,  default=None,
                        help="Number of episodes (default: infinite until the window is closed)")
    parser.add_argument("--random",      action="store_true",
                        help="Use random actions instead of the model (debug)")
    parser.add_argument("--steps",      type=int,  default=2000,
                        help="Max steps per episode (default: 2000)")
    return parser.parse_args()


def _window_open(client: int) -> bool:
    """Return False if the PyBullet window was closed or is not connected."""
    if client is None:
        return False
    try:
        info = p.getConnectionInfo(client)
        return bool(info.get('isConnected', False))
    except Exception:
        return False


def main():
    args = parse_args()
    render_mode = "direct" if args.no_render else "human"
    infinite    = (args.n_episodes is None) and (not args.no_render)

    # Check the model
    model_path = args.model
    if not os.path.exists(model_path + ".zip"):
        print(f"[ERROR] Model not found at: {model_path}.zip")
        print("Train one first with:  python train.py")
        sys.exit(1)

    # Create the environment
    def make_env():
        env = ParkourEnv(render_mode=render_mode, level_seed=args.level_seed)
        return Monitor(env)

    vec_env = DummyVecEnv([make_env])

    if os.path.exists(args.vecnorm):
        vec_env = VecNormalize.load(args.vecnorm, vec_env)
        vec_env.training    = False
        vec_env.norm_reward = False
        print(f"VecNormalize loaded: {args.vecnorm}")
    else:
        print("[WARN] VecNormalize not found, running without normalization")

    model = PPO.load(model_path, env=vec_env, device="cpu")
    print(f"Model loaded         : {model_path}")

    # Grab the PyBullet client so we can detect the window being closed
    inner_env: ParkourEnv = vec_env.envs[0].env

    if infinite:
        print("\nRunning indefinitely - close the PyBullet window to exit.\n")
    else:
        n_ep = args.n_episodes if args.n_episodes else 5
        print(f"\nRunning {n_ep} episode(s).\n")

    ep_rewards = []
    episode    = 0

    try:
        while True:
            # Start/Reset env (this initializes the PyBullet client)
            obs      = vec_env.reset()
            ep_rew   = 0.0
            step     = 0
            done_ep  = False

            # Stop once the fixed number of episodes has been reached
            if not infinite and args.n_episodes and episode >= args.n_episodes:
                break

            # Check if window was closed or failed to open
            if render_mode == "human" and not _window_open(inner_env._client):
                print("\nWindow closed by the user or connection error.")
                break

            while not done_ep and step < args.steps:
                # Check the window is still open at every step
                if render_mode == "human" and not _window_open(inner_env._client):
                    print("\nWindow closed by the user.")
                    return

                if args.random:
                    # sample() returns one action, SB3 wants a list since it is a VecEnv
                    action = np.array([vec_env.action_space.sample()])
                else:
                    action, _ = model.predict(obs, deterministic=True)
                try:
                    obs, reward, done, _ = vec_env.step(action)
                except (p.error, Exception) as e:
                    err_msg = str(e).lower()
                    if "not connected" in err_msg or "physics server" in err_msg:
                        print("\nLost connection with the simulator (window closed).")
                        return
                    raise e

                ep_rew  += float(reward[0])
                step    += 1
                done_ep  = bool(done[0])

                if not args.no_render:
                    time.sleep(1.0 / 60.0)  # ~60 fps

            episode += 1
            ep_rewards.append(ep_rew)
            print(f"Episode {episode:4d} | reward: {ep_rew:8.2f} | steps: {step}")

    except KeyboardInterrupt:
        print("\nInterrupted by the user (Ctrl+C).")

    finally:
        if ep_rewards:
            print(f"\n=== Summary ({len(ep_rewards)} episode(s)) ===")
            print(f"  Mean  : {np.mean(ep_rewards):.2f}")
            print(f"  Std   : {np.std(ep_rewards):.2f}")
            print(f"  Max   : {np.max(ep_rewards):.2f}")
        vec_env.close()


if __name__ == "__main__":
    main()
