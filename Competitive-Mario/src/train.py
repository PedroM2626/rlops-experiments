"""
Training script for Competitive Mario agents.

Trains a single agent using PPO with CnnPolicy on the Mario environment.
Each agent can have its own hyperparameters and movement style.
All training runs are tracked in MLflow.

Usage:
    python src/train.py --agent-name mario_speedster --timesteps 500000
    python src/train.py --agent-name mario_careful --lr 0.0001 --timesteps 1000000
    python src/train.py --config configs/agents.yaml --agent-name mario_speedster
"""

import argparse
import os
import sys
import yaml

import mlflow
import numpy as np
import torch
from dotenv import load_dotenv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack

# Add project root to path
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from src.env.mario_env import MarioCompetitiveEnv
from src.agent.callbacks import MLflowCallback

load_dotenv()


def make_env(level_seed=None, difficulty=0.5, level_width=200):
    """Factory for a single monitored Mario environment."""
    def _init():
        env = MarioCompetitiveEnv(
            level_seed=level_seed,
            difficulty=difficulty,
            level_width=level_width,
            render_mode=None,
        )
        env = Monitor(env)
        return env
    return _init


def load_agent_config(config_path, agent_name):
    """Load agent-specific hyperparameters from YAML config."""
    if config_path is None or not os.path.exists(config_path):
        return {}
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    agents = config.get("agents", {})
    return agents.get(agent_name, {})


def parse_args():
    p = argparse.ArgumentParser(description="Train a Mario agent using PPO")

    # Agent identity
    p.add_argument("--agent-name", type=str, default="mario_default",
                   help="Name of the agent to train")
    p.add_argument("--config", type=str, default="configs/agents.yaml",
                   help="Path to agents YAML config file")

    # MLflow
    p.add_argument("--tracking-uri", type=str,
                   default=os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))
    p.add_argument("--experiment-name", type=str, default="competitive_mario")

    # Environment
    p.add_argument("--level-seed", type=int, default=42,
                   help="Seed for level generation (all agents share this for fair competition)")
    p.add_argument("--difficulty", type=float, default=0.5,
                   help="Level difficulty (0.0 to 1.0)")
    p.add_argument("--level-width", type=int, default=200,
                   help="Level width in tiles")

    # PPO hyperparameters
    p.add_argument("--lr", type=float, default=float(os.getenv("LEARNING_RATE", 3e-4)))
    p.add_argument("--n-steps", type=int, default=int(os.getenv("N_STEPS", 2048)))
    p.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", 64)))
    p.add_argument("--gamma", type=float, default=float(os.getenv("GAMMA", 0.99)))
    p.add_argument("--n-epochs", type=int, default=int(os.getenv("N_EPOCHS", 10)))
    p.add_argument("--gae-lambda", type=float, default=float(os.getenv("GAE_LAMBDA", 0.95)))
    p.add_argument("--clip-range", type=float, default=float(os.getenv("CLIP_RANGE", 0.2)))
    p.add_argument("--ent-coef", type=float, default=float(os.getenv("ENT_COEF", 0.01)))

    # Training
    p.add_argument("--timesteps", type=int, default=500_000,
                   help="Total training timesteps")
    p.add_argument("--n-envs", type=int, default=int(os.getenv("N_ENVS", 4)))
    p.add_argument("--frame-stack", type=int, default=4,
                   help="Number of frames to stack for temporal info")

    # Checkpointing
    p.add_argument("--checkpoint-freq", type=int,
                   default=int(os.getenv("CHECKPOINT_FREQ", 50_000)))
    p.add_argument("--model-dir", type=str, default="models")
    p.add_argument("--model", type=str, default=None,
                   help="Path to pretrained model to continue training")

    return p.parse_args()


def main():
    args = parse_args()

    # load agent config from YAML if available
    config_path = os.path.join(_PROJECT_DIR, args.config)
    agent_cfg = load_agent_config(config_path, args.agent_name)
    if agent_cfg:
        print(f"Loaded config for agent '{args.agent_name}': {agent_cfg}")
        # override args with config values (config takes priority)
        for key, value in agent_cfg.items():
            key_norm = key.replace("-", "_")
            if hasattr(args, key_norm):
                setattr(args, key_norm, value)

    # agent-specific model directory
    agent_model_dir = os.path.join(_PROJECT_DIR, args.model_dir, args.agent_name)
    os.makedirs(agent_model_dir, exist_ok=True)

    # setup MLflow
    tracking_uri = args.tracking_uri
    # if it's a relative path (not a URL scheme), make it absolute to project dir
    if "://" not in tracking_uri and not os.path.isabs(tracking_uri):
        tracking_uri = os.path.abspath(os.path.join(_PROJECT_DIR, tracking_uri))
    # on Windows, MLflow needs file:// prefix for local absolute paths
    if os.path.isabs(tracking_uri) and "://" not in tracking_uri:
        tracking_uri = "file:///" + tracking_uri.replace("\\", "/")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=f"train_{args.agent_name}") as run:
        print(f"MLflow run id: {run.info.run_id}")
        print(f"Agent: {args.agent_name}")
        print(f"Experiment: {args.experiment_name}")

        # log all params
        params = {
            "agent_name": args.agent_name,
            "algorithm": "PPO",
            "policy": "CnnPolicy",
            "model_type": "convolutional",
            "total_timesteps": args.timesteps,
            "learning_rate": args.lr,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "gamma": args.gamma,
            "n_epochs": args.n_epochs,
            "gae_lambda": args.gae_lambda,
            "clip_range": args.clip_range,
            "ent_coef": args.ent_coef,
            "n_envs": args.n_envs,
            "frame_stack": args.frame_stack,
            "level_seed": args.level_seed,
            "difficulty": args.difficulty,
            "level_width": args.level_width,
        }
        mlflow.log_params(params)

        # create vectorized environments
        print(f"\nCreating {args.n_envs} parallel environments...")
        vec_env = DummyVecEnv([
            make_env(
                level_seed=args.level_seed,
                difficulty=args.difficulty,
                level_width=args.level_width,
            )
            for _ in range(args.n_envs)
        ])

        # stack frames for temporal information
        vec_env = VecFrameStack(vec_env, n_stack=args.frame_stack)

        # transpose images for PyTorch (channels first)
        vec_env = VecTransposeImage(vec_env)

        # create callback
        callback = MLflowCallback(
            checkpoint_freq=args.checkpoint_freq,
            model_save_dir=agent_model_dir,
            agent_name=args.agent_name,
            verbose=1,
        )

        if args.model:
            model_path = os.path.join(_PROJECT_DIR, args.model)
            print(f"\nLoading pretrained model: {model_path}")
            model = PPO.load(model_path, env=vec_env, device="auto")
            print(f"Continuing from step {model.num_timesteps:,}")
        else:
            print("\nTraining from scratch...")
            model = PPO(
                policy="CnnPolicy",
                env=vec_env,
                learning_rate=args.lr,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                gamma=args.gamma,
                n_epochs=args.n_epochs,
                gae_lambda=args.gae_lambda,
                clip_range=args.clip_range,
                ent_coef=args.ent_coef,
                verbose=1,
                device="auto",
            )

        device_name = str(next(model.policy.parameters()).device)
        mlflow.log_param("device", device_name)
        print(f"Training on device: {device_name}")
        print(f"Total timesteps: {args.timesteps:,}\n")

        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            progress_bar=True,
        )

        # save final model
        final_path = os.path.join(agent_model_dir, f"ppo_{args.agent_name}_final")
        model.save(final_path)
        mlflow.log_artifact(final_path + ".zip", artifact_path="final_model")

        # log final metrics
        mlflow.log_metrics({
            "final_best_reward": callback.best_reward,
            "final_best_x_pos": callback.best_x_pos,
            "total_episodes": callback.episode_count,
            "total_flags": callback.total_flags,
        })

        print(f"\nTraining complete for agent '{args.agent_name}'")
        print(f"  Best reward: {callback.best_reward:.2f}")
        print(f"  Best x_pos: {callback.best_x_pos}")
        print(f"  Flag completions: {callback.total_flags}")
        print(f"  Total episodes: {callback.episode_count}")
        print(f"  Model saved to: {final_path}.zip")

        vec_env.close()


if __name__ == "__main__":
    main()
