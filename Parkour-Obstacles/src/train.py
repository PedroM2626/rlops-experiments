"""
Training entrypoint for the 3D Parkour RL agent.

Usage:
    python src/train.py [options]

Options:
    --test          Test trained model instead of training
    --episodes N    Number of test episodes (default: 5)
    --model PATH    Path to model for testing or resuming training

MLflow logs:
    - All hyperparameters
    - Per-episode reward and length
    - Model checkpoints as artifacts
"""

import argparse
import os

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from dotenv import load_dotenv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from src.env.parkour_env import ParkourEnv
from src.agent.callbacks import MLflowCallback


load_dotenv()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO agent on Parkour env")

    # Mode
    p.add_argument("--test", action="store_true", help="Test trained model instead of training")
    p.add_argument("--episodes", type=int, default=5, help="Number of test episodes")
    
    # MLflow
    p.add_argument("--experiment-name", type=str,
                   default=os.getenv("EXPERIMENT_NAME", "parkour_ppo"))
    p.add_argument("--tracking-uri", type=str,
                   default=os.getenv("MLFLOW_TRACKING_URI", "mlruns"))

    # Training volume
    p.add_argument("--timesteps", type=int,
                   default=int(os.getenv("TOTAL_TIMESTEPS", 1_000_000)))

    # PPO hyperparameters
    p.add_argument("--lr",          type=float,
                   default=float(os.getenv("LEARNING_RATE", 3e-4)))
    p.add_argument("--n-steps",     type=int,
                   default=int(os.getenv("N_STEPS", 2048)))
    p.add_argument("--batch-size",  type=int,
                   default=int(os.getenv("BATCH_SIZE", 64)))
    p.add_argument("--gamma",       type=float,
                   default=float(os.getenv("GAMMA", 0.99)))
    p.add_argument("--n-epochs",    type=int,
                   default=int(os.getenv("N_EPOCHS", 10)))
    p.add_argument("--gae-lambda",  type=float,
                   default=float(os.getenv("GAE_LAMBDA", 0.95)))
    p.add_argument("--clip-range",  type=float,
                   default=float(os.getenv("CLIP_RANGE", 0.2)))
    p.add_argument("--ent-coef",    type=float,
                   default=float(os.getenv("ENT_COEF", 0.01)))

    # Network
    p.add_argument("--net-arch",    type=str,
                   default=os.getenv("NET_ARCH", "256,256"))
    p.add_argument("--n-envs",      type=int,
                   default=int(os.getenv("N_ENVS", 4)))

    # Checkpointing
    p.add_argument("--checkpoint-freq", type=int,
                   default=int(os.getenv("CHECKPOINT_FREQ", 50_000)))
    p.add_argument("--model-dir",   type=str, default="models")
    p.add_argument("--model",       type=str, default=None,
                   help="Path to pretrained model (.zip) to continue training")

    return p.parse_args()


def make_env(render_mode: str = "direct"):
    """Factory for a single monitored parkour environment."""
    def _init():
        env = ParkourEnv(render_mode=render_mode)
        env = Monitor(env)
        return env
    return _init


def test(env, model, episodes: int = 5):
    """Test trained model in human-rendered environment."""
    print(f"\nTesting for {episodes} episodes...")
    
    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                print(f"Episode {ep+1}: steps={steps}, reward={total_reward:.2f}")
                break
    
    env.close()


def main():
    args = parse_args()

    if args.test:
        if not args.model:
            print("Error: --model required for testing")
            return
        print(f"Loading model: {args.model}")
        model = PPO.load(args.model)
        test_env = make_env("human")()
        test(test_env, model, args.episodes)
        return

    # ------------------------------------------------------------------ MLflow setup
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run() as run:
        print(f"MLflow run id : {run.info.run_id}")
        print(f"Experiment    : {args.experiment_name}")

        # Log hyperparameters
        params = {
            "algorithm"       : "PPO",
            "total_timesteps" : args.timesteps,
            "learning_rate"   : args.lr,
            "n_steps"         : args.n_steps,
            "batch_size"      : args.batch_size,
            "gamma"           : args.gamma,
            "n_epochs"        : args.n_epochs,
            "gae_lambda"      : args.gae_lambda,
            "clip_range"      : args.clip_range,
            "ent_coef"        : args.ent_coef,
            "net_arch"        : args.net_arch,
            "n_envs"          : args.n_envs,
        }
        mlflow.log_params(params)

        # ------------------------------------------------------------------ Env
        print(f"\nCreating {args.n_envs} parallel environments ...")
        vec_env = DummyVecEnv([make_env("direct") for _ in range(args.n_envs)])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

        # ------------------------------------------------------------------ Model
        net_arch = [int(x) for x in args.net_arch.split(",")]
        policy_kwargs = dict(
            net_arch=dict(pi=net_arch, vf=net_arch),
            activation_fn=torch.nn.Tanh,
        )

        if args.model:
            print(f"\nLoading pretrained model: {args.model}")
            model = PPO.load(args.model, env=vec_env, device="auto")
            print(f"Continuing from step {model.num_timesteps:,}")
        else:
            model = PPO(
            policy          = "MlpPolicy",
            env             = vec_env,
            learning_rate   = args.lr,
            n_steps         = args.n_steps,
            batch_size      = args.batch_size,
            gamma           = args.gamma,
            n_epochs        = args.n_epochs,
            gae_lambda      = args.gae_lambda,
            clip_range      = args.clip_range,
            ent_coef        = args.ent_coef,
            policy_kwargs   = policy_kwargs,
            verbose         = 1,
            device          = "auto",
        )

        device_name = str(next(model.policy.parameters()).device)
        mlflow.log_param("device", device_name)
        print(f"Training on device: {device_name}")
        print(f"Total timesteps   : {args.timesteps:,}\n")

        # ------------------------------------------------------------------ Callbacks
        callback = MLflowCallback(
            checkpoint_freq = args.checkpoint_freq,
            model_save_dir  = args.model_dir,
            verbose         = 1,
        )

        # ------------------------------------------------------------------ Train
        model.learn(
            total_timesteps = args.timesteps,
            callback        = callback,
            progress_bar    = True,
        )

        # ------------------------------------------------------------------ Persist
        os.makedirs(args.model_dir, exist_ok=True)
        final_model_path = os.path.join(args.model_dir, "ppo_parkour_final")
        model.save(final_model_path)

        # Save VecNormalize stats
        vecnorm_path = os.path.join(args.model_dir, "vec_normalize.pkl")
        vec_env.save(vecnorm_path)

        mlflow.log_artifact(final_model_path + ".zip", artifact_path="model")
        mlflow.log_artifact(vecnorm_path, artifact_path="model")

        print(f"\nTraining complete. Model saved to: {final_model_path}")
        vec_env.close()


if __name__ == "__main__":
    main()
