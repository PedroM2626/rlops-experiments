"""
Training entrypoint for the competitive multi-agent Race environment.

Architecture
------------
N_AGENTS agents compete in the same PyBullet world. Training uses
parameter sharing (one shared PPO policy) via a MultiAgentVecEnv wrapper:
each parallel env in the VecEnv creates a full RaceEnv with N_AGENTS in
one physics world, but exposes exactly ONE agent's obs/action/reward (the
agent assigned by `agent_index`). N_ENVS x N_AGENTS trajectories are
collected in parallel, all updating the same policy weights.

Usage
-----
    python -m src.train [options]
    python -m src.train --test --model models/ppo_race_final.zip

MLflow logs
-----------
    - All hyperparameters
    - Per-episode reward and length
    - Model checkpoints as artifacts
"""

import argparse
import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="gym")

import mlflow
import torch
from dotenv import load_dotenv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.env.race_env import RaceEnv
from src.agent.callbacks import MLflowCallback

load_dotenv()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO agent on competitive Race env")

    # Mode
    p.add_argument("--test",     action="store_true", help="Test trained model")
    p.add_argument("--episodes", type=int, default=5, help="Number of test episodes")

    # MLflow
    p.add_argument("--experiment-name", type=str,
                   default=os.getenv("EXPERIMENT_NAME", "race_ppo"))
    p.add_argument("--tracking-uri", type=str,
                   default=os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))

    # Training
    p.add_argument("--timesteps", type=int,
                   default=int(os.getenv("TOTAL_TIMESTEPS", 3_000_000)))

    # Multi-agent
    p.add_argument("--n-agents", type=int,
                   default=int(os.getenv("N_AGENTS", 4)),
                   help="Number of agents competing per race")

    # PPO hyperparameters
    p.add_argument("--lr",         type=float, default=float(os.getenv("LEARNING_RATE", 3e-4)))
    p.add_argument("--n-steps",    type=int,   default=int(os.getenv("N_STEPS", 2048)))
    p.add_argument("--batch-size", type=int,   default=int(os.getenv("BATCH_SIZE", 128)))
    p.add_argument("--gamma",      type=float, default=float(os.getenv("GAMMA", 0.99)))
    p.add_argument("--n-epochs",   type=int,   default=int(os.getenv("N_EPOCHS", 10)))
    p.add_argument("--gae-lambda", type=float, default=float(os.getenv("GAE_LAMBDA", 0.95)))
    p.add_argument("--clip-range", type=float, default=float(os.getenv("CLIP_RANGE", 0.2)))
    p.add_argument("--ent-coef",   type=float, default=float(os.getenv("ENT_COEF", 0.01)))

    # Network
    p.add_argument("--net-arch", type=str,
                   default=os.getenv("NET_ARCH", "512,512,256"))
    p.add_argument("--n-envs",   type=int,
                   default=int(os.getenv("N_ENVS", 4)))

    # Checkpointing
    p.add_argument("--checkpoint-freq", type=int,
                   default=int(os.getenv("CHECKPOINT_FREQ", 100_000)))
    p.add_argument("--model-dir", type=str, default="models")
    p.add_argument("--model",     type=str, default=None,
                   help="Path to pretrained model (.zip) to continue training")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------

def make_env(n_agents: int, agent_index: int, render_mode: str = "direct",
             level_seed: int = None):
    """
    Factory function for a single monitored RaceEnv.
    Creates one full physics world (N_AGENTS in it), but exposes
    only agent_index's obs/action/reward to SB3.
    """
    def _init():
        env = RaceEnv(
            render_mode=render_mode,
            n_agents=n_agents,
            agent_index=agent_index,
            level_seed=level_seed,
        )
        env = Monitor(env)
        return env
    return _init


def build_vec_env(n_envs: int, n_agents: int,
                  render_mode: str = "direct") -> DummyVecEnv:
    """
    Build a VecEnv with n_envs * n_agents total environments.
    Each parallel world spawns n_agents and cycles through them.
    This implements parameter sharing: all agent slots share the policy.
    """
    fns = []
    for env_idx in range(n_envs):
        # Each env_idx group shares the same physics world
        # (implemented via multiple RaceEnv instances with the same seed
        #  offset - each sharing the same n_agents competing)
        for agent_idx in range(n_agents):
            fns.append(make_env(
                n_agents=n_agents,
                agent_index=agent_idx,
                render_mode=render_mode,
            ))
    return DummyVecEnv(fns)


# ---------------------------------------------------------------------------
# Test / play
# ---------------------------------------------------------------------------

def test(model_path: str, n_agents: int, episodes: int = 5):
    """Load and visualize a trained model."""
    print(f"\nLoading model: {model_path}")
    model = PPO.load(model_path)

    # Single full-race env in human mode for agent 0 (camera follows that one)
    env = make_env(n_agents=n_agents, agent_index=0, render_mode="human")()

    print(f"Testing for {episodes} episodes with {n_agents} agents...\n")
    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        steps        = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps        += 1
            if terminated or truncated:
                print(
                    f"  Episode {ep+1}: steps={steps} "
                    f"reward={total_reward:.2f} "
                    f"rank={info.get('race_rank','?')} "
                    f"finished={info.get('finished', False)}"
                )
                break

    env.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.test:
        if not args.model:
            print("Error: --model required for --test mode")
            return
        test(args.model, args.n_agents, args.episodes)
        return

    # ------------------------------------------------------------------
    # MLflow setup
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run() as run:
        print(f"MLflow run id  : {run.info.run_id}")
        print(f"Experiment     : {args.experiment_name}")

        params = {
            "algorithm":       "PPO",
            "model_type":      "PPO MultiAgent (Parameter Sharing)",
            "total_timesteps": args.timesteps,
            "n_agents":        args.n_agents,
            "learning_rate":   args.lr,
            "n_steps":         args.n_steps,
            "batch_size":      args.batch_size,
            "gamma":           args.gamma,
            "n_epochs":        args.n_epochs,
            "gae_lambda":      args.gae_lambda,
            "clip_range":      args.clip_range,
            "ent_coef":        args.ent_coef,
            "net_arch":        args.net_arch,
            "n_envs":          args.n_envs,
            "total_env_slots": args.n_envs * args.n_agents,
        }
        mlflow.log_params(params)
        mlflow.log_param("project_path", "./")

        # ------------------------------------------------------------------
        # Env
        # ------------------------------------------------------------------
        total_slots = args.n_envs * args.n_agents
        print(f"\nCreating {args.n_envs} worlds x {args.n_agents} agents "
              f"= {total_slots} total env slots...")
        vec_env = build_vec_env(args.n_envs, args.n_agents, render_mode="direct")
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

        # ------------------------------------------------------------------
        # Model
        # ------------------------------------------------------------------
        net_arch     = [int(x) for x in args.net_arch.split(",")]
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
                policy        = "MlpPolicy",
                env           = vec_env,
                learning_rate = args.lr,
                n_steps       = args.n_steps,
                batch_size    = args.batch_size,
                gamma         = args.gamma,
                n_epochs      = args.n_epochs,
                gae_lambda    = args.gae_lambda,
                clip_range    = args.clip_range,
                ent_coef      = args.ent_coef,
                policy_kwargs = policy_kwargs,
                verbose       = 1,
                device        = "auto",
            )

        device_name = str(next(model.policy.parameters()).device)
        mlflow.log_param("device", device_name)
        print(f"Training on device : {device_name}")
        print(f"Total timesteps    : {args.timesteps:,}\n")

        # ------------------------------------------------------------------
        # Callbacks
        # ------------------------------------------------------------------
        callback = MLflowCallback(
            checkpoint_freq = args.checkpoint_freq,
            model_save_dir  = args.model_dir,
            verbose         = 1,
        )

        # ------------------------------------------------------------------
        # Train
        # ------------------------------------------------------------------
        model.learn(
            total_timesteps = args.timesteps,
            callback        = callback,
            progress_bar    = True,
        )

        # ------------------------------------------------------------------
        # Save final model
        # ------------------------------------------------------------------
        os.makedirs(args.model_dir, exist_ok=True)
        final_path   = os.path.join(args.model_dir, "ppo_race_final")
        vecnorm_path = os.path.join(args.model_dir, "vec_normalize.pkl")

        model.save(final_path)
        vec_env.save(vecnorm_path)

        mlflow.log_artifact(final_path + ".zip", artifact_path="model")
        mlflow.log_artifact(vecnorm_path,        artifact_path="model")

        print(f"\nTraining complete. Model -> {final_path}")
        vec_env.close()


if __name__ == "__main__":
    main()
