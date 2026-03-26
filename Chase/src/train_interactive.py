"""
Interactive training script for Chase environment with in-window controls.

Usage:
    python src/train_interactive.py
    python src/train_interactive.py --model models/ppo_chase_final.zip

Controls (PyBullet window):
    Speed slider         - Training speed multiplier (0.1 - 4.0)
    Timestep Limit slider- Max timesteps (0 = unlimited)
    Pause/Resume button  - Toggle training pause
    Save Checkpoint button - Save model immediately
"""

import argparse
import os
import sys
import threading
import time

import mlflow
import numpy as np
import torch
from dotenv import load_dotenv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

import pybullet as p

from src.env.chase_env import ChaseEnv
from src.agent.callbacks import MLflowCallback

load_dotenv()


def make_env(render_mode: str = "direct"):
    """Factory for a single monitored Chase environment."""
    def _init():
        env = ChaseEnv(render_mode=render_mode)
        env = Monitor(env)
        return env
    return _init


class InteractiveTrainer:
    # PyBullet debug param IDs
    _SLIDER_SPEED    = None
    _SLIDER_LIMIT    = None
    _BTN_PAUSE       = None
    _BTN_SAVE        = None
    _HUD_IDS: list   = []

    def __init__(self, args):
        self.args          = args
        self.paused        = False
        self.running       = True
        self.save_requested = False
        self.current_step  = 0
        self.start_time    = time.time()

        # Stats (updated by callback)
        self.episode_count = 0
        self.last_reward   = 0.0
        self.best_reward   = float("-inf")

        self._setup()
        self._setup_debug_ui()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup(self):
        print(f"\n{'='*60}")
        print("CHASE - INTERACTIVE TRAINING")
        print(f"{'='*60}\n")

        model_path = self.args.model

        print(f"Creating {self.args.n_envs} parallel training environments...")
        vec_env       = DummyVecEnv([make_env("direct") for _ in range(self.args.n_envs)])
        self.vec_env  = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

        # Separate env for visualization (GUI)
        self.render_env = make_env("human")()

        # Grab the physics client ID from the render env so we can add debug widgets
        # The client is created on first reset()
        obs, _ = self.render_env.reset()
        # ChaseEnv wraps Monitor which wraps ChaseEnv; unwrap to get _client
        base_env = self.render_env
        while hasattr(base_env, "env"):
            base_env = base_env.env
        self._physics_client = base_env._client

        net_arch = [int(x) for x in self.args.net_arch.split(",")]
        policy_kwargs = dict(
            net_arch=dict(pi=net_arch, vf=net_arch),
            activation_fn=torch.nn.Tanh,
        )

        if model_path:
            print(f"Loading pretrained model: {model_path}")
            self.model = PPO.load(model_path, env=self.vec_env, device="auto")
            vecnorm_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")
            if os.path.exists(vecnorm_path):
                try:
                    self.vec_env = VecNormalize.load(vecnorm_path, self.vec_env)
                    print(f"Loaded VecNormalize stats: {vecnorm_path}")
                except Exception as e:
                    print(f"Could not load VecNormalize: {e}")
            self.current_step = self.model.num_timesteps
            print(f"Continuing from step {self.current_step:,}")
        else:
            print("Training from scratch...")
            self.model = PPO(
                policy="MlpPolicy",
                env=self.vec_env,
                learning_rate=self.args.lr,
                n_steps=self.args.n_steps,
                batch_size=self.args.batch_size,
                gamma=self.args.gamma,
                n_epochs=self.args.n_epochs,
                gae_lambda=self.args.gae_lambda,
                clip_range=self.args.clip_range,
                ent_coef=self.args.ent_coef,
                policy_kwargs=policy_kwargs,
                verbose=0,
                device="auto",
            )

        os.makedirs(self.args.model_dir, exist_ok=True)
        limit_str = f"{self.args.timesteps:,}" if self.args.timesteps > 0 else "unlimited"
        print(f"Timestep limit: {limit_str}")
        print(f"Model dir: {self.args.model_dir}\n")

    def _setup_debug_ui(self):
        """Add sliders and buttons to the PyBullet debug window."""
        client = self._physics_client

        # Speed slider: 0.1 to 4.0, default 1.0
        self._SLIDER_SPEED = p.addUserDebugParameter(
            "Speed", 0.1, 4.0, 1.0,
            physicsClientId=client)

        # Timestep limit slider: 0 to 5_000_000, default = args.timesteps (or 0 if unlimited)
        default_limit = self.args.timesteps if self.args.timesteps > 0 else 0
        self._SLIDER_LIMIT = p.addUserDebugParameter(
            "Timestep Limit (0=unlimited)", 0, 5_000_000, default_limit,
            physicsClientId=client)

        # Buttons (implemented as sliders that go 0->1 - standard PyBullet trick)
        self._BTN_PAUSE = p.addUserDebugParameter(
            "Pause / Resume", 1, 0, 1,
            physicsClientId=client)

        self._BTN_SAVE = p.addUserDebugParameter(
            "Save Checkpoint", 1, 0, 1,
            physicsClientId=client)

        # Store previous button values to detect clicks
        self._prev_pause_val = p.readUserDebugParameter(self._BTN_PAUSE, physicsClientId=client)
        self._prev_save_val  = p.readUserDebugParameter(self._BTN_SAVE,  physicsClientId=client)

        self._HUD_IDS = []

    # ------------------------------------------------------------------
    # UI polling helpers
    # ------------------------------------------------------------------

    def _read_ui(self):
        """Read slider values and detect button presses."""
        client = self._physics_client
        try:
            speed = float(p.readUserDebugParameter(self._SLIDER_SPEED, physicsClientId=client))
            speed = max(0.1, min(4.0, speed))

            raw_limit = float(p.readUserDebugParameter(self._SLIDER_LIMIT, physicsClientId=client))
            timestep_limit = int(raw_limit)

            # Pause button - detect flip
            pause_val = p.readUserDebugParameter(self._BTN_PAUSE, physicsClientId=client)
            if pause_val != self._prev_pause_val:
                self.paused = not self.paused
                self._prev_pause_val = pause_val

            # Save button - detect flip
            save_val  = p.readUserDebugParameter(self._BTN_SAVE, physicsClientId=client)
            if save_val != self._prev_save_val:
                self.save_requested = True
                self._prev_save_val  = save_val

            return speed, timestep_limit
        except Exception:
            return 1.0, self.args.timesteps

    def _update_hud(self, speed: float, target: int):
        """Refresh HUD text in the 3D scene."""
        client   = self._physics_client
        elapsed  = time.time() - self.start_time
        h, m, s  = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)

        limit_str = f"{target:,}" if target > 0 else "unlimited"
        status    = "PAUSED" if self.paused else "running"

        lines = [
            f"Status:  {status}",
            f"Step:    {self.current_step:,} / {limit_str}",
            f"Episode: {self.episode_count}",
            f"Reward:  last={self.last_reward:.1f}  best={self.best_reward:.1f}",
            f"Speed:   {speed:.1f}x",
            f"Time:    {h:02d}:{m:02d}:{s:02d}",
        ]

        # Remove old text objects
        for tid in self._HUD_IDS:
            try:
                p.removeUserDebugItem(tid, physicsClientId=client)
            except Exception:
                pass
        self._HUD_IDS = []

        # Draw new text (stacked vertically above scene)
        for i, line in enumerate(lines):
            y_pos = 10 - i * 1.2   # stack downward in scene coords
            try:
                tid = p.addUserDebugText(
                    line,
                    textPosition=[0, y_pos, 3.5 - i * 0.35],
                    textColorRGB=[1.0, 1.0, 0.0],
                    textSize=1.2,
                    lifeTime=0,
                    physicsClientId=client,
                )
                self._HUD_IDS.append(tid)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, mlflow_callback: MLflowCallback):
        print("Starting training. Use the PyBullet window controls.\n")

        steps_per_call   = self.args.n_steps * self.args.n_envs
        last_render      = 0
        last_save_time   = time.time()
        last_hud_update  = 0

        # Infinite loop unless a limit is set
        while self.running:
            speed, target_timesteps = self._read_ui()

            # Sync callback stats
            self.episode_count = mlflow_callback.episode_count
            self.last_reward   = mlflow_callback.last_reward
            self.best_reward   = mlflow_callback.best_reward

            # Check timestep limit
            if target_timesteps > 0 and self.current_step >= target_timesteps:
                print(f"\nTimestep limit reached ({target_timesteps:,}). Stopping.")
                break

            # Update HUD ~2x/sec
            if time.time() - last_hud_update > 0.5:
                self._update_hud(speed, target_timesteps)
                last_hud_update = time.time()

            if self.paused:
                time.sleep(0.1)
                continue

            # Determine how many rollout calls to do this tick
            iters = max(1, int(speed))

            for _ in range(iters):
                if target_timesteps > 0 and self.current_step >= target_timesteps:
                    break

                self.model.learn(
                    total_timesteps=steps_per_call,
                    reset_num_timesteps=False,
                    progress_bar=False,
                    callback=mlflow_callback,
                )
                self.current_step += steps_per_call

                # Periodic render
                if self.current_step - last_render > 500:
                    self._render()
                    last_render = self.current_step

            # Auto-save every 60 s
            if time.time() - last_save_time > 60:
                self._save_checkpoint()
                last_save_time = time.time()

            # Handle manual save request
            if self.save_requested:
                self._save_checkpoint()
                self.save_requested = False

            # When speed < 1, apply a sleep to slow things down
            if speed < 1.0:
                time.sleep((1.0 / speed - 1.0) * 0.01)

        print("\nTraining finished.")
        self._save_final()

    # ------------------------------------------------------------------
    # Render & save helpers
    # ------------------------------------------------------------------

    def _render(self):
        """Run a few steps in the render env to show the current policy."""
        try:
            obs, _ = self.render_env.reset()
            for _ in range(50):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, _, _ = self.render_env.step(action)
                if done:
                    obs, _ = self.render_env.reset()
        except Exception as e:
            print(f"Render error: {e}")

    def _save_checkpoint(self):
        checkpoint_name = f"ppo_chase_{self.current_step:010d}"
        path = os.path.join(self.args.model_dir, checkpoint_name)
        self.model.save(path)
        vecnorm_path = os.path.join(self.args.model_dir, "vec_normalize.pkl")
        self.vec_env.save(vecnorm_path)
        print(f"\n>>> Checkpoint saved: {path}")

    def _save_final(self):
        final_path   = os.path.join(self.args.model_dir, "ppo_chase_final")
        vecnorm_path = os.path.join(self.args.model_dir, "vec_normalize.pkl")
        self.model.save(final_path)
        self.vec_env.save(vecnorm_path)
        print(f"Final model saved: {final_path}")

    def close(self):
        try:
            self.render_env.close()
        except Exception:
            pass
        try:
            self.vec_env.close()
        except Exception:
            pass


# ------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------

def parse_args():
    p_arg = argparse.ArgumentParser(description="Interactive training for Chase with PyBullet UI")

    p_arg.add_argument("--model",        type=str,   default=None,   help="Pretrained model to load (.zip)")
    p_arg.add_argument("--timesteps",    type=int,   default=0,      help="Timestep limit (0 = unlimited)")
    p_arg.add_argument("--lr",           type=float, default=float(os.getenv("LEARNING_RATE", 3e-4)))
    p_arg.add_argument("--n-steps",      type=int,   default=int(os.getenv("N_STEPS", 2048)))
    p_arg.add_argument("--batch-size",   type=int,   default=int(os.getenv("BATCH_SIZE", 64)))
    p_arg.add_argument("--gamma",        type=float, default=float(os.getenv("GAMMA", 0.99)))
    p_arg.add_argument("--n-epochs",     type=int,   default=int(os.getenv("N_EPOCHS", 10)))
    p_arg.add_argument("--gae-lambda",   type=float, default=float(os.getenv("GAE_LAMBDA", 0.95)))
    p_arg.add_argument("--clip-range",   type=float, default=float(os.getenv("CLIP_RANGE", 0.2)))
    p_arg.add_argument("--ent-coef",     type=float, default=float(os.getenv("ENT_COEF", 0.01)))
    p_arg.add_argument("--net-arch",     type=str,   default=os.getenv("NET_ARCH", "256,256"))
    p_arg.add_argument("--n-envs",       type=int,   default=int(os.getenv("N_ENVS", 4)))
    p_arg.add_argument("--model-dir",    type=str,   default="models")
    p_arg.add_argument("--checkpoint-freq", type=int, default=int(os.getenv("CHECKPOINT_FREQ", 50_000)))
    p_arg.add_argument("--experiment-name", type=str, default=os.getenv("EXPERIMENT_NAME", "chase_ppo"))
    p_arg.add_argument("--tracking-uri",    type=str, default=os.getenv("MLFLOW_TRACKING_URI", "mlruns"))

    return p_arg.parse_args()


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    args = parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    trainer = InteractiveTrainer(args)

    with mlflow.start_run() as run:
        print(f"MLflow run id: {run.info.run_id}")

        params = {
            "algorithm":   "PPO",
            "learning_rate": args.lr,
            "n_steps":     args.n_steps,
            "batch_size":  args.batch_size,
            "gamma":       args.gamma,
            "n_epochs":    args.n_epochs,
            "gae_lambda":  args.gae_lambda,
            "clip_range":  args.clip_range,
            "ent_coef":    args.ent_coef,
            "net_arch":    args.net_arch,
            "n_envs":      args.n_envs,
            "timestep_limit": args.timesteps if args.timesteps > 0 else "unlimited",
        }
        mlflow.log_params(params)

        device_name = str(next(trainer.model.policy.parameters()).device)
        mlflow.log_param("device", device_name)

        callback = MLflowCallback(
            checkpoint_freq=args.checkpoint_freq,
            model_save_dir=args.model_dir,
            verbose=1,
        )

        try:
            trainer.train(callback)
        finally:
            trainer.close()


if __name__ == "__main__":
    main()
