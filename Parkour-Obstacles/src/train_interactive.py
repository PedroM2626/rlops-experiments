"""
Interactive training script for Parkour environment with in-window controls.

Usage:
    python src/train_interactive.py
    python src/train_interactive.py --model models/ppo_parkour_final.zip --timesteps 3000000

Controls (PyBullet window):
    Speed slider           - Training speed multiplier (0.1 - 4.0)
    Timestep Limit slider  - Max timesteps (0 = unlimited)
    Pause/Resume button    - Toggle training pause
    Save Checkpoint button - Save model immediately
"""

import argparse
import os
import sys
import time

import mlflow
import numpy as np
import torch
from dotenv import load_dotenv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

import pybullet as p

from src.env.parkour_env import ParkourEnv
from src.agent.callbacks import MLflowCallback

load_dotenv()


def make_env(render_mode: str = "direct", seed: int = None):
    """Factory for a single monitored Parkour environment."""
    def _init():
        env = ParkourEnv(render_mode=render_mode, level_seed=seed)
        env = Monitor(env)
        return env
    return _init


class InteractiveUICallback(BaseCallback):
    """Callback that polls PyBullet UI, updates HUD, and controls training pace/pauses."""
    def __init__(self, trainer, verbose=0):
        super().__init__(verbose)
        self.trainer = trainer
        self.last_hud_update = 0
        self.last_save_time = time.time()

    def _on_step(self):
        speed, target_timesteps = self.trainer._read_ui()
        self.trainer.current_step = self.num_timesteps

        # Update HUD ~2x a second
        if time.time() - self.last_hud_update > 0.5:
            self.trainer._update_hud(speed, target_timesteps)
            self.last_hud_update = time.time()

        # Handle pause
        while self.trainer.paused and self.trainer.running:
            time.sleep(0.1)
            # Still need to read UI while paused so we can unpause
            speed, target_timesteps = self.trainer._read_ui()
            if time.time() - self.last_hud_update > 0.5:
                self.trainer._update_hud(speed, target_timesteps)
                self.last_hud_update = time.time()

        if not self.trainer.running:
            return False

        if self.trainer.save_requested:
            self.trainer._save_checkpoint()
            self.trainer.save_requested = False

        if target_timesteps > 0 and self.num_timesteps >= target_timesteps:
            print(f"\nTimestep limit reached ({target_timesteps:,}). Stopping.")
            return False

        if time.time() - self.last_save_time > 60:
            self.trainer._save_checkpoint()
            self.last_save_time = time.time()

        # Step the visual environment to animate the agent live
        self.trainer._render_step()

        if speed < 1.0:
            time.sleep((1.0 / speed - 1.0) * 0.01)

        return True


class InteractiveTrainer:

    def __init__(self, args):
        self.args           = args
        self.paused         = False
        self.running        = True
        self.save_requested = False
        self.current_step   = 0
        self.start_time     = time.time()

        # Stats (synced from callback)
        self.episode_count  = 0
        self.last_reward    = 0.0
        self.best_reward    = float("-inf")

        self.mlflow_callback = None

        # PyBullet debug param IDs
        self._slider_speed  = None
        self._slider_limit  = None
        self._btn_pause     = None
        self._btn_save      = None
        self._hud_ids: list = []
        self._prev_pause_val = None
        self._prev_save_val  = None
        self._physics_client = None

        self._setup()
        self._setup_debug_ui()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup(self):
        print(f"\n{'='*60}")
        print("PARKOUR - INTERACTIVE TRAINING")
        print(f"{'='*60}\n")

        model_path = self.args.model or self.args.load_model

        print(f"Creating {self.args.n_envs} parallel training environments...")
        vec_env      = DummyVecEnv([make_env("direct", self.args.seed) for _ in range(self.args.n_envs)])
        self.vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

        # Separate visualization env
        self.render_env = make_env("human", self.args.seed)()

        # Initialize render env now so we can get the physics client
        self.render_obs, _ = self.render_env.reset()
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

        # Ensure GUI is enabled just in case
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=client)

        self._slider_speed = p.addUserDebugParameter(
            "Speed", 0.1, 4.0, 1.0,
            physicsClientId=client)

        default_limit = self.args.timesteps if self.args.timesteps > 0 else 0
        self._slider_limit = p.addUserDebugParameter(
            "Timestep Limit (0=unlimited)", 0, 5_000_000, default_limit,
            physicsClientId=client)

        self._btn_pause = p.addUserDebugParameter(
            "Pause / Resume", 1, 0, 1,
            physicsClientId=client)

        self._btn_save = p.addUserDebugParameter(
            "Save Checkpoint", 1, 0, 1,
            physicsClientId=client)

        self._prev_pause_val = p.readUserDebugParameter(self._btn_pause, physicsClientId=client)
        self._prev_save_val  = p.readUserDebugParameter(self._btn_save,  physicsClientId=client)

    # ------------------------------------------------------------------
    # UI polling
    # ------------------------------------------------------------------

    def _read_ui(self):
        """Read slider values and detect button clicks. Returns (speed, timestep_limit)."""
        client = self._physics_client
        try:
            speed = float(p.readUserDebugParameter(self._slider_speed, physicsClientId=client))
            speed = max(0.1, min(4.0, speed))

            raw_limit     = float(p.readUserDebugParameter(self._slider_limit, physicsClientId=client))
            timestep_limit = int(raw_limit)

            pause_val = p.readUserDebugParameter(self._btn_pause, physicsClientId=client)
            if pause_val != self._prev_pause_val:
                self.paused = not self.paused
                self._prev_pause_val = pause_val

            save_val = p.readUserDebugParameter(self._btn_save, physicsClientId=client)
            if save_val != self._prev_save_val:
                self.save_requested = True
                self._prev_save_val = save_val

            return speed, timestep_limit
        except Exception:
            return 1.0, self.args.timesteps

    def _update_hud(self, speed: float, target: int):
        """Refresh HUD text rendered inside the 3D scene."""
        client   = self._physics_client
        elapsed  = time.time() - self.start_time
        h, m, s  = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)

        limit_str = f"{target:,}" if target > 0 else "unlimited"
        status    = "PAUSED" if self.paused else "running"

        # Sync stats from MLflow
        if self.mlflow_callback is not None:
            self.episode_count = self.mlflow_callback.episode_count
            self.last_reward   = self.mlflow_callback.last_reward
            self.best_reward   = self.mlflow_callback.best_reward

        lines = [
            f"Status:  {status}",
            f"Step:    {self.current_step:,} / {limit_str}",
            f"Episode: {self.episode_count}",
            f"Reward:  last={self.last_reward:.1f}  best={self.best_reward:.1f}",
            f"Speed:   {speed:.1f}x",
            f"Time:    {h:02d}:{m:02d}:{s:02d}",
        ]

        for tid in self._hud_ids:
            try:
                p.removeUserDebugItem(tid, physicsClientId=client)
            except Exception:
                pass
        self._hud_ids = []

        for i, line in enumerate(lines):
            try:
                tid = p.addUserDebugText(
                    line,
                    textPosition=[0, 8 - i * 1.1, 4.0 - i * 0.35],
                    textColorRGB=[1.0, 1.0, 0.0],
                    textSize=1.2,
                    lifeTime=0,
                    physicsClientId=client,
                )
                self._hud_ids.append(tid)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, mlflow_callback: MLflowCallback):
        print("Starting training. Use the PyBullet window controls.\n")
        self.mlflow_callback = mlflow_callback

        ui_callback = InteractiveUICallback(self)
        callbacks = CallbackList([mlflow_callback, ui_callback])

        huge_timesteps = 5_000_000_000

        try:
            self.model.learn(
                total_timesteps=huge_timesteps,
                reset_num_timesteps=False,
                progress_bar=False,
                callback=callbacks,
            )
        except KeyboardInterrupt:
            print("\nInterrupted by user.")

        print("\nTraining finished.")
        self._save_final()

    # ------------------------------------------------------------------
    # Render & save helpers
    # ------------------------------------------------------------------

    def _render_step(self):
        """Run a single step in the render env to show the current policy live."""
        try:
            action, _ = self.model.predict(self.render_obs, deterministic=True)
            self.render_obs, _, done, _, _ = self.render_env.step(action)
            if done:
                self.render_obs, _ = self.render_env.reset()
        except Exception as e:
            pass

    def _save_checkpoint(self):
        name = f"ppo_parkour_{self.current_step:010d}"
        path = os.path.join(self.args.model_dir, name)
        self.model.save(path)
        vecnorm_path = os.path.join(self.args.model_dir, "vec_normalize.pkl")
        self.vec_env.save(vecnorm_path)
        print(f"\n>>> Checkpoint saved: {path}")

    def _save_final(self):
        final_path   = os.path.join(self.args.model_dir, "ppo_parkour_final")
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
    p_arg = argparse.ArgumentParser(description="Interactive training for Parkour with PyBullet UI")

    p_arg.add_argument("--model",       type=str,   default=None,   help="Pretrained model (.zip)")
    p_arg.add_argument("--load-model",  type=str,   default=None,   help="Alias for --model")
    p_arg.add_argument("--timesteps",   type=int,   default=0,      help="Timestep limit (0 = unlimited)")
    p_arg.add_argument("--lr",          type=float, default=float(os.getenv("LEARNING_RATE", 3e-4)))
    p_arg.add_argument("--n-steps",     type=int,   default=int(os.getenv("N_STEPS", 2048)))
    p_arg.add_argument("--batch-size",  type=int,   default=int(os.getenv("BATCH_SIZE", 64)))
    p_arg.add_argument("--gamma",       type=float, default=float(os.getenv("GAMMA", 0.99)))
    p_arg.add_argument("--n-epochs",    type=int,   default=int(os.getenv("N_EPOCHS", 10)))
    p_arg.add_argument("--gae-lambda",  type=float, default=float(os.getenv("GAE_LAMBDA", 0.95)))
    p_arg.add_argument("--clip-range",  type=float, default=float(os.getenv("CLIP_RANGE", 0.2)))
    p_arg.add_argument("--ent-coef",    type=float, default=float(os.getenv("ENT_COEF", 0.01)))
    p_arg.add_argument("--net-arch",    type=str,   default=os.getenv("NET_ARCH", "256,256"))
    p_arg.add_argument("--n-envs",      type=int,   default=int(os.getenv("N_ENVS", 4)))
    p_arg.add_argument("--model-dir",   type=str,   default="models")
    p_arg.add_argument("--seed",        type=int,   default=None)
    p_arg.add_argument("--checkpoint-freq", type=int, default=int(os.getenv("CHECKPOINT_FREQ", 50_000)))
    p_arg.add_argument("--experiment-name", type=str, default=os.getenv("EXPERIMENT_NAME", "parkour_ppo"))
    p_arg.add_argument("--tracking-uri",    type=str, default=os.getenv("MLFLOW_TRACKING_URI", "mlruns"))

    return p_arg.parse_args()


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    args = parse_args()

    if args.seed is None:
        args.seed = np.random.randint(0, 10000)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    trainer = InteractiveTrainer(args)

    with mlflow.start_run() as run:
        print(f"MLflow run id: {run.info.run_id}")

        params = {
            "algorithm":      "PPO",
            "learning_rate":  args.lr,
            "n_steps":        args.n_steps,
            "batch_size":     args.batch_size,
            "gamma":          args.gamma,
            "n_epochs":       args.n_epochs,
            "gae_lambda":     args.gae_lambda,
            "clip_range":     args.clip_range,
            "ent_coef":       args.ent_coef,
            "net_arch":       args.net_arch,
            "n_envs":         args.n_envs,
            "seed":           args.seed,
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
