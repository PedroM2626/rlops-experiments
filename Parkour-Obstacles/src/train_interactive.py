"""
Interactive training with real-time visualization and controls.

Usage:
    python src/train_interactive.py --model models/ppo_parkour_final.zip --timesteps 3000000

Controls:
    SPACE   - Pause/Resume
    +/-     - Speed up/Slow down
    Q       - Quit and save
    S       - Save checkpoint
"""

import argparse
import os
import sys
import threading
import time

import numpy as np
import torch
from dotenv import load_dotenv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from src.env.parkour_env import ParkourEnv

load_dotenv()


class InteractiveTrainer:
    def __init__(self, args):
        self.args = args
        self.paused = False
        self.speed = 1.0  # 1x speed
        self.running = True
        self.save_requested = False
        self.current_step = 0
        
        self._setup()
        self._setup_input()
    
    def _setup(self):
        print(f"\n{'='*60}")
        print("INTERACTIVE TRAINING")
        print(f"{'='*60}")
        
        # Determine model path
        model_path = self.args.model or self.args.load_model
        
        # Environment
        print(f"\nCreating {self.args.n_envs} parallel environments...")
        vec_env = DummyVecEnv([self._make_env("direct") for _ in range(self.args.n_envs)])
        self.vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        
        # Visualization env (separate, for rendering)
        self.render_env = self._make_env("human")()
        
        # Model
        net_arch = [int(x) for x in self.args.net_arch.split(",")]
        policy_kwargs = dict(
            net_arch=dict(pi=net_arch, vf=net_arch),
            activation_fn=torch.nn.Tanh,
        )
        
        if model_path:
            print(f"\nLoading pretrained model: {model_path}")
            self.model = PPO.load(
                model_path,
                env=self.vec_env,
                device="auto",
            )
            # Try to load vec_normalize stats
            vecnorm_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")
            if os.path.exists(vecnorm_path):
                try:
                    self.vec_env = VecNormalize.load(vecnorm_path, self.vec_env)
                    print(f"Loaded VecNormalize stats from: {vecnorm_path}")
                except Exception as e:
                    print(f"Could not load VecNormalize: {e}")
            
            # Reset the model timesteps to continue training
            self.current_step = self.model.num_timesteps
            print(f"Continuing from step {self.current_step:,}")
        else:
            print("\nTraining from scratch...")
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
        print(f"Total timesteps target: {self.args.timesteps:,}")
        print(f"Model dir: {self.args.model_dir}")
    
    def _make_env(self, render_mode):
        def _init():
            env = ParkourEnv(render_mode=render_mode, level_seed=self.args.seed)
            env = Monitor(env)
            return env
        return _init
    
    def _setup_input(self):
        """Setup keyboard input handling"""
        if sys.platform == 'win32':
            import msvcrt
            self._input_thread = threading.Thread(target=self._input_loop, daemon=True)
            self._input_thread.start()
        else:
            import select
            self._input_thread = threading.Thread(target=self._input_loop, daemon=True)
            self._input_thread.start()
    
    def _input_loop(self):
        """Handle keyboard input"""
        print("\n" + "="*60)
        print("CONTROLS:")
        print("  SPACE   - Pause/Resume training")
        print("  + / -   - Speed up / Slow down")
        print("  S       - Save checkpoint")
        print("  Q       - Quit and save")
        print("="*60 + "\n")
        
        while self.running:
            try:
                if sys.platform == 'win32':
                    import msvcrt
                    if msvcrt.kbhit():
                        key = msvcrt.getch().decode('utf-8', errors='ignore')
                        self._handle_key(key)
                else:
                    import select
                    import tty
                    import termios
                    if select.select([sys.stdin], [], [], 0)[0]:
                        key = sys.stdin.read(1)
                        self._handle_key(key)
            except:
                pass
            time.sleep(0.05)
    
    def _handle_key(self, key):
        key = key.lower()
        
        if key == ' ':
            self.paused = not self.paused
            status = "PAUSED" if self.paused else "RUNNING"
            print(f"\n>>> {status}")
        
        elif key in ['+', '=']:
            self.speed = min(self.speed * 1.5, 10.0)
            print(f"\n>>> Speed: {self.speed:.1f}x")
        
        elif key == '-':
            self.speed = max(self.speed / 1.5, 0.1)
            print(f"\n>>> Speed: {self.speed:.1f}x")
        
        elif key == 's':
            self.save_requested = True
            print(f"\n>>> Save requested...")
        
        elif key == 'q':
            self.running = False
            print(f"\n>>> QUITTING...")
    
    def train(self):
        print(f"\nStarting training... Press SPACE to pause, Q to quit.\n")
        
        last_render = 0
        last_print = time.time()
        last_save = time.time()
        
        target_timesteps = self.args.timesteps
        total_steps = target_timesteps - self.current_step
        
        # Training loop with manual control
        steps_per_call = self.args.n_steps * self.args.n_envs
        
        while self.current_step < target_timesteps and self.running:
            if not self.paused:
                # Adjust iterations based on speed
                iters = max(1, int(self.speed))
                
                for _ in range(iters):
                    if self.current_step >= target_timesteps:
                        break
                    
                    self.model.learn(
                        total_timesteps=steps_per_call,
                        reset_num_timesteps=False,
                        progress_bar=False,
                    )
                    self.current_step += steps_per_call
                    
                    # Render occasionally
                    if self.current_step - last_render > 500 and not self.paused:
                        self._render()
                        last_render = self.current_step
                
                # Save checkpoint periodically
                if time.time() - last_save > 60:  # Every 60 seconds
                    self._save_checkpoint()
                    last_save = time.time()
            
            # Status print
            if time.time() - last_print > 0.5:
                progress = (self.current_step / target_timesteps) * 100
                speed_str = f"{self.speed:.1f}x" if self.speed != 1 else ""
                status = "PAUSED" if self.paused else ""
                print(f"\r  Step: {self.current_step:,} / {target_timesteps:,} ({progress:.1f}%) {speed_str} {status}", end="", flush=True)
                last_print = time.time()
            
            # Handle save request
            if self.save_requested:
                self._save_checkpoint()
                self.save_requested = False
            
            time.sleep(0.01)
        
        print(f"\n\nTraining complete!")
        self._save_final()
    
    def _render(self):
        """Render the visualization environment"""
        try:
            obs, _ = self.render_env.reset()
            for _ in range(30):  # Render a few steps
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, _, _ = self.render_env.step(action)
                if done:
                    obs, _ = self.render_env.reset()
        except Exception as e:
            print(f"Render error: {e}")
    
    def _save_checkpoint(self):
        """Save a checkpoint"""
        checkpoint_name = f"ppo_parkour_{self.current_step:010d}"
        path = os.path.join(self.args.model_dir, checkpoint_name)
        self.model.save(path)
        
        # Save VecNormalize
        vecnorm_path = os.path.join(self.args.model_dir, "vec_normalize.pkl")
        self.vec_env.save(vecnorm_path)
        
        print(f"\n>>> Checkpoint saved: {path}")
    
    def _save_final(self):
        """Save final model"""
        final_path = os.path.join(self.args.model_dir, "ppo_parkour_final")
        self.model.save(final_path)
        
        vecnorm_path = os.path.join(self.args.model_dir, "vec_normalize.pkl")
        self.vec_env.save(vecnorm_path)
        
        print(f"Final model saved to: {final_path}")
        
        self.render_env.close()
        self.vec_env.close()


def parse_args():
    p = argparse.ArgumentParser(description="Interactive training with visualization")
    
    p.add_argument("--model", type=str, default=None,
                   help="Path to pretrained model (.zip)")
    p.add_argument("--load-model", type=str, default=None,
                   help="Path to pretrained model (.zip)")
    p.add_argument("--timesteps", type=int, default=3_000_000,
                   help="Total training timesteps")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--net-arch", type=str, default="256,256")
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--model-dir", type=str, default="models")
    p.add_argument("--seed", type=int, default=None)
    
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.seed is None:
        args.seed = np.random.randint(0, 10000)
    
    trainer = InteractiveTrainer(args)
    trainer.train()
