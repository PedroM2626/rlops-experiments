import time
import sys
sys.path.insert(0, "/home/claude/ppo-benchmark/scripts")
from common import ENV_ID, SEED, TOTAL_TIMESTEPS, PPO_CONFIG, save_result, evaluate_policy, save_eval_result

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor


class RewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.history = []

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            mean_r = sum(ep["r"] for ep in self.model.ep_info_buffer) / len(self.model.ep_info_buffer)
            self.history.append((self.num_timesteps, mean_r))
        return True


import torch.nn as nn

def make_env():
    def _f():
        return Monitor(gym.make(ENV_ID))
    return _f


def main():
    cfg = PPO_CONFIG
    env = DummyVecEnv([make_env() for _ in range(cfg["n_envs"])])

    policy_kwargs = dict(
        net_arch=dict(pi=list(cfg["hidden_sizes"]), vf=list(cfg["hidden_sizes"])),
        activation_fn=nn.Tanh,
        ortho_init=True,  # gains: sqrt(2) hidden, 0.01 policy-out, 1.0 value-out (default do SB3, igual às outras libs)
        optimizer_kwargs=dict(eps=cfg["adam_eps"]),
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=cfg["lr"],
        n_steps=cfg["n_steps"],
        batch_size=(cfg["n_steps"] * cfg["n_envs"]) // cfg["minibatches"],
        n_epochs=cfg["n_epochs"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        clip_range=cfg["clip_coef"],
        ent_coef=cfg["ent_coef"],
        vf_coef=cfg["vf_coef"],
        max_grad_norm=cfg["max_grad_norm"],
        policy_kwargs=policy_kwargs,
        seed=SEED,
        verbose=0,
    )

    logger = RewardLogger()
    t0 = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=logger)
    elapsed = time.time() - t0

    save_result("stable_baselines3", elapsed, logger.history)
    env.close()

    def act_fn(obs):
        action, _ = model.predict(obs, deterministic=False)
        return int(action)

    eval_returns = evaluate_policy(ENV_ID, act_fn)
    save_eval_result("stable_baselines3", elapsed, eval_returns)


if __name__ == "__main__":
    main()
