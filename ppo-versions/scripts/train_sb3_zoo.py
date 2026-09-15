"""SB3 com os hiperparâmetros OFICIAIS do RL Baselines3 Zoo especificamente
pro LunarLander (github.com/DLR-RM/rl-baselines3-zoo, hyperparams/ppo.yml),
em vez dos hiperparâmetros genéricos de common.py (que foram calibrados
pro CartPole). Objetivo: confirmar se reward negativo nas outras
implementações é config ruim pra essa tarefa, não bug.

Hiperparâmetros do Zoo pra LunarLander-v3:
    n_envs: 16, n_steps: 1024, batch_size: 64, gae_lambda: 0.98,
    gamma: 0.999, n_epochs: 4, ent_coef: 0.01
    (lr, clip_range, vf_coef, max_grad_norm, arquitetura: default do SB3)
"""
import time
import sys
sys.path.insert(0, "/home/claude/ppo-benchmark/scripts")
from common import ENV_ID, SEED, save_result, evaluate_policy, save_eval_result

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

ZOO_CONFIG = dict(
    n_envs=16,
    n_steps=1024,
    batch_size=64,
    gae_lambda=0.98,
    gamma=0.999,
    n_epochs=4,
    ent_coef=0.01,
    total_timesteps=1_000_000,
)


class RewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.history = []

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            mean_r = sum(ep["r"] for ep in self.model.ep_info_buffer) / len(self.model.ep_info_buffer)
            self.history.append((self.num_timesteps, mean_r))
        return True


def make_env():
    def _f():
        return Monitor(gym.make(ENV_ID))
    return _f


def main():
    cfg = ZOO_CONFIG
    env = DummyVecEnv([make_env() for _ in range(cfg["n_envs"])])

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        gae_lambda=cfg["gae_lambda"],
        gamma=cfg["gamma"],
        n_epochs=cfg["n_epochs"],
        ent_coef=cfg["ent_coef"],
        seed=SEED,
        verbose=0,
    )

    logger = RewardLogger()
    t0 = time.time()
    model.learn(total_timesteps=cfg["total_timesteps"], callback=logger)
    elapsed = time.time() - t0

    save_result("stable_baselines3_zoo", elapsed, logger.history)

    def act_fn(obs):
        action, _ = model.predict(obs, deterministic=False)
        return int(action)

    eval_returns = evaluate_policy(ENV_ID, act_fn)
    save_eval_result("stable_baselines3_zoo", elapsed, eval_returns)
    env.close()


if __name__ == "__main__":
    main()
