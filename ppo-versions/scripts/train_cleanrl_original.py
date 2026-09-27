"""Runs the official ppo.py script from the CleanRL repository (github.com/vwxyzjn/cleanrl,
master branch) as a subprocess, passing the SAME hyperparameters used by the
other 3 implementations. This is their real implementation, not a
reproduction -- unlike 'cleanrl_style_pytorch', which is merely inspired by its style.
"""
import re
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

sys.path.insert(0, "/home/claude/ppo-benchmark/scripts")
from common import ENV_ID, SEED, TOTAL_TIMESTEPS, PPO_CONFIG, save_result, evaluate_policy, save_eval_result

LINE_RE = re.compile(r"global_step=(\d+), episodic_return=([\-\d.]+)")


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """Exact replica of the Agent class in cleanrl_original_ppo.py, used only to load
    the saved weights and evaluate -- it takes no part in training."""

    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )

    def act(self, obs):
        logits = self.actor(obs)
        return Categorical(logits=logits).sample()


def main():
    cfg = PPO_CONFIG
    script = "/home/claude/ppo-benchmark/scripts/cleanrl_original_ppo.py"

    cmd = [
        sys.executable, script,
        "--env-id", ENV_ID,
        "--seed", str(SEED),
        "--total-timesteps", str(TOTAL_TIMESTEPS),
        "--learning-rate", str(cfg["lr"]),
        "--num-envs", str(cfg["n_envs"]),
        "--num-steps", str(cfg["n_steps"]),
        "--gamma", str(cfg["gamma"]),
        "--gae-lambda", str(cfg["gae_lambda"]),
        "--num-minibatches", str(cfg["minibatches"]),
        "--update-epochs", str(cfg["n_epochs"]),
        "--clip-coef", str(cfg["clip_coef"]),
        "--ent-coef", str(cfg["ent_coef"]),
        "--vf-coef", str(cfg["vf_coef"]),
        "--max-grad-norm", str(cfg["max_grad_norm"]),
        "--no-anneal-lr",     # our other 3 libs use a constant LR -- disabled for an identical comparison
        "--no-clip-vloss",    # none of the other 3 clips the value loss -- disabled for an identical comparison
        "--no-cuda",          # CPU, same as the others (no GPU in the benchmark)
        "--no-torch-deterministic",
    ]

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd="/home/claude/ppo-benchmark")
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print("STDOUT:", proc.stdout[-3000:])
        print("STDERR:", proc.stderr[-3000:])
        raise RuntimeError(f"cleanrl_original failed with code {proc.returncode}")

    episodes = []
    for line in proc.stdout.splitlines():
        m = LINE_RE.search(line)
        if m:
            episodes.append((int(m.group(1)), float(m.group(2))))

    # same smoothing (20-episode window) used by the other implementations
    smoothed = []
    window = []
    for ts, r in episodes:
        window.append(r)
        if len(window) > 20:
            window.pop(0)
        smoothed.append((ts, sum(window) / len(window)))

    save_result("cleanrl_original", elapsed, smoothed)

    import gymnasium as gym
    tmp_env = gym.make(ENV_ID)
    obs_dim = tmp_env.observation_space.shape[0]
    act_dim = tmp_env.action_space.n
    tmp_env.close()

    ckpt_path = f"/home/claude/ppo-benchmark/results/_ckpt_cleanrl_original_seed{SEED}.pt"
    agent = Agent(obs_dim, act_dim)
    agent.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    agent.eval()

    def act_fn(obs):
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = agent.act(obs_t)
        return int(action.item())

    eval_returns = evaluate_policy(ENV_ID, act_fn)
    save_eval_result("cleanrl_original", elapsed, eval_returns)


if __name__ == "__main__":
    main()
