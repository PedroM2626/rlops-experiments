"""Standalone PPO implementation modeled exactly after RLlib's PPO algorithm.

This script extracts the core algorithmic choices of Ray RLlib's PPO
without requiring the massive `ray` dependency, solving Python 3.14
compatibility issues.

Key RLlib specific PPO features implemented here:
1. Value function clipping (vf_clip_param = 10.0 by default in RLlib).
2. Dynamic KL penalty (kl_coeff adjusted based on KL divergence).
3. Huber or MSE loss for value function (MSE by default).
4. Advantage standardization at the minibatch level.
"""

import csv
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import gymnasium
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal

from config import *  # noqa: F403, F401


# RLlib-specific constants
RLLIB_VF_CLIP_PARAM = 10.0
RLLIB_KL_TARGET = 0.01
RLLIB_INITIAL_KL_COEFF = 0.2


# ---------------------------------------------------------------------------
# 1. Policy network
# ---------------------------------------------------------------------------
class RLlibPolicyNetwork(nn.Module):
    """Network architecture mimicking RLlib's FullyConnectedNetwork."""
    def __init__(self, obs_dim: int = 8, act_dim: int = 4, is_continuous: bool = False) -> None:
        super().__init__()
        self.is_continuous = is_continuous
        # By default, RLlib does not share layers between policy and value function
        # if vf_share_layers=False. We'll implement non-shared for closer match to
        # RLlib's standard defaults, though config.py specifies shared in some others.
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, NET_ARCH[0]),
            nn.Tanh(),
            nn.Linear(NET_ARCH[0], NET_ARCH[1]),
            nn.Tanh(),
            nn.Linear(NET_ARCH[1], act_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, NET_ARCH[0]),
            nn.Tanh(),
            nn.Linear(NET_ARCH[0], NET_ARCH[1]),
            nn.Tanh(),
            nn.Linear(NET_ARCH[1], 1)
        )
        if self.is_continuous:
            self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))
        self._init_weights()

    def _init_weights(self) -> None:
        # RLlib uses Xavier uniform by default
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, obs: torch.Tensor):
        logits = self.actor(obs)
        value = self.critic(obs).squeeze(-1)
        return logits, value

    def get_action_and_value(self, obs: torch.Tensor):
        logits = self.actor(obs)
        value = self.critic(obs).squeeze(-1)
        if self.is_continuous:
            action_std = torch.exp(self.actor_logstd.expand_as(logits))
            dist = Normal(logits, action_std)
            action = dist.sample()
            return action, dist.log_prob(action).sum(-1), value
        else:
            dist = Categorical(logits=logits)
            action = dist.sample()
            return action, dist.log_prob(action), value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        logits, value = self.forward(obs)
        if self.is_continuous:
            action_std = torch.exp(self.actor_logstd.expand_as(logits))
            dist = Normal(logits, action_std)
            return dist.log_prob(actions).sum(-1), dist.entropy().sum(-1), value, dist
        else:
            dist = Categorical(logits=logits)
            return dist.log_prob(actions), dist.entropy(), value, dist


# ---------------------------------------------------------------------------
# 2. Rollout buffer (same GAE logic as custom_pytorch)
# ---------------------------------------------------------------------------
@dataclass
class RolloutBuffer:
    capacity: int = N_STEPS
    is_continuous: bool = False
    states: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    dones: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    values: list = field(default_factory=list)
    advantages: np.ndarray | None = None
    returns: np.ndarray | None = None

    def add(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_gae(self, gamma, gae_lambda, last_value):
        n = len(self.rewards)
        self.advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(self.dones[t])
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - float(self.dones[t])
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae
        self.returns = self.advantages + np.array(self.values, dtype=np.float32)

    def get_batches(self, batch_size, device):
        n = len(self.states)
        indices = np.arange(n)
        np.random.shuffle(indices)

        states_arr = np.array(self.states, dtype=np.float32)
        if self.is_continuous:
            actions_arr = np.array(self.actions, dtype=np.float32)
        else:
            actions_arr = np.array(self.actions, dtype=np.int64)
        log_probs_arr = np.array(self.log_probs, dtype=np.float32)
        values_arr = np.array(self.values, dtype=np.float32)

        for start in range(0, n, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            yield (
                torch.tensor(states_arr[batch_idx], device=device),
                torch.tensor(actions_arr[batch_idx], device=device),
                torch.tensor(log_probs_arr[batch_idx], device=device),
                torch.tensor(self.advantages[batch_idx], device=device),
                torch.tensor(self.returns[batch_idx], device=device),
                torch.tensor(values_arr[batch_idx], device=device),
            )

    def reset(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        self.advantages = None
        self.returns = None


# ---------------------------------------------------------------------------
# 3. RLlib PPO Trainer
# ---------------------------------------------------------------------------
class RLlibPPOTrainer:
    def __init__(self, policy: RLlibPolicyNetwork, device: torch.device):
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE, eps=1e-5)
        self.kl_coeff = RLLIB_INITIAL_KL_COEFF

    def update(self, buffer: RolloutBuffer, device: torch.device):
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        n_updates = 0

        for _ in range(N_EPOCHS):
            for batch in buffer.get_batches(BATCH_SIZE, device):
                states, actions, old_log_probs, advantages, returns, old_values = batch

                if NORMALIZE_ADVANTAGE and advantages.numel() > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                new_log_probs, entropy, new_values, dist = self.policy.evaluate_actions(states, actions)

                # RLlib KL Divergence computation
                # KL(old || new) = old_log_prob - new_log_prob (approx) or exact
                kl = (old_log_probs - new_log_probs).mean()
                
                # 1. Policy Loss
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - CLIP_RANGE, 1.0 + CLIP_RANGE) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # RLlib KL Penalty
                mean_kl_loss = self.kl_coeff * kl

                # 2. Value Function Loss (with RLlib clipping)
                value_err = (new_values - returns) ** 2
                value_clipped = old_values + torch.clamp(
                    new_values - old_values, -RLLIB_VF_CLIP_PARAM, RLLIB_VF_CLIP_PARAM
                )
                value_err_clipped = (value_clipped - returns) ** 2
                vf_loss = torch.max(value_err, value_err_clipped).mean()

                # 3. Total Loss
                entropy_mean = entropy.mean()
                loss = policy_loss + mean_kl_loss + VF_COEF * vf_loss - ENT_COEF * entropy_mean

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += vf_loss.item()
                total_entropy += entropy_mean.item()
                total_kl += kl.item()
                n_updates += 1

        # RLlib Dynamic KL Update
        mean_kl = total_kl / n_updates
        if mean_kl > 2.0 * RLLIB_KL_TARGET:
            self.kl_coeff *= 1.5
        elif mean_kl < 0.5 * RLLIB_KL_TARGET:
            self.kl_coeff *= 0.5

        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "kl": mean_kl,
            "kl_coeff": self.kl_coeff,
        }


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_all_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[RLlib Extracted] Using device: {device}")

    env = gymnasium.make(ENV_ID)
    env = gymnasium.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(SEED)
    env.observation_space.seed(SEED)

    obs_dim = env.observation_space.shape[0]
    is_continuous = isinstance(env.action_space, gymnasium.spaces.Box)
    act_dim = env.action_space.shape[0] if is_continuous else env.action_space.n

    policy = RLlibPolicyNetwork(obs_dim, act_dim, is_continuous).to(device)
    trainer = RLlibPPOTrainer(policy, device)
    buffer = RolloutBuffer(capacity=N_STEPS, is_continuous=is_continuous)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    episode_rewards = []
    recent_rewards = deque(maxlen=20)
    curve_rows = []

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CURVES_DIR.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name="rllib_extracted"):
        params = get_hyperparam_dict()
        params["impl"] = "rllib_extracted"
        params["rllib_vf_clip_param"] = RLLIB_VF_CLIP_PARAM
        params["rllib_kl_target"] = RLLIB_KL_TARGET
        mlflow.log_params(params)

        global_step = 0
        obs, _ = env.reset(seed=SEED)
        start_time = time.time()

        print(f"[RLlib Extracted] Starting training for {TOTAL_TIMESTEPS} timesteps")

        while global_step < TOTAL_TIMESTEPS:
            buffer.reset()
            for _ in range(N_STEPS):
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action_tensor, log_prob_tensor, value_tensor = policy.get_action_and_value(obs_tensor)

                if is_continuous:
                    action = action_tensor.cpu().numpy().squeeze(0)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                else:
                    action = action_tensor.item()
                    next_obs, reward, terminated, truncated, info = env.step(action)
            
                done = terminated or truncated

                buffer.add(obs, action, reward, done, log_prob_tensor.item(), value_tensor.item())
                global_step += 1

                if "episode" in info:
                    ep_reward = float(info["episode"]["r"])
                    episode_rewards.append(ep_reward)
                    recent_rewards.append(ep_reward)

                obs = next_obs
                if done:
                    obs, _ = env.reset()

                if global_step >= TOTAL_TIMESTEPS:
                    break

            with torch.no_grad():
                last_obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                _, last_value = policy(last_obs_tensor)
                last_value = last_value.item()

            buffer.compute_gae(GAMMA, GAE_LAMBDA, last_value)
            losses = trainer.update(buffer, device)

            if len(recent_rewards) > 0:
                mean_reward = float(np.mean(recent_rewards))
            else:
                mean_reward = 0.0

            mlflow.log_metrics({
                "mean_reward": mean_reward,
                "policy_loss": losses["policy_loss"],
                "value_loss": losses["value_loss"],
                "entropy": losses["entropy"],
                "kl": losses["kl"],
                "kl_coeff": losses["kl_coeff"]
            }, step=global_step)

            curve_rows.append({
                "step": global_step,
                "mean_reward": mean_reward,
                "policy_loss": losses["policy_loss"],
                "value_loss": losses["value_loss"]
            })

            print(f"[RLlib Extracted] Step {global_step}/{TOTAL_TIMESTEPS} | Mean reward (20 ep): {mean_reward:.2f} | KL: {losses['kl']:.4f}")

        # Save model & curve
        model_path = MODELS_DIR / "rllib_extracted_ppo.pt"
        # Format for evaluate.py
        unified_state = {}
        for k, v in policy.actor.state_dict().items(): unified_state[f"actor.{k}"] = v
        for k, v in policy.critic.state_dict().items(): unified_state[f"critic.{k}"] = v
        torch.save({"model_state_dict": unified_state}, str(model_path))

        csv_path = CURVES_DIR / "rllib_extracted_curve.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(curve_rows[0].keys()))
            writer.writeheader()
            writer.writerows(curve_rows)

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(csv_path))
        print("[RLlib Extracted] Training complete.")

if __name__ == "__main__":
    main()
