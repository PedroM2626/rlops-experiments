"""CleanRL-style PPO implementation for LunarLander-v3.

Single-file, from-scratch PPO using PyTorch and Gymnasium.
All hyperparameters are imported from config.py to ensure a fair comparison
across implementations. Training curves and model artifacts are logged to
MLflow.

Reference: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
"""

import csv
import random
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal

from config import (
    BATCH_SIZE,
    CLIP_RANGE,
    CURVES_DIR,
    ENT_COEF,
    ENV_ID,
    GAE_LAMBDA,
    GAMMA,
    LEARNING_RATE,
    LOG_INTERVAL,
    MAX_GRAD_NORM,
    MLFLOW_EXPERIMENT,
    MLFLOW_URI,
    MODELS_DIR,
    N_EPOCHS,
    N_STEPS,
    NORMALIZE_ADVANTAGE,
    SEED,
    TOTAL_TIMESTEPS,
    VF_COEF,
    get_hyperparam_dict,
)


# ---------------------------------------------------------------------------
# Utility: layer initialisation
# ---------------------------------------------------------------------------

def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Apply orthogonal initialisation to a linear layer.

    Args:
        layer: The nn.Linear module to initialise.
        std: Standard deviation (gain) for orthogonal init.
        bias_const: Constant value for the bias.

    Returns:
        The initialised layer (same object, mutated in place).
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


# ---------------------------------------------------------------------------
# Agent (Actor-Critic network)
# ---------------------------------------------------------------------------

class Agent(nn.Module):
    """PPO actor-critic agent with separate networks for policy and value.

    Architecture (per branch):
        Linear(obs_dim, 64) -> Tanh -> Linear(64, 64) -> Tanh -> head

    The actor head produces logits for a Categorical distribution (4 actions).
    The critic head produces a single scalar state-value estimate.
    """

    def __init__(self, obs_dim: int, act_dim: int, is_continuous: bool = False) -> None:
        super().__init__()
        self.is_continuous = is_continuous

        # Critic network
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        # Actor network
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )
        if self.is_continuous:
            self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Return the critic's state-value estimate for observations *x*.

        Args:
            x: Observation tensor of shape (batch, obs_dim).

        Returns:
            Value tensor of shape (batch, 1).
        """
        return self.critic(x)

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action (or evaluate a given action) and return diagnostics.

        Args:
            x: Observation tensor of shape (batch, obs_dim).
            action: Optional pre-selected action tensor.  When ``None`` an
                action is sampled from the policy.

        Returns:
            action: The (sampled or given) action tensor.
            log_prob: Log-probability of the action under the current policy.
            entropy: Entropy of the action distribution.
            value: Critic value estimate.
        """
        action_logits = self.actor(x)
        
        if self.is_continuous:
            action_logstd = self.actor_logstd.expand_as(action_logits)
            action_std = torch.exp(action_logstd)
            dist = Normal(action_logits, action_std)
        else:
            dist = Categorical(logits=action_logits)

        if action is None:
            action = dist.sample()
            
        if self.is_continuous:
            log_prob = dist.log_prob(action).sum(-1)
            entropy = dist.entropy().sum(-1)
        else:
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

        return action, log_prob, entropy, self.critic(x)


# ---------------------------------------------------------------------------
# Rollout storage
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Fixed-size buffer that stores one rollout of N_STEPS transitions.

    All data is kept as flat numpy arrays and converted to tensors only when
    needed for the PPO update.
    """

    def __init__(self, n_steps: int, obs_dim: int, act_dim: int, is_continuous: bool, device: torch.device) -> None:
        """Initialise rollout buffer."""
        self.obs = np.zeros((n_steps, obs_dim), dtype=np.float32)
        if is_continuous:
            self.actions = np.zeros((n_steps, act_dim), dtype=np.float32)
        else:
            self.actions = np.zeros(n_steps, dtype=np.float32)
        self.log_probs = np.zeros(n_steps, dtype=np.float32)
        self.rewards = np.zeros(n_steps, dtype=np.float32)
        self.dones = np.zeros(n_steps, dtype=np.float32)
        self.values = np.zeros(n_steps, dtype=np.float32)

        self.ptr = 0
        self.device = device
        self.n_steps = n_steps

    def store(
        self,
        obs: np.ndarray,
        action: np.ndarray | int,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        """Store a single transition at the current pointer and advance."""
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.values[self.ptr] = value
        self.ptr += 1

    def reset(self) -> None:
        """Reset the pointer so the buffer can be reused for the next rollout."""
        self.ptr = 0

    def compute_gae(
        self,
        last_value: float,
        last_done: bool,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and discounted returns.

        Args:
            last_value: V(s_{T+1}) -- bootstrap value at the end of the rollout.
            last_done: Whether the last state was terminal.
            gamma: Discount factor.
            gae_lambda: GAE lambda.

        Returns:
            advantages: Tensor of shape (n_steps,).
            returns: Tensor of shape (n_steps,).
        """
        advantages = np.zeros(self.n_steps, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + self.values
        return (
            torch.tensor(advantages, device=self.device),
            torch.tensor(returns, device=self.device),
        )

    def get_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return observations, actions, and log_probs as tensors."""
        return (
            torch.tensor(self.obs, device=self.device),
            torch.tensor(self.actions, device=self.device),
            torch.tensor(self.log_probs, device=self.device),
        )


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def ppo_update(
    agent: Agent,
    optimizer: optim.Adam,
    buffer: RolloutBuffer,
    advantages: torch.Tensor,
    returns: torch.Tensor,
) -> dict[str, float]:
    """Perform multiple epochs of PPO clipped updates on the collected rollout.

    Args:
        agent: The actor-critic network.
        optimizer: Adam optimiser.
        buffer: Filled rollout buffer.
        advantages: GAE advantage tensor (n_steps,).
        returns: Discounted return tensor (n_steps,).

    Returns:
        Dictionary of mean loss components for logging.
    """
    obs_t, actions_t, old_log_probs_t = buffer.get_tensors()
    n_steps = buffer.n_steps
    batch_indices = np.arange(n_steps)

    # Accumulators for logging
    total_pg_loss = 0.0
    total_vf_loss = 0.0
    total_entropy = 0.0
    total_updates = 0

    for _epoch in range(N_EPOCHS):
        np.random.shuffle(batch_indices)

        for start in range(0, n_steps, BATCH_SIZE):
            end = start + BATCH_SIZE
            mb_idx = batch_indices[start:end]

            mb_obs = obs_t[mb_idx]
            mb_actions = actions_t[mb_idx]
            mb_old_log_probs = old_log_probs_t[mb_idx]
            mb_advantages = advantages[mb_idx]
            mb_returns = returns[mb_idx]

            # Normalise advantages at the minibatch level
            if NORMALIZE_ADVANTAGE and len(mb_advantages) > 1:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Forward pass
            _, new_log_probs, entropy, new_values = agent.get_action_and_value(mb_obs, mb_actions)
            new_values = new_values.squeeze(-1)

            # Ratio and clipped surrogate loss
            log_ratio = new_log_probs - mb_old_log_probs
            ratio = log_ratio.exp()

            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - CLIP_RANGE, 1.0 + CLIP_RANGE)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value function loss
            vf_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()

            # Entropy bonus (maximise entropy -> subtract from loss)
            entropy_loss = entropy.mean()

            # Combined loss
            loss = pg_loss + VF_COEF * vf_loss - ENT_COEF * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            total_pg_loss += pg_loss.item()
            total_vf_loss += vf_loss.item()
            total_entropy += entropy_loss.item()
            total_updates += 1

    return {
        "pg_loss": total_pg_loss / total_updates,
        "vf_loss": total_vf_loss / total_updates,
        "entropy": total_entropy / total_updates,
    }


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------

def set_all_seeds(seed: int) -> None:
    """Set seeds for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    """Run PPO training on LunarLander-v3 in CleanRL style."""

    # -- Device selection ---------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CleanRL PPO] Using device: {device}")

    # -- Seeding ------------------------------------------------------------
    set_all_seeds(SEED)

    # -- Environment --------------------------------------------------------
    env = gym.make(ENV_ID)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    obs_dim = env.observation_space.shape[0]  # e.g., 8
    is_continuous = isinstance(env.action_space, gym.spaces.Box)
    act_dim = env.action_space.shape[0] if is_continuous else env.action_space.n

    # Seed the environment
    obs, info = env.reset(seed=SEED)

    # -- Agent & optimiser --------------------------------------------------
    agent = Agent(obs_dim, act_dim, is_continuous).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

    # -- Rollout buffer -----------------------------------------------------
    buffer = RolloutBuffer(N_STEPS, obs_dim, act_dim, is_continuous, device)

    # -- Tracking -----------------------------------------------------------
    episode_rewards: deque[float] = deque(maxlen=20)
    curve_data: list[tuple[int, float]] = []
    global_step = 0
    last_log_step = 0
    num_updates = TOTAL_TIMESTEPS // N_STEPS
    start_time = time.time()

    # -- Ensure output dirs exist -------------------------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CURVES_DIR.mkdir(parents=True, exist_ok=True)

    # -- MLflow setup -------------------------------------------------------
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="cleanrl_ppo"):
        # Log hyperparameters
        params = get_hyperparam_dict()
        params["implementation"] = "cleanrl"
        mlflow.log_params(params)

        # ==================================================================
        # Training loop: collect rollout -> update -> repeat
        # ==================================================================
        for update in range(1, num_updates + 1):
            buffer.reset()

            # -- Collect N_STEPS transitions --------------------------------
            for step in range(N_STEPS):
                global_step += 1

                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

                with torch.no_grad():
                    action, log_prob, _, value = agent.get_action_and_value(obs_tensor)

                log_prob_val = log_prob.item()
                value_val = value.item()

                if is_continuous:
                    next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy().squeeze(0))
                else:
                    next_obs, reward, terminated, truncated, info = env.step(action.item())
                
                done = terminated or truncated

                buffer.store(obs, action.detach().cpu().numpy().squeeze(0), log_prob_val, reward, done, value_val)

                # Track completed episodes via RecordEpisodeStatistics
                if "episode" in info:
                    ep_return = info["episode"]["r"]
                    ep_length = info["episode"]["l"]
                    episode_rewards.append(float(ep_return))
                    print(
                        f"  [Step {global_step:>7d}] Episode finished: "
                        f"return={ep_return:.1f}, length={ep_length}, "
                        f"rolling_mean_20={np.mean(episode_rewards):.1f}"
                    )

                # Advance observation
                if done:
                    obs, info = env.reset()
                else:
                    obs = next_obs

            # -- Compute GAE ------------------------------------------------
            with torch.no_grad():
                last_obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                last_value = agent.get_value(last_obs_tensor).item()

            # 'done' here is from the very last step of the rollout
            advantages, returns = buffer.compute_gae(last_value, done, GAMMA, GAE_LAMBDA)

            # -- PPO update -------------------------------------------------
            loss_info = ppo_update(agent, optimizer, buffer, advantages, returns)

            # -- Logging ----------------------------------------------------
            if global_step - last_log_step >= LOG_INTERVAL and len(episode_rewards) > 0:
                mean_reward = float(np.mean(episode_rewards))
                elapsed = time.time() - start_time
                sps = global_step / elapsed if elapsed > 0 else 0.0

                print(
                    f"[Update {update:>3d}/{num_updates}] "
                    f"step={global_step:>7d}/{TOTAL_TIMESTEPS}  "
                    f"mean_reward={mean_reward:>7.1f}  "
                    f"pg_loss={loss_info['pg_loss']:.4f}  "
                    f"vf_loss={loss_info['vf_loss']:.4f}  "
                    f"entropy={loss_info['entropy']:.4f}  "
                    f"SPS={sps:.0f}"
                )

                mlflow.log_metrics(
                    {
                        "mean_reward": mean_reward,
                        "pg_loss": loss_info["pg_loss"],
                        "vf_loss": loss_info["vf_loss"],
                        "entropy": loss_info["entropy"],
                        "sps": sps,
                    },
                    step=global_step,
                )

                curve_data.append((global_step, mean_reward))
                last_log_step = global_step

        # ==================================================================
        # Post-training: save artefacts
        # ==================================================================
        elapsed_total = time.time() - start_time
        print(f"\n[CleanRL PPO] Training finished in {elapsed_total:.1f}s ({global_step} steps).")

        # -- Save training curve CSV ----------------------------------------
        curve_path = CURVES_DIR / "cleanrl_curve.csv"
        with open(curve_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "mean_reward"])
            writer.writerows(curve_data)
        print(f"[CleanRL PPO] Training curve saved to {curve_path}")

        # -- Save model state_dict ------------------------------------------
        model_path = MODELS_DIR / "cleanrl_ppo.pt"
        torch.save(agent.state_dict(), model_path)
        print(f"[CleanRL PPO] Model saved to {model_path}")

        # -- Log artefacts to MLflow ----------------------------------------
        mlflow.log_artifact(str(curve_path))
        mlflow.log_artifact(str(model_path))
        mlflow.log_metric("training_time_s", elapsed_total)

        if len(episode_rewards) > 0:
            mlflow.log_metric("final_mean_reward", float(np.mean(episode_rewards)))

    env.close()
    print("[CleanRL PPO] Done.")


if __name__ == "__main__":
    main()
