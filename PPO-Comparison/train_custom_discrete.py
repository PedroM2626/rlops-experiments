"""Custom PPO implementation from scratch using PyTorch and NumPy.

This is a minimal, educational PPO agent for LunarLander-v3.
No external RL libraries are used -- only PyTorch, NumPy, and Gymnasium.
All hyperparameters are imported from config.py for fair comparison.
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

from config import *  # noqa: F403, F401 -- shared hyperparams


# ---------------------------------------------------------------------------
# 1. Policy network (shared trunk, two heads)
# ---------------------------------------------------------------------------
class PolicyNetwork(nn.Module):
    """Actor-Critic network with shared hidden layers.

    Architecture
    ------------
    obs (8) -> Linear(64) -> Tanh -> Linear(64) -> Tanh
        -> actor head  (Linear -> 4 logits)
        -> critic head (Linear -> 1 scalar value)
    """

    def __init__(self, obs_dim: int = 8, act_dim: int = 4, is_continuous: bool = False) -> None:
        super().__init__()
        self.is_continuous = is_continuous
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

    # -- weight initialisation ------------------------------------------------
    def _init_weights(self) -> None:
        """Apply Orthogonal init to hidden layers, small-scale init to heads."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        # Actor head -- small init keeps initial policy close to uniform
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        # Critic head -- standard init
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    # -- forward pass ---------------------------------------------------------
    def forward(self, obs: torch.Tensor):
        logits = self.actor(obs)
        value = self.critic(obs).squeeze(-1)
        return logits, value

    def get_action_and_value(self, obs: torch.Tensor):
        """Sample an action and return (action, log_prob, value)."""
        logits, value = self.forward(obs)
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
        """Given stored obs and actions, return (log_prob, entropy, value)."""
        logits, value = self.forward(obs)
        if self.is_continuous:
            action_std = torch.exp(self.actor_logstd.expand_as(logits))
            dist = Normal(logits, action_std)
            return dist.log_prob(actions).sum(-1), dist.entropy().sum(-1), value
        else:
            dist = Categorical(logits=logits)
            return dist.log_prob(actions), dist.entropy(), value


# ---------------------------------------------------------------------------
# 2. Rollout buffer with GAE
# ---------------------------------------------------------------------------
@dataclass
class RolloutBuffer:
    """Fixed-length buffer that stores one rollout and computes GAE.

    Usage
    -----
    1. Call ``add()`` for each timestep.
    2. After ``N_STEPS`` transitions, call ``compute_gae()``.
    3. Iterate over random minibatches via ``get_batches()``.
    """

    capacity: int = N_STEPS
    is_continuous: bool = False
    states: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    dones: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    values: list = field(default_factory=list)
    # Computed after the rollout is complete
    advantages: np.ndarray | None = None
    returns: np.ndarray | None = None

    def add(
        self,
        state: np.ndarray,
        action: Any,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        """Append a single transition to the buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_gae(
        self,
        gamma: float,
        gae_lambda: float,
        last_value: float,
    ) -> None:
        """Compute Generalized Advantage Estimation (GAE-lambda).

        Parameters
        ----------
        gamma : float
            Discount factor.
        gae_lambda : float
            GAE smoothing parameter (1.0 = Monte-Carlo, 0.0 = TD(0)).
        last_value : float
            V(s_{T+1}) bootstrap value for the state after the last step.
        """
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
            delta = (
                self.rewards[t]
                + gamma * next_value * next_non_terminal
                - self.values[t]
            )
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae
        self.returns = self.advantages + np.array(self.values, dtype=np.float32)

    def get_batches(self, batch_size: int, device: torch.device):
        """Yield random minibatches of (states, actions, old_log_probs, advantages, returns).

        The entire rollout is shuffled and split into batches of ``batch_size``.
        """
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

    def reset(self) -> None:
        """Clear all stored data for the next rollout."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        self.advantages = None
        self.returns = None


# ---------------------------------------------------------------------------
# 3. PPO trainer
# ---------------------------------------------------------------------------
class PPOTrainer:
    """Proximal Policy Optimization with clipped surrogate objective.

    This class owns the optimizer and performs the PPO update step
    given a filled ``RolloutBuffer``.
    """

    def __init__(
        self,
        policy: PolicyNetwork,
        lr: float = LEARNING_RATE,
        clip_range: float = CLIP_RANGE,
        n_epochs: int = N_EPOCHS,
        batch_size: int = BATCH_SIZE,
        vf_coef: float = VF_COEF,
        ent_coef: float = ENT_COEF,
        max_grad_norm: float = MAX_GRAD_NORM,
        normalize_advantage: bool = NORMALIZE_ADVANTAGE,
    ) -> None:
        self.policy = policy
        self.clip_range = clip_range
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.normalize_advantage = normalize_advantage

        self.optimizer = optim.Adam(policy.parameters(), lr=lr, eps=1e-5)

    def update(self, buffer: RolloutBuffer, device: torch.device) -> dict[str, float]:
        """Run ``n_epochs`` of PPO updates over the rollout data.

        Returns
        -------
        dict
            Aggregated loss statistics averaged over all minibatch updates.
        """
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_loss = 0.0
        n_updates = 0

        for _epoch in range(self.n_epochs):
            for batch in buffer.get_batches(self.batch_size, device):
                states, actions, old_log_probs, advantages, returns, old_values = batch

                # Optionally normalise advantages per minibatch
                if self.normalize_advantage and advantages.numel() > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                # Evaluate current policy on stored transitions
                new_log_probs, entropy, new_values = self.policy.evaluate_actions(
                    states, actions
                )

                # -- policy (actor) loss ------------------------------------
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # -- value (critic) loss ------------------------------------
                v_loss_unclipped = (new_values - returns) ** 2
                v_clipped = old_values + torch.clamp(
                    new_values - old_values, -self.clip_range, self.clip_range
                )
                v_loss_clipped = (v_clipped - returns) ** 2
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                # -- entropy bonus ------------------------------------------
                entropy_mean = entropy.mean()

                # -- combined loss ------------------------------------------
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_mean

                # -- gradient step ------------------------------------------
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Accumulate stats
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_mean.item()
                total_loss += loss.item()
                n_updates += 1

        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "total_loss": total_loss / n_updates,
        }


# ---------------------------------------------------------------------------
# 4. Seed helper
# ---------------------------------------------------------------------------
def set_all_seeds(seed: int) -> None:
    """Set seeds for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# 5. Main training loop
# ---------------------------------------------------------------------------
def main() -> None:
    """Train a PPO agent on LunarLander-v3 from scratch and log everything."""

    # -- reproducibility ----------------------------------------------------
    set_all_seeds(SEED)

    # -- device -------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_custom] Using device: {device}")
    # -- environment --------------------------------------------------------
    env = gymnasium.make(ENV_ID)
    env = gymnasium.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(SEED)
    env.observation_space.seed(SEED)
    obs_dim = env.observation_space.shape[0]  # 8
    is_continuous = isinstance(env.action_space, gymnasium.spaces.Box)
    act_dim = env.action_space.shape[0] if is_continuous else env.action_space.n  # 4

    # 3. Setup models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = PolicyNetwork(obs_dim, act_dim, is_continuous).to(device)
    trainer = PPOTrainer(policy)

    # 4. Rollout Buffer
    buffer = RolloutBuffer(capacity=N_STEPS, is_continuous=is_continuous)

    # -- MLflow setup -------------------------------------------------------
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # -- episode tracking ---------------------------------------------------
    episode_rewards: list[float] = []
    recent_rewards: deque[float] = deque(maxlen=20)
    curve_rows: list[dict[str, Any]] = []  # for CSV export

    # -- ensure output dirs exist -------------------------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CURVES_DIR.mkdir(parents=True, exist_ok=True)

    # -- start MLflow run ---------------------------------------------------
    with mlflow.start_run(run_name="custom_discrete"):
        params = get_hyperparam_dict()
        params["impl"] = "custom_discrete"
        params["device"] = str(device)
        mlflow.log_params(params)

        global_step = 0
        num_updates = 0
        total_updates = TOTAL_TIMESTEPS // N_STEPS
        obs, _info = env.reset(seed=SEED)
        start_time = time.time()

        print(f"[train_custom] Starting training for {TOTAL_TIMESTEPS} timesteps")

        while global_step < TOTAL_TIMESTEPS:
            # -- LR annealing -----------------------------------------------
            frac = 1.0 - (num_updates / total_updates)
            lrnow = LEARNING_RATE * frac
            trainer.optimizer.param_groups[0]["lr"] = lrnow

            # -- collect rollout of N_STEPS transitions ---------------------
            buffer.reset()
            for _step in range(N_STEPS):
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action_tensor, log_prob_tensor, value_tensor = policy.get_action_and_value(obs_tensor)

                if is_continuous:
                    action = action_tensor.cpu().numpy().squeeze(0)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                else:
                    action = action_tensor.item()
                    next_obs, reward, terminated, truncated, info = env.step(action)
                
                done = terminated or truncated

                buffer.add(
                    state=obs,
                    action=action,
                    reward=reward,
                    done=done,
                    log_prob=log_prob_tensor.item(),
                    value=value_tensor.item(),
                )
                global_step += 1

                # Track completed episodes via RecordEpisodeStatistics
                if "episode" in info:
                    ep_reward = float(info["episode"]["r"])
                    ep_len = int(info["episode"]["l"])
                    episode_rewards.append(ep_reward)
                    recent_rewards.append(ep_reward)

                obs = next_obs
                if done:
                    obs, _info = env.reset()

                if global_step >= TOTAL_TIMESTEPS:
                    break

            # -- bootstrap value for GAE ------------------------------------
            with torch.no_grad():
                last_obs_tensor = torch.tensor(
                    obs, dtype=torch.float32, device=device
                ).unsqueeze(0)
                _, last_value = policy(last_obs_tensor)
                last_value = last_value.item()

            buffer.compute_gae(GAMMA, GAE_LAMBDA, last_value)

            # -- PPO update -------------------------------------------------
            losses = trainer.update(buffer, device)
            num_updates += 1

            # -- logging ----------------------------------------------------
            elapsed = time.time() - start_time
            fps = int(global_step / max(elapsed, 1e-8))

            if len(recent_rewards) > 0:
                mean_reward_20 = float(np.mean(recent_rewards))
            else:
                mean_reward_20 = 0.0

            # Log to MLflow every update
            mlflow.log_metrics(
                {
                    "rollout/mean_reward_20ep": mean_reward_20,
                    "rollout/episodes_total": len(episode_rewards),
                    "train/policy_loss": losses["policy_loss"],
                    "train/value_loss": losses["value_loss"],
                    "train/entropy": losses["entropy"],
                    "train/total_loss": losses["total_loss"],
                    "train/fps": fps,
                },
                step=global_step,
            )

            # Store curve data
            curve_rows.append(
                {
                    "step": global_step,
                    "mean_reward": mean_reward_20,
                    "episodes_total": len(episode_rewards),
                    "policy_loss": losses["policy_loss"],
                    "value_loss": losses["value_loss"],
                    "entropy": losses["entropy"],
                    "fps": fps,
                    "elapsed_s": round(elapsed, 2),
                }
            )

            # Console output at LOG_INTERVAL granularity
            if num_updates == 1 or global_step % LOG_INTERVAL < N_STEPS:
                print(
                    f"  step={global_step:>7d}/{TOTAL_TIMESTEPS}  "
                    f"episodes={len(episode_rewards):<4d}  "
                    f"mean_r(20)={mean_reward_20:>8.2f}  "
                    f"pi_loss={losses['policy_loss']:.4f}  "
                    f"v_loss={losses['value_loss']:.4f}  "
                    f"ent={losses['entropy']:.4f}  "
                    f"fps={fps}"
                )

        # -- training complete ----------------------------------------------
        total_time = time.time() - start_time
        print(
            f"\n[train_custom] Training complete in {total_time:.1f}s  "
            f"({len(episode_rewards)} episodes)"
        )
        if recent_rewards:
            print(f"[train_custom] Final mean reward (last 20): {np.mean(recent_rewards):.2f}")

        # -- save model -----------------------------------------------------
        model_path = MODELS_DIR / "custom_discrete_ppo.pt"
        torch.save(
            {
                "model_state_dict": policy.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "global_step": global_step,
                "episodes": len(episode_rewards),
            },
            str(model_path),
        )
        mlflow.log_artifact(str(model_path), artifact_path="models")
        print(f"[train_custom] Model saved to {model_path}")

        # -- save training curve CSV ----------------------------------------
        csv_path = CURVES_DIR / "custom_discrete_curve.csv"
        fieldnames = list(curve_rows[0].keys()) if curve_rows else []
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(curve_rows)
        mlflow.log_artifact(str(csv_path), artifact_path="curves")
        print(f"[train_custom] Curve CSV saved to {csv_path}")

        # -- final summary metrics ------------------------------------------
        final_mean = float(np.mean(recent_rewards)) if recent_rewards else 0.0
        mlflow.log_metrics(
            {
                "final/mean_reward_20ep": final_mean,
                "final/total_episodes": len(episode_rewards),
                "final/total_time_s": round(total_time, 2),
                "final/total_updates": num_updates,
            },
            step=global_step,
        )

    env.close()
    print("[train_custom] Done.")


if __name__ == "__main__":
    main()
