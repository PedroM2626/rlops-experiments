"""Shared configuration for all implementations, to keep the comparison fair."""
import os

ENV_ID = os.environ.get("PPO_ENV", "CartPole-v1")
SEED = int(os.environ.get("PPO_SEED", 42))
TOTAL_TIMESTEPS = int(os.environ.get("PPO_TIMESTEPS", 150_000))
EVAL_EPISODES = int(os.environ.get("PPO_EVAL_EPISODES", 50))
RUN_TAG = os.environ.get("PPO_RUN_TAG", "")  # suffix so different configs don't get mixed up in the same env/seed
RESULTS_SUBDIR = ENV_ID.replace("/", "_") + RUN_TAG

# "Canonical" PPO hyperparameters (same values across all libs, where applicable).
# Part 12: RL Baselines3 Zoo values for LunarLander (n_steps, n_envs,
# gae_lambda, gamma, n_epochs, ent_coef, batch_size via the derived "minibatches")
# instead of the generic ones inherited from CartPole -- see README Parts 11/12.
PPO_CONFIG = dict(
    n_steps=1024,          # steps per env before an update (Zoo: 1024)
    n_envs=16,             # parallel envs (Zoo: 16)
    n_epochs=4,            # optimization epochs per batch (Zoo: 4)
    gamma=0.999,           # Zoo: 0.999 (was 0.99 -- long episodes need more weight on the future)
    gae_lambda=0.98,       # Zoo: 0.98 (was 0.95)
    clip_coef=0.2,
    ent_coef=0.01,         # Zoo: 0.01 (same as what we were already using)
    vf_coef=0.5,
    lr=3e-4,
    minibatches=256,       # (1024*16)/256 = batch_size 64, same as the Zoo
    max_grad_norm=0.5,
    # architecture -- identical across libs (SB3 default; the Zoo doesn't override it)
    hidden_sizes=(64, 64),
    activation="tanh",
    adam_eps=1e-5,
    # CleanRL/SB3-style orthogonal initialization (gain per layer type)
    ortho_gain_hidden=2 ** 0.5,
    ortho_gain_actor_out=0.01,
    ortho_gain_critic_out=1.0,
)


def evaluate_policy(env_id, act_fn, n_episodes=None, base_seed=100_000):
    """Runs N episodes with an ALREADY TRAINED (frozen) policy and returns the list
    of per-episode returns. act_fn(obs) -> action (int). Uses stochastic sampling
    (not greedy) to capture the real variance exhibited by the trained model."""
    import gymnasium as gym

    n_episodes = n_episodes or EVAL_EPISODES
    returns = []
    for i in range(n_episodes):
        env = gym.make(env_id)
        obs, _ = env.reset(seed=base_seed + i)
        done = False
        total = 0.0
        while not done:
            action = act_fn(obs)
            obs, r, terminated, truncated, _ = env.step(action)
            total += r
            done = terminated or truncated
        returns.append(total)
        env.close()
    return returns


def save_eval_result(name, train_elapsed_s, eval_returns, path=None):
    import json
    import os
    import numpy as np

    if path is None:
        path = f"/home/claude/ppo-benchmark/results_eval/{RESULTS_SUBDIR}"
    os.makedirs(path, exist_ok=True)
    out = {
        "name": name,
        "env_id": ENV_ID,
        "seed": SEED,
        "total_timesteps": TOTAL_TIMESTEPS,
        "train_elapsed_seconds": train_elapsed_s,
        "eval_returns": eval_returns,
        "eval_mean": float(np.mean(eval_returns)),
        "eval_std": float(np.std(eval_returns)),
    }
    with open(f"{path}/{name}_seed{SEED}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[EVAL {name} seed={SEED} env={ENV_ID}] {len(eval_returns)} episodes | "
          f"mean={out['eval_mean']:.1f} std={out['eval_std']:.1f}")
    return out


def save_result(name, elapsed_s, rewards, path=None):
    """rewards: list of (timestep, mean_reward) collected during training."""
    import json
    import os

    if path is None:
        path = f"/home/claude/ppo-benchmark/results/{RESULTS_SUBDIR}"
    os.makedirs(path, exist_ok=True)
    out = {
        "name": name,
        "env_id": ENV_ID,
        "seed": SEED,
        "elapsed_seconds": elapsed_s,
        "rewards": rewards,
        "final_reward": rewards[-1][1] if rewards else None,
    }
    with open(f"{path}/{name}_seed{SEED}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{name} seed={SEED} env={ENV_ID}] done in {elapsed_s:.1f}s | final mean reward: {out['final_reward']}")
    return out
