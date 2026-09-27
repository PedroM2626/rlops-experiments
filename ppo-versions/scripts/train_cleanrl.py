"""Minimalist CleanRL-style PPO: a direct implementation, with no RL framework on top.
Just PyTorch + gymnasium. That makes every single implementation decision visible."""
import time
import sys
sys.path.insert(0, "/home/claude/ppo-benchmark/scripts")
from common import ENV_ID, SEED, TOTAL_TIMESTEPS, PPO_CONFIG, save_result, evaluate_policy, save_eval_result

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from torch.distributions import Categorical


def make_env():
    return gym.make(ENV_ID)


def layer_init(layer, gain):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, cfg):
        super().__init__()
        gh, ga, gc = cfg["ortho_gain_hidden"], cfg["ortho_gain_actor_out"], cfg["ortho_gain_critic_out"]
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64), gh), nn.Tanh(),
            layer_init(nn.Linear(64, 64), gh), nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), ga),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64), gh), nn.Tanh(),
            layer_init(nn.Linear(64, 64), gh), nn.Tanh(),
            layer_init(nn.Linear(64, 1), gc),
        )

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(x).squeeze(-1)


def main():
    cfg = PPO_CONFIG
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    envs = gym.vector.SyncVectorEnv([make_env for _ in range(cfg["n_envs"])])
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.n

    agent = ActorCritic(obs_dim, act_dim, cfg)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg["lr"], eps=cfg["adam_eps"])

    n_steps, n_envs = cfg["n_steps"], cfg["n_envs"]
    batch_size = n_steps * n_envs
    minibatch_size = batch_size // cfg["minibatches"]

    obs_buf = torch.zeros((n_steps, n_envs, obs_dim))
    act_buf = torch.zeros((n_steps, n_envs))
    logp_buf = torch.zeros((n_steps, n_envs))
    rew_buf = torch.zeros((n_steps, n_envs))
    done_buf = torch.zeros((n_steps, n_envs))
    val_buf = torch.zeros((n_steps, n_envs))

    next_obs, _ = envs.reset(seed=SEED)
    next_obs = torch.tensor(next_obs, dtype=torch.float32)
    next_done = torch.zeros(n_envs)

    ep_returns = np.zeros(n_envs)
    reward_history = []
    global_step = 0

    t0 = time.time()
    num_updates = TOTAL_TIMESTEPS // batch_size

    for update in range(num_updates):
        for step in range(n_steps):
            global_step += n_envs
            obs_buf[step] = next_obs
            done_buf[step] = next_done

            with torch.no_grad():
                action, logp, _, value = agent.get_action_and_value(next_obs)
            val_buf[step] = value
            act_buf[step] = action
            logp_buf[step] = logp

            next_obs_np, reward, terminated, truncated, infos = envs.step(action.numpy())
            done = np.logical_or(terminated, truncated)
            rew_buf[step] = torch.tensor(reward, dtype=torch.float32)
            ep_returns += reward

            for i, d in enumerate(done):
                if d:
                    reward_history.append((global_step, ep_returns[i]))
                    ep_returns[i] = 0.0

            next_obs = torch.tensor(next_obs_np, dtype=torch.float32)
            next_done = torch.tensor(done, dtype=torch.float32)

        # GAE
        with torch.no_grad():
            next_value = agent.get_action_and_value(next_obs)[3]
            advantages = torch.zeros_like(rew_buf)
            lastgaelam = 0
            for t in reversed(range(n_steps)):
                if t == n_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - done_buf[t + 1]
                    nextvalues = val_buf[t + 1]
                delta = rew_buf[t] + cfg["gamma"] * nextvalues * nextnonterminal - val_buf[t]
                advantages[t] = lastgaelam = delta + cfg["gamma"] * cfg["gae_lambda"] * nextnonterminal * lastgaelam
            returns = advantages + val_buf

        b_obs = obs_buf.reshape(-1, obs_dim)
        b_act = act_buf.reshape(-1)
        b_logp = logp_buf.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)
        b_val = val_buf.reshape(-1)

        b_inds = np.arange(batch_size)
        for epoch in range(cfg["n_epochs"]):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                mb_inds = b_inds[start:start + minibatch_size]

                _, newlogp, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_act[mb_inds]
                )
                logratio = newlogp - b_logp[mb_inds]
                ratio = logratio.exp()

                mb_adv = b_adv[mb_inds]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - cfg["clip_coef"], 1 + cfg["clip_coef"])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newvalue - b_ret[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = pg_loss - cfg["ent_coef"] * entropy_loss + cfg["vf_coef"] * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg["max_grad_norm"])
                optimizer.step()

    elapsed = time.time() - t0

    # smooths reward_history over 20-episode windows for a cleaner comparison
    smoothed = []
    if reward_history:
        window = []
        for ts, r in reward_history:
            window.append(r)
            if len(window) > 20:
                window.pop(0)
            smoothed.append((ts, float(np.mean(window))))

    save_result("cleanrl_style_pytorch", elapsed, smoothed)
    envs.close()

    def act_fn(obs):
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action, _, _, _ = agent.get_action_and_value(obs_t)
        return int(action.item())

    eval_returns = evaluate_policy(ENV_ID, act_fn)
    save_eval_result("cleanrl_style_pytorch", elapsed, eval_returns)


if __name__ == "__main__":
    main()
