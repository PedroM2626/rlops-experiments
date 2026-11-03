"""PPO in pure JAX: network and update compiled with jit, while the env stays gymnasium
(gym's CartPole isn't jittable the way Brax's is, but this way we compare the SAME physics
across the 3 implementations -- fairness > purity)."""
import time
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ENV_ID, SEED, TOTAL_TIMESTEPS, PPO_CONFIG, save_result, evaluate_policy, save_eval_result

import numpy as np
import jax
import jax.numpy as jnp
import optax
import gymnasium as gym
from functools import partial


def make_env():
    return gym.make(ENV_ID)


def init_params(key, obs_dim, act_dim, cfg):
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    gh, ga, gc = cfg["ortho_gain_hidden"], cfg["ortho_gain_actor_out"], cfg["ortho_gain_critic_out"]

    def dense(k, in_d, out_d, gain):
        w = jax.nn.initializers.orthogonal(scale=gain)(k, (in_d, out_d))
        b = jnp.zeros(out_d)
        return w, b

    return {
        "a1": dense(k1, obs_dim, 64, gh), "a2": dense(k2, 64, 64, gh), "a3": dense(k3, 64, act_dim, ga),
        "c1": dense(k4, obs_dim, 64, gh), "c2": dense(k5, 64, 64, gh), "c3": dense(k6, 64, 1, gc),
    }


def forward_actor(params, x):
    w, b = params["a1"]; x = jnp.tanh(x @ w + b)
    w, b = params["a2"]; x = jnp.tanh(x @ w + b)
    w, b = params["a3"]; logits = x @ w + b
    return logits


def forward_critic(params, x):
    w, b = params["c1"]; x = jnp.tanh(x @ w + b)
    w, b = params["c2"]; x = jnp.tanh(x @ w + b)
    w, b = params["c3"]; v = (x @ w + b).squeeze(-1)
    return v


@jax.jit
def get_action_value(params, obs, key):
    logits = forward_actor(params, obs)
    value = forward_critic(params, obs)
    action = jax.random.categorical(key, logits)
    logp_all = jax.nn.log_softmax(logits)
    logp = jnp.take_along_axis(logp_all, action[:, None], axis=1).squeeze(-1)
    return action, logp, value


@jax.jit
def evaluate_actions(params, obs, actions):
    logits = forward_actor(params, obs)
    value = forward_critic(params, obs)
    logp_all = jax.nn.log_softmax(logits)
    logp = jnp.take_along_axis(logp_all, actions[:, None].astype(jnp.int32), axis=1).squeeze(-1)
    probs = jax.nn.softmax(logits)
    entropy = -jnp.sum(probs * logp_all, axis=-1)
    return logp, entropy, value


def ppo_loss(params, obs, actions, old_logp, advantages, returns, cfg):
    logp, entropy, value = evaluate_actions(params, obs, actions)
    ratio = jnp.exp(logp - old_logp)
    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    pg1 = -adv * ratio
    pg2 = -adv * jnp.clip(ratio, 1 - cfg["clip_coef"], 1 + cfg["clip_coef"])
    pg_loss = jnp.maximum(pg1, pg2).mean()
    v_loss = 0.5 * jnp.mean((value - returns) ** 2)
    ent_loss = entropy.mean()
    return pg_loss - cfg["ent_coef"] * ent_loss + cfg["vf_coef"] * v_loss


def main():
    cfg = PPO_CONFIG
    key = jax.random.PRNGKey(SEED)
    np.random.seed(SEED)

    envs = gym.vector.SyncVectorEnv([make_env for _ in range(cfg["n_envs"])])
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.n

    key, init_key = jax.random.split(key)
    params = init_params(init_key, obs_dim, act_dim, cfg)
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg["max_grad_norm"]),
        optax.adam(cfg["lr"], eps=cfg["adam_eps"]),
    )
    opt_state = optimizer.init(params)

    @jax.jit
    def update_step(params, opt_state, obs, actions, old_logp, advantages, returns):
        loss, grads = jax.value_and_grad(ppo_loss)(
            params, obs, actions, old_logp, advantages, returns, cfg
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    n_steps, n_envs = cfg["n_steps"], cfg["n_envs"]
    batch_size = n_steps * n_envs
    minibatch_size = batch_size // cfg["minibatches"]

    next_obs, _ = envs.reset(seed=SEED)
    next_done = np.zeros(n_envs, dtype=np.float32)
    ep_returns = np.zeros(n_envs)
    reward_history = []
    global_step = 0

    t0 = time.time()
    num_updates = TOTAL_TIMESTEPS // batch_size

    for update in range(num_updates):
        obs_buf = np.zeros((n_steps, n_envs, obs_dim), dtype=np.float32)
        act_buf = np.zeros((n_steps, n_envs), dtype=np.int32)
        logp_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
        rew_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
        done_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
        val_buf = np.zeros((n_steps, n_envs), dtype=np.float32)

        for step in range(n_steps):
            global_step += n_envs
            obs_buf[step] = next_obs
            done_buf[step] = next_done
            key, subkey = jax.random.split(key)
            action, logp, value = get_action_value(params, jnp.array(next_obs), subkey)
            action_np = np.array(action)

            next_obs, reward, terminated, truncated, infos = envs.step(action_np)
            done = np.logical_or(terminated, truncated)

            act_buf[step] = action_np
            logp_buf[step] = np.array(logp)
            val_buf[step] = np.array(value)
            rew_buf[step] = reward
            ep_returns += reward

            for i, d in enumerate(done):
                if d:
                    reward_history.append((global_step, ep_returns[i]))
                    ep_returns[i] = 0.0

            next_done = done.astype(np.float32)

        _, _, next_value = get_action_value(params, jnp.array(next_obs), key)
        next_value = np.array(next_value)

        advantages = np.zeros_like(rew_buf)
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

        b_inds = np.arange(batch_size)
        for epoch in range(cfg["n_epochs"]):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                mb = b_inds[start:start + minibatch_size]
                params, opt_state, loss = update_step(
                    params, opt_state,
                    jnp.array(b_obs[mb]), jnp.array(b_act[mb]), jnp.array(b_logp[mb]),
                    jnp.array(b_adv[mb]), jnp.array(b_ret[mb]),
                )

    elapsed = time.time() - t0

    smoothed = []
    if reward_history:
        window = []
        for ts, r in reward_history:
            window.append(r)
            if len(window) > 20:
                window.pop(0)
            smoothed.append((ts, float(np.mean(window))))

    save_result("jax_pure", elapsed, smoothed)
    envs.close()

    eval_key = [key]  # list so it can be reassigned inside the closure

    def act_fn(obs):
        eval_key[0], subkey = jax.random.split(eval_key[0])
        logits = forward_actor(params, jnp.array(obs, dtype=jnp.float32))
        action = jax.random.categorical(subkey, logits)
        return int(action)

    eval_returns = evaluate_policy(ENV_ID, act_fn)
    save_eval_result("jax_pure", elapsed, eval_returns)


if __name__ == "__main__":
    main()
