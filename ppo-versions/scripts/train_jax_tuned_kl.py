"""Igual ao train_jax_tuned.py (normalização de obs/reward, rede 128x128, LR
annealing), MAS com os 2 truques específicos do RLlib que ficaram como
hipótese em aberto no README:

5. Value function clipping (RLLIB_VF_CLIP_PARAM = 10.0) -- clipa o quanto a
   estimativa de valor pode se mover em relação ao valor coletado no rollout,
   igual ao clipping de política, só que pra value function.
6. Penalidade de KL dinâmica (RLLIB_KL_TARGET = 0.01, RLLIB_INITIAL_KL_COEFF
   = 0.2) -- adiciona kl_coeff * KL(antiga||nova) na loss, e ajusta kl_coeff
   a cada update: sobe se a KL medida passou do alvo, desce se ficou abaixo.

Objetivo: isolar se esses 2 mecanismos específicos (e não outra coisa
qualquer do RLlib) explicam o gap que sobrou entre jax_tuned (45.1) e o
RLlib de verdade (158.9) no LunarLander.
"""
import time
import sys
sys.path.insert(0, "/home/claude/ppo-benchmark/scripts")
from common import ENV_ID, SEED, TOTAL_TIMESTEPS, PPO_CONFIG, save_result, evaluate_policy, save_eval_result

import numpy as np
import jax
import jax.numpy as jnp
import optax
import gymnasium as gym

HIDDEN = 128
RLLIB_VF_CLIP_PARAM = 10.0
RLLIB_KL_TARGET = 0.01
RLLIB_INITIAL_KL_COEFF = 0.2


class RunningMeanStd:
    """Welford/Chan's algorithm -- média e variância rodantes, atualizadas em batch."""
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        self.mean = new_mean
        self.var = M2 / tot_count
        self.count = tot_count


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
        "a1": dense(k1, obs_dim, HIDDEN, gh), "a2": dense(k2, HIDDEN, HIDDEN, gh), "a3": dense(k3, HIDDEN, act_dim, ga),
        "c1": dense(k4, obs_dim, HIDDEN, gh), "c2": dense(k5, HIDDEN, HIDDEN, gh), "c3": dense(k6, HIDDEN, 1, gc),
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


def ppo_loss(params, obs, actions, old_logp, old_value, advantages, returns, kl_coeff, cfg):
    logp, entropy, value = evaluate_actions(params, obs, actions)
    logratio = logp - old_logp
    ratio = jnp.exp(logratio)

    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    pg1 = -adv * ratio
    pg2 = -adv * jnp.clip(ratio, 1 - cfg["clip_coef"], 1 + cfg["clip_coef"])
    pg_loss = jnp.maximum(pg1, pg2).mean()

    # value function clipping, igual ao clipping de política só que pro crítico
    v_clipped = old_value + jnp.clip(value - old_value, -RLLIB_VF_CLIP_PARAM, RLLIB_VF_CLIP_PARAM)
    v_loss_unclipped = (value - returns) ** 2
    v_loss_clipped = (v_clipped - returns) ** 2
    v_loss = 0.5 * jnp.mean(jnp.maximum(v_loss_unclipped, v_loss_clipped))

    ent_loss = entropy.mean()

    # penalidade de KL: estimador de Schulman (k3), baixa variância, não exige
    # os logits antigos completos -- só old_logp da ação amostrada
    approx_kl = jnp.mean((ratio - 1) - logratio)

    loss = (pg_loss - cfg["ent_coef"] * ent_loss + cfg["vf_coef"] * v_loss
            + kl_coeff * approx_kl)
    return loss, approx_kl


def normalize_obs(obs, rms, eps=1e-8):
    return np.clip((obs - rms.mean) / np.sqrt(rms.var + eps), -10.0, 10.0)


def main():
    cfg = PPO_CONFIG
    key = jax.random.PRNGKey(SEED)
    np.random.seed(SEED)

    envs = gym.vector.SyncVectorEnv([make_env for _ in range(cfg["n_envs"])])
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.n

    key, init_key = jax.random.split(key)
    params = init_params(init_key, obs_dim, act_dim, cfg)

    n_steps, n_envs = cfg["n_steps"], cfg["n_envs"]
    batch_size = n_steps * n_envs
    minibatch_size = batch_size // cfg["minibatches"]
    num_updates = TOTAL_TIMESTEPS // batch_size

    # LR com annealing linear -- optax schedule
    lr_schedule = optax.linear_schedule(
        init_value=cfg["lr"], end_value=0.0, transition_steps=num_updates * cfg["n_epochs"] * cfg["minibatches"]
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg["max_grad_norm"]),
        optax.adam(lr_schedule, eps=cfg["adam_eps"]),
    )
    opt_state = optimizer.init(params)

    @jax.jit
    def update_step(params, opt_state, obs, actions, old_logp, old_value, advantages, returns, kl_coeff):
        (loss, approx_kl), grads = jax.value_and_grad(ppo_loss, has_aux=True)(
            params, obs, actions, old_logp, old_value, advantages, returns, kl_coeff, cfg
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, approx_kl

    obs_rms = RunningMeanStd(shape=(obs_dim,))
    ret_rms = RunningMeanStd(shape=())
    running_ret = np.zeros(n_envs)
    kl_coeff = RLLIB_INITIAL_KL_COEFF

    raw_obs, _ = envs.reset(seed=SEED)
    obs_rms.update(raw_obs)
    next_obs = normalize_obs(raw_obs, obs_rms)
    next_done = np.zeros(n_envs, dtype=np.float32)
    ep_returns = np.zeros(n_envs)  # retorno RAW (não-normalizado), pra log comparável com as outras libs
    reward_history = []
    global_step = 0

    t0 = time.time()

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

            raw_obs, raw_reward, terminated, truncated, infos = envs.step(action_np)
            done = np.logical_or(terminated, truncated)

            # normalização de reward: divide pelo std do retorno descontado rodante
            running_ret = running_ret * cfg["gamma"] + raw_reward
            ret_rms.update(running_ret.reshape(-1, 1).squeeze(-1) if running_ret.ndim else running_ret)
            norm_reward = np.clip(raw_reward / np.sqrt(ret_rms.var + 1e-8), -10.0, 10.0)

            obs_rms.update(raw_obs)
            next_obs = normalize_obs(raw_obs, obs_rms)

            act_buf[step] = action_np
            logp_buf[step] = np.array(logp)
            val_buf[step] = np.array(value)
            rew_buf[step] = norm_reward
            ep_returns += raw_reward  # log continua em escala RAW, comparável com as outras libs

            for i, d in enumerate(done):
                if d:
                    reward_history.append((global_step, ep_returns[i]))
                    ep_returns[i] = 0.0
                    running_ret[i] = 0.0

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
        b_val = val_buf.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)

        b_inds = np.arange(batch_size)
        kl_this_update = []
        for epoch in range(cfg["n_epochs"]):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                mb = b_inds[start:start + minibatch_size]
                params, opt_state, loss, approx_kl = update_step(
                    params, opt_state,
                    jnp.array(b_obs[mb]), jnp.array(b_act[mb]), jnp.array(b_logp[mb]),
                    jnp.array(b_val[mb]), jnp.array(b_adv[mb]), jnp.array(b_ret[mb]),
                    kl_coeff,
                )
                kl_this_update.append(float(approx_kl))

        # ajuste dinâmico do coeficiente de KL, igual ao RLlib: sobe se passou
        # do alvo, desce se ficou bem abaixo -- reavaliado a cada iteração de coleta+update
        mean_kl = float(np.mean(kl_this_update))
        if mean_kl > 1.5 * RLLIB_KL_TARGET:
            kl_coeff *= 1.5
        elif mean_kl < RLLIB_KL_TARGET / 1.5:
            kl_coeff *= 0.5
        kl_coeff = float(np.clip(kl_coeff, 1e-4, 100.0))

    elapsed = time.time() - t0

    smoothed = []
    if reward_history:
        window = []
        for ts, r in reward_history:
            window.append(r)
            if len(window) > 20:
                window.pop(0)
            smoothed.append((ts, float(np.mean(window))))

    save_result("jax_tuned_kl", elapsed, smoothed)
    envs.close()

    # congela obs_rms no estado final de treino -- avaliação usa as MESMAS estatísticas
    eval_key = [key]

    def act_fn(obs):
        eval_key[0], subkey = jax.random.split(eval_key[0])
        norm_obs = normalize_obs(np.array(obs), obs_rms)
        logits = forward_actor(params, jnp.array(norm_obs, dtype=jnp.float32))
        action = jax.random.categorical(subkey, logits)
        return int(action)

    eval_returns = evaluate_policy(ENV_ID, act_fn)
    save_eval_result("jax_tuned_kl", elapsed, eval_returns)


if __name__ == "__main__":
    main()
