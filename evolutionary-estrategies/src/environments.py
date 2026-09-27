"""
Wrappers around gymnax for vectorized fitness evaluation.

Each environment exposes `rollout(flat_params, rng) -> mean_return`.
The policy is a 2-layer MLP shared by all algorithms:
    obs (obs_dim) -> Dense(32) -> tanh -> Dense(32) -> tanh -> Dense(act_dim)

For discrete actions: argmax(logits)
For continuous actions: tanh(logits) * action_scale
"""
from __future__ import annotations
from typing import Callable, Dict, Any
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Environment metadata
# ---------------------------------------------------------------------------

ENV_META: Dict[str, Dict[str, Any]] = {
    "CartPole-v1":              dict(obs_dim=4,  act_dim=2,  discrete=True,  max_steps=500,  act_scale=None),
    "Acrobot-v1":               dict(obs_dim=6,  act_dim=3,  discrete=True,  max_steps=500,  act_scale=None),
    "Pendulum-v1":              dict(obs_dim=3,  act_dim=1,  discrete=False, max_steps=200,  act_scale=2.0),
    "MountainCarContinuous-v0": dict(obs_dim=2,  act_dim=1,  discrete=False, max_steps=999,  act_scale=1.0),
}

HIDDEN = 32  # neurons in the hidden layers


def param_count(env_name: str) -> int:
    """Total number of MLP policy parameters for the given environment."""
    m = ENV_META[env_name]
    obs, act = m["obs_dim"], m["act_dim"]
    return (obs * HIDDEN + HIDDEN) + (HIDDEN * HIDDEN + HIDDEN) + (HIDDEN * act + act)


def unpack_params(flat: jnp.ndarray, obs_dim: int, act_dim: int):
    """Unpacks a flat vector into the MLP weights/biases."""
    h = HIDDEN
    idx = 0
    W1 = flat[idx: idx + obs_dim * h].reshape(obs_dim, h); idx += obs_dim * h
    b1 = flat[idx: idx + h];                                idx += h
    W2 = flat[idx: idx + h * h].reshape(h, h);             idx += h * h
    b2 = flat[idx: idx + h];                                idx += h
    W3 = flat[idx: idx + h * act_dim].reshape(h, act_dim); idx += h * act_dim
    b3 = flat[idx: idx + act_dim];                          idx += act_dim
    return W1, b1, W2, b2, W3, b3


def forward_mlp(flat: jnp.ndarray, obs: jnp.ndarray,
                obs_dim: int, act_dim: int) -> jnp.ndarray:
    """Forward pass of the MLP."""
    W1, b1, W2, b2, W3, b3 = unpack_params(flat, obs_dim, act_dim)
    x = jnp.tanh(obs @ W1 + b1)
    x = jnp.tanh(x @ W2 + b2)
    return x @ W3 + b3


def make_rollout_fn(env_name: str, n_rollouts: int = 8) -> Callable:
    """
    Returns the JIT-compiled function:
        rollout(flat_params: [n_params], rng) -> scalar mean return

    Parameters
    ----------
    env_name   : name of the gymnax environment
    n_rollouts : number of episodes used for the average (reduces variance)
    """
    import gymnax
    env, env_params = gymnax.make(env_name)
    meta      = ENV_META[env_name]
    obs_dim   = meta["obs_dim"]
    act_dim   = meta["act_dim"]
    discrete  = meta["discrete"]
    max_steps = meta["max_steps"]
    act_scale = meta.get("act_scale", 1.0) or 1.0

    def single_episode(flat: jnp.ndarray, rng: jax.Array) -> jnp.ndarray:
        rng_reset, rng_run = jax.random.split(rng)
        obs, state = env.reset(rng_reset, env_params)

        def step_fn(carry, _):
            obs, state, done, total_r, rng = carry
            rng, sub = jax.random.split(rng)
            logits = forward_mlp(flat, obs, obs_dim, act_dim)
            if discrete:
                action = jnp.argmax(logits)
            else:
                action = jnp.tanh(logits) * act_scale
            obs, state, reward, terminated, truncated, _ = env.step(sub, state, action, env_params)
            done_step = terminated | truncated
            done_new  = done | done_step
            total_r   = total_r + jnp.where(done, 0.0, reward)
            return (obs, state, done_new, total_r, rng), None

        init = (obs, state, jnp.bool_(False), jnp.float32(0.0), rng_run)
        (_, _, _, total_r, _), _ = jax.lax.scan(step_fn, init, None, length=max_steps)
        return total_r

    def rollout(flat: jnp.ndarray, rng: jax.Array) -> jnp.ndarray:
        rngs    = jax.random.split(rng, n_rollouts)
        returns = jax.vmap(single_episode, in_axes=(None, 0))(flat, rngs)
        return jnp.mean(returns)

    rollout_jit = jax.jit(rollout)
    rollout_jit.n_params  = param_count(env_name)
    rollout_jit.env_name  = env_name
    rollout_jit.obs_dim   = obs_dim
    rollout_jit.act_dim   = act_dim
    return rollout_jit


def make_pop_eval_fn(env_name: str, n_rollouts: int = 8) -> Callable:
    """
    Returns a function that evaluates an entire population in parallel
    (vmap over individuals):
        evaluate(pop [pop_size, n_params], rng) -> fitness [pop_size]
    """
    rollout_fn = make_rollout_fn(env_name, n_rollouts)

    def evaluate(pop: jnp.ndarray, rng: jax.Array) -> jnp.ndarray:
        pop_size = pop.shape[0]
        rngs     = jax.random.split(rng, pop_size)
        return jax.vmap(rollout_fn)(pop, rngs)

    evaluate_jit = jax.jit(evaluate)
    evaluate_jit.n_params = rollout_fn.n_params
    evaluate_jit.env_name = env_name
    return evaluate_jit
