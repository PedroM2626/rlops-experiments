"""
Rigorous Statistical Evaluation and Out-of-Sample Validation module.

Runs a strict evaluation of the trained policies over N=100 test episodes with
independent pseudo-random seeds that were never seen during training.

Computes formal statistical metrics according to the state of the art in RL
(Henderson et al. 2018; Machado et al. 2018; Agarwal et al. 2021):
  - Central tendency: sample mean and median
  - Dispersion and uncertainty: standard deviation, variance, SEM, 95% bootstrap CI, IQR
  - Noise/reliability metrics: Signal-to-Noise Ratio (SNR), Coefficient of Variation (CV)
  - Generalization vs overfitting: optimism gap / winner's curse (f_train - R_test)
  - Task completion / success rate (Success Rate)
"""
from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, Any, Callable

from src.environments import ENV_META, forward_mlp, param_count


SUCCESS_THRESHOLDS = {
    "CartPole-v1":              475.0,   # Standard Gym criterion (holding for 475+ steps)
    "Acrobot-v1":              -100.0,   # At least -100 steps to reach the target bar
    "Pendulum-v1":             -200.0,   # Stabilization close to the upright equilibrium
    "MountainCarContinuous-v0":  90.0,   # Full traversal of the valley with the peak reached
}


def make_test_eval_fn(env_name: str, n_episodes: int = 100) -> Callable:
    """
    Returns a JIT-compiled function that evaluates a policy over n_episodes
    independent episodes:
        test_eval(flat_params, rng) -> returns [n_episodes]
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

    def evaluate_test(flat: jnp.ndarray, rng: jax.Array) -> jnp.ndarray:
        rngs = jax.random.split(rng, n_episodes)
        return jax.vmap(single_episode, in_axes=(None, 0))(flat, rngs)

    eval_jit = jax.jit(evaluate_test)
    eval_jit.env_name = env_name
    eval_jit.n_params = param_count(env_name)
    return eval_jit


def evaluate_program_100(policy_fn: Callable, env_name: str,
                         n_episodes: int = 100, seed: int = 99999) -> np.ndarray:
    """Evaluates a symbolic (GP) program over n_episodes test episodes in Python."""
    import gymnax
    env, env_params = gymnax.make(env_name)
    meta      = ENV_META[env_name]
    discrete  = meta["discrete"]
    max_steps = meta["max_steps"]
    act_scale = meta.get("act_scale", 1.0) or 1.0

    step_jit = jax.jit(env.step)
    reset_jit = jax.jit(env.reset)

    returns = []
    for ep in range(n_episodes):
        rng = jax.random.PRNGKey(seed + ep * 7)
        rng_r, rng_s = jax.random.split(rng)
        obs, state = reset_jit(rng_r, env_params)
        obs_np = np.array(obs, dtype=np.float32)

        total_r = 0.0
        for _ in range(max_steps):
            action_raw = policy_fn(obs_np)
            if discrete:
                action = int(np.argmax(action_raw))
            else:
                action = np.clip(np.tanh(action_raw) * act_scale, -act_scale, act_scale).astype(np.float32)
            rng_s, sub = jax.random.split(rng_s)
            obs, state, reward, terminated, truncated, _ = step_jit(sub, state, action, env_params)
            total_r += float(reward)
            if bool(terminated) or bool(truncated):
                break
            obs_np = np.array(obs, dtype=np.float32)
        returns.append(total_r)
    return np.array(returns, dtype=np.float32)


def compute_statistical_metrics(
    returns: np.ndarray,
    f_train_best: float | None = None,
    env_name: str | None = None,
    n_bootstraps: int = 2000,
    seed: int = 1234,
) -> Dict[str, Any]:
    """
    Computes rigorous validation metrics over the empirical distribution of 100 returns.
    """
    returns = np.asarray(returns, dtype=np.float64)
    n = len(returns)

    # 1. Measures of central tendency
    mean_val   = float(np.mean(returns))
    median_val = float(np.median(returns))

    # 2. Measures of dispersion
    std_val = float(np.std(returns, ddof=1)) if n > 1 else 0.0
    var_val = float(np.var(returns, ddof=1)) if n > 1 else 0.0
    sem_val = float(std_val / np.sqrt(n))   if n > 0 else 0.0

    # 3. 95% confidence interval via non-parametric bootstrap (Agarwal et al. 2021)
    rng_np = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstraps)
    for b in range(n_bootstraps):
        boot_sample = rng_np.choice(returns, size=n, replace=True)
        boot_means[b] = np.mean(boot_sample)
    ci95_low  = float(np.percentile(boot_means, 2.5))
    ci95_high = float(np.percentile(boot_means, 97.5))

    # 4. Order statistics and quantiles
    q25 = float(np.percentile(returns, 25))
    q75 = float(np.percentile(returns, 75))
    iqr_val = q75 - q25
    min_val = float(np.min(returns))
    max_val = float(np.max(returns))

    # 5. Noise and stability metrics
    # SNR = |μ| / (σ + ε)
    snr_val = float(abs(mean_val) / (std_val + 1e-8))
    # Coefficient of variation (relative dispersion) = σ / (|μ| + ε)
    cv_val = float(std_val / (abs(mean_val) + 1e-8))
    # Fraction of stochastic variance (noise fraction)
    noise_fraction = float(var_val / (var_val + mean_val**2 + 1e-8))

    # 6. Success rate
    threshold = SUCCESS_THRESHOLDS.get(env_name, 0.0)
    success_rate = float(np.mean(returns >= threshold)) * 100.0

    # 7. Optimism gap (winner's curse / selection bias)
    optimism_gap = float(f_train_best - mean_val) if f_train_best is not None else 0.0

    return {
        "n_episodes":       n,
        "test_mean":        round(mean_val, 2),
        "test_median":      round(median_val, 2),
        "test_std":         round(std_val, 2),
        "test_sem":         round(sem_val, 2),
        "ci95_low":         round(ci95_low, 2),
        "ci95_high":        round(ci95_high, 2),
        "iqr":              round(iqr_val, 2),
        "min":              round(min_val, 2),
        "max":              round(max_val, 2),
        "snr":              round(snr_val, 2),
        "cv":               round(cv_val, 3),
        "noise_fraction":   round(noise_fraction, 4),
        "success_rate_pct": round(success_rate, 1),
        "f_train_best":     round(f_train_best, 2) if f_train_best is not None else None,
        "optimism_gap":     round(optimism_gap, 2) if f_train_best is not None else None,
        "raw_returns":      returns.tolist(),
    }
