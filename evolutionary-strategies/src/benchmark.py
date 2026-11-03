"""
Benchmark module: evolutionary training loop for every algorithm.

All 3 families (Family 1: Direct Solutions, Family 2: Symbolic Programs, and 
Family 3: Probabilistic Models/EDA) are accelerated via pure JAX vectorized 
compilation (vmap + lax.scan). A fallback Python interpreted mode is retained 
for debugging.

Returns: dict with the fitness history (mean, max, std) per generation.
"""
from __future__ import annotations
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Callable, Any
import numpy as np
import jax
import jax.numpy as jnp

from src.environments import ENV_META, make_pop_eval_fn, make_rollout_fn


# ---------------------------------------------------------------------------
# Result of a single run
# ---------------------------------------------------------------------------
class RunResult:
    def __init__(self, algo_name: str, env_name: str, family: int):
        self.algo_name   = algo_name
        self.env_name    = env_name
        self.family      = family
        self.generations: List[int]   = []
        self.mean_fitness: List[float] = []
        self.max_fitness:  List[float] = []
        self.std_fitness:  List[float] = []
        self.best_fitness: List[float] = []
        self.wall_time:    List[float] = []
        self.total_time:   float = 0.0
        self.n_evals:      int   = 0
        self.best_params:  Any   = None
        self.best_program: Any   = None
        self.policy_fn:    Any   = None


# ---------------------------------------------------------------------------
# Benchmark for Families 1 and 3 (real parameter vectors — evaluation in JAX)
# ---------------------------------------------------------------------------

def run_param_based(
    algo,              # GA / DE / CMA-ES / OpenAI-ES / PBIL instance
    algo_name: str,
    env_name: str,
    family: int,
    n_generations: int = 200,
    n_rollouts: int = 4,
    seed: int = 42,
    log_every: int = 10,
) -> RunResult:
    """
    Training loop for Family 1 (Direct) and Family 3 (EDA) algorithms.
    Fitness evaluation is performed in JAX (vmap over the population).
    """
    result = RunResult(algo_name, env_name, family)
    rng    = jax.random.PRNGKey(seed)

    # Set up the environment and the evaluation function
    eval_fn  = make_pop_eval_fn(env_name, n_rollouts=n_rollouts)
    n_params = eval_fn.n_params

    # Initialize the algorithm
    rng, rng_init = jax.random.split(rng)
    if hasattr(algo, '_hyperparams'):  # CMA-ES needs n to compute pop_size
        state = algo.init(rng_init, n_params)
        pop_size = algo._hp["lam"]
    else:
        state = algo.init(rng_init, n_params)
        pop_size = algo.pop_size

    # First JAX compilation (warm-up)
    rng, rng_ask, rng_eval = jax.random.split(rng, 3)

    # For DE, the trial vector must be kept separately
    is_de = hasattr(algo, 'F')

    t0 = time.time()
    for gen in range(n_generations):
        t_gen = time.time()
        rng, rng_ask, rng_eval = jax.random.split(rng, 3)

        if is_de:
            trial, state_with_trial = algo.ask(state, rng_ask)
            fitness = eval_fn(trial, rng_eval)
            fitness_np = np.array(fitness)
            state = algo.tell(state, fitness, trial)
        else:
            pop, state = algo.ask(state, rng_ask)
            fitness = eval_fn(pop, rng_eval)
            fitness_np = np.array(fitness)

            if isinstance(algo, type(None)):  # never
                pass
            elif hasattr(algo, 'tell'):
                sig = algo.tell.__code__.co_varnames
                if 'rng' in sig:
                    state = algo.tell(state, fitness, rng_ask)
                elif 'population' in sig:
                    state = algo.tell(state, fitness, pop, rng_ask)
                else:
                    state = algo.tell(state, fitness, rng_ask)

        result.generations.append(gen)
        result.mean_fitness.append(float(np.mean(fitness_np)))
        result.max_fitness.append(float(np.max(fitness_np)))
        result.std_fitness.append(float(np.std(fitness_np)))
        result.best_fitness.append(float(state.best_fitness))
        result.wall_time.append(time.time() - t_gen)
        result.n_evals += pop_size

        if gen % log_every == 0 or gen == n_generations - 1:
            print(f"  [{algo_name:12s}|{env_name:28s}] gen={gen:3d}  "
                  f"best={float(state.best_fitness):8.2f}  "
                  f"mean={float(np.mean(fitness_np)):8.2f}  "
                  f"t={time.time()-t_gen:.2f}s")

    result.total_time = time.time() - t0
    return result


def run_param_based_v2(
    algo,
    algo_name: str,
    env_name: str,
    family: int,
    n_generations: int = 200,
    n_rollouts: int = 4,
    seed: int = 42,
    log_every: int = 10,
) -> RunResult:
    """Corrected version that dispatches tell() according to the algorithm type."""
    from src.family1_direct import DE, SimpleGA, OpenAIES
    from src.family3_eda    import CMAES, PBIL

    result = RunResult(algo_name, env_name, family)
    rng    = jax.random.PRNGKey(seed)

    eval_fn  = make_pop_eval_fn(env_name, n_rollouts=n_rollouts)
    n_params = eval_fn.n_params

    rng, rng_init = jax.random.split(rng)
    state = algo.init(rng_init, n_params)

    if isinstance(algo, CMAES):
        pop_size = algo._hp["lam"]
    else:
        pop_size = algo.pop_size

    t0 = time.time()
    for gen in range(n_generations):
        t_gen = time.time()
        rng, rng_ask, rng_eval = jax.random.split(rng, 3)

        if isinstance(algo, DE):
            trial, _    = algo.ask(state, rng_ask)
            fitness     = eval_fn(trial, rng_eval)
            state       = algo.tell(state, fitness, trial)

        elif isinstance(algo, SimpleGA):
            pop, state  = algo.ask(state, rng_ask)
            fitness     = eval_fn(pop, rng_eval)
            state       = algo.tell(state, fitness, rng_ask)

        elif isinstance(algo, CMAES):
            pop, state  = algo.ask(state, rng_ask)
            fitness     = eval_fn(pop, rng_eval)
            state       = algo.tell(state, pop, fitness)

        elif isinstance(algo, OpenAIES):
            pop, state  = algo.ask(state, rng_ask)
            fitness     = eval_fn(pop, rng_eval)
            state       = algo.tell(state, fitness)

        elif isinstance(algo, PBIL):
            pop, state  = algo.ask(state, rng_ask)
            fitness     = eval_fn(pop, rng_eval)
            state       = algo.tell(state, fitness, pop, rng_ask)

        else:
            raise ValueError(f"Unknown algorithm: {type(algo)}")

        fitness_np = np.array(fitness)
        result.generations.append(gen)
        result.mean_fitness.append(float(np.mean(fitness_np)))
        result.max_fitness.append(float(np.max(fitness_np)))
        result.std_fitness.append(float(np.std(fitness_np)))
        result.best_fitness.append(float(state.best_fitness))
        result.wall_time.append(time.time() - t_gen)
        result.n_evals += pop_size

        if gen % log_every == 0 or gen == n_generations - 1:
            print(f"  [{algo_name:12s}|{env_name:28s}] gen={gen:3d}  "
                  f"best={float(state.best_fitness):8.2f}  "
                  f"mean={float(np.mean(fitness_np)):8.2f}  "
                  f"t={time.time()-t_gen:.2f}s")

    result.total_time = time.time() - t0
    result.best_params = np.array(algo.best_params(state))
    return result


# ---------------------------------------------------------------------------
# Benchmark Family 2 (Programs — Pure JAX Vectorized Rollout & Evaluation)
# ---------------------------------------------------------------------------

def _rollout_program_python(policy_fn: Callable, env_name: str,
                             n_rollouts: int, seed: int) -> float:
    """
    Evaluate a program-based policy through gymnax in Python (fallback interpreter).
    """
    import gymnax
    env, env_params = gymnax.make(env_name)
    meta     = ENV_META[env_name]
    discrete = meta["discrete"]
    max_steps= meta["max_steps"]
    act_scale= meta.get("act_scale", 1.0) or 1.0

    step_jit = jax.jit(env.step)
    reset_jit = jax.jit(env.reset)

    total_returns = []
    for ep in range(n_rollouts):
        rng = jax.random.PRNGKey(seed * 100 + ep)
        rng_r, rng_s = jax.random.split(rng)
        obs, state = reset_jit(rng_r, env_params)
        obs_np = np.array(obs, dtype=np.float32)

        total_r = 0.0
        done    = False
        for _ in range(max_steps):
            action_raw = policy_fn(obs_np)
            if discrete:
                action = int(np.argmax(action_raw))
            else:
                action = np.clip(
                    np.tanh(action_raw) * act_scale,
                    -act_scale, act_scale
                ).astype(np.float32)
            rng_s, sub = jax.random.split(rng_s)
            obs, state, reward, terminated, truncated, _ = step_jit(sub, state, action, env_params)
            done_step = bool(terminated) or bool(truncated)
            obs_np   = np.array(obs, dtype=np.float32)
            total_r += float(reward)
            if bool(done_step):
                break
        total_returns.append(total_r)
    return float(np.mean(total_returns))


def run_program_based(
    algo,
    algo_name: str,
    env_name: str,
    family: int,
    n_generations: int = 100,
    n_rollouts: int = 4,
    seed: int = 42,
    log_every: int = 10,
    pure_jax: bool = True,
) -> RunResult:
    """
    Training loop for Family 2 (Programs: LinearGP and CartesianGP).
    Accelerated via pure JAX vectorized compilation (vmap + lax.scan) across
    episodes and individuals.
    """
    from src.environments import make_program_pop_eval_fn

    result  = RunResult(algo_name, env_name, family)
    rng     = jax.random.PRNGKey(seed)

    rng, rng_init = jax.random.split(rng)
    state = algo.init(rng_init)

    if pure_jax:
        eval_fn = make_program_pop_eval_fn(env_name, algo, n_rollouts=n_rollouts)

    t0 = time.time()
    for gen in range(n_generations):
        t_gen = time.time()
        rng, rng_eval = jax.random.split(rng)

        if pure_jax:
            # Entire population evaluated in parallel in pure JAX
            fitness = eval_fn(state.programs, rng_eval)
            fitness.block_until_ready()
            fitness_np = np.array(fitness, dtype=np.float32)
        else:
            # Fallback Python interpreted evaluation
            programs = np.array(state.programs)
            fitness_list = []
            for i in range(len(programs)):
                prog = programs[i]
                pf   = algo.get_policy_fn(prog)
                f    = _rollout_program_python(pf, env_name, n_rollouts, seed + gen * 1000 + i)
                fitness_list.append(f)
            fitness_np = np.array(fitness_list, dtype=np.float32)

        # Update the algorithm
        state = algo.tell(state, jnp.array(fitness_np))

        result.generations.append(gen)
        result.mean_fitness.append(float(np.mean(fitness_np)))
        result.max_fitness.append(float(np.max(fitness_np)))
        result.std_fitness.append(float(np.std(fitness_np)))
        result.best_fitness.append(float(state.best_fitness))
        result.wall_time.append(time.time() - t_gen)
        result.n_evals += len(state.programs)

        if gen % log_every == 0 or gen == n_generations - 1:
            print(f"  [{algo_name:12s}|{env_name:28s}] gen={gen:3d}  "
                  f"best={float(state.best_fitness):8.2f}  "
                  f"mean={float(np.mean(fitness_np)):8.2f}  "
                  f"t={time.time()-t_gen:.2f}s")

    result.total_time = time.time() - t0
    result.best_program = np.array(state.best_program)
    result.policy_fn = algo.get_policy_fn(state.best_program)
    return result
