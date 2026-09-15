"""Config compartilhada entre todas as implementações, pra manter comparação justa."""
import os

ENV_ID = os.environ.get("PPO_ENV", "CartPole-v1")
SEED = int(os.environ.get("PPO_SEED", 42))
TOTAL_TIMESTEPS = int(os.environ.get("PPO_TIMESTEPS", 150_000))
EVAL_EPISODES = int(os.environ.get("PPO_EVAL_EPISODES", 50))
RUN_TAG = os.environ.get("PPO_RUN_TAG", "")  # sufixo pra não misturar configs diferentes no mesmo env/seed
RESULTS_SUBDIR = ENV_ID.replace("/", "_") + RUN_TAG

# Hiperparâmetros PPO "canônicos" (mesmos valores em todas as libs, onde aplicável).
# Parte 12: valores do RL Baselines3 Zoo pra LunarLander (n_steps, n_envs,
# gae_lambda, gamma, n_epochs, ent_coef, batch_size via "minibatches" derivado)
# em vez dos genéricos herdados do CartPole -- ver README Parte 11/12.
PPO_CONFIG = dict(
    n_steps=1024,          # steps por env antes de update (Zoo: 1024)
    n_envs=16,             # envs paralelos (Zoo: 16)
    n_epochs=4,            # epochs de otimização por batch (Zoo: 4)
    gamma=0.999,           # Zoo: 0.999 (era 0.99 -- episódios longos precisam de mais peso no futuro)
    gae_lambda=0.98,       # Zoo: 0.98 (era 0.95)
    clip_coef=0.2,
    ent_coef=0.01,         # Zoo: 0.01 (igual ao que já usávamos)
    vf_coef=0.5,
    lr=3e-4,
    minibatches=256,       # (1024*16)/256 = batch_size 64, igual ao Zoo
    max_grad_norm=0.5,
    # arquitetura -- igual nas libs (default do SB3, o Zoo não sobrescreve isso)
    hidden_sizes=(64, 64),
    activation="tanh",
    adam_eps=1e-5,
    # inicialização ortogonal estilo CleanRL/SB3 (ganho por tipo de camada)
    ortho_gain_hidden=2 ** 0.5,
    ortho_gain_actor_out=0.01,
    ortho_gain_critic_out=1.0,
)


def evaluate_policy(env_id, act_fn, n_episodes=None, base_seed=100_000):
    """Roda N episódios com uma política JÁ TREINADA (congelada) e retorna a lista
    de retornos por episódio. act_fn(obs) -> action (int). Usa amostragem estocástica
    (não greedy) pra capturar a variância real que o modelo treinado exibe."""
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
    print(f"[EVAL {name} seed={SEED} env={ENV_ID}] {len(eval_returns)} episódios | "
          f"mean={out['eval_mean']:.1f} std={out['eval_std']:.1f}")
    return out


def save_result(name, elapsed_s, rewards, path=None):
    """rewards: lista de (timestep, mean_reward) durante o treino."""
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
