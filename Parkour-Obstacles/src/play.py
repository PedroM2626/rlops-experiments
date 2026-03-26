"""
Play script - loads a trained PPO model and renderiza em tempo real.

Uso:
    python play.py                                 # roda até fechar janela (usa modelo padrão)
    python play.py --model models/meu_modelo       # modelo específico
    python play.py --no-render --n-episodes 5      # benchmark headless

Sem --n-episodes, roda indefinidamente até o usuário fechar a janela.
"""

import argparse
import os
import sys
import time

import numpy as np
import pybullet as p
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.env.parkour_env import ParkourEnv
from stable_baselines3.common.monitor import Monitor

DEFAULT_MODEL  = "models/ppo_parkour_final"
DEFAULT_VECNORM = "models/vec_normalize.pkl"


def parse_args():
    parser = argparse.ArgumentParser(description="Watch the trained parkour agent")
    parser.add_argument("--model",      type=str,  default=DEFAULT_MODEL,
                        help="Caminho do modelo .zip (sem extensão)")
    parser.add_argument("--vecnorm",    type=str,  default=DEFAULT_VECNORM,
                        help="Caminho do VecNormalize .pkl")
    parser.add_argument("--no-render",  action="store_true",
                        help="Rodar sem janela GUI (benchmark headless)")
    parser.add_argument("--level-seed", type=int,  default=None,
                        help="Seed do level para reproducibilidade")
    parser.add_argument("--n-episodes", type=int,  default=None,
                        help="Número de episódios (padrão: infinito até fechar janela)")
    parser.add_argument("--random",      action="store_true",
                        help="Usar acoes aleatorias em vez do modelo (debug)")
    parser.add_argument("--steps",      type=int,  default=2000,
                        help="Máx steps por episódio (padrão: 2000)")
    return parser.parse_args()


def _window_open(client: int) -> bool:
    """Retorna False se a janela PyBullet foi fechada ou nao conectada."""
    if client is None:
        return False
    try:
        info = p.getConnectionInfo(client)
        return bool(info.get('isConnected', False))
    except Exception:
        return False


def main():
    args = parse_args()
    render_mode = "direct" if args.no_render else "human"
    infinite    = (args.n_episodes is None) and (not args.no_render)

    # Verificar modelo
    model_path = args.model
    if not os.path.exists(model_path + ".zip"):
        print(f"[ERRO] Modelo nao encontrado em: {model_path}.zip")
        print("Treine primeiro com:  python train.py")
        sys.exit(1)

    # Criar ambiente
    def make_env():
        env = ParkourEnv(render_mode=render_mode, level_seed=args.level_seed)
        return Monitor(env)

    vec_env = DummyVecEnv([make_env])

    if os.path.exists(args.vecnorm):
        vec_env = VecNormalize.load(args.vecnorm, vec_env)
        vec_env.training    = False
        vec_env.norm_reward = False
        print(f"VecNormalize carregado: {args.vecnorm}")
    else:
        print("[AVISO] VecNormalize nao encontrado, rodando sem normalizacao")

    model = PPO.load(model_path, env=vec_env, device="cpu")
    print(f"Modelo carregado     : {model_path}")

    # Pegar o client PyBullet para detectar fechamento da janela
    inner_env: ParkourEnv = vec_env.envs[0].env

    if infinite:
        print("\nRodando indefinidamente - feche a janela PyBullet para sair.\n")
    else:
        n_ep = args.n_episodes if args.n_episodes else 5
        print(f"\nRodando {n_ep} episodio(s).\n")

    ep_rewards = []
    episode    = 0

    try:
        while True:
            # Start/Reset env (this initializes the PyBullet client)
            obs      = vec_env.reset()
            ep_rew   = 0.0
            step     = 0
            done_ep  = False

            # Parar se número fixo de episódios foi atingido
            if not infinite and args.n_episodes and episode >= args.n_episodes:
                break

            # Check if window was closed or failed to open
            if render_mode == "human" and not _window_open(inner_env._client):
                print("\nJanela fechada pelo usuario ou erro na conexao.")
                break

            while not done_ep and step < args.steps:
                # Verificar se a janela ainda esta aberta a cada passo
                if render_mode == "human" and not _window_open(inner_env._client):
                    print("\nJanela fechada pelo usuario.")
                    return

                if args.random:
                    # sample() returns one action, SB3 wants a list since it is a VecEnv
                    action = np.array([vec_env.action_space.sample()])
                else:
                    action, _ = model.predict(obs, deterministic=True)
                try:
                    obs, reward, done, _ = vec_env.step(action)
                except (p.error, Exception) as e:
                    err_msg = str(e).lower()
                    if "not connected" in err_msg or "physics server" in err_msg:
                        print("\nConexao perdida com o simulador (janela fechada).")
                        return
                    raise e

                ep_rew  += float(reward[0])
                step    += 1
                done_ep  = bool(done[0])

                if not args.no_render:
                    time.sleep(1.0 / 60.0)  # ~60 fps

            episode += 1
            ep_rewards.append(ep_rew)
            print(f"Episodio {episode:4d} | reward: {ep_rew:8.2f} | steps: {step}")

    except KeyboardInterrupt:
        print("\nInterrompido pelo usuario (Ctrl+C).")

    finally:
        if ep_rewards:
            print(f"\n=== Resumo ({len(ep_rewards)} episodio(s)) ===")
            print(f"  Media : {np.mean(ep_rewards):.2f}")
            print(f"  Std   : {np.std(ep_rewards):.2f}")
            print(f"  Max   : {np.max(ep_rewards):.2f}")
        vec_env.close()


if __name__ == "__main__":
    main()
