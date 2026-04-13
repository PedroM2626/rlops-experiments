"""
Visualization for Competitive Mario races.

Renders all competing agents side-by-side in a Pygame window,
showing each agent playing the same level simultaneously with
a live scoreboard.

Usage:
    python src/watch_race.py --agents mario_speedster mario_careful
    python src/watch_race.py --models-dir models --level-seed 42
"""

import argparse
import glob
import os
import sys
import time

import numpy as np

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.monitor import Monitor

from src.env.mario_env import MarioCompetitiveEnv, VIEWPORT_W, VIEWPORT_H


def find_agent_model(models_dir, agent_name):
    """Find the final model for an agent."""
    agent_dir = os.path.join(models_dir, agent_name)
    if not os.path.isdir(agent_dir):
        return None
    final = os.path.join(agent_dir, f"ppo_{agent_name}_final.zip")
    if os.path.exists(final):
        return final
    zips = sorted(glob.glob(os.path.join(agent_dir, "*.zip")))
    return zips[-1] if zips else None


def parse_args():
    p = argparse.ArgumentParser(description="Watch a Mario race between agents")
    p.add_argument("--agents", nargs="+", default=None)
    p.add_argument("--models-dir", type=str, default="models")
    p.add_argument("--level-seed", type=int, default=42)
    p.add_argument("--difficulty", type=float, default=0.5)
    p.add_argument("--level-width", type=int, default=200)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--scale", type=float, default=1.0,
                   help="Scale factor for each agent viewport")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        import pygame
    except ImportError:
        print("Error: pygame or pygame-ce is required. Install with: pip install pygame-ce")
        return

    models_dir = os.path.join(_PROJECT_DIR, args.models_dir)

    # discover agents
    if args.agents:
        agent_names = args.agents
    else:
        if not os.path.isdir(models_dir):
            print(f"Error: Models directory not found: {models_dir}")
            return
        agent_names = [
            d for d in os.listdir(models_dir)
            if os.path.isdir(os.path.join(models_dir, d))
        ]

    if not agent_names:
        print("Error: No agents found. Train some agents first.")
        return

    # load models and create environments
    agents = []
    for name in agent_names:
        model_path = find_agent_model(models_dir, name)
        if model_path is None:
            print(f"Warning: No model for '{name}', skipping.")
            continue

        # create env for this agent
        env = MarioCompetitiveEnv(
            level_seed=args.level_seed,
            difficulty=args.difficulty,
            level_width=args.level_width,
            render_mode="rgb_array",
        )
        env = Monitor(env)
        vec_env = DummyVecEnv([lambda e=env: e])
        vec_env = VecFrameStack(vec_env, n_stack=4)
        vec_env = VecTransposeImage(vec_env)

        model = PPO.load(model_path, device="auto")
        obs = vec_env.reset()

        agents.append({
            "name": name,
            "model": model,
            "vec_env": vec_env,
            "env": env,
            "obs": obs,
            "done": False,
            "info": {},
            "x_pos": 0,
            "reward": 0.0,
            "flag": False,
            "dead": False,
            "frame": 0,
        })
        print(f"Loaded agent: {name}")

    if len(agents) < 2:
        print("Error: Need at least 2 agents.")
        return

    n_agents = len(agents)

    # layout: agents in a grid
    cols = min(n_agents, 3)
    rows = (n_agents + cols - 1) // cols
    scale = args.scale
    tile_w = int(VIEWPORT_W * scale)
    tile_h = int(VIEWPORT_H * scale)
    scoreboard_h = 120
    window_w = cols * tile_w
    window_h = rows * tile_h + scoreboard_h

    pygame.init()
    pygame.display.set_caption("Competitive Mario -- Race")
    screen = pygame.display.set_mode((window_w, window_h))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16, bold=True)
    title_font = pygame.font.SysFont("monospace", 22, bold=True)

    running = True
    race_finished = False
    winner = None
    start_time = time.time()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    # restart race
                    for agent in agents:
                        agent["obs"] = agent["vec_env"].reset()
                        agent["done"] = False
                        agent["reward"] = 0.0
                        agent["x_pos"] = 0
                        agent["flag"] = False
                        agent["dead"] = False
                        agent["frame"] = 0
                    race_finished = False
                    winner = None
                    start_time = time.time()

        # step each agent
        all_done = True
        for agent in agents:
            if not agent["done"]:
                all_done = False
                action, _ = agent["model"].predict(agent["obs"], deterministic=True)
                agent["obs"], reward, dones, infos = agent["vec_env"].step(action)
                agent["reward"] += float(reward[0])
                agent["frame"] += 1

                if dones[0]:
                    agent["done"] = True
                    info = infos[0]
                    agent["x_pos"] = info.get("x_pos", 0)
                    agent["flag"] = info.get("flag_get", False)
                    agent["dead"] = info.get("is_dead", False)
                    agent["info"] = info
                    if agent["flag"] and winner is None:
                        winner = agent["name"]
                else:
                    info = infos[0]
                    agent["x_pos"] = info.get("x_pos", 0)
                    agent["info"] = info

        if all_done and not race_finished:
            race_finished = True
            if winner is None:
                # winner is the one who went furthest
                best_agent = max(agents, key=lambda a: a["x_pos"])
                winner = best_agent["name"]

        # render
        screen.fill((20, 20, 40))

        for idx, agent in enumerate(agents):
            row = idx // cols
            col = idx % cols
            x_off = col * tile_w
            y_off = row * tile_h

            # get frame from env
            frame = agent["env"]._render_frame()
            if frame is not None:
                # convert to pygame surface
                surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                if scale != 1.0:
                    surface = pygame.transform.scale(surface, (tile_w, tile_h))
                screen.blit(surface, (x_off, y_off))

            # agent name label
            color = (0, 255, 0) if agent["flag"] else (255, 0, 0) if agent["dead"] else (255, 255, 255)
            label = font.render(agent["name"], True, color)
            screen.blit(label, (x_off + 5, y_off + 5))

            # x position
            pos_label = font.render(f"X: {agent['x_pos']}", True, (255, 220, 100))
            screen.blit(pos_label, (x_off + 5, y_off + 22))

            # status
            if agent["flag"]:
                status = font.render("FLAG!", True, (0, 255, 0))
            elif agent["dead"]:
                status = font.render("DEAD", True, (255, 0, 0))
            elif agent["done"]:
                status = font.render("TIME UP", True, (200, 200, 0))
            else:
                status = font.render("RUNNING", True, (100, 255, 100))
            screen.blit(status, (x_off + 5, y_off + 39))

        # scoreboard
        sb_y = rows * tile_h
        pygame.draw.rect(screen, (30, 30, 60), (0, sb_y, window_w, scoreboard_h))
        pygame.draw.line(screen, (100, 100, 200), (0, sb_y), (window_w, sb_y), 2)

        elapsed = time.time() - start_time
        title = title_font.render(
            f"COMPETITIVE MARIO RACE  |  Seed: {args.level_seed}  |  {elapsed:.1f}s",
            True, (255, 215, 0),
        )
        screen.blit(title, (10, sb_y + 5))

        # per-agent stats in scoreboard
        sorted_agents = sorted(agents, key=lambda a: a["x_pos"], reverse=True)
        for i, agent in enumerate(sorted_agents):
            rank_str = f"#{i+1}"
            flag_str = "FLAG" if agent["flag"] else "DEAD" if agent["dead"] else f"X:{agent['x_pos']}"
            entry = font.render(
                f"{rank_str} {agent['name']:20s} {flag_str:>10s}  R:{agent['reward']:.1f}",
                True, (200, 200, 200),
            )
            screen.blit(entry, (10, sb_y + 30 + i * 18))

        if race_finished and winner:
            winner_text = title_font.render(f"WINNER: {winner}!", True, (0, 255, 0))
            text_rect = winner_text.get_rect(center=(window_w // 2, sb_y + scoreboard_h - 15))
            screen.blit(winner_text, text_rect)

        pygame.display.flip()
        clock.tick(args.fps)

    # cleanup
    for agent in agents:
        agent["vec_env"].close()
    pygame.quit()


if __name__ == "__main__":
    main()
