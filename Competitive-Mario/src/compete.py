"""
Competition engine for Competitive Mario.

Loads multiple trained agents and runs them on the same level,
collecting performance metrics and producing a ranking.

Usage:
    python src/compete.py --agents mario_speedster mario_careful mario_balanced
    python src/compete.py --models-dir models --level-seed 42
"""

import argparse
import glob
import os
import sys
import json
import time

import mlflow
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.monitor import Monitor

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from src.env.mario_env import MarioCompetitiveEnv

load_dotenv()


def find_agent_model(models_dir, agent_name):
    """Find the final model for an agent in the models directory."""
    agent_dir = os.path.join(models_dir, agent_name)
    if not os.path.isdir(agent_dir):
        return None
    # look for final model first
    final = os.path.join(agent_dir, f"ppo_{agent_name}_final.zip")
    if os.path.exists(final):
        return final
    # fallback: latest checkpoint
    zips = sorted(glob.glob(os.path.join(agent_dir, "*.zip")))
    if zips:
        return zips[-1]
    return None


def run_agent(model_path, level_seed, difficulty, level_width, n_episodes=5, frame_stack=4):
    """
    Run a trained agent and collect performance metrics.

    Returns
    -------
    dict
        Aggregated metrics for this agent.
    """
    env = MarioCompetitiveEnv(
        level_seed=level_seed,
        difficulty=difficulty,
        level_width=level_width,
        render_mode=None,
    )
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecFrameStack(vec_env, n_stack=frame_stack)
    vec_env = VecTransposeImage(vec_env)

    model = PPO.load(model_path, device="auto")

    results = {
        "x_positions": [],
        "flag_gets": [],
        "scores": [],
        "coins": [],
        "times_left": [],
        "episode_lengths": [],
        "episode_rewards": [],
    }

    for ep in range(n_episodes):
        obs = vec_env.reset()
        ep_reward = 0.0
        ep_steps = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = vec_env.step(action)
            ep_reward += float(reward[0])
            ep_steps += 1
            done = dones[0]

            if done:
                info = infos[0]
                results["x_positions"].append(info.get("x_pos", 0))
                results["flag_gets"].append(info.get("flag_get", False))
                results["scores"].append(info.get("score", 0))
                results["coins"].append(info.get("coins", 0))
                results["times_left"].append(info.get("time", 0))
                results["episode_lengths"].append(ep_steps)
                results["episode_rewards"].append(ep_reward)

    vec_env.close()

    # aggregate
    return {
        "avg_x_pos": float(np.mean(results["x_positions"])) if results["x_positions"] else 0,
        "max_x_pos": int(np.max(results["x_positions"])) if results["x_positions"] else 0,
        "flag_rate": float(np.mean(results["flag_gets"])) if results["flag_gets"] else 0,
        "avg_score": float(np.mean(results["scores"])) if results["scores"] else 0,
        "avg_coins": float(np.mean(results["coins"])) if results["coins"] else 0,
        "avg_time_left": float(np.mean(results["times_left"])) if results["times_left"] else 0,
        "avg_reward": float(np.mean(results["episode_rewards"])) if results["episode_rewards"] else 0,
        "avg_length": float(np.mean(results["episode_lengths"])) if results["episode_lengths"] else 0,
        "n_episodes": n_episodes,
        "raw": results,
    }


def compute_ranking(agent_results):
    """
    Compute a composite ranking score for each agent.

    Scoring:
        - 50% flag completion rate
        - 30% average x position (normalized)
        - 10% average score (normalized)
        - 10% average time left (normalized -- faster = better)
    """
    if not agent_results:
        return []

    # get max values for normalization
    max_x = max(r["avg_x_pos"] for r in agent_results.values()) or 1
    max_score = max(r["avg_score"] for r in agent_results.values()) or 1
    max_time = max(r["avg_time_left"] for r in agent_results.values()) or 1

    ranking = []
    for name, metrics in agent_results.items():
        composite = (
            0.50 * metrics["flag_rate"]
            + 0.30 * (metrics["avg_x_pos"] / max_x)
            + 0.10 * (metrics["avg_score"] / max_score)
            + 0.10 * (metrics["avg_time_left"] / max_time)
        )
        ranking.append({
            "agent": name,
            "composite_score": round(composite, 4),
            "flag_rate": metrics["flag_rate"],
            "avg_x_pos": metrics["avg_x_pos"],
            "max_x_pos": metrics["max_x_pos"],
            "avg_score": metrics["avg_score"],
            "avg_reward": metrics["avg_reward"],
            "avg_coins": metrics["avg_coins"],
            "avg_time_left": metrics["avg_time_left"],
        })

    ranking.sort(key=lambda r: r["composite_score"], reverse=True)
    for i, entry in enumerate(ranking):
        entry["rank"] = i + 1

    return ranking


def generate_charts(ranking, output_dir):
    """Generate comparison charts and save as PNG files."""
    os.makedirs(output_dir, exist_ok=True)

    agents = [r["agent"] for r in ranking]
    n = len(agents)

    # 1. Bar chart: composite score
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, n))
    bars = ax.bar(agents, [r["composite_score"] for r in ranking], color=colors)
    ax.set_ylabel("Composite Score")
    ax.set_title("Competition Results -- Composite Score")
    ax.set_ylim(0, 1.1)
    for bar, r in zip(bars, ranking):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"#{r['rank']}",
            ha="center", fontweight="bold",
        )
    plt.tight_layout()
    path = os.path.join(output_dir, "ranking_composite.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)

    # 2. Bar chart: x position
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(agents, [r["avg_x_pos"] for r in ranking], color=colors)
    ax.set_ylabel("Average X Position (pixels)")
    ax.set_title("Competition Results -- Distance Traveled")
    plt.tight_layout()
    path = os.path.join(output_dir, "ranking_xpos.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)

    # 3. Radar chart
    categories = ["Flag Rate", "X Position", "Score", "Coins", "Time Left"]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    max_x = max(r["avg_x_pos"] for r in ranking) or 1
    max_s = max(r["avg_score"] for r in ranking) or 1
    max_c = max(r["avg_coins"] for r in ranking) or 1
    max_t = max(r["avg_time_left"] for r in ranking) or 1

    for i, r in enumerate(ranking):
        values = [
            r["flag_rate"],
            r["avg_x_pos"] / max_x,
            r["avg_score"] / max_s,
            r["avg_coins"] / max_c,
            r["avg_time_left"] / max_t,
        ]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=r["agent"], color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_ylim(0, 1.1)
    ax.set_title("Agent Performance Radar", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    path = os.path.join(output_dir, "ranking_radar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Charts saved to: {output_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Run a competition between Mario agents")

    p.add_argument("--agents", nargs="+", default=None,
                   help="Names of agents to compete (auto-discover from models-dir if not set)")
    p.add_argument("--models-dir", type=str, default="models",
                   help="Directory containing agent model subdirectories")
    p.add_argument("--level-seed", type=int, default=42,
                   help="Seed for level generation (same for all agents)")
    p.add_argument("--difficulty", type=float, default=0.5)
    p.add_argument("--level-width", type=int, default=200)
    p.add_argument("--n-episodes", type=int, default=5,
                   help="Number of episodes per agent")
    p.add_argument("--output-dir", type=str, default="results",
                   help="Directory to save results and charts")
    p.add_argument("--tracking-uri", type=str,
                   default=os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))
    p.add_argument("--experiment-name", type=str, default="competitive_mario_race")

    return p.parse_args()


def main():
    args = parse_args()

    models_dir = os.path.join(_PROJECT_DIR, args.models_dir)
    output_dir = os.path.join(_PROJECT_DIR, args.output_dir)

    # discover agents
    if args.agents:
        agent_names = args.agents
    else:
        # auto-discover from models directory
        if not os.path.isdir(models_dir):
            print(f"Error: Models directory not found: {models_dir}")
            return
        agent_names = [
            d for d in os.listdir(models_dir)
            if os.path.isdir(os.path.join(models_dir, d))
        ]

    if not agent_names:
        print("Error: No agents found. Train some agents first with src/train.py")
        return

    print(f"Competition: {len(agent_names)} agents")
    print(f"Level seed: {args.level_seed}, Difficulty: {args.difficulty}")
    print(f"Episodes per agent: {args.n_episodes}")
    print(f"Agents: {', '.join(agent_names)}")
    print("-" * 60)

    # find models
    agent_models = {}
    for name in agent_names:
        model_path = find_agent_model(models_dir, name)
        if model_path is None:
            print(f"  [Warning] No model found for agent '{name}', skipping...")
            continue
        agent_models[name] = model_path
        print(f"  {name}: {model_path}")

    if len(agent_models) < 2:
        print("\nError: Need at least 2 agents with trained models to compete.")
        return

    # setup MLflow
    tracking_uri = args.tracking_uri
    if "://" not in tracking_uri and not os.path.isabs(tracking_uri):
        tracking_uri = os.path.abspath(os.path.join(_PROJECT_DIR, tracking_uri))
    if os.path.isabs(tracking_uri) and "://" not in tracking_uri:
        tracking_uri = "file:///" + tracking_uri.replace("\\", "/")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=f"race_{args.level_seed}_{int(time.time())}"):
        mlflow.log_params({
            "level_seed": args.level_seed,
            "difficulty": args.difficulty,
            "n_episodes": args.n_episodes,
            "n_agents": len(agent_models),
            "agents": ",".join(agent_models.keys()),
        })

        # run each agent
        agent_results = {}
        for name, model_path in agent_models.items():
            print(f"\nRunning agent '{name}'...")
            results = run_agent(
                model_path=model_path,
                level_seed=args.level_seed,
                difficulty=args.difficulty,
                level_width=args.level_width,
                n_episodes=args.n_episodes,
            )
            agent_results[name] = results
            print(f"  Avg X: {results['avg_x_pos']:.0f}, "
                  f"Max X: {results['max_x_pos']}, "
                  f"Flags: {results['flag_rate']:.0%}, "
                  f"Score: {results['avg_score']:.0f}")

        # compute ranking
        ranking = compute_ranking(agent_results)

        # print results
        print("\n" + "=" * 60)
        print("COMPETITION RESULTS")
        print("=" * 60)
        for r in ranking:
            flag_str = f"{r['flag_rate']:.0%}"
            print(
                f"  #{r['rank']} {r['agent']:20s} "
                f"Score: {r['composite_score']:.4f} "
                f"Flags: {flag_str:>5s} "
                f"X: {r['avg_x_pos']:.0f} "
                f"Game Score: {r['avg_score']:.0f}"
            )
        print("=" * 60)

        winner = ranking[0]["agent"] if ranking else "N/A"
        print(f"\nWinner: {winner}")

        # save results
        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(output_dir, "ranking.json")
        with open(results_path, "w") as f:
            json.dump(ranking, f, indent=2)
        mlflow.log_artifact(results_path)

        # log metrics per agent
        for r in ranking:
            mlflow.log_metrics({
                f"{r['agent']}_rank": r["rank"],
                f"{r['agent']}_composite": r["composite_score"],
                f"{r['agent']}_flag_rate": r["flag_rate"],
                f"{r['agent']}_avg_x_pos": r["avg_x_pos"],
            })

        # generate charts
        generate_charts(ranking, output_dir)
        for chart_file in glob.glob(os.path.join(output_dir, "*.png")):
            mlflow.log_artifact(chart_file, artifact_path="charts")

        print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
