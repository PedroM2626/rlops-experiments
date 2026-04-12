from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from game_env import OBS_SIZE, SurvivalTrainingEnv
from policy import PolicyNetwork, flatten_parameters, infer_action, set_parameters_from_flat


def str_to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train humanoid survival policy and export to ONNX")
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--generations", type=int, default=60)
    parser.add_argument("--population", type=int, default=28)
    parser.add_argument("--episode-seconds", type=float, default=45.0)
    parser.add_argument("--sigma", type=float, default=0.08)
    parser.add_argument("--learning-rate", type=float, default=0.045)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--reuse", type=str, default="true")
    return parser.parse_args()


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)


def append_history(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def run_episode(env: SurvivalTrainingEnv, model: PolicyNetwork) -> tuple[float, dict]:
    obs = env.reset()
    done = False
    total_reward = 0.0
    info: dict = {}

    while not done:
        action = infer_action(model, obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward

    return total_reward, info


def evaluate_candidate(
    model: PolicyNetwork,
    parameter_vector: torch.Tensor,
    seeds: np.ndarray,
    episode_seconds: float,
) -> tuple[float, float]:
    set_parameters_from_flat(model, parameter_vector)

    rewards = []
    distances = []
    for seed in seeds:
        env = SurvivalTrainingEnv(episode_seconds=episode_seconds, seed=int(seed))
        reward, info = run_episode(env, model)
        rewards.append(reward)
        distances.append(float(info.get("distance", 0.0)))

    return float(np.mean(rewards)), float(np.mean(distances))


def main(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = model_dir / "agent_policy.pt"
    onnx_path = model_dir / "agent_policy.onnx"
    status_path = model_dir / "training_status.json"
    history_path = model_dir / "training_history.jsonl"

    model = PolicyNetwork(obs_size=OBS_SIZE, hidden_size=args.hidden_size)
    model.eval()

    reuse_checkpoint = str_to_bool(args.reuse)
    reused_checkpoint = False

    best_reward = -1e9
    start_generation = 0

    if reuse_checkpoint and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
            best_reward = float(checkpoint.get("best_reward", best_reward))
            start_generation = int(checkpoint.get("generation", -1)) + 1
            reused_checkpoint = True

    params = flatten_parameters(model)
    best_params = params.clone()

    if best_reward <= -1e8:
        eval_seed = np.array([args.seed], dtype=np.int64)
        best_reward, _ = evaluate_candidate(model, best_params, eval_seed, args.episode_seconds)

    rng = np.random.default_rng(args.seed)
    started_at = time.time()

    running_status = {
        "state": "running",
        "generation": start_generation,
        "best_reward": best_reward,
        "current_reward": best_reward,
        "population": args.population,
        "episode_seconds": args.episode_seconds,
        "reused_checkpoint": reused_checkpoint,
        "onnx_path": str(onnx_path),
        "checkpoint_path": str(checkpoint_path),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json_atomic(status_path, running_status)

    for generation_offset in range(args.generations):
        generation = start_generation + generation_offset

        evaluation_seeds = rng.integers(
            low=0,
            high=1_000_000,
            size=max(args.eval_episodes, 1),
            dtype=np.int64,
        )

        noise = rng.standard_normal((args.population, params.numel())).astype(np.float32)
        rewards = np.zeros(args.population, dtype=np.float32)

        for i in range(args.population):
            candidate = params + torch.from_numpy(noise[i]) * args.sigma
            reward, _ = evaluate_candidate(model, candidate, evaluation_seeds, args.episode_seconds)
            rewards[i] = reward

            if reward > best_reward:
                best_reward = float(reward)
                best_params = candidate.detach().clone()

        std = float(rewards.std())
        if std < 1e-8:
            normalized = rewards - rewards.mean()
        else:
            normalized = (rewards - rewards.mean()) / (std + 1e-8)

        gradient = (noise.T @ normalized) / float(args.population)
        params = params + torch.from_numpy(gradient).float() * (args.learning_rate / max(args.sigma, 1e-8))

        current_reward, current_distance = evaluate_candidate(model, params, evaluation_seeds, args.episode_seconds)
        if current_reward > best_reward:
            best_reward = float(current_reward)
            best_params = params.detach().clone()

        generation_status = {
            "state": "running",
            "generation": generation,
            "generation_in_run": generation_offset + 1,
            "best_reward": best_reward,
            "current_reward": float(current_reward),
            "distance": float(current_distance),
            "population": args.population,
            "episode_seconds": args.episode_seconds,
            "reused_checkpoint": reused_checkpoint,
            "onnx_path": str(onnx_path),
            "checkpoint_path": str(checkpoint_path),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        write_json_atomic(status_path, generation_status)
        append_history(history_path, generation_status)

        set_parameters_from_flat(model, best_params)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "best_reward": best_reward,
                "generation": generation,
                "obs_size": OBS_SIZE,
                "action_size": 8,
                "hidden_size": args.hidden_size,
                "updated_at": time.time(),
            },
            checkpoint_path,
        )

        set_parameters_from_flat(model, params)

    set_parameters_from_flat(model, best_params)
    model.eval()

    dummy_input = torch.zeros(1, OBS_SIZE, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
    )

    finished_status = {
        "state": "finished",
        "generation": start_generation + max(args.generations - 1, 0),
        "best_reward": best_reward,
        "current_reward": best_reward,
        "population": args.population,
        "episode_seconds": args.episode_seconds,
        "reused_checkpoint": reused_checkpoint,
        "onnx_path": str(onnx_path),
        "checkpoint_path": str(checkpoint_path),
        "training_seconds": round(time.time() - started_at, 3),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json_atomic(status_path, finished_status)
    append_history(history_path, finished_status)


def guarded_main() -> None:
    args = parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        interrupted_status = {
            "state": "interrupted",
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        model_dir = Path(args.model_dir)
        write_json_atomic(model_dir / "training_status.json", interrupted_status)
    except Exception as exc:
        error_status = {
            "state": "error",
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        model_dir = Path(args.model_dir)
        write_json_atomic(model_dir / "training_status.json", error_status)
        raise


if __name__ == "__main__":
    guarded_main()
