"""Engine-ready model export helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ml_games_engine.brains import ObservationShellEnv
from ml_games_engine.control import utc_now_iso, write_json
from ml_games_engine.scenarios import get_scenario

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional at import time
    ort = None


class PolicyInferenceWrapper(torch.nn.Module):
    """PPO policy wrapper with VecNormalize baked into the forward path."""

    def __init__(self, model: PPO, vecnormalize: VecNormalize | None):
        super().__init__()
        self.policy = model.policy
        self.normalize_observation = vecnormalize is not None
        obs_shape = model.observation_space.shape
        action_low = np.asarray(model.action_space.low, dtype=np.float32)
        action_high = np.asarray(model.action_space.high, dtype=np.float32)

        if vecnormalize is not None:
            obs_mean = np.asarray(vecnormalize.obs_rms.mean, dtype=np.float32)
            obs_var = np.asarray(vecnormalize.obs_rms.var, dtype=np.float32)
            obs_clip = float(vecnormalize.clip_obs)
            obs_eps = float(vecnormalize.epsilon)
        else:
            obs_mean = np.zeros(obs_shape, dtype=np.float32)
            obs_var = np.ones(obs_shape, dtype=np.float32)
            obs_clip = 1_000_000.0
            obs_eps = 0.0

        self.register_buffer("obs_mean", torch.tensor(obs_mean, dtype=torch.float32))
        self.register_buffer("obs_var", torch.tensor(obs_var, dtype=torch.float32))
        self.register_buffer("obs_clip", torch.tensor(obs_clip, dtype=torch.float32))
        self.register_buffer("obs_eps", torch.tensor(obs_eps, dtype=torch.float32))
        self.register_buffer("action_low", torch.tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(action_high, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.normalize_observation:
            obs = torch.clamp(
                (obs - self.obs_mean) / torch.sqrt(self.obs_var + self.obs_eps),
                -self.obs_clip,
                self.obs_clip,
            )
        features = self.policy.extract_features(obs)
        latent_pi = self.policy.mlp_extractor.forward_actor(features)
        actions = self.policy.action_net(latent_pi)
        return torch.clamp(actions, self.action_low, self.action_high)


def _load_vecnormalize(model: PPO, vecnorm_path: str | None) -> VecNormalize | None:
    if not vecnorm_path or not Path(vecnorm_path).exists():
        return None

    shell_env = DummyVecEnv(
        [lambda: ObservationShellEnv(model.observation_space, model.action_space)]
    )
    vec = VecNormalize.load(vecnorm_path, shell_env)
    vec.training = False
    vec.norm_reward = False
    return vec


def _sample_observation(scenario_id: str, reward_cfg: dict, world_cfg: dict, seed: int | None, fallback_dim: int) -> np.ndarray:
    scenario = get_scenario(scenario_id)
    env = None
    try:
        env = scenario.make_env(
            render_mode="direct",
            reward_config=reward_cfg,
            world_config=world_cfg,
            seed=seed,
        )
        obs, _ = env.reset()
        return np.asarray(obs, dtype=np.float32).reshape(1, -1)
    except Exception:
        return np.zeros((1, fallback_dim), dtype=np.float32)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def _normalization_payload(vecnormalize: VecNormalize | None, observation_dim: int) -> dict:
    if vecnormalize is None:
        return {
            "type": "identity",
            "enabled": False,
            "mean": [0.0] * observation_dim,
            "var": [1.0] * observation_dim,
            "epsilon": 0.0,
            "clip_obs": None,
        }
    return {
        "type": "vecnormalize",
        "enabled": True,
        "mean": np.asarray(vecnormalize.obs_rms.mean, dtype=np.float32).tolist(),
        "var": np.asarray(vecnormalize.obs_rms.var, dtype=np.float32).tolist(),
        "epsilon": float(vecnormalize.epsilon),
        "clip_obs": float(vecnormalize.clip_obs),
    }


def _export_readme(manifest: dict) -> str:
    scenario = manifest["scenario"]
    contract = manifest["contract"]
    return f"""# Engine Export Bundle

This bundle is intended to run the trained policy both outside a game engine and inside Unity, Godot, or Unreal.

## Files

- `policy.onnx`: engine-ready deterministic policy with observation normalization baked in
- `manifest.json`: I/O contract, action bounds, scenario metadata, and source paths
- `normalization.json`: exported normalization stats for auditing and custom runtimes
- `sample_observation.json`: sample raw observation vector accepted by the ONNX model
- `validation.json`: comparison between PyTorch and ONNXRuntime outputs
- `python/run_policy.py`: standalone ONNXRuntime example
- `unity/MLGamesPolicyRunner.cs`: Unity-side wrapper contract
- `godot/ml_games_policy_runner.gd`: Godot-side wrapper contract
- `unreal/MLGamesPolicyRunner.h/.cpp`: Unreal-side wrapper contract

## Contract

- Scenario: `{scenario["label"]}` (`{scenario["scenario_id"]}`)
- Observation size: `{contract["observation_dim"]}`
- Action size: `{contract["action_dim"]}`
- Input tensor name: `{contract["input_name"]}`
- Output tensor name: `{contract["output_name"]}`
- Input format: raw observation vector as `float32`
- Output format: deterministic clipped action vector as `float32`

## Notes

- The exported ONNX model already includes observation normalization from VecNormalize.
- To reproduce the same behavior in Unity, Godot, or Unreal, your engine must build the same observation vector described in `manifest.json`.
- If you want to keep using the native Python stack, the original SB3 checkpoint remains valid and the ONNX bundle is optional.
"""


def _python_runtime_template(manifest: dict) -> str:
    return f'''"""Standalone ONNXRuntime inference example for this export bundle."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
SESSION = ort.InferenceSession(str(ROOT / "policy.onnx"), providers=["CPUExecutionProvider"])
INPUT_NAME = MANIFEST["contract"]["input_name"]


def predict(observation: np.ndarray) -> np.ndarray:
    observation = np.asarray(observation, dtype=np.float32).reshape(1, -1)
    output = SESSION.run(None, {{INPUT_NAME: observation}})[0]
    return output[0]


if __name__ == "__main__":
    sample = np.asarray(json.loads((ROOT / "sample_observation.json").read_text(encoding="utf-8")), dtype=np.float32)
    action = predict(sample)
    print("sample observation size:", sample.shape[0])
    print("predicted action:", action.tolist())
'''


def _unity_template(manifest: dict) -> str:
    obs_dim = manifest["contract"]["observation_dim"]
    action_dim = manifest["contract"]["action_dim"]
    return f"""using System;
using System.Collections.Generic;

namespace MLGames.EngineExport
{{
    public interface IMLGamesOnnxBackend
    {{
        float[] Run(float[] observation);
    }}

    public sealed class MLGamesPolicyRunner
    {{
        public const int ObservationSize = {obs_dim};
        public const int ActionSize = {action_dim};

        private readonly IMLGamesOnnxBackend _backend;

        public MLGamesPolicyRunner(IMLGamesOnnxBackend backend)
        {{
            _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        }}

        public float[] Predict(IReadOnlyList<float> observation)
        {{
            if (observation == null || observation.Count != ObservationSize)
            {{
                throw new ArgumentException($"Expected {{ObservationSize}} observation values.");
            }}

            float[] input = new float[ObservationSize];
            for (int i = 0; i < ObservationSize; i++)
            {{
                input[i] = observation[i];
            }}

            float[] action = _backend.Run(input);
            if (action == null || action.Length != ActionSize)
            {{
                throw new InvalidOperationException($"Backend must return {{ActionSize}} action values.");
            }}

            return action;
        }}
    }}
}}
"""


def _godot_template(manifest: dict) -> str:
    obs_dim = manifest["contract"]["observation_dim"]
    action_dim = manifest["contract"]["action_dim"]
    return f"""class_name MLGamesPolicyRunner
extends RefCounted

const OBSERVATION_SIZE := {obs_dim}
const ACTION_SIZE := {action_dim}

var backend: Callable

func _init(inference_backend: Callable) -> void:
\tbackend = inference_backend

func predict(observation: PackedFloat32Array) -> PackedFloat32Array:
\tassert(observation.size() == OBSERVATION_SIZE)
\tvar action = backend.call(observation)
\tassert(action.size() == ACTION_SIZE)
\treturn action
"""


def _unreal_header_template(manifest: dict) -> str:
    obs_dim = manifest["contract"]["observation_dim"]
    action_dim = manifest["contract"]["action_dim"]
    return f"""#pragma once

#include \"CoreMinimal.h\"
#include \"UObject/Object.h\"
#include \"MLGamesPolicyRunner.generated.h\"

DECLARE_DELEGATE_RetVal_OneParam(TArray<float>, FMLGamesBackendDelegate, const TArray<float>&);

UCLASS(BlueprintType)
class UMLGamesPolicyRunner : public UObject
{{
\tGENERATED_BODY()

public:
\tstatic constexpr int32 ObservationSize = {obs_dim};
\tstatic constexpr int32 ActionSize = {action_dim};

\tvoid SetBackend(FMLGamesBackendDelegate InBackend);
\tTArray<float> Predict(const TArray<float>& Observation) const;

private:
\tFMLGamesBackendDelegate Backend;
}};
"""


def _unreal_cpp_template() -> str:
    return """#include \"MLGamesPolicyRunner.h\"

void UMLGamesPolicyRunner::SetBackend(FMLGamesBackendDelegate InBackend)
{
\tBackend = InBackend;
}

TArray<float> UMLGamesPolicyRunner::Predict(const TArray<float>& Observation) const
{
\tcheck(Observation.Num() == ObservationSize);
\tcheck(Backend.IsBound());
\tTArray<float> Action = Backend.Execute(Observation);
\tcheck(Action.Num() == ActionSize);
\treturn Action;
}
"""


def _validation_payload(wrapper: PolicyInferenceWrapper, onnx_path: Path, sample_observation: np.ndarray) -> dict:
    if ort is None:
        raise RuntimeError("onnxruntime is not installed. Install it to validate engine exports.")

    wrapper.eval()
    torch_output = wrapper(torch.tensor(sample_observation, dtype=torch.float32)).detach().cpu().numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_output = session.run(None, {"obs": sample_observation.astype(np.float32)})[0]
    max_abs_error = float(np.max(np.abs(torch_output - onnx_output)))
    return {
        "validated_at": utc_now_iso(),
        "backend": "onnxruntime",
        "max_abs_error": max_abs_error,
        "torch_output": torch_output[0].tolist(),
        "onnx_output": onnx_output[0].tolist(),
        "within_tolerance": bool(max_abs_error < 1e-4),
    }


def export_engine_package(
    *,
    scenario_id: str,
    model_path: str,
    output_dir: str | Path,
    vecnorm_path: str | None = None,
    run_id: str | None = None,
    reward_cfg: dict | None = None,
    world_cfg: dict | None = None,
    seed: int | None = None,
) -> dict:
    """Export an engine-ready bundle for a PPO checkpoint."""
    if ort is None:
        raise RuntimeError("onnxruntime is not installed. Install it to export validated engine bundles.")

    scenario = get_scenario(scenario_id)
    reward_cfg = dict(scenario.default_reward if reward_cfg is None else reward_cfg)
    world_cfg = dict(scenario.default_world if world_cfg is None else world_cfg)

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "python").mkdir(parents=True, exist_ok=True)
    (output_dir / "unity").mkdir(parents=True, exist_ok=True)
    (output_dir / "godot").mkdir(parents=True, exist_ok=True)
    (output_dir / "unreal").mkdir(parents=True, exist_ok=True)

    model = PPO.load(model_path, device="cpu")
    vecnormalize = _load_vecnormalize(model, vecnorm_path)
    wrapper = PolicyInferenceWrapper(model, vecnormalize).eval()

    observation_dim = int(np.prod(model.observation_space.shape))
    action_dim = int(np.prod(model.action_space.shape))
    sample_observation = _sample_observation(
        scenario_id=scenario_id,
        reward_cfg=reward_cfg,
        world_cfg=world_cfg,
        seed=seed,
        fallback_dim=observation_dim,
    )

    onnx_path = output_dir / "policy.onnx"
    dummy_input = torch.tensor(sample_observation, dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        dummy_input,
        str(onnx_path),
        input_names=["obs"],
        output_names=["action"],
        opset_version=17,
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        dynamo=False,
    )

    normalization = _normalization_payload(vecnormalize, observation_dim)
    write_json(output_dir / "normalization.json", normalization)
    write_json(output_dir / "sample_observation.json", sample_observation[0].astype(np.float32).tolist())

    manifest = {
        "exported_at": utc_now_iso(),
        "run_id": run_id,
        "source": {
            "model_path": str(Path(model_path).resolve()),
            "vecnorm_path": str(Path(vecnorm_path).resolve()) if vecnorm_path else None,
        },
        "scenario": {
            "scenario_id": scenario.scenario_id,
            "label": scenario.label,
            "dimension": scenario.dimension,
            "description": scenario.description,
            "viewport": scenario.viewport,
            "info": scenario.info,
        },
        "contract": {
            "input_name": "obs",
            "output_name": "action",
            "observation_dim": observation_dim,
            "action_dim": action_dim,
            "action_low": np.asarray(model.action_space.low, dtype=np.float32).tolist(),
            "action_high": np.asarray(model.action_space.high, dtype=np.float32).tolist(),
            "deterministic": True,
            "normalization_baked_into_onnx": normalization["enabled"],
        },
        "compatibility": {
            "standalone": ["python", "onnxruntime", "original_sb3_checkpoint"],
            "game_engines": ["unity", "godot", "unreal"],
            "recommended_exchange_format": "onnx",
        },
    }
    write_json(output_dir / "manifest.json", manifest)

    validation = _validation_payload(wrapper, onnx_path, sample_observation)
    write_json(output_dir / "validation.json", validation)

    (output_dir / "README.md").write_text(_export_readme(manifest), encoding="utf-8")
    (output_dir / "python" / "run_policy.py").write_text(_python_runtime_template(manifest), encoding="utf-8")
    (output_dir / "unity" / "MLGamesPolicyRunner.cs").write_text(_unity_template(manifest), encoding="utf-8")
    (output_dir / "godot" / "ml_games_policy_runner.gd").write_text(_godot_template(manifest), encoding="utf-8")
    (output_dir / "unreal" / "MLGamesPolicyRunner.h").write_text(_unreal_header_template(manifest), encoding="utf-8")
    (output_dir / "unreal" / "MLGamesPolicyRunner.cpp").write_text(_unreal_cpp_template(), encoding="utf-8")

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an SB3 PPO checkpoint as an engine-ready ONNX bundle.")
    parser.add_argument("--scenario", required=True, help="Scenario id such as arena2d, chase, or parkour.")
    parser.add_argument("--model", required=True, help="Path to the .zip PPO checkpoint.")
    parser.add_argument("--out", required=True, help="Directory for the exported bundle.")
    parser.add_argument("--vecnorm", default=None, help="Optional VecNormalize pickle path.")
    parser.add_argument("--run-id", default=None, help="Optional originating run id.")
    parser.add_argument("--seed", type=int, default=None, help="Seed used to generate the sample observation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_engine_package(
        scenario_id=args.scenario,
        model_path=args.model,
        output_dir=args.out,
        vecnorm_path=args.vecnorm,
        run_id=args.run_id,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
