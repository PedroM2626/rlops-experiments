# Engine Export Bundle

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

- Scenario: `Arena 2D` (`arena2d`)
- Observation size: `10`
- Action size: `2`
- Input tensor name: `obs`
- Output tensor name: `action`
- Input format: raw observation vector as `float32`
- Output format: deterministic clipped action vector as `float32`

## Notes

- The exported ONNX model already includes observation normalization from VecNormalize.
- To reproduce the same behavior in Unity, Godot, or Unreal, your engine must build the same observation vector described in `manifest.json`.
- If you want to keep using the native Python stack, the original SB3 checkpoint remains valid and the ONNX bundle is optional.
