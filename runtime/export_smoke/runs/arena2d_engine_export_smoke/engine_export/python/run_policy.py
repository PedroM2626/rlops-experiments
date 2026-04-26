"""Standalone ONNXRuntime inference example for this export bundle."""

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
    output = SESSION.run(None, {INPUT_NAME: observation})[0]
    return output[0]


if __name__ == "__main__":
    sample = np.asarray(json.loads((ROOT / "sample_observation.json").read_text(encoding="utf-8")), dtype=np.float32)
    action = predict(sample)
    print("sample observation size:", sample.shape[0])
    print("predicted action:", action.tolist())
