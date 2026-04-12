from __future__ import annotations

import argparse
import json
import socket
import sys

import numpy as np

try:
    import onnxruntime as ort
except ImportError as exc:
    print(f"onnxruntime not installed: {exc}")
    sys.exit(2)

ACTION_SIZE = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UDP ONNX policy inference server")
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--host", default="127.0.0.1", type=str)
    parser.add_argument("--port", default=8765, type=int)
    return parser.parse_args()


def sanitize_action(raw_action: np.ndarray) -> list[float]:
    flat = np.asarray(raw_action, dtype=np.float32).reshape(-1)
    if flat.size < ACTION_SIZE:
        padded = np.zeros(ACTION_SIZE, dtype=np.float32)
        padded[: flat.size] = flat
        flat = padded
    elif flat.size > ACTION_SIZE:
        flat = flat[:ACTION_SIZE]

    return [float(v) for v in flat]


def run_server(model_path: str, host: str, port: int) -> None:
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))

    print(f"ONNX UDP server running at {host}:{port}")
    print(f"Loaded model: {model_path}")

    while True:
        packet, addr = sock.recvfrom(65535)
        if not packet:
            continue

        try:
            payload = json.loads(packet.decode("utf-8"))
        except json.JSONDecodeError:
            continue

        if isinstance(payload, dict) and payload.get("ping"):
            sock.sendto(json.dumps({"ok": True}).encode("utf-8"), addr)
            continue

        if not isinstance(payload, dict) or "obs" not in payload:
            continue

        obs = np.asarray(payload.get("obs", []), dtype=np.float32).reshape(1, -1)

        try:
            output = session.run([output_name], {input_name: obs})[0]
            action = sanitize_action(output)
            response = {
                "action": action,
            }
            sock.sendto(json.dumps(response).encode("utf-8"), addr)
        except Exception as inference_error:
            response = {
                "action": [0.0] * ACTION_SIZE,
                "error": str(inference_error),
            }
            sock.sendto(json.dumps(response).encode("utf-8"), addr)


def main() -> None:
    args = parse_args()
    run_server(args.model, args.host, args.port)


if __name__ == "__main__":
    main()
