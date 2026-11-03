"""
train_neat.py
=============
Cross-platform paired benchmark manager for the NEAT agent (MarIO.lua).

Runs BizHawk in turbo mode (speedmode 600) for a specified duration (default 180s),
monitors the evolution of species and genomes, persists the checkpoints, and extracts
the fitness metrics for direct comparison.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from env_config import get_bizhawk_environment

REPO_DIR = Path(__file__).resolve().parent


def run_neat_benchmark(duration_seconds: int = 180, custom_dir: str = None, custom_bin: str = None):
    env = get_bizhawk_environment(custom_dir=custom_dir, custom_bin=custom_bin)
    if not env.is_available:
        print(f"[ERROR] {env.message}")
        sys.exit(1)

    lua_script = env.bizhawk_dir / "MarIO.lua"
    log_file = env.bizhawk_dir / "neat_training_results.txt"

    # Ensure lua script is synced
    src_lua = REPO_DIR / "MarIO.lua"
    if src_lua.exists():
        shutil.copy2(src_lua, lua_script)

    print("=" * 65)
    print("  STARTING PAIRED NEAT AGENT BENCHMARK SESSION")
    print("=" * 65)
    print(f"Planned duration: {duration_seconds} seconds (speedmode 600)")
    print(f"Platform:         {env.platform}")
    print(f"Command:          {' '.join(env.exec_cmd)}")
    print(f"Script:           {lua_script.name}")
    print(f"ROM:              {env.rom_path.name}\n")

    if log_file.exists():
        log_file.unlink()

    cmd = list(env.exec_cmd) + [f"--lua={lua_script.name}", str(env.rom_path)]
    proc = subprocess.Popen(cmd, cwd=str(env.bizhawk_dir))

    start_time = time.time()
    last_count = 0

    try:
        while time.time() - start_time < duration_seconds:
            time.sleep(5)
            elapsed = int(time.time() - start_time)
            if log_file.exists():
                try:
                    lines = log_file.read_text(encoding="utf-8", errors="ignore").splitlines()
                    if len(lines) > last_count:
                        last_count = len(lines)
                        print(f"[{elapsed:3d}s] Evaluations: {len(lines)} | Last: {lines[-1]}")
                except Exception:
                    pass
    except KeyboardInterrupt:
        print("\nInterrupt received.")
    finally:
        print(f"\nEnding NEAT session ({int(time.time() - start_time)}s elapsed)...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    # Sync back the log file to the repository if generated
    repo_log = REPO_DIR / "neat_training_results.txt"
    if log_file.exists():
        shutil.copy2(log_file, repo_log)
        print(f"Results synced to {repo_log.name}")


def main():
    parser = argparse.ArgumentParser(description="Cross-platform benchmark runner for NEAT MarIO")
    parser.add_argument("duration", type=int, nargs="?", default=180, help="Benchmark duration in seconds (default: 180)")
    parser.add_argument("--bizhawk-dir", type=str, default=None, help="Custom BizHawk directory")
    parser.add_argument("--bizhawk-bin", type=str, default=None, help="Custom BizHawk binary")
    args = parser.parse_args()

    run_neat_benchmark(args.duration, custom_dir=args.bizhawk_dir, custom_bin=args.bizhawk_bin)


if __name__ == "__main__":
    main()
