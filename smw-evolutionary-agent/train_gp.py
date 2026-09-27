"""
train_gp.py
===========
Autonomous training manager for the Genetic Programming agent (MarIO_GP).

Runs BizHawk in turbo mode, monitors the evolution log produced by the GP
generations, persists the checkpoints and extracts the final explicit policy.
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

BIZHAWK_DIR = Path(r"C:\BizHawk")
REPO_DIR = Path(__file__).resolve().parent
EMUHAWK_EXE = BIZHAWK_DIR / "EmuHawk.exe"
ROM_FILE = BIZHAWK_DIR / "Super Mario World (USA).sfc"
LUA_SCRIPT = BIZHAWK_DIR / "MarIO_GP.lua"
LOG_FILE = BIZHAWK_DIR / "gp_training_results.txt"


def run_training(duration_seconds: int = 180):
    print("=" * 65)
    print("  STARTING A REAL TRAINING SESSION FOR THE GP AGENT")
    print("=" * 65)
    print(f"Planned duration: ~{duration_seconds} seconds (turbo mode)")
    print(f"Emulator: {EMUHAWK_EXE}")
    print(f"Script:   {LUA_SCRIPT}")
    print(f"ROM:      {ROM_FILE.name}\n")

    # Clear the previous log if it exists
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    cmd = [str(EMUHAWK_EXE), f"--lua={LUA_SCRIPT}", str(ROM_FILE)]
    proc = subprocess.Popen(cmd, cwd=str(BIZHAWK_DIR))

    start_time = time.time()
    last_printed_lines = 0

    try:
        while time.time() - start_time < duration_seconds:
            time.sleep(3)
            elapsed = int(time.time() - start_time)
            if LOG_FILE.exists():
                try:
                    lines = LOG_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()
                    if len(lines) > last_printed_lines:
                        for l in lines[last_printed_lines:]:
                            if l.startswith("Gen ") or l.startswith("--- BEST SYMBOLIC POLICY"):
                                print(f"[{elapsed:3d}s] {l}")
                        last_printed_lines = len(lines)
                except Exception:
                    pass
    except KeyboardInterrupt:
        print("\nUser interrupt received.")
    finally:
        print(f"\nEnding the emulation session ({int(time.time() - start_time)}s elapsed)...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    # Copy checkpoints and logs into the repository
    if LOG_FILE.exists():
        shutil.copy2(LOG_FILE, REPO_DIR / "gp_training_results.txt")
        print("\n[OK] Results written to gp_training_results.txt")

    # Copy the generated .gppool files
    pool_files = list(BIZHAWK_DIR.glob("*.gppool"))
    for f in pool_files:
        shutil.copy2(f, REPO_DIR / f.name)
        print(f"[OK] Checkpoint preserved: {f.name}")


if __name__ == "__main__":
    dur = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    run_training(dur)
