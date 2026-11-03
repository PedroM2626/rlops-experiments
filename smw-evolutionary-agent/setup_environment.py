"""
setup_environment.py
====================
Cross-platform environment validation and preparation for smw-evolutionary-agent.

Checks:
  1. Cryptographic integrity of the Super Mario World (USA) ROM via SHA-1.
  2. Presence and configuration of the BizHawk 2.9+ executables and libraries.
  3. Integrity and format of the native DP1.state savestate.
  4. Presence of the evolutionary scripts: MarIO.lua (NEAT) and MarIO_GP.lua (Genetic Programming).
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

from env_config import BizHawkEnvironment, get_bizhawk_environment

REPO_DIR = Path(__file__).resolve().parent
ROM_REPO = REPO_DIR / "Super Mario World (USA).sfc"
STATE_REPO = REPO_DIR / "DP1.state"
ROM_SHA1_USA = "6b47bb75d16514b6a476aa0c73a683a2a4c18765"


def check_rom(env: BizHawkEnvironment) -> bool:
    print("\n[1/4] Checking ROM integrity...")
    if not ROM_REPO.exists():
        print(f"  [ERROR] ROM not found at: {ROM_REPO}")
        return False

    data = ROM_REPO.read_bytes()
    sha1 = hashlib.sha1(data).hexdigest()
    print(f"  File    : {ROM_REPO.name}")
    print(f"  Size    : {len(data) // 1024} KB")
    print(f"  SHA-1   : {sha1}")

    if sha1 == ROM_SHA1_USA:
        print("  [OK] Official SHA-1 verified (dump identical to reference benchmark).")
    else:
        print(f"  [WARNING] SHA-1 differs from expected ({ROM_SHA1_USA}).")

    if not env.rom_path.exists() or env.rom_path.stat().st_size != len(data):
        env.bizhawk_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROM_REPO, env.rom_path)
        print(f"  [OK] ROM synchronized with {env.bizhawk_dir}")
    return True


def check_emulator(env: BizHawkEnvironment) -> bool:
    print(f"\n[2/4] Checking BizHawk emulator on {env.platform}...")
    if not env.is_available:
        print(f"  [ERROR] {env.message}")
        return False

    print(f"  [OK] Executable command: {' '.join(env.exec_cmd)}")
    print(f"  [OK] BizHawk Directory:  {env.bizhawk_dir}")
    return True


def check_savestate(env: BizHawkEnvironment) -> bool:
    print("\n[3/4] Checking native savestate (DP1.state)...")
    valid_in_repo = STATE_REPO.exists() and STATE_REPO.stat().st_size > 10_000
    valid_in_biz = env.state_path.exists() and env.state_path.stat().st_size > 10_000

    if valid_in_repo:
        with open(STATE_REPO, "rb") as f:
            magic = f.read(4)
        if magic.startswith(b"PK") or magic.startswith(b"BZh"):
            print(f"  [OK] Native savestate verified in repo ({STATE_REPO.stat().st_size // 1024} KB).")
            if not valid_in_biz:
                env.bizhawk_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(STATE_REPO, env.state_path)
                print(f"  [OK] DP1.state synchronized with {env.bizhawk_dir}")
            return True

    print("  [INFO] DP1.state must be generated on first launch.")
    return True


def check_scripts(env: BizHawkEnvironment) -> bool:
    print("\n[4/4] Checking evolutionary scripts...")
    scripts = ["MarIO.lua", "MarIO_GP.lua"]
    all_ok = True
    for s in scripts:
        src = REPO_DIR / s
        dst = env.bizhawk_dir / s
        if src.exists():
            env.bizhawk_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"  [OK] {s} ({src.stat().st_size // 1024} KB) copied to {env.bizhawk_dir.name}.")
        else:
            print(f"  [ERROR] {s} not found in repository.")
            all_ok = False
    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Cross-platform environment validation and preparation for smw-evolutionary-agent"
    )
    parser.add_argument(
        "--bizhawk-dir",
        type=str,
        default=None,
        help="Custom path to BizHawk directory"
    )
    parser.add_argument(
        "--bizhawk-bin",
        type=str,
        default=None,
        help="Custom path to BizHawk binary"
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  SMW EVOLUTIONARY AGENT - ENVIRONMENT VERIFICATION")
    print("=" * 65)

    env = get_bizhawk_environment(custom_dir=args.bizhawk_dir, custom_bin=args.bizhawk_bin)

    ok_rom = check_rom(env)
    ok_emu = check_emulator(env)
    ok_state = check_savestate(env)
    ok_scripts = check_scripts(env)

    print("\n" + "-" * 65)
    if ok_rom and ok_emu and ok_scripts:
        print("  [SUCCESS] Environment is fully configured and ready to run.")
        print("  Execute: python launch_mario.py")
        print("-" * 65 + "\n")
        return 0
    else:
        print("  [WARNING] Some components require attention before launch.")
        print("-" * 65 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
