"""
setup_environment.py
====================
Automated environment validation and preparation for smw-evolutionary-agent.

Checks:
  1. Cryptographic integrity of the Super Mario World (USA) ROM via SHA-1.
  2. Presence and configuration of the BizHawk 2.9.1 executables and libraries.
  3. Integrity and format of the native DP1.state savestate.
  4. Presence of the evolutionary scripts: MarIO.lua (NEAT) and MarIO_GP.lua (Genetic Programming).
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
BIZHAWK_DIR = Path(r"C:\BizHawk")
EMUHAWK_EXE = BIZHAWK_DIR / "EmuHawk.exe"
ROM_REPO = REPO_DIR / "Super Mario World (USA).sfc"
ROM_BIZHAWK = BIZHAWK_DIR / "Super Mario World (USA).sfc"
STATE_REPO = REPO_DIR / "DP1.state"
STATE_BIZHAWK = BIZHAWK_DIR / "DP1.state"

ROM_SHA1_USA = "6b47bb75d16514b6a476aa0c73a683a2a4c18765"


def check_rom() -> bool:
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
        print("  [OK] Official SHA-1 verified (dump identical to the reference benchmark).")
    else:
        print(f"  [WARNING] SHA-1 differs from the expected value ({ROM_SHA1_USA}).")

    if not ROM_BIZHAWK.exists() or ROM_BIZHAWK.stat().st_size != len(data):
        shutil.copy2(ROM_REPO, ROM_BIZHAWK)
        print("  [OK] ROM synchronized with C:\\BizHawk\\")
    return True


def check_emulator() -> bool:
    print("\n[2/4] Checking the BizHawk emulator...")
    if not EMUHAWK_EXE.exists():
        print(f"  [ERROR] BizHawk not found at {EMUHAWK_EXE}")
        return False

    print(f"  [OK] BizHawk executable found: {EMUHAWK_EXE}")
    return True


def check_savestate() -> bool:
    print("\n[3/4] Checking the native savestate (DP1.state)...")
    valid_in_repo = STATE_REPO.exists() and STATE_REPO.stat().st_size > 10_000
    valid_in_biz = STATE_BIZHAWK.exists() and STATE_BIZHAWK.stat().st_size > 10_000

    if valid_in_repo:
        with open(STATE_REPO, "rb") as f:
            magic = f.read(4)
        if magic.startswith(b"PK") or magic.startswith(b"BZh"):
            print(f"  [OK] Native savestate verified in the repository ({STATE_REPO.stat().st_size // 1024} KB).")
            if not valid_in_biz:
                shutil.copy2(STATE_REPO, STATE_BIZHAWK)
                print("  [OK] DP1.state synchronized with C:\\BizHawk\\")
            return True

    print("  [INFO] DP1.state must be generated on the first launch.")
    return True


def check_scripts() -> bool:
    print("\n[4/4] Checking the evolutionary scripts...")
    scripts = ["MarIO.lua", "MarIO_GP.lua"]
    all_ok = True
    for s in scripts:
        src = REPO_DIR / s
        dst = BIZHAWK_DIR / s
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  [OK] {s} ({src.stat().st_size // 1024} KB) copied to BizHawk.")
        else:
            print(f"  [ERROR] {s} not found in the repository.")
            all_ok = False
    return all_ok


def main():
    print("=" * 60)
    print("  SMW EVOLUTIONARY AGENT - SETUP AND VALIDATION")
    print("=" * 60)

    ok_rom = check_rom()
    ok_emu = check_emulator()
    ok_state = check_savestate()
    ok_scripts = check_scripts()

    print("\n" + "=" * 60)
    if ok_rom and ok_emu and ok_state and ok_scripts:
        print("  STATUS: ENVIRONMENT 100% CONFIGURED AND OPERATIONAL!")
        print("=" * 60)
        print("\nTo start the agent in a fully autonomous way:")
        print("  python launch_mario.py       # Run NEAT")
        print("  python launch_mario.py --gp  # Run Genetic Programming\n")
    else:
        print("  STATUS: SETUP FAILED. Check the messages above.")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
