"""
launch_mario.py
===============
100% autonomous launch pipeline for Super Mario World with evolutionary agents.
Supports NEAT (MarIO.lua) and Genetic Programming (MarIO_GP.lua).

Autonomous execution flow:
  1. Integrity validation of the ROM and of the BizHawk emulator binaries.
  2. Compatibility check of the 'DP1.state' savestate (native BizHawk format).
     If it is missing or incompatible, run the deterministic Lua bootstrap
     to navigate the SMW menus and save the state at the exact level-start frame.
  3. Direct launch of BizHawk with the selected evolutionary script (--lua=MarIO.lua or --lua=MarIO_GP.lua).

Usage:
  python launch_mario.py           # Start MarI/O (NEAT)
  python launch_mario.py --gp      # Start MarI/O Genetic Programming (Interpretable Tree-GP)
  python launch_mario.py --reboot-state  # Force DP1.state regeneration
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Directory and path constants
REPO_DIR = Path(__file__).resolve().parent
BIZHAWK_DIR = Path(r"C:\BizHawk")
EMUHAWK_EXE = BIZHAWK_DIR / "EmuHawk.exe"
ROM_FILE = BIZHAWK_DIR / "Super Mario World (USA).sfc"
STATE_FILE = BIZHAWK_DIR / "DP1.state"

LUA_NEAT = BIZHAWK_DIR / "MarIO.lua"
LUA_GP = BIZHAWK_DIR / "MarIO_GP.lua"

ROM_SHA1_USA = "6b47bb75d16514b6a476aa0c73a683a2a4c18765"


def log(msg: str):
    print(f"[SMW-AGENT] {msg}")


def check_rom_and_binaries() -> bool:
    """Validate and synchronize the ROM and scripts into the BizHawk directory."""
    if not EMUHAWK_EXE.exists():
        log(f"ERROR: BizHawk executable not found at {EMUHAWK_EXE}")
        return False

    # Copy the ROM from the repository to BizHawk if needed
    rom_src = REPO_DIR / "Super Mario World (USA).sfc"
    if not ROM_FILE.exists() and rom_src.exists():
        shutil.copy2(rom_src, ROM_FILE)
        log("ROM copied to C:\\BizHawk\\")

    if not ROM_FILE.exists():
        log(f"ERROR: ROM not found at {ROM_FILE}")
        return False

    # Validate SHA-1
    sha1 = hashlib.sha1(ROM_FILE.read_bytes()).hexdigest()
    if sha1 == ROM_SHA1_USA:
        log("ROM verified successfully (SHA-1 matches the official USA dump).")
    else:
        log(f"WARNING: ROM SHA-1 ({sha1}) differs from the expected value ({ROM_SHA1_USA}).")

    # Make sure the latest Lua scripts are present in the BizHawk directory
    for script_name in ["MarIO.lua", "MarIO_GP.lua"]:
        src = REPO_DIR / script_name
        dst = BIZHAWK_DIR / script_name
        if src.exists():
            shutil.copy2(src, dst)
            log(f"Script updated: {dst.name}")

    # Remove a corrupted or outdated config.ini if present, to avoid dialog boxes
    cfg_file = BIZHAWK_DIR / "config.ini"
    if cfg_file.exists():
        # Check whether config.ini triggers a version popup
        try:
            content = cfg_file.read_text(encoding="utf-8", errors="ignore")
            if "DispMethodGL" not in content or "MainFormWidth" not in content:
                cfg_file.unlink(missing_ok=True)
                log("Legacy config.ini removed to ensure a popup-free startup.")
        except Exception:
            cfg_file.unlink(missing_ok=True)

    return True


def is_valid_bizhawk_state(state_path: Path) -> bool:
    """Check whether the file is a valid native BizHawk savestate."""
    if not state_path.exists() or state_path.stat().st_size < 10_000:
        return False

    with open(state_path, "rb") as f:
        magic = f.read(16)

    # Libretro/Snes9x format starts with #!s9xsnp (not supported by savestate.load in BizHawk)
    if magic.startswith(b"#!s9xsn"):
        return False

    # BizHawk savestates use a zip/bzip2 container (PK\x03\x04 or BZh)
    return magic.startswith(b"PK") or magic.startswith(b"BZh") or magic.startswith(b"BizState")


def generate_native_savestate() -> bool:
    """
    Navigate the SMW ROM menus in a fully autonomous way and save
    DP1.state in native BizHawk format at the first frame of the level.
    """
    log("Generating the native DP1.state savestate autonomously...")

    bootstrap_lua = BIZHAWK_DIR / "auto_bootstrap_dp1.lua"
    lua_code = """-- auto_bootstrap_dp1.lua
-- Autonomous DP1.state savestate generator for BizHawk
local frame = 0
local logfile = io.open("C:/BizHawk/bootstrap.log", "w")

while frame < 1500 do
    frame = frame + 1
    
    -- Pulse Start at regular intervals to advance the title screen and select the save file
    if (frame % 30) < 6 then
        joypad.set({["P1 Start"] = true})
    else
        joypad.set({["P1 Start"] = false})
    end
    
    -- On the Overworld map, move right and press the enter button
    if frame > 380 and (frame % 20) < 10 then
        joypad.set({["P1 Right"] = true, ["P1 Start"] = true, ["P1 B"] = true})
    end
    
    local mode = memory.readbyte(0x0100)
    local marioX = memory.read_s16_le(0x94)
    
    -- Game Mode 0x14 = In-Level (active gameplay)
    if mode == 0x14 and marioX >= 100 then
        savestate.save("C:/BizHawk/DP1.state")
        if logfile then
            logfile:write(string.format("SUCCESS: level loaded at frame %d (Mode 0x14, MarioX=%d)\\n", frame, marioX))
            logfile:close()
        end
        client.closeemulator()
        return
    end
    
    emu.frameadvance()
end

if logfile then
    logfile:write("TIMEOUT: failed to reach the level within 1500 frames.\\n")
    logfile:close()
end
client.closeemulator()
"""
    bootstrap_lua.write_text(lua_code, encoding="utf-8")

    cmd = [str(EMUHAWK_EXE), f"--lua={bootstrap_lua}", str(ROM_FILE)]
    proc = subprocess.Popen(cmd, cwd=str(BIZHAWK_DIR))

    try:
        proc.wait(timeout=25)
    except subprocess.TimeoutExpired:
        proc.kill()

    # Copy the generated state back into the repository
    if is_valid_bizhawk_state(STATE_FILE):
        repo_state = REPO_DIR / "DP1.state"
        shutil.copy2(STATE_FILE, repo_state)
        log(f"Native DP1.state generated successfully ({STATE_FILE.stat().st_size // 1024} KB).")
        return True
    else:
        log("FAILED to generate the native DP1.state.")
        return False


def launch_agent(use_gp: bool = False):
    """Run BizHawk with the selected evolutionary agent."""
    script_path = LUA_GP if use_gp else LUA_NEAT
    algo_name = "Genetic Programming (Interpretable Symbolic Tree)" if use_gp else "NEAT (Evolving Neural Topology)"

    log(f"Starting emulation with {algo_name}...")
    log(f"Executable: {EMUHAWK_EXE}")
    log(f"Script:     {script_path.name}")
    log(f"ROM:        {ROM_FILE.name}")
    log("The emulator will open now. Watch the agent evolve in its window.")

    cmd = [
        str(EMUHAWK_EXE),
        f"--lua={script_path}",
        str(ROM_FILE)
    ]

    proc = subprocess.Popen(cmd, cwd=str(BIZHAWK_DIR))

    try:
        proc.wait()
    except KeyboardInterrupt:
        log("Interrupt requested by the user. Closing BizHawk...")
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()

    log("Emulation session finished.")


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous launcher for the Super Mario World evolutionary agent (NEAT / GP)"
    )
    parser.add_argument(
        "--gp",
        action="store_true",
        help="Run the Genetic Programming approach (MarIO_GP.lua)"
    )
    parser.add_argument(
        "--reboot-state",
        action="store_true",
        help="Regenerate the DP1.state savestate even if it already exists"
    )
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  SUPER MARIO WORLD - AUTONOMOUS EVOLUTIONARY AGENT")
    print("=" * 65 + "\n")

    if not check_rom_and_binaries():
        sys.exit(1)

    # Check whether the state needs to be regenerated
    need_state = args.reboot_state or not is_valid_bizhawk_state(STATE_FILE)

    # If a valid copy exists in the repo, restore it directly
    repo_state = REPO_DIR / "DP1.state"
    if need_state and is_valid_bizhawk_state(repo_state) and not args.reboot_state:
        shutil.copy2(repo_state, STATE_FILE)
        log("Valid DP1.state restored from the repository.")
        need_state = False

    if need_state:
        success = generate_native_savestate()
        if not success:
            log("CRITICAL ERROR: could not obtain or generate a valid DP1.state.")
            sys.exit(1)

    launch_agent(use_gp=args.gp)


if __name__ == "__main__":
    main()
