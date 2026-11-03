"""
env_config.py
=============
Cross-platform environment configuration and discovery engine for
smw-evolutionary-agent (supporting Windows, Linux, and macOS).

Abstracts platform-specific emulator executables, standard directory layouts,
and launcher commands (including Mono and native Linux BizHawk variants).
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class BizHawkEnvironment:
    """Resolved BizHawk environment and executable command wrapper."""
    bizhawk_dir: Path
    exec_cmd: List[str]
    rom_path: Path
    state_path: Path
    is_available: bool
    platform: str
    message: str


def find_bizhawk_executable(base_dir: Optional[Path] = None, custom_bin: Optional[Path] = None) -> Optional[List[str]]:
    """
    Detect the appropriate BizHawk execution command for the current OS.
    Returns a command list (e.g. ['EmuHawk.exe'], ['mono', 'EmuHawk.exe'], or ['EmuHawkMono.sh']).
    """
    if custom_bin and custom_bin.exists():
        if sys.platform != "win32" and custom_bin.suffix.lower() == ".exe":
            mono_bin = shutil.which("mono")
            if mono_bin:
                return [mono_bin, str(custom_bin)]
        return [str(custom_bin)]

    # 1. Search in PATH
    for candidate in ["EmuHawk", "EmuHawk.exe", "EmuHawkMono.sh", "bizhawk"]:
        path_hit = shutil.which(candidate)
        if path_hit:
            p = Path(path_hit)
            if sys.platform != "win32" and p.suffix.lower() == ".exe":
                mono_bin = shutil.which("mono")
                if mono_bin:
                    return [mono_bin, str(p)]
            return [str(p)]

    # 2. Check candidate directories
    search_dirs: List[Path] = []
    if base_dir:
        search_dirs.append(base_dir)

    env_home = os.environ.get("BIZHAWK_HOME") or os.environ.get("BIZHAWK_DIR")
    if env_home:
        search_dirs.append(Path(env_home))

    home = Path.home()
    if sys.platform == "win32":
        search_dirs.extend([
            Path(r"C:\BizHawk"),
            home / "BizHawk",
            home / "AppData" / "Local" / "BizHawk",
            Path(r"C:\Program Files\BizHawk"),
            Path(r"C:\Program Files (x86)\BizHawk"),
        ])
        exe_names = ["EmuHawk.exe", "BizHawk.exe"]
    else:  # Linux / macOS
        search_dirs.extend([
            Path("/opt/bizhawk"),
            home / "bizhawk",
            home / ".local" / "share" / "bizhawk",
            Path("/usr/local/bizhawk"),
        ])
        exe_names = ["EmuHawkMono.sh", "bizhawk", "EmuHawk.exe"]

    for d in search_dirs:
        if not d.exists() or not d.is_dir():
            continue
        for name in exe_names:
            target = d / name
            if target.exists():
                if sys.platform != "win32" and target.suffix.lower() == ".exe":
                    mono_bin = shutil.which("mono")
                    if mono_bin:
                        return [mono_bin, str(target)]
                return [str(target)]

    return None


def get_bizhawk_environment(
    custom_dir: Optional[str] = None,
    custom_bin: Optional[str] = None
) -> BizHawkEnvironment:
    """
    Resolve and validate the BizHawk environment in a cross-platform manner.
    """
    repo_dir = Path(__file__).resolve().parent
    base_dir: Optional[Path] = None

    if custom_dir:
        base_dir = Path(custom_dir)
    else:
        env_dir = os.environ.get("BIZHAWK_HOME") or os.environ.get("BIZHAWK_DIR")
        if env_dir:
            base_dir = Path(env_dir)
        elif sys.platform == "win32" and Path(r"C:\BizHawk").exists():
            base_dir = Path(r"C:\BizHawk")
        elif (repo_dir / "bizhawk").exists():
            base_dir = repo_dir / "bizhawk"

    custom_bin_path = Path(custom_bin) if custom_bin else None
    exec_cmd = find_bizhawk_executable(base_dir=base_dir, custom_bin=custom_bin_path)

    if exec_cmd:
        # Determine the effective BizHawk directory from the executable path
        target_path = Path(exec_cmd[-1])
        effective_dir = target_path.parent if target_path.is_file() else (base_dir or repo_dir)
        is_avail = True
        msg = f"BizHawk executable resolved: {' '.join(exec_cmd)}"
    else:
        effective_dir = base_dir or (Path(r"C:\BizHawk") if sys.platform == "win32" else Path("/opt/bizhawk"))
        exec_cmd = [str(effective_dir / ("EmuHawk.exe" if sys.platform == "win32" else "EmuHawkMono.sh"))]
        is_avail = False
        msg = (
            f"BizHawk executable not found in standard paths for {sys.platform}. "
            "Please install BizHawk (2.9+) and set the BIZHAWK_HOME environment variable or pass --bizhawk-dir."
        )

    rom_path = effective_dir / "Super Mario World (USA).sfc"
    state_path = effective_dir / "DP1.state"

    return BizHawkEnvironment(
        bizhawk_dir=effective_dir,
        exec_cmd=exec_cmd,
        rom_path=rom_path,
        state_path=state_path,
        is_available=is_avail,
        platform=sys.platform,
        message=msg
    )
