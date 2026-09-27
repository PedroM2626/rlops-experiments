"""Subprocess helper for run_all.py: runs a script from this directory with a
chosen interpreter (torch vs JAX venv) and reports whether it succeeded.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def sh(python: str, script_args: list[str]) -> bool:
    print(f"\n$ {python} {' '.join(script_args)}", flush=True)
    r = subprocess.run([python, *script_args], cwd=str(HERE))
    return r.returncode == 0
