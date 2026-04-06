"""JSON-backed control and runtime state helpers."""

from __future__ import annotations

import copy
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def utc_now_iso() -> str:
    """Return a compact UTC timestamp for state files."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path, default: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Load JSON from disk, falling back to a defensive copy of ``default``."""
    if not path.exists():
        return copy.deepcopy(default or {})
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return copy.deepcopy(default or {})


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Atomically write JSON so the dashboard never reads a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f"{path.stem}_",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@dataclass(frozen=True)
class RuntimePaths:
    """Runtime file contract shared by the runner and the dashboard."""

    control: Path
    state: Path
    metrics: Path
    frame: Path
    artifacts_dir: Path

    @classmethod
    def from_control_path(cls, control_path: str | Path) -> "RuntimePaths":
        control = Path(control_path).resolve()
        runtime_dir = control.parent
        return cls(
            control=control,
            state=runtime_dir / "state.json",
            metrics=runtime_dir / "metrics.json",
            frame=runtime_dir / "latest_frame.npy",
            artifacts_dir=runtime_dir / "artifacts",
        )
