"""JSON-backed control and runtime state helpers."""

from __future__ import annotations

import copy
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


def utc_now_iso() -> str:
    """Return a compact UTC timestamp for state files."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path, default: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Load JSON from disk, falling back to a defensive copy of ``default``."""
    if not path.exists():
        return copy.deepcopy(default or {})
    
    max_retries = 5
    delay = 0.05
    for attempt in range(max_retries):
        try:
            with path.open("r", encoding="utf-8-sig") as handle:
                return json.load(handle)
        except (PermissionError, OSError):
            if attempt == max_retries - 1:
                return copy.deepcopy(default or {})
            time.sleep(delay)
        except json.JSONDecodeError:
            return copy.deepcopy(default or {})
    return copy.deepcopy(default or {})


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Atomically write JSON so the dashboard never reads a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    max_retries = 5
    delay = 0.05
    last_exc = None
    
    for attempt in range(max_retries):
        fd, tmp_path = tempfile.mkstemp(
            prefix=f"{path.stem}_",
            suffix=".tmp",
            dir=str(path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
            os.replace(tmp_path, path)
            return
        except (PermissionError, OSError) as exc:
            last_exc = exc
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            time.sleep(delay)
            
    if last_exc:
        raise last_exc


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    """Append a single JSON object to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    max_retries = 5
    delay = 0.05
    last_exc = None
    
    for attempt in range(max_retries):
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True))
                handle.write("\n")
            return
        except (PermissionError, OSError) as exc:
            last_exc = exc
            time.sleep(delay)
            
    if last_exc:
        raise last_exc


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read JSONL content, skipping malformed lines defensively."""
    if not path.exists():
        return []

    max_retries = 5
    delay = 0.05
    for attempt in range(max_retries):
        try:
            rows: List[Dict[str, Any]] = []
            with path.open("r", encoding="utf-8-sig") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            return rows
        except (PermissionError, OSError):
            if attempt == max_retries - 1:
                return []
            time.sleep(delay)
    return []


def sanitize_slug(value: str) -> str:
    """Convert arbitrary labels into filesystem-friendly slugs."""
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip())
    slug = slug.strip("-._")
    return slug or "run"


@dataclass(frozen=True)
class RuntimePaths:
    """Runtime file contract shared by the runner and the dashboard."""

    control: Path
    state: Path
    metrics: Path
    frame: Path
    artifacts_dir: Path
    runs_dir: Path

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
            runs_dir=runtime_dir / "runs",
        )
