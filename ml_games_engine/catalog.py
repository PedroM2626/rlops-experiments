"""Artifact and run discovery helpers for the dashboard."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

from ml_games_engine.control import load_json
from ml_games_engine.scenarios import ROOT_DIR, ScenarioSpec, get_scenario


@dataclass(frozen=True)
class ModelAsset:
    """Discovered model checkpoint plus its paired VecNormalize stats."""

    label: str
    model_path: str
    vecnorm_path: str | None
    scenario_id: str
    source: str
    updated_at: float
    step_hint: int | None
    is_final: bool

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class VecNormAsset:
    """Standalone VecNormalize file."""

    label: str
    path: str
    scenario_id: str
    source: str
    updated_at: float

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def _candidate_roots(spec: ScenarioSpec) -> List[Path]:
    roots = []
    for relative in spec.model_roots:
        root = ROOT_DIR / relative
        if root.exists():
            roots.append(root)
    return roots


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT_DIR.resolve()))
    except ValueError:
        return str(path.resolve())


def _step_hint(name: str) -> int | None:
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else None


def discover_model_assets(scenario_id: str) -> List[ModelAsset]:
    """Return checkpoint options for a scenario, newest first."""
    spec = get_scenario(scenario_id)
    assets: List[ModelAsset] = []

    for root in _candidate_roots(spec):
        for model_path in root.rglob("*.zip"):
            name = model_path.stem
            if spec.model_prefix not in name:
                continue

            vecnorm_path = model_path.with_name("vec_normalize.pkl")
            pair = str(vecnorm_path.resolve()) if vecnorm_path.exists() else None
            source = _rel(root)
            label = f"{model_path.name} | {_rel(model_path.parent)}"
            if pair:
                label += " | vecnorm"
            assets.append(
                ModelAsset(
                    label=label,
                    model_path=str(model_path.resolve()),
                    vecnorm_path=pair,
                    scenario_id=scenario_id,
                    source=source,
                    updated_at=model_path.stat().st_mtime,
                    step_hint=_step_hint(name),
                    is_final="final" in name.lower(),
                )
            )

    assets.sort(key=lambda item: (item.updated_at, item.step_hint or -1), reverse=True)
    dedup: Dict[str, ModelAsset] = {}
    for asset in assets:
        dedup.setdefault(asset.model_path, asset)
    return list(dedup.values())


def discover_vecnorm_assets(scenario_id: str) -> List[VecNormAsset]:
    """Return VecNormalize files for a scenario, newest first."""
    spec = get_scenario(scenario_id)
    assets: List[VecNormAsset] = []

    for root in _candidate_roots(spec):
        for vecnorm_path in root.rglob("vec_normalize.pkl"):
            assets.append(
                VecNormAsset(
                    label=f"vec_normalize.pkl | {_rel(vecnorm_path.parent)}",
                    path=str(vecnorm_path.resolve()),
                    scenario_id=scenario_id,
                    source=_rel(root),
                    updated_at=vecnorm_path.stat().st_mtime,
                )
            )

    assets.sort(key=lambda item: item.updated_at, reverse=True)
    dedup: Dict[str, VecNormAsset] = {}
    for asset in assets:
        dedup.setdefault(asset.path, asset)
    return list(dedup.values())


def discover_saved_runs() -> List[dict]:
    """Return archived runs with lightweight manifest/summary metadata."""
    runs_dir = ROOT_DIR / "runtime" / "runs"
    if not runs_dir.exists():
        return []

    rows: List[dict] = []
    for run_dir in sorted(runs_dir.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
        if not run_dir.is_dir():
            continue
        manifest = load_json(run_dir / "manifest.json", default={})
        summary = load_json(run_dir / "summary.json", default={})
        latest_state = load_json(run_dir / "state_latest.json", default={})
        rows.append(
            {
                "run_id": run_dir.name,
                "path": str(run_dir.resolve()),
                "manifest": manifest,
                "summary": summary,
                "latest_state": latest_state,
            }
        )
    return rows
