"""Streamlit dashboard for the ML games experimentation engine."""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from ml_games_engine.catalog import (
    discover_engine_exports,
    discover_model_assets,
    discover_saved_runs,
    discover_vecnorm_assets,
)
from ml_games_engine.control import RuntimePaths, load_json, read_jsonl, write_json
from ml_games_engine.scenarios import SCENARIOS, get_scenario


ROOT_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = ROOT_DIR / "runtime"
CONTROL_PATH = RUNTIME_DIR / "control.json"
PATHS = RuntimePaths.from_control_path(CONTROL_PATH)


def ensure_defaults() -> None:
    st.session_state.setdefault("pilot_version", 0)
    st.session_state.setdefault("reward_version", 0)
    st.session_state.setdefault("world_version", 0)
    st.session_state.setdefault("save_version", 0)
    st.session_state.setdefault("runner_process", None)


@st.cache_data(ttl=2)
def cached_model_assets(scenario_id: str) -> list[dict]:
    return [asset.as_dict() for asset in discover_model_assets(scenario_id)]


@st.cache_data(ttl=2)
def cached_vecnorm_assets(scenario_id: str) -> list[dict]:
    return [asset.as_dict() for asset in discover_vecnorm_assets(scenario_id)]


@st.cache_data(ttl=2)
def cached_saved_runs() -> list[dict]:
    return discover_saved_runs()


@st.cache_data(ttl=2)
def cached_engine_exports(scenario_id: str) -> list[dict]:
    return [asset.as_dict() for asset in discover_engine_exports(scenario_id)]


def current_control() -> dict:
    return load_json(PATHS.control, default={})


def make_run_id(scenario_id: str) -> str:
    return f"{scenario_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def build_control_payload(
    scenario_id: str,
    training: dict,
    reward_values: dict,
    world_values: dict,
    pilot: dict,
    *,
    paused: bool,
    speed: float,
    run_id: str,
    stop_requested: bool = False,
) -> dict:
    return {
        "scenario_id": scenario_id,
        "training": training,
        "reward": {
            "version": st.session_state.reward_version,
            "values": reward_values,
        },
        "world": {
            "version": st.session_state.world_version,
            "values": world_values,
        },
        "pilot": {
            "version": st.session_state.pilot_version,
            **pilot,
        },
        "runtime": {
            "paused": paused,
            "speed": speed,
            "stop_requested": stop_requested,
            "save_version": st.session_state.save_version,
            "run_id": run_id,
        },
    }


def start_runner() -> None:
    runner = st.session_state.runner_process
    if runner is not None and runner.poll() is None:
        control = current_control()
        if control:
            runtime_cfg = dict(control.get("runtime", {}))
            runtime_cfg["stop_requested"] = True
            control["runtime"] = runtime_cfg
            write_json(PATHS.control, control)
            time.sleep(1.0)
        runner.terminate()
        try:
            runner.wait(timeout=10)
        except Exception:
            pass

    PATHS.control.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "ml_games_engine.runner", "--control", str(PATHS.control)]
    st.session_state.runner_process = subprocess.Popen(command, cwd=str(ROOT_DIR))


def load_runtime_state() -> tuple[dict, dict]:
    state = load_json(PATHS.state, default={})
    metrics = load_json(PATHS.metrics, default={"history": []})
    return state, metrics


def safe_line_chart(data: pd.DataFrame, *, x: str, y: str | list[str]) -> None:
    try:
        st.line_chart(data, x=x, y=y)
    except TypeError as exc:
        if "_TypedDictMeta.__new__() got an unexpected keyword argument 'closed'" not in str(exc):
            raise
        y_columns = [y] if isinstance(y, str) else list(y)
        fallback_columns = [x] + [col for col in y_columns if col in data.columns]
        st.warning("Charts are temporarily unavailable in this Python/Altair combo. Showing table fallback.")
        st.dataframe(data[fallback_columns] if fallback_columns else data, use_container_width=True)


def render_config_editor(prefix: str, values: dict, *, columns: int = 2) -> dict:
    output: dict = {}
    col_blocks = st.columns(columns)
    for index, (key, value) in enumerate(values.items()):
        target = col_blocks[index % columns]
        label = key.replace("_", " ").title()
        widget_key = f"{prefix}_{key}"
        with target:
            if isinstance(value, bool):
                output[key] = st.checkbox(label, value=value, key=widget_key)
            elif isinstance(value, int) and not isinstance(value, bool):
                output[key] = int(st.number_input(label, value=int(value), step=1, key=widget_key))
            else:
                step = max(abs(float(value)) * 0.05, 0.01)
                output[key] = float(st.number_input(label, value=float(value), step=step, format="%.4f", key=widget_key))
    return output


def asset_label(asset: dict) -> str:
    return asset["label"]


def selected_asset(options: list[dict], selection_index: int | None) -> dict | None:
    if selection_index is None:
        return None
    if selection_index < 0 or selection_index >= len(options):
        return None
    return options[selection_index]


def format_run_label(run: dict) -> str:
    summary = run.get("summary", {})
    manifest = run.get("manifest", {})
    state = run.get("latest_state", {})
    scenario = manifest.get("scenario", {}).get("label", state.get("scenario_label", "unknown"))
    status = summary.get("status", state.get("status", "unknown"))
    steps = summary.get("steps", state.get("step", 0))
    return f"{run['run_id']} | {scenario} | {status} | step {steps:,}"


def bundle_file_rows(bundle_dir: Path) -> list[dict]:
    if not bundle_dir.exists():
        return []
    rows = []
    for path in sorted(bundle_dir.rglob("*"), key=lambda item: str(item)):
        if path.is_file():
            rows.append(
                {
                    "file": str(path.relative_to(bundle_dir)),
                    "size_kb": round(path.stat().st_size / 1024.0, 2),
                }
            )
    return rows


st.set_page_config(page_title="ML Games Engine", layout="wide")
ensure_defaults()

state, metrics = load_runtime_state()
scenario_options = list(SCENARIOS.keys())
selected_scenario = st.sidebar.selectbox(
    "Scenario",
    scenario_options,
    format_func=lambda key: SCENARIOS[key].label,
)
scenario = get_scenario(selected_scenario)

model_assets = cached_model_assets(selected_scenario)
vecnorm_assets = cached_vecnorm_assets(selected_scenario)
engine_exports = cached_engine_exports(selected_scenario)

st.title("ML Games Engine")
st.caption("Core desacoplado + dashboard hot-reload com catalogo de checkpoints, fisica configuravel, historico completo e exportacao ONNX para uso standalone, Unity, Godot ou Unreal.")

st.sidebar.markdown(f"**Dimensao:** {scenario.dimension}")
st.sidebar.markdown(f"**Viewport:** {scenario.viewport}")
st.sidebar.caption(scenario.description)
st.sidebar.json(
    {
        "scenario_info": scenario.info,
        "world_defaults": scenario.default_world,
    },
    expanded=False,
)

default_training = dict(scenario.default_training)
default_reward = dict(scenario.default_reward)
default_world = dict(scenario.default_world)

tabs = st.tabs(["Experiment", "Live Monitor", "Saved Runs", "Assets"])
tab_experiment, tab_live, tab_saved, tab_assets = tabs

with tab_experiment:
    metric_cols = st.columns(4)
    metric_cols[0].metric("Action Dim", scenario.info.get("action_dim", "-"))
    metric_cols[1].metric("Obs Dim", scenario.info.get("observation_dim", "-"))
    metric_cols[2].metric("Physics Hz", scenario.info.get("physics_hz", "-"))
    metric_cols[3].metric("Max Steps", scenario.info.get("max_steps", "-"))

    setup_left, setup_right = st.columns([1.2, 1.0])

    with setup_left:
        st.subheader("Training Setup")
        st.info("Cada run final exporta automaticamente um bundle ONNX com manifesto de contrato, normalizacao embutida e stubs para Unity, Godot, Unreal e runtime Python.")
        train_mode = st.radio(
            "Training source",
            ["Novo modelo", "Retomar checkpoint"],
            horizontal=True,
            key=f"train_mode_{selected_scenario}",
        )
        resume_model = None
        resume_vecnorm = None
        if train_mode == "Retomar checkpoint":
            if not model_assets:
                st.warning("Nenhum checkpoint encontrado para este scenario. O treino vai usar um modelo novo.")
            else:
                selected_model_index = st.selectbox(
                    "Checkpoint",
                    options=list(range(len(model_assets))),
                    format_func=lambda idx: asset_label(model_assets[idx]),
                    key=f"resume_model_{selected_scenario}",
                )
                resume_model = selected_asset(model_assets, selected_model_index)
                vecnorm_mode = st.radio(
                    "VecNormalize",
                    ["Auto pareado", "Escolher outro", "Nenhum"],
                    horizontal=True,
                    key=f"vecnorm_mode_{selected_scenario}",
                )
                if resume_model and vecnorm_mode == "Auto pareado":
                    resume_vecnorm = resume_model.get("vecnorm_path")
                elif vecnorm_mode == "Escolher outro" and vecnorm_assets:
                    vecnorm_index = st.selectbox(
                        "VecNormalize file",
                        options=list(range(len(vecnorm_assets))),
                        format_func=lambda idx: vecnorm_assets[idx]["label"],
                        key=f"resume_vecnorm_{selected_scenario}",
                    )
                    chosen_vecnorm = selected_asset(vecnorm_assets, vecnorm_index)
                    resume_vecnorm = chosen_vecnorm["path"] if chosen_vecnorm else None
                elif vecnorm_mode == "Escolher outro":
                    st.info("Nenhum VecNormalize catalogado para este scenario.")

        train_col1, train_col2 = st.columns(2)
        with train_col1:
            timesteps = st.number_input("Timesteps", min_value=0, value=int(default_training["timesteps"]), step=1000)
            n_envs = st.number_input("Parallel envs", min_value=1, value=int(default_training["n_envs"]), step=1)
            lr = st.number_input("Learning rate", min_value=0.0, value=float(default_training["lr"]), format="%.6f")
            gamma = st.number_input("Gamma", min_value=0.0, max_value=1.0, value=float(default_training["gamma"]), format="%.4f")
            n_steps = st.number_input("Rollout steps", min_value=16, value=int(default_training["n_steps"]), step=16)
        with train_col2:
            batch_size = st.number_input("Batch size", min_value=8, value=int(default_training["batch_size"]), step=8)
            n_epochs = st.number_input("Epochs", min_value=1, value=int(default_training["n_epochs"]), step=1)
            checkpoint_freq = st.number_input("Checkpoint freq", min_value=100, value=int(default_training["checkpoint_freq"]), step=100)
            net_arch = st.text_input("Net arch", value=str(default_training["net_arch"]))
            seed = st.number_input("Seed", min_value=0, value=int(default_training["seed"]), step=1)

        st.subheader("Brain Switcher")
        pilot_name = st.selectbox(
            "Viewport brain",
            ["live_policy", "random", "checkpoint_ppo", "disabled"],
            help="Troca o cerebro da visualizacao em tempo real, ou desliga o preview.",
            key=f"pilot_name_{selected_scenario}",
        )
        pilot_model = None
        pilot_vecnorm = None
        if pilot_name == "checkpoint_ppo":
            if not model_assets:
                st.warning("Nao ha checkpoints catalogados para usar como brain de viewport.")
            else:
                pilot_index = st.selectbox(
                    "Viewport checkpoint",
                    options=list(range(len(model_assets))),
                    format_func=lambda idx: asset_label(model_assets[idx]),
                    key=f"pilot_model_{selected_scenario}",
                )
                pilot_model = selected_asset(model_assets, pilot_index)
                pilot_vecnorm = pilot_model.get("vecnorm_path") if pilot_model else None

    with setup_right:
        st.subheader("World & Physics")
        world_values = render_config_editor(f"world_{selected_scenario}", default_world, columns=1)
        st.subheader("Reward Hot Reload")
        reward_values = render_config_editor(f"reward_{selected_scenario}", default_reward, columns=1)
        speed = st.slider("Preview speed", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
        auto_refresh = st.checkbox("Auto refresh", value=True)
        refresh_seconds = st.slider("Refresh interval", min_value=1, max_value=5, value=2, step=1)

    training_payload = {
        "timesteps": int(timesteps),
        "n_envs": int(n_envs),
        "lr": float(lr),
        "gamma": float(gamma),
        "n_steps": int(n_steps),
        "batch_size": int(batch_size),
        "n_epochs": int(n_epochs),
        "gae_lambda": float(default_training["gae_lambda"]),
        "clip_range": float(default_training["clip_range"]),
        "ent_coef": float(default_training["ent_coef"]),
        "checkpoint_freq": int(checkpoint_freq),
        "net_arch": net_arch,
        "seed": int(seed),
        "model_path": resume_model["model_path"] if resume_model else None,
        "vecnorm_path": resume_vecnorm,
    }
    pilot_payload = {
        "name": pilot_name,
        "checkpoint_path": pilot_model["model_path"] if pilot_model else None,
        "vecnorm_path": pilot_vecnorm,
    }

    actions_top = st.columns(4)
    if actions_top[0].button("Start / Restart Run", use_container_width=True):
        st.session_state.reward_version += 1
        st.session_state.world_version += 1
        st.session_state.pilot_version += 1
        run_id = make_run_id(selected_scenario)
        control = build_control_payload(
            selected_scenario,
            training_payload,
            reward_values,
            world_values,
            pilot_payload,
            paused=False,
            speed=speed,
            run_id=run_id,
        )
        write_json(PATHS.control, control)
        start_runner()

    if actions_top[1].button("Pause / Resume", use_container_width=True):
        control = current_control()
        runtime_cfg = dict(control.get("runtime", {}))
        paused = not bool(runtime_cfg.get("paused", False))
        run_id = runtime_cfg.get("run_id", make_run_id(selected_scenario))
        control = build_control_payload(
            control.get("scenario_id", selected_scenario),
            training_payload,
            reward_values,
            world_values,
            pilot_payload,
            paused=paused,
            speed=float(runtime_cfg.get("speed", speed)),
            run_id=run_id,
        )
        write_json(PATHS.control, control)

    if actions_top[2].button("Apply Reward", use_container_width=True):
        st.session_state.reward_version += 1
        control = current_control()
        runtime_cfg = dict(control.get("runtime", {}))
        run_id = runtime_cfg.get("run_id", make_run_id(selected_scenario))
        control = build_control_payload(
            control.get("scenario_id", selected_scenario),
            training_payload,
            reward_values,
            control.get("world", {}).get("values", world_values),
            {
                "name": control.get("pilot", {}).get("name", pilot_name),
                "checkpoint_path": control.get("pilot", {}).get("checkpoint_path"),
                "vecnorm_path": control.get("pilot", {}).get("vecnorm_path"),
            },
            paused=bool(runtime_cfg.get("paused", False)),
            speed=float(runtime_cfg.get("speed", speed)),
            run_id=run_id,
        )
        write_json(PATHS.control, control)

    if actions_top[3].button("Apply Physics", use_container_width=True):
        st.session_state.world_version += 1
        control = current_control()
        runtime_cfg = dict(control.get("runtime", {}))
        run_id = runtime_cfg.get("run_id", make_run_id(selected_scenario))
        control = build_control_payload(
            control.get("scenario_id", selected_scenario),
            training_payload,
            control.get("reward", {}).get("values", reward_values),
            world_values,
            {
                "name": control.get("pilot", {}).get("name", pilot_name),
                "checkpoint_path": control.get("pilot", {}).get("checkpoint_path"),
                "vecnorm_path": control.get("pilot", {}).get("vecnorm_path"),
            },
            paused=bool(runtime_cfg.get("paused", False)),
            speed=float(runtime_cfg.get("speed", speed)),
            run_id=run_id,
        )
        write_json(PATHS.control, control)

    actions_bottom = st.columns(4)
    if actions_bottom[0].button("Switch Brain", use_container_width=True):
        st.session_state.pilot_version += 1
        control = current_control()
        runtime_cfg = dict(control.get("runtime", {}))
        run_id = runtime_cfg.get("run_id", make_run_id(selected_scenario))
        control = build_control_payload(
            control.get("scenario_id", selected_scenario),
            training_payload,
            control.get("reward", {}).get("values", reward_values),
            control.get("world", {}).get("values", world_values),
            pilot_payload,
            paused=bool(runtime_cfg.get("paused", False)),
            speed=float(runtime_cfg.get("speed", speed)),
            run_id=run_id,
        )
        write_json(PATHS.control, control)

    if actions_bottom[1].button("Save Checkpoint", use_container_width=True):
        st.session_state.save_version += 1
        control = current_control()
        runtime_cfg = dict(control.get("runtime", {}))
        runtime_cfg["save_version"] = st.session_state.save_version
        control["runtime"] = runtime_cfg
        write_json(PATHS.control, control)

    if actions_bottom[2].button("Stop Run", use_container_width=True):
        control = current_control()
        runtime_cfg = dict(control.get("runtime", {}))
        run_id = runtime_cfg.get("run_id", make_run_id(selected_scenario))
        control = build_control_payload(
            control.get("scenario_id", selected_scenario),
            training_payload,
            control.get("reward", {}).get("values", reward_values),
            control.get("world", {}).get("values", world_values),
            {
                "name": control.get("pilot", {}).get("name", pilot_name),
                "checkpoint_path": control.get("pilot", {}).get("checkpoint_path"),
                "vecnorm_path": control.get("pilot", {}).get("vecnorm_path"),
            },
            paused=True,
            speed=float(runtime_cfg.get("speed", speed)),
            run_id=run_id,
            stop_requested=True,
        )
        write_json(PATHS.control, control)

    if actions_bottom[3].button("Refresh Now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

with tab_live:
    live_dimension = state.get("dimension", scenario.dimension)
    status_cols = st.columns(6)
    status_cols[0].metric("Status", state.get("status", "idle"))
    status_cols[1].metric("Steps", f"{int(state.get('step', 0)):,}")
    status_cols[2].metric("Episodes", int(state.get("episodes", 0)))
    status_cols[3].metric("Last Reward", f"{float(state.get('last_reward', 0.0)):.2f}")
    status_cols[4].metric("Best Reward", f"{float(state.get('best_reward') or 0.0):.2f}")
    gravity_display = state.get("world", {}).get("gravity_z", scenario.default_world.get("gravity_z", "n/a"))
    status_cols[5].metric("Gravity Z", gravity_display)

    live_left, live_right = st.columns([1.2, 1.0])
    history = metrics.get("history", [])
    episode_rows = [row for row in history if row.get("kind") == "episode"]
    train_rows = [row for row in history if row.get("kind") == "train"]

    with live_left:
        st.subheader("Live Metrics")
        if episode_rows:
            df_rewards = pd.DataFrame(episode_rows)
            safe_line_chart(df_rewards, x="step", y="reward")
        else:
            st.info("Os graficos vao aparecer assim que os primeiros episodios forem fechados.")

        if train_rows:
            df_losses = pd.DataFrame(train_rows)
            chart_cols = [col for col in ["loss", "value_loss", "policy_loss", "approx_kl"] if col in df_losses.columns]
            if chart_cols:
                safe_line_chart(df_losses, x="step", y=chart_cols)

        st.subheader("State Snapshot")
        st.json(state or {"status": "idle"}, expanded=False)

    with live_right:
        st.subheader("Viewport")
        if state.get("pilot_name") == "disabled":
            st.info("Preview desativado neste run.")
        elif live_dimension == "2D" and PATHS.frame.exists():
            st.image(np.load(PATHS.frame), caption="Arena 2D live preview", use_container_width=True)
        elif live_dimension == "3D":
            st.info("O viewport 3D abre em uma janela PyBullet separada. O dashboard salva o historico completo do run em `runtime/runs/`.")
        else:
            st.info("Nenhum frame disponivel ainda.")

        st.subheader("Events")
        for event in state.get("events", []):
            st.code(event)

with tab_saved:
    saved_runs = cached_saved_runs()
    if not saved_runs:
        st.info("Nenhum run salvo ainda.")
    else:
        selected_run_index = st.selectbox(
            "Saved runs",
            options=list(range(len(saved_runs))),
            format_func=lambda idx: format_run_label(saved_runs[idx]),
        )
        selected_run = saved_runs[selected_run_index]
        run_dir = Path(selected_run["path"])
        manifest = load_json(run_dir / "manifest.json", default={})
        summary = load_json(run_dir / "summary.json", default={})
        latest_state = load_json(run_dir / "state_latest.json", default={})
        episode_history = read_jsonl(run_dir / "episode_history.jsonl")
        train_history = read_jsonl(run_dir / "train_history.jsonl")
        control_history = read_jsonl(run_dir / "control_history.jsonl")
        export_dir = run_dir / "engine_export"
        export_manifest = load_json(export_dir / "manifest.json", default={})
        export_validation = load_json(export_dir / "validation.json", default={})
        export_files = bundle_file_rows(export_dir)

        saved_metrics = st.columns(6)
        saved_metrics[0].metric("Run", selected_run["run_id"])
        saved_metrics[1].metric("Status", summary.get("status", latest_state.get("status", "unknown")))
        saved_metrics[2].metric("Steps", f"{int(summary.get('steps', latest_state.get('step', 0))):,}")
        saved_metrics[3].metric("Episodes", int(summary.get("episodes", latest_state.get("episodes", 0))))
        saved_metrics[4].metric("Best Reward", f"{float(summary.get('best_reward') or 0.0):.2f}")
        saved_metrics[5].metric("Engine Export", "ready" if export_manifest else "missing")

        saved_left, saved_right = st.columns([1.2, 1.0])
        with saved_left:
            st.subheader("Saved Charts")
            chart_reward = run_dir / "charts" / "episode_reward.png"
            chart_train = run_dir / "charts" / "training_metrics.png"
            if chart_reward.exists():
                st.image(str(chart_reward), caption="Episode reward", use_container_width=True)
            elif episode_history:
                safe_line_chart(pd.DataFrame(episode_history), x="step", y="reward")
            if chart_train.exists():
                st.image(str(chart_train), caption="Training metrics", use_container_width=True)
            elif train_history:
                df_train = pd.DataFrame(train_history)
                cols = [col for col in ["loss", "value_loss", "policy_loss", "approx_kl"] if col in df_train.columns]
                if cols:
                    safe_line_chart(df_train, x="step", y=cols)

            if episode_history:
                st.subheader("Episode History")
                st.dataframe(pd.DataFrame(episode_history).tail(100), use_container_width=True)
            if train_history:
                st.subheader("Train History")
                st.dataframe(pd.DataFrame(train_history).tail(100), use_container_width=True)

        with saved_right:
            preview_path = run_dir / "previews" / "latest_frame.png"
            if preview_path.exists():
                st.subheader("Preview Snapshot")
                st.image(str(preview_path), use_container_width=True)

            st.subheader("Manifest")
            st.json(manifest, expanded=False)
            st.subheader("Summary")
            st.json(summary or latest_state, expanded=False)
            st.subheader("Engine Export")
            if export_manifest:
                contract = export_manifest.get("contract", {})
                engine_cols = st.columns(3)
                engine_cols[0].metric("Obs Dim", contract.get("observation_dim", "-"))
                engine_cols[1].metric("Action Dim", contract.get("action_dim", "-"))
                engine_cols[2].metric("Validated", "yes" if export_validation.get("within_tolerance") else "no")
                if export_files:
                    st.dataframe(pd.DataFrame(export_files), use_container_width=True)
                st.json(export_manifest, expanded=False)
                st.json(export_validation, expanded=False)
            else:
                st.info("Este run ainda nao possui bundle ONNX exportado.")
            st.subheader("Control History")
            if control_history:
                st.dataframe(pd.DataFrame(control_history).tail(50), use_container_width=True)

with tab_assets:
    st.subheader("Checkpoint Catalog")
    if model_assets:
        st.dataframe(pd.DataFrame(model_assets), use_container_width=True)
    else:
        st.info("Nenhum checkpoint catalogado para este scenario.")

    st.subheader("VecNormalize Catalog")
    if vecnorm_assets:
        st.dataframe(pd.DataFrame(vecnorm_assets), use_container_width=True)
    else:
        st.info("Nenhum VecNormalize catalogado para este scenario.")

    st.subheader("Engine Export Catalog")
    if engine_exports:
        st.dataframe(pd.DataFrame(engine_exports), use_container_width=True)
        selected_export_index = st.selectbox(
            "Engine bundle",
            options=list(range(len(engine_exports))),
            format_func=lambda idx: engine_exports[idx]["label"],
            key=f"engine_export_{selected_scenario}",
        )
        selected_export = engine_exports[selected_export_index]
        export_manifest = load_json(Path(selected_export["manifest_path"]), default={})
        export_validation = load_json(Path(selected_export["validation_path"]), default={}) if selected_export.get("validation_path") else {}
        asset_left, asset_right = st.columns([1.1, 1.0])
        with asset_left:
            st.json(export_manifest, expanded=False)
        with asset_right:
            st.json(export_validation, expanded=False)
            bundle_rows = bundle_file_rows(Path(selected_export["path"]))
            if bundle_rows:
                st.dataframe(pd.DataFrame(bundle_rows), use_container_width=True)
    else:
        st.info("Nenhum bundle ONNX catalogado para este scenario.")

if "auto_refresh" not in locals():
    auto_refresh = True
if "refresh_seconds" not in locals():
    refresh_seconds = 2

if auto_refresh:
    time.sleep(refresh_seconds)
    st.cache_data.clear()
    st.rerun()
