"""Streamlit dashboard for the ML games experimentation engine."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from ml_games_engine.control import RuntimePaths, load_json, write_json
from ml_games_engine.scenarios import SCENARIOS, get_scenario


ROOT_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = ROOT_DIR / "runtime"
CONTROL_PATH = RUNTIME_DIR / "control.json"
PATHS = RuntimePaths.from_control_path(CONTROL_PATH)


def ensure_defaults() -> None:
    st.session_state.setdefault("pilot_version", 0)
    st.session_state.setdefault("reward_version", 0)
    st.session_state.setdefault("save_version", 0)
    st.session_state.setdefault("runner_process", None)


def build_control_payload(
    scenario_id: str,
    training: dict,
    reward_values: dict,
    pilot: dict,
    *,
    paused: bool,
    speed: float,
    stop_requested: bool = False,
) -> dict:
    return {
        "scenario_id": scenario_id,
        "training": training,
        "reward": {
            "version": st.session_state.reward_version,
            "values": reward_values,
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
        },
    }


def start_runner() -> None:
    runner = st.session_state.runner_process
    if runner is not None and runner.poll() is None:
        runner.terminate()
        try:
            runner.wait(timeout=10)
        except Exception:
            pass

    PATHS.control.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "ml_games_engine.runner", "--control", str(PATHS.control)]
    st.session_state.runner_process = subprocess.Popen(command, cwd=str(ROOT_DIR))


def stop_runner() -> None:
    runner = st.session_state.runner_process
    if runner is not None and runner.poll() is None:
        runner.terminate()


def load_runtime_state() -> tuple[dict, dict]:
    state = load_json(PATHS.state, default={})
    metrics = load_json(PATHS.metrics, default={"history": []})
    return state, metrics


def safe_line_chart(data: pd.DataFrame, *, x: str, y: str | list[str]) -> None:
    try:
        st.line_chart(data, x=x, y=y)
    except TypeError as exc:
        # Python 3.14 + current Altair may fail on TypedDict(closed=...).
        if "_TypedDictMeta.__new__() got an unexpected keyword argument 'closed'" not in str(exc):
            raise

        y_columns = [y] if isinstance(y, str) else list(y)
        fallback_columns = [x] + [col for col in y_columns if col in data.columns]
        st.warning("Charts are temporarily unavailable in this Python/Altair combo. Showing table fallback.")
        st.dataframe(data[fallback_columns] if fallback_columns else data, use_container_width=True)


st.set_page_config(page_title="ML Games Engine", layout="wide")
ensure_defaults()

st.title("ML Games Engine")
st.caption("Core desacoplado + dashboard hot-reload para mini-projetos 2D e 3D.")

scenario_options = list(SCENARIOS.keys())
selected_scenario = st.sidebar.selectbox(
    "Scenario",
    scenario_options,
    format_func=lambda key: SCENARIOS[key].label,
)
scenario = get_scenario(selected_scenario)

st.sidebar.markdown(f"**Dimensao:** {scenario.dimension}")
st.sidebar.markdown(f"**Viewport:** {scenario.viewport}")
st.sidebar.caption(scenario.description)

default_training = dict(scenario.default_training)
default_reward = dict(scenario.default_reward)

with st.sidebar.expander("Training Setup", expanded=True):
    timesteps = st.number_input("Timesteps", min_value=0, value=int(default_training["timesteps"]), step=1000)
    n_envs = st.number_input("Parallel envs", min_value=1, value=int(default_training["n_envs"]), step=1)
    lr = st.number_input("Learning rate", min_value=0.0, value=float(default_training["lr"]), format="%.6f")
    gamma = st.number_input("Gamma", min_value=0.0, max_value=1.0, value=float(default_training["gamma"]), format="%.4f")
    n_steps = st.number_input("Rollout steps", min_value=16, value=int(default_training["n_steps"]), step=16)
    batch_size = st.number_input("Batch size", min_value=8, value=int(default_training["batch_size"]), step=8)
    n_epochs = st.number_input("Epochs", min_value=1, value=int(default_training["n_epochs"]), step=1)
    checkpoint_freq = st.number_input(
        "Checkpoint freq",
        min_value=100,
        value=int(default_training["checkpoint_freq"]),
        step=100,
    )
    net_arch = st.text_input("Net arch", value=str(default_training["net_arch"]))
    model_path = st.text_input("Resume model path", value="")
    vecnorm_path = st.text_input("Resume VecNormalize path", value="")
    seed = st.number_input("Seed", min_value=0, value=int(default_training["seed"]), step=1)

with st.sidebar.expander("Brain Switcher", expanded=True):
    pilot_name = st.selectbox(
        "Viewport brain",
        ["live_policy", "random", "checkpoint_ppo"],
        index=0,
        help="Troca o cerebro usado na visualizacao sem fechar a simulacao.",
    )
    checkpoint_path = st.text_input("Checkpoint path", value="", disabled=(pilot_name != "checkpoint_ppo"))
    checkpoint_vecnorm = st.text_input(
        "Checkpoint VecNormalize",
        value="",
        disabled=(pilot_name != "checkpoint_ppo"),
    )

with st.sidebar.expander("Runtime Controls", expanded=True):
    speed = st.slider("Preview speed", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
    auto_refresh = st.checkbox("Auto refresh", value=True)
    refresh_seconds = st.slider("Refresh interval", min_value=1, max_value=5, value=2, step=1)

reward_values = {}
with st.sidebar.expander("Reward Hot Reload", expanded=True):
    for key, value in default_reward.items():
        reward_values[key] = st.number_input(
            key,
            value=float(value),
            step=max(abs(float(value)) * 0.05, 0.01),
            format="%.4f",
        )

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
    "model_path": model_path or None,
    "vecnorm_path": vecnorm_path or None,
}
pilot_payload = {
    "name": pilot_name,
    "checkpoint_path": checkpoint_path or None,
    "vecnorm_path": checkpoint_vecnorm or None,
}

top_left, top_mid, top_right, top_extra = st.columns(4)

if top_left.button("Start / Restart Run", use_container_width=True):
    st.session_state.reward_version += 1
    st.session_state.pilot_version += 1
    control = build_control_payload(
        selected_scenario,
        training_payload,
        reward_values,
        pilot_payload,
        paused=False,
        speed=speed,
    )
    write_json(PATHS.control, control)
    start_runner()

if top_mid.button("Pause / Resume", use_container_width=True):
    control = load_json(PATHS.control, default={})
    current_runtime = dict(control.get("runtime", {}))
    paused = not bool(current_runtime.get("paused", False))
    control = build_control_payload(
        control.get("scenario_id", selected_scenario),
        control.get("training", training_payload),
        control.get("reward", {}).get("values", reward_values),
        {
            "name": control.get("pilot", {}).get("name", pilot_name),
            "checkpoint_path": control.get("pilot", {}).get("checkpoint_path"),
            "vecnorm_path": control.get("pilot", {}).get("vecnorm_path"),
        },
        paused=paused,
        speed=float(current_runtime.get("speed", speed)),
    )
    write_json(PATHS.control, control)

if top_right.button("Apply Reward Changes", use_container_width=True):
    st.session_state.reward_version += 1
    control = load_json(PATHS.control, default={})
    control = build_control_payload(
        control.get("scenario_id", selected_scenario),
        control.get("training", training_payload),
        reward_values,
        {
            "name": control.get("pilot", {}).get("name", pilot_name),
            "checkpoint_path": control.get("pilot", {}).get("checkpoint_path"),
            "vecnorm_path": control.get("pilot", {}).get("vecnorm_path"),
        },
        paused=bool(control.get("runtime", {}).get("paused", False)),
        speed=float(control.get("runtime", {}).get("speed", speed)),
    )
    write_json(PATHS.control, control)

if top_extra.button("Switch Brain", use_container_width=True):
    st.session_state.pilot_version += 1
    control = load_json(PATHS.control, default={})
    control = build_control_payload(
        control.get("scenario_id", selected_scenario),
        control.get("training", training_payload),
        control.get("reward", {}).get("values", reward_values),
        pilot_payload,
        paused=bool(control.get("runtime", {}).get("paused", False)),
        speed=float(control.get("runtime", {}).get("speed", speed)),
    )
    write_json(PATHS.control, control)

ops_left, ops_mid, ops_right = st.columns(3)

if ops_left.button("Save Checkpoint", use_container_width=True):
    st.session_state.save_version += 1
    control = load_json(PATHS.control, default={})
    current_runtime = dict(control.get("runtime", {}))
    current_runtime["save_version"] = st.session_state.save_version
    control["runtime"] = current_runtime
    write_json(PATHS.control, control)

if ops_mid.button("Stop Run", use_container_width=True):
    control = load_json(PATHS.control, default={})
    control = build_control_payload(
        control.get("scenario_id", selected_scenario),
        control.get("training", training_payload),
        control.get("reward", {}).get("values", reward_values),
        {
            "name": control.get("pilot", {}).get("name", pilot_name),
            "checkpoint_path": control.get("pilot", {}).get("checkpoint_path"),
            "vecnorm_path": control.get("pilot", {}).get("vecnorm_path"),
        },
        paused=True,
        speed=float(control.get("runtime", {}).get("speed", speed)),
        stop_requested=True,
    )
    write_json(PATHS.control, control)

if ops_right.button("Refresh Now", use_container_width=True):
    st.rerun()

state, metrics = load_runtime_state()

status_cols = st.columns(5)
status_cols[0].metric("Status", state.get("status", "idle"))
status_cols[1].metric("Steps", f"{int(state.get('step', 0)):,}")
status_cols[2].metric("Episodes", int(state.get("episodes", 0)))
status_cols[3].metric("Last Reward", f"{float(state.get('last_reward', 0.0)):.2f}")
status_cols[4].metric("Best Reward", f"{float(state.get('best_reward') or 0.0):.2f}")

left, right = st.columns([1.2, 1.0])

history = metrics.get("history", [])
episode_rows = [row for row in history if row.get("kind") == "episode"]
train_rows = [row for row in history if row.get("kind") == "train"]

with left:
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

    st.subheader("Runtime State")
    st.json(state or {"status": "idle"})

with right:
    st.subheader("Viewport")
    if scenario.dimension == "2D" and PATHS.frame.exists():
        st.image(np.load(PATHS.frame), caption="Arena 2D live preview", use_container_width=True)
    elif scenario.dimension == "3D":
        st.info("O viewport 3D abre em uma janela PyBullet separada enquanto o dashboard controla o experimento.")
    else:
        st.info("Nenhum frame disponivel ainda.")

    st.subheader("Events")
    for event in state.get("events", []):
        st.code(event)

if auto_refresh:
    time.sleep(refresh_seconds)
    st.rerun()
