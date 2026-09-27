"""Utilities shared by every training/evaluation script in this study.

Covers: seeding, env construction, GAE, rolling-mean logging, meta.json,
curve CSV I/O, and discrete-policy helpers for the torch arms.
"""

from __future__ import annotations

import csv
import json
import platform
import random
from collections import deque
from pathlib import Path

import numpy as np


def set_all_seeds(seed: int, torch=None) -> None:
    """Seed Python, NumPy and (optionally) PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass


def make_env(env_id: str, seed: int):
    """Create a single gymnasium env with episode statistics.

    We deliberately use ``RecordEpisodeStatistics`` (not SB3's Monitor) in
    the manual arms so episode returns come from the same wrapper family.
    The SB3 arm uses SB3's own Monitor/VecEnv stack (documented confound,
    part of the "library" factor) but logs the identical rolling metric.
    """
    import gymnasium as gym

    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    obs, _ = env.reset(seed=seed)
    return env


def obs_act_dims(env) -> tuple[int, int, bool]:
    """Return (obs_dim, act_dim, is_continuous) for an env."""
    import gymnasium as gym

    obs_dim = int(np.prod(env.observation_space.shape))
    is_cont = isinstance(env.action_space, gym.spaces.Box)
    act_dim = int(env.action_space.shape[0]) if is_cont else int(env.action_space.n)
    return obs_dim, act_dim, is_cont


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    last_done: bool,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalised Advantage Estimation (single-env, NumPy).

    Args:
        rewards: (T,) step rewards (already timeout-bootstrapped if SB3 spec).
        values:  (T,) value estimates V(s_t).
        dones:   (T,) 1.0 where the episode ENDED (terminated OR truncated).
        last_value: V(s_T) of the obs following the rollout.
        last_done: whether the rollout ended in a terminal state.
    Returns:
        (advantages, returns) with returns = advantages + values.
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float64)
    last_gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            next_non_term = 1.0 - float(last_done)
            next_val = float(last_value)
        else:
            next_non_term = 1.0 - float(dones[t + 1])
            next_val = float(values[t + 1])
        delta = float(rewards[t]) + gamma * next_val * next_non_term - float(values[t])
        last_gae = delta + gamma * gae_lambda * next_non_term * last_gae
        adv[t] = last_gae
    ret = adv + values.astype(np.float64)
    return adv.astype(np.float32), ret.astype(np.float32)


class RollingMean:
    """Rolling mean over the last `window` episode returns."""

    def __init__(self, window: int = 20) -> None:
        self.buf: deque[float] = deque(maxlen=window)

    def add(self, r: float) -> None:
        self.buf.append(float(r))

    def mean(self) -> float:
        return float(np.mean(self.buf)) if self.buf else 0.0

    def __len__(self) -> int:
        return len(self.buf)


def write_curve_csv(path: Path, rows: list[tuple[int, float]]) -> None:
    """Writes a learning curve as ``step,mean_reward`` CSV, creating parent dirs and truncating
    any file already there."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "mean_reward"])
        w.writerows(rows)


def read_curve_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Counterpart of ``write_curve_csv``: looks the two columns up by header name, so column
    order in the file is irrelevant. Returns ``(steps, mean_rewards)``."""
    steps, rews = [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            rews.append(float(row["mean_reward"]))
    return np.asarray(steps), np.asarray(rews)


def write_meta_json(path: Path, payload: dict) -> None:
    """Writes run metadata as sorted, indented JSON, defaulting only the ``platform`` and
    ``python`` keys; the caller's dict is copied, never mutated."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("platform", platform.platform())
    payload.setdefault("python", platform.python_version())
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def package_versions(names: list[str]) -> dict[str, str]:
    """Best-effort installed versions (never raises)."""
    out: dict[str, str] = {}
    try:
        from importlib import metadata as md
    except ImportError:
        import importlib_metadata as md  # type: ignore
    for n in names:
        try:
            out[n] = md.version(n)
        except Exception:
            out[n] = "unknown"
    return out
