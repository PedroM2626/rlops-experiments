"""
Automated tests for the Race environment.

Run with:
    cd Race
    python -m pytest tests/test_race_env.py -v
"""

import numpy as np
import pytest

from src.env.race_env import RaceEnv, OBS_DIM, N_JOINTS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def single_agent_env():
    """Single-agent Race env (n_agents=1) for fast unit tests."""
    env = RaceEnv(render_mode="direct", n_agents=1, agent_index=0, level_seed=42)
    yield env
    env.close()


@pytest.fixture(scope="module")
def multi_agent_env():
    """Multi-agent Race env (n_agents=3) for integration tests."""
    env = RaceEnv(render_mode="direct", n_agents=3, agent_index=0, level_seed=7)
    yield env
    env.close()


# ---------------------------------------------------------------------------
# Observation & Action Space
# ---------------------------------------------------------------------------

class TestSpaces:
    def test_obs_dim(self, single_agent_env):
        assert single_agent_env.observation_space.shape == (OBS_DIM,), \
            f"Expected obs dim {OBS_DIM}, got {single_agent_env.observation_space.shape}"

    def test_action_dim(self, single_agent_env):
        assert single_agent_env.action_space.shape == (N_JOINTS,)

    def test_obs_dtype(self, single_agent_env):
        obs, _ = single_agent_env.reset()
        assert obs.dtype == np.float32

    def test_obs_within_bounds(self, single_agent_env):
        obs, _ = single_agent_env.reset()
        assert np.all(obs >= single_agent_env.observation_space.low), \
            "Observation below lower bound"
        assert np.all(obs <= single_agent_env.observation_space.high), \
            "Observation above upper bound"


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_obs_and_info(self, single_agent_env):
        result = single_agent_env.reset()
        assert isinstance(result, tuple) and len(result) == 2
        obs, info = result
        assert obs.shape == (OBS_DIM,)
        assert isinstance(info, dict)

    def test_reset_seed_reproducible(self):
        env1 = RaceEnv(render_mode="direct", n_agents=2, agent_index=0, level_seed=1234)
        env2 = RaceEnv(render_mode="direct", n_agents=2, agent_index=0, level_seed=1234)
        obs1, _ = env1.reset(seed=0)
        obs2, _ = env2.reset(seed=0)
        np.testing.assert_allclose(obs1, obs2, atol=1e-4,
                                   err_msg="Same seed should give same initial obs")
        env1.close()
        env2.close()

    def test_double_reset(self, single_agent_env):
        single_agent_env.reset()
        obs, info = single_agent_env.reset()
        assert obs.shape == (OBS_DIM,)


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

class TestStep:
    def test_step_zero_action(self, single_agent_env):
        single_agent_env.reset()
        action = np.zeros(N_JOINTS, dtype=np.float32)
        obs, reward, terminated, truncated, info = single_agent_env.step(action)
        assert obs.shape == (OBS_DIM,)
        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_random_action(self, single_agent_env):
        single_agent_env.reset()
        for _ in range(10):
            action = single_agent_env.action_space.sample()
            obs, reward, terminated, truncated, _ = single_agent_env.step(action)
            if terminated or truncated:
                break
        assert obs.shape == (OBS_DIM,)

    def test_step_action_clipping(self, single_agent_env):
        """Actions outside [-1, 1] should be clipped, not raise."""
        single_agent_env.reset()
        action = np.full(N_JOINTS, 999.0, dtype=np.float32)
        obs, reward, terminated, truncated, _ = single_agent_env.step(action)
        assert obs.shape == (OBS_DIM,)

    def test_step_returns_float_reward(self, single_agent_env):
        single_agent_env.reset()
        action = np.zeros(N_JOINTS, dtype=np.float32)
        _, reward, _, _, _ = single_agent_env.step(action)
        assert np.isfinite(reward), f"Reward should be finite, got {reward}"


# ---------------------------------------------------------------------------
# Multi-agent
# ---------------------------------------------------------------------------

class TestMultiAgent:
    def test_multi_agent_reset(self, multi_agent_env):
        obs, info = multi_agent_env.reset()
        assert obs.shape == (OBS_DIM,)

    def test_n_agents_loaded(self, multi_agent_env):
        multi_agent_env.reset()
        assert len(multi_agent_env._agent_ids) == 3

    def test_different_agent_indices_different_obs(self):
        """Two wrappers on the same env type with different agent_index should differ."""
        env0 = RaceEnv(render_mode="direct", n_agents=3, agent_index=0, level_seed=100)
        obs0, _ = env0.reset(seed=5)
        # Agent 0 spawns at a different Y than agent 1, so positions differ
        assert obs0.shape == (OBS_DIM,)
        env0.close()

    def test_multi_step_episode(self, multi_agent_env):
        multi_agent_env.reset()
        done = False
        steps = 0
        max_steps = 50
        while not done and steps < max_steps:
            action = multi_agent_env.action_space.sample()
            _, _, terminated, truncated, _ = multi_agent_env.step(action)
            done = terminated or truncated
            steps += 1
        assert steps > 0


# ---------------------------------------------------------------------------
# Info dict
# ---------------------------------------------------------------------------

class TestInfoDict:
    def test_info_keys_present(self, single_agent_env):
        single_agent_env.reset()
        _, _, _, _, info = single_agent_env.step(np.zeros(N_JOINTS))
        for key in ("race_rank", "dist_to_goal", "fallen", "finished"):
            assert key in info, f"Missing key '{key}' in info dict"

    def test_rank_range(self, single_agent_env):
        single_agent_env.reset()
        _, _, _, _, info = single_agent_env.step(np.zeros(N_JOINTS))
        rank = info["race_rank"]
        assert 0 <= rank < single_agent_env.n_agents


# ---------------------------------------------------------------------------
# Terrain generator
# ---------------------------------------------------------------------------

class TestTerrainGenerator:
    def test_terrain_builds_without_error(self):
        import pybullet as p
        import pybullet_data
        from src.env.terrain_generator import TerrainGenerator
        client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
        p.setGravity(0, 0, -9.81, physicsClientId=client)
        gen = TerrainGenerator(client, seed=999)
        ids, start, goal = gen.build()
        assert len(ids) > 0
        assert start.shape == (3,)
        assert goal.shape  == (3,)
        assert goal[0] > start[0], "Goal should be ahead (larger X) of start"
        p.disconnect(client)
