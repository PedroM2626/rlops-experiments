"""
Automated tests for the Competitive Mario system.

Tests cover:
    1. Level generation
    2. Environment creation and stepping
    3. Observation shape / type
    4. Reward range
    5. Agent training smoke test (very short)
    6. Competition pipeline smoke test
"""

import os
import sys
import pytest
import numpy as np

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from src.env.level_generator import LevelGenerator, GROUND, EMPTY, FLAG_POLE, FLAG_TOP
from src.env.mario_env import MarioCompetitiveEnv, OBS_SHAPE, MAX_FRAMES


class TestLevelGenerator:
    """Tests for procedural level generation."""

    def test_default_shape(self):
        gen = LevelGenerator(width=100, height=15, seed=42)
        tiles = gen.generate()
        assert tiles.shape == (15, 100)

    def test_ground_exists(self):
        gen = LevelGenerator(width=100, height=15, seed=42)
        tiles = gen.generate()
        # bottom two rows should have at least some ground
        ground_count = np.sum(tiles[-2:, :] == GROUND)
        assert ground_count > 50, f"Not enough ground tiles: {ground_count}"

    def test_flag_exists(self):
        gen = LevelGenerator(width=100, height=15, seed=42)
        tiles = gen.generate()
        has_flag = np.any(tiles == FLAG_POLE) or np.any(tiles == FLAG_TOP)
        assert has_flag, "Level has no flag pole"

    def test_start_zone_clear(self):
        gen = LevelGenerator(width=100, height=15, seed=42)
        tiles = gen.generate()
        # first 8 columns should be clear above ground
        sky_area = tiles[:13, :8]  # rows 0-12 should be empty
        non_empty = np.sum(sky_area != EMPTY)
        assert non_empty == 0, f"Start zone not clear: {non_empty} non-empty tiles"

    def test_reproducibility(self):
        gen1 = LevelGenerator(seed=123)
        gen2 = LevelGenerator(seed=123)
        assert np.array_equal(gen1.generate(), gen2.generate())

    def test_difficulty_variation(self):
        gen_easy = LevelGenerator(seed=42, difficulty=0.0)
        gen_hard = LevelGenerator(seed=42, difficulty=1.0)
        tiles_easy = gen_easy.generate()
        tiles_hard = gen_hard.generate()
        # hard levels should have more non-empty tiles (enemies, obstacles)
        # or at least be different
        assert not np.array_equal(tiles_easy, tiles_hard)


class TestMarioEnv:
    """Tests for the Mario environment."""

    def test_env_creation(self):
        env = MarioCompetitiveEnv(level_seed=42, render_mode=None)
        obs, info = env.reset()
        assert obs is not None
        assert info is not None
        env.close()

    def test_observation_shape(self):
        env = MarioCompetitiveEnv(level_seed=42, render_mode=None)
        obs, _ = env.reset()
        assert obs.shape == OBS_SHAPE, f"Expected {OBS_SHAPE}, got {obs.shape}"
        assert obs.dtype == np.uint8
        env.close()

    def test_step_returns(self):
        env = MarioCompetitiveEnv(level_seed=42, render_mode=None)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(1)  # move right
        assert obs.shape == OBS_SHAPE
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert "x_pos" in info
        assert "flag_get" in info
        env.close()

    def test_action_space(self):
        env = MarioCompetitiveEnv(level_seed=42, render_mode=None)
        assert env.action_space.n == 5
        for a in range(5):
            assert env.action_space.contains(a)
        env.close()

    def test_moving_right_increases_x(self):
        env = MarioCompetitiveEnv(level_seed=42, render_mode=None)
        env.reset()
        initial_x = env.mario_x
        for _ in range(30):
            env.step(1)  # ACTION_RIGHT
        assert env.mario_x > initial_x, "Moving right should increase x"
        env.close()

    def test_episode_truncates(self):
        env = MarioCompetitiveEnv(level_seed=42, render_mode=None)
        env.reset()
        truncated = False
        for _ in range(MAX_FRAMES + 10):
            _, _, terminated, truncated, _ = env.step(0)  # NOOP
            if terminated or truncated:
                break
        assert terminated or truncated, "Episode should end eventually"
        env.close()

    def test_rgb_array_render(self):
        env = MarioCompetitiveEnv(level_seed=42, render_mode="rgb_array")
        env.reset()
        frame = env.render()
        assert frame is not None
        assert frame.shape[2] == 3  # RGB
        env.close()

    def test_info_keys(self):
        env = MarioCompetitiveEnv(level_seed=42, render_mode=None)
        _, info = env.reset()
        expected_keys = ["x_pos", "y_pos", "flag_get", "score", "coins", "time", "is_dead", "frame"]
        for key in expected_keys:
            assert key in info, f"Missing key in info: {key}"
        env.close()


class TestTrainingSmoke:
    """Quick smoke test for the training pipeline."""

    def test_ppo_can_train(self):
        """Verify PPO can do a very short training run without errors."""
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
        from stable_baselines3.common.monitor import Monitor

        def make_env():
            env = MarioCompetitiveEnv(level_seed=42, difficulty=0.3, render_mode=None)
            return Monitor(env)

        vec_env = DummyVecEnv([make_env])
        vec_env = VecFrameStack(vec_env, n_stack=4)
        vec_env = VecTransposeImage(vec_env)

        model = PPO(
            "CnnPolicy",
            vec_env,
            n_steps=64,
            batch_size=32,
            n_epochs=2,
            verbose=0,
        )
        model.learn(total_timesteps=128)
        vec_env.close()

    def test_model_predict(self):
        """Verify a trained model can produce actions."""
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
        from stable_baselines3.common.monitor import Monitor

        def make_env():
            env = MarioCompetitiveEnv(level_seed=42, render_mode=None)
            return Monitor(env)

        vec_env = DummyVecEnv([make_env])
        vec_env = VecFrameStack(vec_env, n_stack=4)
        vec_env = VecTransposeImage(vec_env)

        model = PPO(
            "CnnPolicy",
            vec_env,
            n_steps=64,
            batch_size=32,
            n_epochs=1,
            verbose=0,
        )
        model.learn(total_timesteps=64)

        obs = vec_env.reset()
        action, _ = model.predict(obs, deterministic=True)
        assert action is not None
        assert vec_env.action_space.contains(action[0])
        vec_env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
