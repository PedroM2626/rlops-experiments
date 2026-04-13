"""
Custom Mario-style competitive gymnasium environment.

A pure-Python side-scrolling platformer environment for reinforcement learning.
Each agent controls a Mario character and must navigate a procedurally generated
level as fast as possible. The environment renders frames as RGB images suitable
for CNN-based policies.

Observation : (84, 84, 3) uint8 RGB image (resized viewport around the player)
Action space: Discrete(5) -- [NOOP, RIGHT, RIGHT+JUMP, JUMP, LEFT]
"""

import math
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .level_generator import (
    LevelGenerator,
    EMPTY, GROUND, BRICK, QUESTION, PIPE_BOTTOM, PIPE_TOP,
    ENEMY, COIN, FLAG_POLE, FLAG_TOP,
)

# -----------------------------------------------------------------------
# Physics constants (in pixels)
# -----------------------------------------------------------------------
TILE_SIZE = 16
GRAVITY = 0.7
JUMP_FORCE = -10.0
MOVE_SPEED = 3.0
MAX_FALL_SPEED = 12.0
FRICTION = 0.85

# Viewport (pixels before resizing)
VIEWPORT_W = 256
VIEWPORT_H = 240

# Output observation shape
OBS_WIDTH = 84
OBS_HEIGHT = 84
OBS_SHAPE = (OBS_HEIGHT, OBS_WIDTH, 3)

# Time limit per episode (in frames at ~60 fps)
MAX_FRAMES = 3000

# -----------------------------------------------------------------------
# Color palette (RGB)
# -----------------------------------------------------------------------
SKY_COLOR = np.array([92, 148, 252], dtype=np.uint8)
GROUND_COLOR = np.array([160, 88, 32], dtype=np.uint8)
BRICK_COLOR = np.array([200, 120, 60], dtype=np.uint8)
QUESTION_COLOR = np.array([252, 200, 44], dtype=np.uint8)
PIPE_COLOR = np.array([0, 168, 0], dtype=np.uint8)
PIPE_TOP_COLOR = np.array([0, 200, 0], dtype=np.uint8)
ENEMY_COLOR = np.array([180, 80, 60], dtype=np.uint8)
COIN_COLOR = np.array([252, 216, 68], dtype=np.uint8)
FLAG_COLOR = np.array([0, 200, 0], dtype=np.uint8)
FLAG_TOP_COLOR = np.array([200, 0, 0], dtype=np.uint8)
MARIO_COLOR = np.array([228, 52, 28], dtype=np.uint8)  # red
MARIO_SKIN = np.array([252, 188, 116], dtype=np.uint8)

TILE_COLORS = {
    EMPTY: SKY_COLOR,
    GROUND: GROUND_COLOR,
    BRICK: BRICK_COLOR,
    QUESTION: QUESTION_COLOR,
    PIPE_BOTTOM: PIPE_COLOR,
    PIPE_TOP: PIPE_TOP_COLOR,
    ENEMY: ENEMY_COLOR,
    COIN: COIN_COLOR,
    FLAG_POLE: FLAG_COLOR,
    FLAG_TOP: FLAG_TOP_COLOR,
}

# Action definitions
ACTION_NOOP = 0
ACTION_RIGHT = 1
ACTION_RIGHT_JUMP = 2
ACTION_JUMP = 3
ACTION_LEFT = 4


class MarioCompetitiveEnv(gym.Env):
    """
    Mario-style side-scrolling platformer environment for RL.

    Parameters
    ----------
    level_seed : int or None
        Seed for procedural level generation. If None, random.
    difficulty : float
        Level difficulty (0.0 to 1.0).
    level_width : int
        Level width in tiles.
    render_mode : str or None
        'human', 'rgb_array', or None.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        level_seed: int = None,
        difficulty: float = 0.5,
        level_width: int = 200,
        render_mode: str = None,
    ):
        super().__init__()

        self.level_seed = level_seed
        self.difficulty = difficulty
        self.level_width = level_width
        self.render_mode = render_mode

        # spaces
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=OBS_SHAPE, dtype=np.uint8
        )

        # pygame init (lazy, only when render_mode == 'human')
        self._pygame_screen = None
        self._pygame_clock = None

        # state placeholders
        self.tiles = None
        self.mario_x = 0.0
        self.mario_y = 0.0
        self.mario_vx = 0.0
        self.mario_vy = 0.0
        self.on_ground = False
        self.frame_count = 0
        self.score = 0
        self.coins_collected = 0
        self.flag_reached = False
        self.is_dead = False
        self._x_position_last = 0.0
        self._time_last = MAX_FRAMES

        # enemies are stored as a list of [x, y, vx, alive]
        self.enemies = []

        # collected coins tracking (set of (row, col) tuples)
        self._collected_coins = set()
        self._hit_questions = set()

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """Reset the environment and return the initial observation."""
        super().reset(seed=seed)

        # generate level
        level_seed = self.level_seed
        if level_seed is None:
            level_seed = self.np_random.integers(0, 2**31)
        gen = LevelGenerator(
            width=self.level_width,
            height=15,
            seed=int(level_seed),
            difficulty=self.difficulty,
        )
        self.tiles = gen.generate()

        # place Mario at the start
        ground_row = self.tiles.shape[0] - 2  # two rows of ground
        self.mario_x = 3.0 * TILE_SIZE
        self.mario_y = float((ground_row - 1) * TILE_SIZE)
        self.mario_vx = 0.0
        self.mario_vy = 0.0
        self.on_ground = True

        self.frame_count = 0
        self.score = 0
        self.coins_collected = 0
        self.flag_reached = False
        self.is_dead = False
        self._x_position_last = self.mario_x
        self._time_last = MAX_FRAMES
        self._collected_coins = set()
        self._hit_questions = set()

        # spawn enemies
        self._spawn_enemies()

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action):
        """Execute one environment step."""
        assert self.action_space.contains(action), f"Invalid action: {action}"

        if self.is_dead or self.flag_reached:
            # episode already over, return terminal obs
            obs = self._get_observation()
            return obs, 0.0, True, False, self._get_info()

        # apply action
        self._apply_action(action)

        # update physics
        self._update_physics()

        # update enemies
        self._update_enemies()

        # check collisions
        self._check_collisions()

        self.frame_count += 1

        # compute reward
        reward = self._compute_reward()

        # check termination
        terminated = self.is_dead or self.flag_reached
        truncated = self.frame_count >= MAX_FRAMES

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the current frame."""
        frame = self._render_frame()
        if self.render_mode == "human":
            self._render_human(frame)
        return frame if self.render_mode == "rgb_array" else None

    def close(self):
        """Clean up resources."""
        if self._pygame_screen is not None:
            import pygame
            pygame.quit()
            self._pygame_screen = None

    # ------------------------------------------------------------------
    # Action handling
    # ------------------------------------------------------------------

    def _apply_action(self, action):
        """Translate discrete action into velocity changes."""
        if action == ACTION_RIGHT:
            self.mario_vx = MOVE_SPEED
        elif action == ACTION_RIGHT_JUMP:
            self.mario_vx = MOVE_SPEED
            if self.on_ground:
                self.mario_vy = JUMP_FORCE
                self.on_ground = False
        elif action == ACTION_JUMP:
            if self.on_ground:
                self.mario_vy = JUMP_FORCE
                self.on_ground = False
        elif action == ACTION_LEFT:
            self.mario_vx = -MOVE_SPEED
        # ACTION_NOOP: do nothing

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def _update_physics(self):
        """Update Mario's position and velocity."""
        # gravity
        self.mario_vy += GRAVITY
        self.mario_vy = min(self.mario_vy, MAX_FALL_SPEED)

        # horizontal movement with friction
        new_x = self.mario_x + self.mario_vx
        new_y = self.mario_y + self.mario_vy

        # clamp to level bounds
        new_x = max(0, new_x)
        max_x = (self.tiles.shape[1] - 1) * TILE_SIZE
        new_x = min(new_x, max_x)

        # collision with level geometry
        new_x, new_y = self._resolve_collisions(new_x, new_y)

        self.mario_x = new_x
        self.mario_y = new_y

        # friction
        self.mario_vx *= FRICTION

        # fall-death check (below ground)
        if self.mario_y > self.tiles.shape[0] * TILE_SIZE:
            self.is_dead = True

    def _resolve_collisions(self, new_x, new_y):
        """Resolve collisions between Mario and solid tiles."""
        mario_w = TILE_SIZE - 2
        mario_h = TILE_SIZE - 1

        # horizontal collision
        rect_x = new_x
        rect_y = self.mario_y
        if self._check_solid_overlap(rect_x, rect_y, mario_w, mario_h):
            new_x = self.mario_x
            self.mario_vx = 0

        # vertical collision
        rect_x = new_x
        rect_y = new_y
        if self._check_solid_overlap(rect_x, rect_y, mario_w, mario_h):
            if self.mario_vy > 0:
                # landing on ground
                tile_row = int((new_y + mario_h) / TILE_SIZE)
                new_y = (tile_row * TILE_SIZE) - mario_h - 1
                self.mario_vy = 0
                self.on_ground = True
            elif self.mario_vy < 0:
                # head bump
                tile_row = int(new_y / TILE_SIZE)
                new_y = (tile_row + 1) * TILE_SIZE
                self.mario_vy = 0
                # check if we hit a question block
                self._hit_block_above(new_x, new_y, mario_w)
        else:
            if self.mario_vy > 0:
                self.on_ground = False

        return new_x, new_y

    def _check_solid_overlap(self, x, y, w, h):
        """Check if a rectangle overlaps any solid tile."""
        solid_tiles = {GROUND, BRICK, PIPE_BOTTOM, PIPE_TOP}

        col_start = max(0, int(x / TILE_SIZE))
        col_end = min(self.tiles.shape[1] - 1, int((x + w) / TILE_SIZE))
        row_start = max(0, int(y / TILE_SIZE))
        row_end = min(self.tiles.shape[0] - 1, int((y + h) / TILE_SIZE))

        for row in range(row_start, row_end + 1):
            for col in range(col_start, col_end + 1):
                tile = self.tiles[row, col]
                # question blocks that haven't been hit are also solid
                if tile in solid_tiles or (tile == QUESTION and (row, col) not in self._hit_questions):
                    return True
        return False

    def _hit_block_above(self, x, y, w):
        """When Mario bumps a block from below, handle it."""
        col_start = max(0, int(x / TILE_SIZE))
        col_end = min(self.tiles.shape[1] - 1, int((x + w) / TILE_SIZE))
        row = max(0, int(y / TILE_SIZE) - 1)

        for col in range(col_start, col_end + 1):
            if 0 <= row < self.tiles.shape[0]:
                tile = self.tiles[row, col]
                if tile == QUESTION and (row, col) not in self._hit_questions:
                    self._hit_questions.add((row, col))
                    self.coins_collected += 1
                    self.score += 100

    # ------------------------------------------------------------------
    # Enemies
    # ------------------------------------------------------------------

    def _spawn_enemies(self):
        """Spawn enemies from the level tiles."""
        self.enemies = []
        for row in range(self.tiles.shape[0]):
            for col in range(self.tiles.shape[1]):
                if self.tiles[row, col] == ENEMY:
                    ex = float(col * TILE_SIZE)
                    ey = float(row * TILE_SIZE)
                    vx = -1.0  # enemies walk left
                    self.enemies.append([ex, ey, vx, True])
                    self.tiles[row, col] = EMPTY  # remove from tilemap

    def _update_enemies(self):
        """Move enemies and handle basic collision with ground."""
        for enemy in self.enemies:
            if not enemy[3]:  # not alive
                continue
            ex, ey, vx = enemy[0], enemy[1], enemy[2]
            ex += vx

            # reverse direction at walls / edges
            col = int(ex / TILE_SIZE)
            row = int(ey / TILE_SIZE)
            ground_row = row + 1
            if (
                col < 0
                or col >= self.tiles.shape[1]
                or (ground_row < self.tiles.shape[0] and self.tiles[ground_row, col] == EMPTY)
            ):
                vx = -vx

            # check wall collision
            next_col = int((ex + (TILE_SIZE if vx > 0 else 0)) / TILE_SIZE)
            if 0 <= next_col < self.tiles.shape[1] and 0 <= row < self.tiles.shape[0]:
                if self.tiles[row, next_col] in {GROUND, BRICK, PIPE_BOTTOM, PIPE_TOP}:
                    vx = -vx

            enemy[0] = ex
            enemy[2] = vx

    def _check_collisions(self):
        """Check collision between Mario and enemies / coins / flag."""
        mario_w = TILE_SIZE - 2
        mario_h = TILE_SIZE - 1

        # enemy collision
        for enemy in self.enemies:
            if not enemy[3]:
                continue
            ex, ey = enemy[0], enemy[1]
            if self._rects_overlap(
                self.mario_x, self.mario_y, mario_w, mario_h,
                ex + 2, ey + 2, TILE_SIZE - 4, TILE_SIZE - 4,
            ):
                # landing on top of enemy: kill it
                if self.mario_vy > 0 and self.mario_y + mario_h < ey + TILE_SIZE // 2:
                    enemy[3] = False
                    self.mario_vy = JUMP_FORCE * 0.6
                    self.score += 200
                else:
                    self.is_dead = True
                    return

        # coin / question collection
        col_start = max(0, int(self.mario_x / TILE_SIZE))
        col_end = min(
            self.tiles.shape[1] - 1,
            int((self.mario_x + mario_w) / TILE_SIZE),
        )
        row_start = max(0, int(self.mario_y / TILE_SIZE))
        row_end = min(
            self.tiles.shape[0] - 1,
            int((self.mario_y + mario_h) / TILE_SIZE),
        )
        for row in range(row_start, row_end + 1):
            for col in range(col_start, col_end + 1):
                tile = self.tiles[row, col]
                if tile == COIN and (row, col) not in self._collected_coins:
                    self._collected_coins.add((row, col))
                    self.coins_collected += 1
                    self.score += 50
                elif tile in (FLAG_POLE, FLAG_TOP):
                    self.flag_reached = True
                    self.score += 1000

    @staticmethod
    def _rects_overlap(x1, y1, w1, h1, x2, y2, w2, h2):
        """Check if two axis-aligned rectangles overlap."""
        return x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self):
        """
        Reward function:
          - x_progress: reward for moving right
          - time_penalty: small penalty per frame
          - death_penalty: large penalty for dying
          - flag_bonus: large bonus for reaching the flag
          - coin_bonus: small bonus for collecting coins (already in score)
        """
        x_progress = (self.mario_x - self._x_position_last) / TILE_SIZE
        self._x_position_last = self.mario_x

        # clamp outlier jumps (e.g. after reset)
        if abs(x_progress) > 5:
            x_progress = 0

        time_penalty = -0.01
        death_penalty = -15.0 if self.is_dead else 0.0
        flag_bonus = 50.0 if self.flag_reached else 0.0

        reward = x_progress + time_penalty + death_penalty + flag_bonus
        return float(np.clip(reward, -15, 50))

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_observation(self):
        """Render the viewport and resize to the observation shape."""
        frame = self._render_frame()
        # resize to OBS_SHAPE
        obs = self._resize_frame(frame, OBS_WIDTH, OBS_HEIGHT)
        return obs

    def _render_frame(self):
        """Render the full viewport as an RGB numpy array."""
        frame = np.full((VIEWPORT_H, VIEWPORT_W, 3), SKY_COLOR, dtype=np.uint8)

        # camera follows Mario
        cam_x = int(self.mario_x - VIEWPORT_W // 3)
        cam_x = max(0, cam_x)
        max_cam = self.tiles.shape[1] * TILE_SIZE - VIEWPORT_W
        cam_x = min(cam_x, max(0, max_cam))

        cam_y = 0  # fixed vertical camera

        # draw tiles
        col_start = max(0, cam_x // TILE_SIZE)
        col_end = min(self.tiles.shape[1], (cam_x + VIEWPORT_W) // TILE_SIZE + 1)
        row_start = max(0, cam_y // TILE_SIZE)
        row_end = min(self.tiles.shape[0], (cam_y + VIEWPORT_H) // TILE_SIZE + 1)

        for row in range(row_start, row_end):
            for col in range(col_start, col_end):
                tile = self.tiles[row, col]
                if tile == EMPTY:
                    continue
                if tile == COIN and (row, col) in self._collected_coins:
                    continue
                if tile == QUESTION and (row, col) in self._hit_questions:
                    tile = BRICK  # show as used brick

                px = col * TILE_SIZE - cam_x
                py = row * TILE_SIZE - cam_y

                color = TILE_COLORS.get(tile, SKY_COLOR)
                self._draw_rect(frame, px, py, TILE_SIZE, TILE_SIZE, color)

        # draw enemies
        for enemy in self.enemies:
            if not enemy[3]:
                continue
            ex = int(enemy[0]) - cam_x
            ey = int(enemy[1]) - cam_y
            self._draw_rect(frame, ex, ey, TILE_SIZE, TILE_SIZE, ENEMY_COLOR)
            # eyes
            self._draw_rect(frame, ex + 3, ey + 3, 3, 3, np.array([255, 255, 255], dtype=np.uint8))
            self._draw_rect(frame, ex + 10, ey + 3, 3, 3, np.array([255, 255, 255], dtype=np.uint8))

        # draw Mario
        mx = int(self.mario_x) - cam_x
        my = int(self.mario_y) - cam_y
        # body
        self._draw_rect(frame, mx + 2, my + 4, TILE_SIZE - 4, TILE_SIZE - 4, MARIO_COLOR)
        # head / skin
        self._draw_rect(frame, mx + 4, my, TILE_SIZE - 8, 6, MARIO_SKIN)
        # hat
        self._draw_rect(frame, mx + 2, my - 2, TILE_SIZE - 2, 3, MARIO_COLOR)

        return frame

    @staticmethod
    def _draw_rect(frame, x, y, w, h, color):
        """Draw a filled rectangle onto the frame array."""
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(frame.shape[1], int(x + w))
        y2 = min(frame.shape[0], int(y + h))
        if x1 < x2 and y1 < y2:
            frame[y1:y2, x1:x2] = color

    @staticmethod
    def _resize_frame(frame, target_w, target_h):
        """Simple nearest-neighbor resize using NumPy (no OpenCV dependency)."""
        src_h, src_w = frame.shape[:2]
        row_indices = (np.arange(target_h) * src_h / target_h).astype(int)
        col_indices = (np.arange(target_w) * src_w / target_w).astype(int)
        row_indices = np.clip(row_indices, 0, src_h - 1)
        col_indices = np.clip(col_indices, 0, src_w - 1)
        return frame[row_indices[:, None], col_indices[None, :]]

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def _get_info(self):
        """Return diagnostic info dictionary."""
        time_left = max(0, MAX_FRAMES - self.frame_count)
        return {
            "x_pos": int(self.mario_x),
            "y_pos": int(self.mario_y),
            "flag_get": self.flag_reached,
            "score": self.score,
            "coins": self.coins_collected,
            "time": time_left,
            "is_dead": self.is_dead,
            "frame": self.frame_count,
        }

    # ------------------------------------------------------------------
    # Human rendering (Pygame)
    # ------------------------------------------------------------------

    def _render_human(self, frame):
        """Render frame to a Pygame window."""
        try:
            import pygame
        except ImportError:
            raise ImportError("pygame or pygame-ce is required for human rendering")

        if self._pygame_screen is None:
            pygame.init()
            pygame.display.set_caption("Competitive Mario")
            self._pygame_screen = pygame.display.set_mode((VIEWPORT_W * 2, VIEWPORT_H * 2))
            self._pygame_clock = pygame.time.Clock()

        # scale frame 2x
        surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        surface = pygame.transform.scale(surface, (VIEWPORT_W * 2, VIEWPORT_H * 2))
        self._pygame_screen.blit(surface, (0, 0))
        pygame.display.flip()
        self._pygame_clock.tick(self.metadata["render_fps"])

        # handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
