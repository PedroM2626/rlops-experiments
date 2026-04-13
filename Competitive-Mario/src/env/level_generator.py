"""
Procedural level generator for the Mario competitive environment.

Generates tile-based side-scrolling levels with platforms, gaps, pipes,
enemies, coins, and a flag pole at the end. Each tile is represented as
an integer code in a 2D NumPy array.

Tile codes:
    0 = empty (sky)
    1 = ground
    2 = brick
    3 = question block (coin)
    4 = pipe (bottom)
    5 = pipe (top)
    6 = enemy (goomba)
    7 = coin (floating)
    8 = flag pole
    9 = flag top
"""

import numpy as np


# Tile constants
EMPTY = 0
GROUND = 1
BRICK = 2
QUESTION = 3
PIPE_BOTTOM = 4
PIPE_TOP = 5
ENEMY = 6
COIN = 7
FLAG_POLE = 8
FLAG_TOP = 9

# Default level dimensions (in tiles)
DEFAULT_WIDTH = 200
DEFAULT_HEIGHT = 15
GROUND_HEIGHT = 2  # number of rows of ground at the bottom


class LevelGenerator:
    """
    Procedural generator for Mario-style side-scrolling levels.

    Parameters
    ----------
    width : int
        Level width in tiles.
    height : int
        Level height in tiles.
    seed : int or None
        Random seed for reproducibility.
    difficulty : float
        Difficulty multiplier (0.0 to 1.0). Higher values produce more
        gaps, enemies, and tighter platforming.
    """

    def __init__(
        self,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        seed: int = None,
        difficulty: float = 0.5,
    ):
        self.width = width
        self.height = height
        self.difficulty = np.clip(difficulty, 0.0, 1.0)
        self.rng = np.random.RandomState(seed)

    def generate(self) -> np.ndarray:
        """
        Generate a complete level as a 2D array of tile codes.

        Returns
        -------
        np.ndarray
            Shape (height, width) with integer tile codes.
        """
        tiles = np.zeros((self.height, self.width), dtype=np.int32)

        # lay the ground
        self._place_ground(tiles)

        # add gaps
        self._place_gaps(tiles)

        # add pipes
        self._place_pipes(tiles)

        # add floating brick / question platforms
        self._place_platforms(tiles)

        # add enemies on ground
        self._place_enemies(tiles)

        # add floating coins
        self._place_coins(tiles)

        # add flag pole at the end
        self._place_flag(tiles)

        # ensure starting area is safe (first 8 columns)
        self._clear_start_zone(tiles)

        return tiles

    # ------------------------------------------------------------------
    # Internal generators
    # ------------------------------------------------------------------

    def _place_ground(self, tiles: np.ndarray) -> None:
        """Fill the bottom rows with ground tiles."""
        for row in range(self.height - GROUND_HEIGHT, self.height):
            tiles[row, :] = GROUND

    def _place_gaps(self, tiles: np.ndarray) -> None:
        """Cut gaps in the ground. Wider / more frequent with difficulty."""
        gap_chance = 0.04 + 0.06 * self.difficulty
        col = 12  # start after the safe zone
        while col < self.width - 15:
            if self.rng.random() < gap_chance:
                gap_width = self.rng.randint(2, 4 + int(2 * self.difficulty))
                gap_width = min(gap_width, self.width - 15 - col)
                for row in range(self.height - GROUND_HEIGHT, self.height):
                    tiles[row, col: col + gap_width] = EMPTY
                col += gap_width + 4  # minimum spacing after a gap
            else:
                col += 1

    def _place_pipes(self, tiles: np.ndarray) -> None:
        """Place vertical pipes of varying heights."""
        pipe_chance = 0.02 + 0.02 * self.difficulty
        col = 15
        while col < self.width - 15:
            if self.rng.random() < pipe_chance:
                # only place pipes on solid ground
                ground_row = self.height - GROUND_HEIGHT
                if tiles[ground_row, col] == GROUND and tiles[ground_row, col + 1] == GROUND:
                    pipe_h = self.rng.randint(2, 5)
                    for row in range(ground_row - pipe_h, ground_row):
                        tiles[row, col] = PIPE_BOTTOM
                        tiles[row, col + 1] = PIPE_BOTTOM
                    tiles[ground_row - pipe_h, col] = PIPE_TOP
                    tiles[ground_row - pipe_h, col + 1] = PIPE_TOP
                    col += 6
                    continue
            col += 1

    def _place_platforms(self, tiles: np.ndarray) -> None:
        """Place floating brick and question-block platforms."""
        platform_chance = 0.03 + 0.03 * self.difficulty
        col = 10
        while col < self.width - 15:
            if self.rng.random() < platform_chance:
                plat_len = self.rng.randint(2, 6)
                plat_row = self.rng.randint(
                    self.height - GROUND_HEIGHT - 7,
                    self.height - GROUND_HEIGHT - 3,
                )
                plat_row = max(2, plat_row)
                for c in range(col, min(col + plat_len, self.width - 1)):
                    if self.rng.random() < 0.3:
                        tiles[plat_row, c] = QUESTION
                    else:
                        tiles[plat_row, c] = BRICK
                col += plat_len + 3
            else:
                col += 1

    def _place_enemies(self, tiles: np.ndarray) -> None:
        """Place enemies on the ground surface."""
        enemy_chance = 0.02 + 0.04 * self.difficulty
        ground_row = self.height - GROUND_HEIGHT - 1
        for col in range(12, self.width - 15):
            if (
                self.rng.random() < enemy_chance
                and tiles[ground_row + 1, col] == GROUND
                and tiles[ground_row, col] == EMPTY
            ):
                tiles[ground_row, col] = ENEMY

    def _place_coins(self, tiles: np.ndarray) -> None:
        """Place floating coins in mid-air."""
        coin_chance = 0.015
        for col in range(10, self.width - 15):
            if self.rng.random() < coin_chance:
                coin_row = self.rng.randint(
                    self.height - GROUND_HEIGHT - 8,
                    self.height - GROUND_HEIGHT - 3,
                )
                coin_row = max(1, coin_row)
                if tiles[coin_row, col] == EMPTY:
                    tiles[coin_row, col] = COIN

    def _place_flag(self, tiles: np.ndarray) -> None:
        """Place a flag pole near the end of the level."""
        flag_col = self.width - 5
        ground_row = self.height - GROUND_HEIGHT
        # ensure ground exists under the flag
        for row in range(ground_row, self.height):
            tiles[row, flag_col] = GROUND
        # place pole
        pole_top = ground_row - 9
        pole_top = max(1, pole_top)
        for row in range(pole_top, ground_row):
            tiles[row, flag_col] = FLAG_POLE
        tiles[pole_top, flag_col] = FLAG_TOP

    def _clear_start_zone(self, tiles: np.ndarray) -> None:
        """Ensure the first 8 columns are flat ground with nothing above."""
        for col in range(8):
            for row in range(self.height - GROUND_HEIGHT):
                tiles[row, col] = EMPTY
            for row in range(self.height - GROUND_HEIGHT, self.height):
                tiles[row, col] = GROUND
