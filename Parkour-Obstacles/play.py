"""Launcher for play mode: puts this project on sys.path and delegates to
src.play.main, which loads a trained PPO agent and runs it in PyBullet.
"""

import sys
import os

# Add src to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from src.play import main

if __name__ == "__main__":
    main()
