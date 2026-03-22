"""Entry point wrapper - ensures src is importable from project root."""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from src.train import main

if __name__ == "__main__":
    main()
