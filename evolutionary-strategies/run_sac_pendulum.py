"""
SAC baseline on Pendulum-v1 (Section 6.5.3): thin wrapper around rl_baselines.py / run_rl_baselines.py.
Standard defaults, untuned; 5 seeds, 30,000 environment steps, evaluation at 5k/10k/20k/30k steps.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_rl_baselines import main

if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "sac_pendulum.json")
    main(["--algo", "sac", "--env", "Pendulum-v1", "--steps", "30000", "--eval-at", "5000", "10000", "20000", "30000", "--out", out] + sys.argv[1:])
