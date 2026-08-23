"""Orchestrator script: trains all PPO implementations, evaluates, and plots.

Usage:
    python run_all.py              # Run everything
    python run_all.py --skip-train # Skip training, just evaluate and plot
    python run_all.py --only sb3   # Train only SB3
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path




def run_script(script: str, label: str) -> bool:
    """Run a training/evaluation script and return True on success."""
    print(f"\n{'='*60}")
    print(f"  Running: {label}")
    print(f"  Script:  {script}")
    print(f"{'='*60}\n")

    start = time.time()
    result = subprocess.run(
        [sys.executable, script],
        cwd=str(Path(__file__).resolve().parent),
    )
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n  [{label}] Completed successfully in {elapsed:.1f}s")
        return True
    else:
        print(f"\n  [{label}] FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all PPO comparison experiments")
    parser.add_argument("--skip-train", action="store_true", help="Skip training phase")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation phase")
    parser.add_argument("--skip-plot", action="store_true", help="Skip plot generation")
    parser.add_argument("--only", type=str, default=None,
                        help="Train only the specified implementation (e.g. sb3, cleanrl)")
    parser.add_argument("--env", type=str, default="LunarLander-v3",
                        help="Gym Environment ID to run the experiment on")
    args = parser.parse_args()

    # Set the environment variable BEFORE importing config so that paths are built correctly
    import os
    os.environ["PPO_ENV_ID"] = args.env

    # We import config inside main to ensure the env var is set first
    from config import IMPLEMENTATIONS, MODELS_DIR, CURVES_DIR, PLOTS_DIR, RESULTS_DIR

    # Ensure directories exist
    for d in [MODELS_DIR, CURVES_DIR, PLOTS_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    total_start = time.time()
    results: dict[str, bool] = {}

    # Phase 1: Training
    if not args.skip_train:
        print("\n" + "#" * 60)
        print("#  PHASE 1: TRAINING")
        print("#" * 60)

        for impl in IMPLEMENTATIONS:
            impl_id = impl["id"]
            label = impl["label"]
            script = impl["script"]

            if args.only and args.only != impl_id:
                print(f"\n  Skipping {label} (--only {args.only})")
                continue

            success = run_script(script, label)
            results[impl_id] = success
    else:
        print("\n  Skipping training phase (--skip-train)")

    # Phase 2: Evaluation
    if not args.skip_eval:
        print("\n" + "#" * 60)
        print("#  PHASE 2: EVALUATION")
        print("#" * 60)
        run_script("evaluate.py", "Unified Evaluation")
    else:
        print("\n  Skipping evaluation phase (--skip-eval)")

    # Phase 3: Plot generation
    if not args.skip_plot:
        print("\n" + "#" * 60)
        print("#  PHASE 3: PLOT GENERATION")
        print("#" * 60)
        run_script("plot_results.py", "Plot Generation")
    else:
        print("\n  Skipping plot generation (--skip-plot)")

    # Summary
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print("  EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    if results:
        print("\n  Training results:")
        for impl_id, success in results.items():
            status = "OK" if success else "FAILED"
            print(f"    {impl_id}: {status}")
    print(f"\n  Results directory: {RESULTS_DIR}")
    print(f"  Plots directory:   {PLOTS_DIR}")
    print()


if __name__ == "__main__":
    main()
