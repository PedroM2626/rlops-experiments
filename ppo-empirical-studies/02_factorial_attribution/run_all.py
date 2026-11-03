"""Orchestrator: train -> evaluate -> analyse -> plot."""
from __future__ import annotations
import argparse
import subprocess
import sys
import time
from pathlib import Path
from run_helpers import sh, HERE  # noqa: F401


def parse_seeds(s: str | None):
    if not s:
        import config as _c
        return list(_c.SEEDS)
    return [int(x) for x in s.split(",") if x.strip() != ""]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--timesteps-scale", type=float, default=1.0)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--python-torch", default=sys.executable)
    ap.add_argument("--python-jax", default=None)
    args = ap.parse_args()

    import config as _c
    import variants as _v
    _v.ensure_dirs()
    py_jax = args.python_jax or args.python_torch
    envs = [args.env] if args.env else list(_c.ENVS)
    seeds = parse_seeds(args.seeds)
    sel = [v for v in _v.VARIANTS if not args.only or v["id"] == args.only]
    assert sel, f"unknown --only {args.only!r}"
    t0 = time.time()
    ok = True

    if not args.skip_train:
        for env_id in envs:
            for v in sel:
                py = (py_jax if v["id"].startswith("jax")
                      else args.python_torch)
                for seed in seeds:
                    cmd = [v["script"], "--env", env_id, "--seed",
                           str(seed), "--timesteps-scale",
                           str(args.timesteps_scale)]
                    if v["mode"]:
                        cmd += ["--mode", v["mode"]]
                    if v.get("abl"):
                        cmd += ["--abl", v["abl"]]
                    ok &= sh(py, cmd)
    if not args.skip_eval:
        for env_id in envs:
            ok &= sh(args.python_torch, ["evaluate.py", "--env", env_id])
            ok &= sh(args.python_torch, ["analyze.py", "--env", env_id])
            ok &= sh(args.python_torch, ["plot_results.py", "--env", env_id])
    print(f"\nEXPERIMENT {'OK' if ok else 'WITH FAILURES'} "
          f"in {(time.time()-t0)/60:.1f} min")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
