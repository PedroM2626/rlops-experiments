"""Background runner: executes all missing (variant, env, seed) jobs in parallel."""
from __future__ import annotations
import subprocess
import sys
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = r"C:\Users\Acer\AppData\Local\Programs\Python\Python311\python.exe"

import config as _c
import variants as _v

def model_exists(vid: str, env_id: str, seed: int) -> bool:
    if vid == "sb3_torch":
        return (_v.MODELS_DIR / f"{vid}_{env_id}_seed{seed}.zip").exists()
    if vid in ("cleanrl_torch", "cleanrl_sb3mode_torch"):
        return (_v.MODELS_DIR / f"{vid}_{env_id}_seed{seed}.pt").exists()
    return (_v.MODELS_DIR / f"{vid}_{env_id}_seed{seed}.npz").exists()

def job_cmd(vid: str, env_id: str, seed: int) -> list[str]:
    v = next(x for x in _v.VARIANTS if x["id"] == vid)
    cmd = [v["script"], "--env", env_id, "--seed", str(seed),
           "--timesteps-scale", "1.0"]
    if v["mode"]:
        cmd += ["--mode", v["mode"]]
    if v.get("abl"):
        cmd += ["--abl", v["abl"]]
    return cmd

def run_one(job):
    vid, env_id, seed = job
    cmd = job_cmd(vid, env_id, seed)
    log = HERE / "results" / "logs" / f"{vid}_{env_id}_seed{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    # JAX jobs: single thread pool is fastest on this machine (JIT dominated)
    if vid.startswith("jax"):
        env["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
    # torch jobs also single-thread: tiny nets, avoids thrash when parallel
    else:
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
    with open(log, "w", encoding="utf-8") as f:
        f.write(f"$ {PY} {' '.join(cmd)}\n")
        f.flush()
        r = subprocess.run([PY, *cmd], cwd=str(HERE), stdout=f,
                           stderr=subprocess.STDOUT, text=True, env=env)
    return (job, r.returncode)

def main(max_workers: int = 6, only: set[str] | None = None,
         seeds: list[int] | None = None):
    _v.ensure_dirs()
    seed_list = seeds if seeds else list(_c.SEEDS)
    jobs = []
    for env_id in _c.ENVS:
        for v in _v.VARIANTS:
            if only and v["id"] not in only:
                continue
            for seed in seed_list:
                if not model_exists(v["id"], env_id, seed):
                    jobs.append((v["id"], env_id, seed))
    print(f"[runner] missing jobs: {len(jobs)}", flush=True)
    for j in jobs:
        print(f"  {j[0]} {j[1]} seed={j[2]}", flush=True)
    ok = True
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for (job, rc) in ex.map(run_one, jobs):
            status = "OK" if rc == 0 else f"FAIL({rc})"
            print(f"[runner] done {job} -> {status}", flush=True)
            ok &= (rc == 0)
    print(f"[runner] ALL {'OK' if ok else 'WITH FAILURES'}", flush=True)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--env", default=None)
    ap.add_argument("--only", default=None,
                    help="comma-separated variant ids (e.g. jax_sb3,jax_cleanrl)")
    ap.add_argument("--seeds", default=None,
                    help="comma-separated seeds (default: config.SEEDS)")
    _a = ap.parse_args()
    if _a.env:
        _c.ENVS = [_a.env]
    _only = set(_a.only.split(",")) if _a.only else None
    _seeds = ([int(x) for x in _a.seeds.split(",") if x.strip()]
              if _a.seeds else None)
    main(_a.workers, _only, _seeds)
