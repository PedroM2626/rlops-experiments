# RLOps Experiments

A monorepo of self-contained machine-learning experiments around reinforcement learning and
evolutionary computation. The GitHub repository is `PedroM2626/rlops-experiments`; the working
copy is still checked out in a folder named `ML-Games`.

The experiments fall into three groups:

- **PyBullet humanoid control** (`Chase`, `Parkour-Obstacles`, `Race`) — Gymnasium environments
  trained with PPO from Stable Baselines3, tracked in MLflow, with Docker images for headless runs.
- **PPO implementation studies** (`ppo-versions`, `PPO-SB3-CleanRL-JAX`) — controlled comparisons
  of many PPO implementations across libraries and numerical stacks.
- **Evolutionary and neuroevolution work** (`evolutionary-strategies`, `smw-evolutionary-agent`)
  — genetic programming / EDA / evolution-strategy benchmarks in JAX, and NEAT vs. GP agents
  playing *Super Mario World* in an emulator.

Each project is standalone: its own README, its own dependencies, its own entry points, and no
shared code with the others. This file is only the index and the machine-level setup.

## Experiment index

### [Chase](Chase/README.md)

Humanoid evasion task: a 10-DOF PyBullet agent flees a rule-based chaser that walks with a
programmed gait and closes in over time. Trains with SB3 PPO, logs to MLflow, and ships an
interactive PyBullet GUI trainer with live sliders, a pause/save panel and an in-scene HUD.

### [Parkour-Obstacles](Parkour-Obstacles/README.md)

Procedural parkour: the same humanoid class must cross platforms, gaps, ledges, ramps, stairs,
stepping stones and low-friction "icy" tiles, with a fresh course generated every episode. The
most fully documented project here (observation/action spec, reward terms, runtime physics
overrides, checkpoint layout) and the one the root `Dockerfile` builds.

### [Race](Race/README.md)

Multi-agent variant: several humanoids race over procedurally generated terrain with rocks,
hills, pillars and ramps. Uses parameter sharing — one PPO policy updated from all agent
trajectories — plus rank and finishing-order rewards, and a pytest suite for the environment.

### [ppo-versions](ppo-versions/README.md)

The PPO Benchmark: a twelve-part comparison of PPO as implemented in Stable Baselines3, CleanRL
(both ours and the official script), pure PyTorch, pure JAX, RLlib, TorchRL and Tianshou on
CartPole and LunarLander. Its README records the experimental arc and the conceptual conclusions;
`results/` and `results_eval/` hold the raw curves and evaluation stats. This is the successor of
the folder formerly called `PPO-Comparison`.

### [PPO-SB3-CleanRL-JAX](PPO-SB3-CleanRL-JAX/README.md)

An attribution study rather than a leaderboard: a factorial design that holds the algorithm spec
fixed while swapping the stack (Torch vs. JAX) and holds the stack fixed while swapping the spec
(SB3 vs. CleanRL), plus a hand-written "bridge" implementation and single-delta ablations to
localise where a performance gap actually comes from.

### [evolutionary-strategies](evolutionary-strategies/README.md)

Benchmark of three evolutionary families on canonical RL tasks with JAX-vectorised fitness
evaluation (`vmap` + `jit` over `gymnax`): direct parameter vectors (SimpleGA, Differential
Evolution, OpenAI-ES), symbolic programs (LinearGP, CartesianGP) and probability models
(CMA-ES, PBIL), followed by a 100-episode out-of-sample validation protocol that quantifies the
optimism gap between in-training fitness and true generalisation.

### [smw-evolutionary-agent](smw-evolutionary-agent/README.md)

NEAT versus Genetic Programming on *Super Mario World* (Yoshi's Island 1), driven through Lua
scripts inside the BizHawk emulator, with both agents sharing one fitness function and one tile
grid observation so the representations — an opaque network graph versus readable per-button
expression trees — are the only difference. Windows-only; see its README for emulator and ROM
setup. [docs/comparison.md](smw-evolutionary-agent/docs/comparison.md) is the write-up.

## Technology stack

| Concern | Technology | Used by |
|---|---|---|
| Physics simulation | PyBullet (articulated 10-DOF URDF humanoid) | Chase, Parkour, Race |
| RL interface | Gymnasium + Stable Baselines3 PPO (`DummyVecEnv`, `VecNormalize`) | Chase, Parkour, Race |
| Experiment tracking | MLflow (local file store per project) | Chase, Parkour, Race |
| Alternative PPO stacks | CleanRL-style Torch, pure PyTorch, RLlib/Ray, TorchRL, Tianshou, JAX + Optax | ppo-versions, PPO-SB3-CleanRL-JAX |
| Vectorised evolution search | JAX (`vmap`/`jit`/`lax.scan`) + gymnax | evolutionary-strategies, PPO-SB3-CleanRL-JAX |
| Evolutionary algorithms | GA, DE, OpenAI-ES, CMA-ES, PBIL, Linear/Cartesian GP, NEAT | evolutionary-strategies, smw-evolutionary-agent |
| Emulation / game interface | BizHawk 2.9.x + Lua API, Snes9x core | smw-evolutionary-agent |
| Statistics / figures | Matplotlib + NumPy everywhere; `scipy.stats` for the hypothesis tests in `ppo-versions`, NumPy bootstrap resampling in `evolutionary-strategies` | ppo-versions, PPO-SB3-CleanRL-JAX, evolutionary-strategies |
| Containerisation | Docker + Docker Compose (CUDA image path available) | Chase, Parkour, Race |

## Repository layout

```
.
├── Chase/                        # PyBullet evasion env (SB3 + MLflow)
│   ├── src/env/chase_env.py
│   ├── src/agent/callbacks.py    # MLflow callback
│   ├── src/train_interactive.py  # GUI trainer (run as -m src.train_interactive)
│   ├── train.py                  # batch training entry point (project root)
│   ├── play_chase.py             # playback of a trained policy
│   ├── test_chase.py             # headless smoke test
│   ├── Dockerfile, requirements.txt
│   └── models/, mlruns/          # git-ignored
│
├── Parkour-Obstacles/            # Procedural parkour env (SB3 + MLflow)
│   ├── src/env/parkour_env.py
│   ├── src/env/level_generator.py
│   ├── src/agent/callbacks.py
│   ├── src/train.py, src/train_interactive.py, src/play.py
│   ├── train.py, play.py         # thin sys.path wrappers around the src modules
│   ├── test_env.py
│   ├── assets/humanoid.urdf
│   └── models/, mlruns/          # git-ignored
│
├── Race/                         # Multi-agent race (SB3 + MLflow)
│   ├── src/env/race_env.py, src/env/terrain_generator.py
│   ├── src/agent/callbacks.py
│   ├── src/train.py, src/play.py
│   ├── tests/                    # pytest suite
│   ├── Dockerfile, requirements.txt, .env.example
│   └── models/, mlruns/          # git-ignored
│
├── ppo-versions/                 # PPO Benchmark (12 parts)
│   ├── scripts/common.py         # shared config, evaluate_policy, save_result
│   ├── scripts/train_*.py        # one script per implementation/variant
│   ├── scripts/{compare,boxplot,stats_test,eval_stats}.py
│   ├── scripts/run_all.sh        # multi-seed sweep used during the study
│   ├── results/, results_eval/   # committed JSON curves + evaluation stats
│   └── requirements.txt
│
├── PPO-SB3-CleanRL-JAX/          # Spec-vs-stack attribution study
│   ├── config.py, specs.py, variants.py, common.py, run_helpers.py
│   ├── train_sb3.py, train_cleanrl_torch.py, train_jax_ppo.py
│   ├── jax_core.py, jax_run.py, jax_run_fast.py, jax_fwd.py, sb3_shim.py
│   ├── run_all.py                # sequential orchestrator
│   ├── run_full_background.py    # resume-safe parallel runner
│   ├── evaluate.py, analyze*.py, plot_*.py
│   ├── requirements-torch.txt, requirements-jax.txt
│   └── results/                  # curves/tables/plots/meta committed, models ignored
│
├── evolutionary-strategies/      # 3 evolutionary families, JAX/gymnax
│   ├── src/{environments,family1_direct,family2_programs,family3_eda}.py
│   ├── src/{benchmark,evaluation,visualization}.py
│   ├── run_benchmark.py, run_gp_only.py
│   ├── run_validation_100.py, generate_final_plots.py
│   └── results/                  # committed JSON/CSV/PNG
│
├── smw-evolutionary-agent/       # NEAT vs GP on Super Mario World
│   ├── MarIO.lua                 # NEAT agent (Lua, runs inside BizHawk)
│   ├── MarIO_GP.lua              # Genetic Programming agent
│   ├── launch_mario.py           # autonomous setup + launch pipeline
│   ├── setup_environment.py      # ROM / emulator / savestate integrity check
│   ├── train_neat.py, train_gp.py
│   ├── docs/comparison.md
│   └── *.sfc, *.dll, *.state, *.pool, *.gppool   # git-ignored
│
├── Dockerfile                    # Parkour training image (installs root requirements.txt)
├── docker-compose.yml            # trainers + MLflow UI services
├── requirements.txt              # shared deps; Parkour is the project that reads it
├── .env.example                  # hyperparameter defaults for the PyBullet projects
└── .dockerignore, .editorconfig, .gitattributes, .gitignore
```

`PPO-Comparison/`, the directory named in older versions of this file, no longer exists: it was
removed in `6cb19b0` and replaced by `ppo-versions/` and `PPO-SB3-CleanRL-JAX/`.

## Quick start

The projects pin different versions of the same libraries (for example SB3 2.7.1 vs 2.9.0, and
three different Gymnasium lines), so use one virtualenv per project rather than one shared env.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

| Project | Dependencies |
|---|---|
| Parkour-Obstacles | `pip install -r requirements.txt` (root file) |
| Chase | `pip install -r Chase/requirements.txt` |
| Race | `pip install -r Race/requirements.txt` |
| ppo-versions | `pip install -r ppo-versions/requirements.txt` |
| PPO-SB3-CleanRL-JAX | `pip install -r requirements-torch.txt` **and** `-r requirements-jax.txt` into a single interpreter |
| evolutionary-strategies | no requirements file; the pinned install line is in its README |
| smw-evolutionary-agent | Python standard library only; the heavy dependency is BizHawk 2.9.x plus a legally dumped SMW ROM |

Environment configuration for the three PyBullet projects is optional:

```bash
cp .env.example .env     # repo root
```

`train.py` / `src.train` read it through `python-dotenv`, which searches upward from the script's
directory, so the root `.env` supplies defaults for `EXPERIMENT_NAME`, `MLFLOW_TRACKING_URI`,
`TOTAL_TIMESTEPS`, `LEARNING_RATE`, `N_STEPS`, `BATCH_SIZE`, `GAMMA`, `N_EPOCHS`, `GAE_LAMBDA`,
`CLIP_RANGE`, `ENT_COEF`, `NET_ARCH`, `N_ENVS` and `CHECKPOINT_FREQ`. Command-line flags win over
`.env`. `Race` also ships its own `.env.example` with the extra `N_AGENTS` key.

## How to run each experiment

All commands are copied from the working entry points; each project README lists the full flag set.

### Chase (from `Chase/`)

```bash
python test_chase.py                                   # env smoke test, no training
python train.py                                        # batch PPO -> models/ppo_chase_final
python train.py --n-steps 4096 --n-epochs 20 --net-arch 512,512
python train.py --test --model models/ppo_chase_final.zip --episodes 5
python play_chase.py                                   # GUI playback (random if no checkpoint)
python -m src.train_interactive --timesteps 500000     # interactive GUI trainer
```

`train.py` has no `--timesteps` flag: its budget is derived as `n_steps * n_epochs * 100`, so
resize the run through `--n-steps` / `--n-epochs`. `src/train_interactive.py` uses `--timesteps`
(0 = unlimited) and must be launched as a module, or with `PYTHONPATH=.` exported.

### Parkour-Obstacles (from `Parkour-Obstacles/`)

```bash
python test_env.py                                                    # headless smoke test
python -m src.train                                                   # defaults: 1M steps, 4 envs
python -m src.train --timesteps 3000000 --n-envs 8 --net-arch 512,512
python -m src.train --model models/ppo_parkour_0000500000.zip         # resume
python -m src.train_interactive --timesteps 3000000                   # GUI sliders + HUD
python -m src.play --model models/ppo_parkour_final --n-episodes 3    # watch the policy
python -m src.play --no-render --n-episodes 20 --level-seed 42        # headless benchmark
python -m src.train --test --model models/ppo_parkour_final.zip --episodes 5
```

Use the `-m src.xxx` form. The modules import each other as `src.env...`, so
`python src/train_interactive.py` fails with `ModuleNotFoundError: No module named 'src'` unless
`PYTHONPATH=.` is exported; the root `train.py` and `play.py` wrappers exist to avoid that.

### Race (from `Race/`)

```bash
python -m pytest tests/test_race_env.py -v
python -m src.play --random --n-agents 4                       # untrained agents, no training run
python -m src.train --n-agents 4 --n-envs 4 --timesteps 5000000
python -m src.train --model models/ppo_race_0000500000.zip     # resume
python -m src.play --model models/ppo_race_final.zip --episodes 3
```

### PPO-SB3-CleanRL-JAX (from `PPO-SB3-CleanRL-JAX/`)

```bash
# smoke test: one env, one seed, 1% of the budget
python run_all.py --env CartPole-v1 --seeds 0 --timesteps-scale 0.01

python run_all.py                                  # full sequential run (slow, CPU)
python run_full_background.py --workers 8           # resume-safe parallel runner
python run_full_background.py --workers 8 --env LunarLander-v3 --only jax_sb3,jax_cleanrl

python evaluate.py       # deterministic evaluation per model -> results/tables
python analyze.py        # seed-paired bootstrap contrasts
python plot_results.py   # curves and boxplots -> results/plots
```

`run_all.py` also accepts `--only`, `--skip-train`, `--skip-eval` and, if you split Torch and JAX
across interpreters, `--python-torch` / `--python-jax`. Individual arms can be run directly, e.g.
`python train_jax_ppo.py --env LunarLander-v3 --seed 0 --mode cleanrl`.

### ppo-versions (from `ppo-versions/`)

The scripts take no CLI flags; they are configured through environment variables read by
`scripts/common.py`: `PPO_ENV` (default `CartPole-v1`), `PPO_SEED`, `PPO_TIMESTEPS`,
`PPO_EVAL_EPISODES` and `PPO_RUN_TAG` (a suffix that keeps different configurations from
colliding in `results/`).

```bash
PPO_ENV=LunarLander-v3 PPO_SEED=42 python scripts/train_sb3_zoo.py
PPO_ENV=LunarLander-v3 PPO_TIMESTEPS=1000000 PPO_SEED=42 PPO_RUN_TAG=-zoo python scripts/train_torchrl.py
PPO_ENV=LunarLander-v3 PPO_SEED=42 PPO_RUN_TAG=-zoo python scripts/eval_stats.py
```

Output paths resolve relative to the script, so a re-run writes inside the project:
`results/` and `results_eval/` for the JSON, `figures/` for the PNGs the post-processing
scripts produce. The study itself ran on a throwaway Linux sandbox that wrote to
`/home/claude/ppo-benchmark` and `/mnt/user-data/outputs`; the nine figures under `figures/`
were regenerated from the committed JSON alone, and their statistics match the tables in the
project README. One arm is missing from the LunarLander plots:
`results/LunarLander-v3/tianshou_seed42.json` stores an empty curve and a null final reward, so
the post-processors skip it and name it in the figure title instead of dropping it silently. Note that `results/` and `results_eval/`
are the committed record of the benchmark, and a re-run overwrites files with the same name in
place — set `PPO_RUN_TAG` to keep a new run separate.

### evolutionary-strategies (from `evolutionary-strategies/`)

```bash
python run_validation_100.py                     # out-of-sample 100-episode protocol
python run_benchmark.py --quick --no-gp          # families 1 and 3, short run
python run_benchmark.py --env CartPole-v1 --seed 7
python run_gp_only.py                            # family 2 (LinearGP + CartesianGP)
python generate_final_plots.py                   # regenerate results/ figures
```

Install the pinned JAX/gymnax stack first (see section 10 of the project README); it targets
Python 3.11.

### smw-evolutionary-agent (from `smw-evolutionary-agent/`, Windows only)

```bash
python setup_environment.py           # validate ROM SHA-1, BizHawk, savestate, Lua scripts
python launch_mario.py                # NEAT run
python launch_mario.py --gp           # Genetic Programming run
python launch_mario.py --reboot-state # regenerate DP1.state from scratch
python train_neat.py 180              # paired 180 s benchmark run (arg = seconds)
python train_gp.py 180
```

The launcher scripts expect the emulator and the ROM under a fixed `C:\BizHawk` directory and copy
the Lua agents there themselves; BizHawk 2.9.x is required because newer releases break the Lua
API these scripts use.

## MLflow tracking

Only the three PyBullet projects use MLflow; the comparison and evolutionary studies write JSON,
CSV and PNG under their own `results/` directories instead.

Each of `Chase`, `Parkour-Obstacles` and `Race` keeps an independent local file store in its own
directory. From the project dir:

```bash
mlflow ui --backend-store-uri ./mlruns     # http://localhost:5000
```

Runs log the PPO hyperparameters and the resolved device, `episode_reward` and `episode_length`
per completed episode, and periodic checkpoints (default every 50k steps) plus the final policy
and `VecNormalize` stats as artifacts. The experiment name comes from `EXPERIMENT_NAME` /
`--experiment-name`, the store location from `MLFLOW_TRACKING_URI` / `--tracking-uri`.

## Docker

`docker-compose.yml` at the repo root defines six services:

| Service | Build | Command | Notes |
|---|---|---|---|
| `chase-trainer` | `Chase/Dockerfile` | `python train.py` | bind-mounts `Chase/models`, `Chase/mlruns` |
| `chase-mlflow` | `python:3.10-slim` | `mlflow ui` | port 5000 |
| `parkour-trainer` | root `Dockerfile` | `python -m src.train` | reads root `.env`; mounts `models`, `mlruns` |
| `parkour-mlflow` | `python:3.10-slim` | `mlflow ui` | port 5003 |
| `race-trainer` | `Race/Dockerfile` | `python -m src.train` | reads `Race/.env` |
| `race-mlflow` | `python:3.10-slim` | `mlflow ui` | port 5001 |

```bash
docker compose up --build chase-trainer
docker compose up parkour-trainer
docker compose up race-trainer
docker compose up chase-mlflow parkour-mlflow race-mlflow
```

Caveats:

- `race-trainer` loads `env_file: ./Race/.env`, and `parkour-trainer` loads the root `.env`;
  `.env` files are ignored, so copy the matching `.env.example` before starting those services.
- Every trainer declares an NVIDIA GPU reservation. On a CPU-only host, build and run directly
  instead, e.g. `docker build -t parkour-ml .` from the repo root then
  `docker run --rm parkour-ml python test_env.py`, or `docker build -t chase-ml .` from `Chase/`.
  Training itself works on CPU (`device="auto"`).

## What is deliberately not versioned

Per `.gitignore`:

- **Model artifacts**: `*.zip`, `*.pkl`, `*.pt`, `*.npz`, `*.onnx`, plus `**/results/logs/` and
  `runtime/`. Checkpoints are regenerable by training and would add hundreds of MB.
- **MLflow stores**: `mlruns/` at any depth.
- **Virtualenvs and caches**: `.venv/`, `venv/`, `/ENV/`, `__pycache__/`, `*.pyc`, `*.pyo`.
- **Emulator payloads**: `*.sfc`, `*.nes` (ROMs must be dumped by the user) and `*.dll` native cores.
- **Emulator run output**: `*.state` savestates, `*.pool` and `*.gppool` evolved population files —
  these are per-session state that `launch_mario.py` regenerates or restores.
- **Secrets and editor state**: `.env`, `Race/.env`, `.DS_Store`, `Thumbs.db`, `.vscode/`, `.idea/`.
  The two env files were committed before those rules existed and are now untracked; neither
  holds a credential — the root one is identical to `.env.example`, and `Race/.env` differs from
  its example only by lacking a commented-out DagsHub block.

Committed on purpose, in contrast: the study outputs that *are* the record of the non-PyBullet
experiments — `ppo-versions/results*`, `evolutionary-strategies/results`, and the
`curves/tables/plots/meta` subtrees plus runner logs under `PPO-SB3-CleanRL-JAX/results`
(its `results/logs/` directory is ignored). `.dockerignore` mirrors the artifact exclusions so
images are not built with checkpoints inside.

## Python and tooling versions

| Project | Interpreter / platform | Key pins |
|---|---|---|
| Parkour-Obstacles | Python 3.10+ (root images use `python:3.10-slim`) | PyBullet 3.2.6, SB3 2.7.1, Gymnasium 1.1.1, Torch 2.10.0, MLflow 3.10.1 |
| Chase | Python 3.10+ | same family, per `Chase/requirements.txt` |
| Race | Python 3.10+ | same family plus matplotlib, pandas, pytest |
| PPO-SB3-CleanRL-JAX | Python 3.10–3.11, one interpreter with Torch **and** JAX installed together, CPU only | `requirements-torch.txt`, `requirements-jax.txt` |
| ppo-versions | recorded on Python 3.12.3 / Ubuntu 24.04, CPU only | dated snapshot in `ppo-versions/requirements.txt` (SB3, JAX, Ray, TorchRL, Tianshou, box2d) |
| evolutionary-strategies | Python 3.11 recommended | JAX, gymnax, NumPy, matplotlib (its install line also pins `flax` and `orbax-checkpoint`) |
| smw-evolutionary-agent | Windows 10/11 x64; Python standard library only | BizHawk 2.9.1, .NET 8 runtime, VC++ 2015-2022 redist, DirectX end-user runtime, `Super Mario World (USA).sfc` |

Paths in every project are relative to that project's own directory: run commands from inside the
experiment folder, not from the repo root.
