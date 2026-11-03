# RLOps Experiments: Empirical Benchmarks and Comparative Studies in RL & Neuroevolution

A curated monorepo dedicated to reproducible, controlled experimental studies across Deep Reinforcement Learning (Proximal Policy Optimization) and Evolutionary Computation (Direct Search, Genetic Programming, Estimation of Distribution Algorithms). 

The upstream GitHub repository is [`PedroM2626/rlops-experiments`](https://github.com/PedroM2626/rlops-experiments); the local workspace is checked out in `ML-Games`.

---

## Scientific Scope and Architectural Overview

The repository focuses strictly on mathematically grounded, reproducible benchmarks adhering to academic empirical standards:

1. **PPO Multi-Library Empirical Spectrum** (`ppo-versions`) — A 12-part controlled study evaluating how algorithmic design choices, framework ecosystems (CleanRL, Stable-Baselines3, RLlib, TorchRL, Tianshou, PyTorch, JAX), and hyperparameter tuning regimes impact sample efficiency, asymptotic stability, and variance across continuous and discrete environments.
2. **PPO Factorial Attribution & GAE Mechanics** (`PPO-SB3-CleanRL-JAX`) — A $2 \times 2$ factorial attribution study isolating the algorithmic specification (SB3 spec vs. CleanRL spec) from the computation engine (PyTorch vs. JAX + Optax), complemented by single-delta ablations and an empirical probe isolating Generalized Advantage Estimation (GAE) episode-boundary indexing differences.
3. **Evolutionary Computation Taxonomy & JAX Benchmark** (`evolutionary-strategies`) — A formal taxonomy and comparative evaluation of three evolutionary computation paradigms (Direct Parameter Search, Symbolic Genetic Programming, and Probabilistic Models / Estimation of Distribution Algorithms like CMA-ES and PBIL) accelerated via JAX (`vmap` + `jit`) on canonical control tasks, with a 100-episode out-of-sample generalization protocol.
4. **Neuroevolution vs. Symbolic Programs in Complex Video Games** (`smw-evolutionary-agent`) — An empirical study comparing Neuroevolution of Augmenting Topologies (NEAT) against Genetic Programming (GP) playing *Super Mario World* via the BizHawk emulator, contrasting topological neural networks against interpretable symbolic decision trees under identical sensory receptors ($13 \times 13$ tile grid).

> [!NOTE]
> **Repository Curation & Deprecated Modules**:
> Early procedural 3D robotics and locomotion prototypes (`Chase`, `Race`, and `Parkour-Obstacles`) were permanently removed from the repository. Rigorous empirical auditing revealed fundamental physical and algorithmic discrepancies (such as non-physical kinematic chaser resets, unphysical lateral spring stabilizers $F_y = -80 \cdot y$, 0% goal completion rates, and severe policy collapse after $2.1\times 10^6$ steps). The monorepo has been streamlined to retain only scientifically validated, fully converging benchmarks.

---

## Sub-Project Directory

### 1. [`ppo-versions`](ppo-versions/README.md)
* **Objective**: Investigate the variance and performance discrepancies among 7 distinct PPO implementations on `CartPole-v1` and `LunarLander-v3`.
* **Implementations Tested**:
  - `cleanrl_original`: Canonical single-file implementation.
  - `cleanrl_style_pytorch`: Modularized PyTorch clean implementation.
  - `stable_baselines3`: Standard production baseline (`VecEnv` + modular policy).
  - `stable_baselines3_zoo`: Stable-Baselines3 with tuned RL Zoo hyperparameters.
  - `jax_pure`: Hardware-accelerated functional PPO using JAX and Optax.
  - `rllib`: Scalable distributed implementation via Ray RLlib.
  - `torchrl`: TensorDict-driven high-throughput implementation.
  - `tianshou`: Modular replay-driven framework.
* **Methodology**: 10 seeds per configuration, formal statistical significance testing (Mann-Whitney U tests, Levene's variance homogeneity tests, Bonferroni multiplicity corrections), and empirical evaluation of RL Zoo hyperparameter sets.

### 2. [`PPO-SB3-CleanRL-JAX`](PPO-SB3-CleanRL-JAX/README.md)
* **Objective**: Disentangle the contributions of algorithmic specification vs. software runtime.
* **Factorial Matrix**:
  - Spec SB3 $\times$ Engine PyTorch
  - Spec SB3 $\times$ Engine JAX
  - Spec CleanRL $\times$ Engine PyTorch
  - Spec CleanRL $\times$ Engine JAX
* **Ablations & Probes**: Value-function loss clipping, learning rate annealing, MSE factor conventions ($0.5$ vs. $1.0$), timeout bootstrapping, and a dedicated `probe_gae` study quantifying off-by-one boundary indexing in advantage computations.

### 3. [`evolutionary-strategies`](evolutionary-strategies/README.md)
* **Objective**: Comparative empirical evaluation of evolutionary algorithms classified by their mathematical object of variation:
  - **Family 1: Direct Solutions** (Direct Representation): Simple Genetic Algorithm (SimpleGA), Differential Evolution (DE), OpenAI Evolution Strategy (OpenAI-ES).
  - **Family 2: Symbolic Programs** (Genetic Programming): Linear Genetic Programming (LinearGP), Cartesian Genetic Programming (CartesianGP).
  - **Family 3: Probabilistic Models** (Estimation of Distribution Algorithms - EDA): Covariance Matrix Adaptation Evolution Strategy (CMA-ES), Population-Based Incremental Learning (PBIL).
* **Speedup & Evaluation**: Vectorized rollouts via JAX `vmap` and `jit` over `gymnax` environments (`CartPole-v1`, `Pendulum-v1`, `Acrobot-v1`, `MountainCar-v0`, `MountainCarContinuous-v0`). Includes a 100-episode out-of-sample evaluation protocol to measure the generalization gap.

### 4. [`smw-evolutionary-agent`](smw-evolutionary-agent/README.md)
* **Objective**: Compare topological neural network neuroevolution (NEAT) against symbolic tree genetic programming (GP) navigating *Super Mario World* (Yoshi's Island 1).
* **Architecture**: BizHawk emulator (Snes9x core) running custom Lua controllers (`MarIO.lua` and `MarIO_GP.lua`) interfacing with Python test harnesses.
* **Perception & Fitness**: Equivalent $13 \times 13$ tile perception window around Mario, with identical fitness metrics based on horizontal progress ($\Delta X$), timeout penalties, and velocity bonuses.

---

## Technology Stack

| Concern | Technologies | Active Projects |
|---|---|---|
| **Core DL & Tensors** | PyTorch (`torch`), JAX (`jax`, `jaxlib`, `optax`) | `ppo-versions`, `PPO-SB3-CleanRL-JAX`, `evolutionary-strategies` |
| **RL Baselines & Stacks** | Stable-Baselines3, CleanRL, Ray/RLlib, TorchRL, Tianshou | `ppo-versions`, `PPO-SB3-CleanRL-JAX` |
| **Vectorized RL Simulation** | `gymnax`, `lax.scan`, `vmap`, `jit` | `evolutionary-strategies`, `PPO-SB3-CleanRL-JAX` |
| **Evolutionary Frameworks** | Custom JAX-accelerated ES/EDA/GA, Lua MarIO engine | `evolutionary-strategies`, `smw-evolutionary-agent` |
| **Emulation & Environment** | BizHawk 2.9.x, LuaJIT / Lua 5.1, Snes9x core | `smw-evolutionary-agent` |
| **Statistical Analysis** | SciPy, NumPy, Pandas (Mann-Whitney U, Levene, bootstrap confidence intervals, Cliff's delta) | `ppo-versions`, `PPO-SB3-CleanRL-JAX`, `evolutionary-strategies` |
| **Visualization** | Matplotlib, Seaborn | All projects |

---

## Repository Tree

```text
.
├── ppo-versions/                 # 12-part PPO multi-framework benchmark
│   ├── scripts/
│   │   ├── common.py             # Shared logging, evaluation protocol, serialization
│   │   ├── train_*.py            # CleanRL, SB3, PyTorch, JAX, RLlib, TorchRL, Tianshou
│   │   ├── compare.py            # Comparative curve plotting
│   │   ├── boxplot.py            # Return distribution boxplots
│   │   ├── stats_test.py         # Mann-Whitney U & Levene hypothesis tests
│   │   └── eval_stats.py         # Evaluation statistics aggregation
│   ├── results/                  # Committed seed training curves (JSON)
│   ├── results_eval/             # Committed evaluation distributions (JSON)
│   ├── figures/                  # Publication-ready plots (.png)
│   └── requirements.txt
│
├── PPO-SB3-CleanRL-JAX/          # 2x2 Factorial attribution study (spec vs. stack)
│   ├── config.py, specs.py       # Algorithmic specifications and run hyperparameters
│   ├── train_sb3.py              # Stable-Baselines3 driver
│   ├── train_cleanrl_torch.py    # CleanRL PyTorch driver
│   ├── train_jax_ppo.py          # Pure JAX PPO driver
│   ├── run_all.py                # Sequential benchmark runner
│   ├── run_full_background.py    # Resilient parallel runner
│   ├── evaluate.py               # Deterministic multi-seed evaluation
│   ├── analyze.py                # Bootstrap resampling and attribution contrasts
│   ├── plot_results.py           # Training curves and attribution boxplots
│   ├── probe_gae/                # Empirical GAE boundary indexing isolation study
│   │   ├── analyze.py, run_probe.sh, README.md, data/
│   ├── requirements-torch.txt    # PyTorch stack requirements
│   ├── requirements-jax.txt      # JAX stack requirements
│   └── results/                  # Committed tables (.csv, .md), curves (.npz), plots (.png)
│
├── evolutionary-strategies/      # 3 Evolutionary families benchmarked with JAX
│   ├── src/
│   │   ├── environments.py       # gymnax environment wrappers and JAX rollouts
│   │   ├── family1_direct.py     # SimpleGA, Differential Evolution, OpenAI-ES
│   │   ├── family2_programs.py   # LinearGP, CartesianGP
│   │   ├── family3_eda.py        # CMA-ES, PBIL
│   │   ├── benchmark.py          # JAX-accelerated training loops
│   │   ├── evaluation.py         # 100-episode out-of-sample validation engine
│   │   └── visualization.py      # Curve and family comparison renderers
│   ├── run_benchmark.py          # Benchmark driver across 5 control tasks
│   ├── run_gp_only.py            # Targeted GP evaluation script
│   ├── run_validation_100.py     # 100-episode out-of-sample validation runner
│   ├── generate_final_plots.py   # Publication figures generator
│   ├── requirements.txt          # Python dependencies (jax, numpy, scipy, matplotlib)
│   └── results/                  # Committed training curves, logs, and validation summaries
│
├── smw-evolutionary-agent/       # NEAT vs. GP on Super Mario World (BizHawk)
│   ├── env_config.py             # Cross-platform environment resolution (Windows/Linux/macOS)
│   ├── MarIO.lua                 # NEAT agent controller (BizHawk Lua script)
│   ├── MarIO_GP.lua              # Genetic Programming symbolic tree controller
│   ├── launch_mario.py           # Cross-platform supervisor and savestate bootstrapper
│   ├── setup_environment.py      # ROM validation (SHA-1 checksum verification)
│   ├── train_neat.py             # Timed NEAT training harness
│   ├── train_gp.py               # Timed GP training harness
│   ├── plot_comparative_progress.py # Unified analytical visualization & Pareto engine
│   ├── neat_training_results.txt # Empirical generation logs (NEAT)
│   ├── gp_training_results.txt   # Empirical generation logs (GP)
│   ├── docs/
│   │   ├── comparison.md         # Mathematical comparison and complexity analysis
│   │   ├── comparative_metrics_summary.md # Generational contrast table
│   │   └── figures/              # Pareto & trajectory plots
│   └── README.md
│
├── requirements.txt              # Shared foundational Python dependencies
└── .gitignore, .gitattributes, .editorconfig
```

---

## Quick Start & Setup

Because subprojects explore different computational paradigms (PyTorch, JAX, BizHawk Lua), dependencies can be installed either via individual project requirements or via a shared virtual environment:

### Core Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# Linux / macOS
source venv/bin/activate

# Install shared core dependencies
pip install -r requirements.txt
```

### Sub-Project Environments

| Sub-Project | Requirements File | Notes |
|---|---|---|
| **`ppo-versions`** | [`ppo-versions/requirements.txt`](ppo-versions/requirements.txt) | Includes PyTorch, JAX, Tianshou, TorchRL, RLlib, SB3. |
| **`PPO-SB3-CleanRL-JAX`** | [`PPO-SB3-CleanRL-JAX/requirements-torch.txt`](PPO-SB3-CleanRL-JAX/requirements-torch.txt)<br>[`PPO-SB3-CleanRL-JAX/requirements-jax.txt`](PPO-SB3-CleanRL-JAX/requirements-jax.txt) | Separate files allow isolated CPU/CUDA wheels for Torch and JAX. |
| **`evolutionary-strategies`** | [`evolutionary-strategies/requirements.txt`](evolutionary-strategies/requirements.txt) | Pure JAX, NumPy, SciPy, Matplotlib. |
| **`smw-evolutionary-agent`** | Python Standard Library + BizHawk 2.9+ | Cross-platform (Windows, Linux via Mono/native, macOS), SNES ROM. |

---

## Execution Guide

### 1. `ppo-versions`
From inside `ppo-versions/`:
```bash
# Execute CleanRL style on LunarLander-v3 with seed 42
python scripts/train_cleanrl.py --env-id LunarLander-v3 --seed 42

# Execute Stable-Baselines3 Zoo configuration
PPO_ENV=LunarLander-v3 PPO_SEED=42 python scripts/train_sb3_zoo.py

# Run hypothesis testing (Mann-Whitney U / Levene) across collected seeds
python scripts/stats_test.py
```

### 2. `PPO-SB3-CleanRL-JAX`
From inside `PPO-SB3-CleanRL-JAX/`:
```bash
# Smoke test (CartPole-v1, 1 seed, 1% timestep budget)
python run_all.py --env CartPole-v1 --seeds 0 --timesteps-scale 0.01

# Execute complete background run with multi-worker parallelism
python run_full_background.py --workers 4 --env LunarLander-v3

# Statistical evaluation, bootstrap resampling, and plot generation
python evaluate.py
python analyze.py
python plot_results.py
```

### 3. `evolutionary-strategies`
From inside `evolutionary-strategies/`:
```bash
# Run benchmark across direct solutions and probability models
python run_benchmark.py --quick

# Run 100-episode out-of-sample generalization validation
python run_validation_100.py

# Regenerate publication figures from committed JSONs
python generate_final_plots.py
```

### 4. `smw-evolutionary-agent`
From inside `smw-evolutionary-agent/` (Windows, Linux, macOS):
```bash
# Verify emulator setup and ROM SHA-1 integrity
python setup_environment.py

# Launch NEAT agent
python launch_mario.py

# Launch Genetic Programming agent
python launch_mario.py --gp

# Generate analytical Pareto plots and contrast tables
python plot_comparative_progress.py

# Run timed 180-second benchmark comparison
python train_neat.py 180
python train_gp.py 180
```

---

## Versioning & Artifact Policy

In accordance with scientific open-source best practices, large transient binary files are excluded from Git via [`.gitignore`](.gitignore):
- Model weights, checkpoints, and archives (`*.zip`, `*.pt`, `*.pkl`, `*.npz`)
- Video and render outputs (`*.mp4`, `*.gif`)
- Local virtual environments (`.venv/`, `venv/`)
- Emulator binary cores, savestates, and population dump pools (`*.sfc`, `*.state`, `*.pool`, `*.gppool`)

Preserved as primary scientific records:
- Evaluated metric summaries, CSV contrast tables, and benchmark JSON data.
- High-resolution comparative plots and figures.
