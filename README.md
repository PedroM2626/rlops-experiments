# RLOps Experiments: Empirical Benchmarks and Comparative Studies in RL & Neuroevolution

A curated monorepo dedicated to reproducible, controlled experimental studies across Deep Reinforcement Learning (Proximal Policy Optimization) and Evolutionary Computation (Direct Search, Genetic Programming, Estimation of Distribution Algorithms). 

The upstream GitHub repository is [`PedroM2626/rlops-experiments`](https://github.com/PedroM2626/rlops-experiments); the local workspace is checked out in `ML-Games`.

---

## Scientific Scope and Architectural Overview

The repository focuses strictly on mathematically grounded, reproducible benchmarks adhering to academic empirical standards:

1. **PPO Empirical Studies: Reproducibility, Ecosystem Variance & Factorial Attribution** (`ppo-empirical-studies`) — A rigorous two-phase empirical investigation into Proximal Policy Optimization:
   - **Phase 1: Multi-Library Ecosystem Benchmark** (`01_ecosystem_benchmark`): Evaluating empirical divergence, sample efficiency, and variance across 7 PPO implementations (CleanRL, Stable-Baselines3, RLlib, TorchRL, Tianshou, PyTorch, JAX) on `CartPole-v1` and `LunarLander-v3` with formal non-parametric hypothesis testing (Mann-Whitney U, Levene, Bonferroni).
   - **Phase 2: Factorial Attribution & Advantage Mechanics** (`02_factorial_attribution`): Disentangling algorithmic specification from software runtime via a $2 \times 2$ factorial matrix (SB3 vs. CleanRL spec $\times$ PyTorch vs. JAX engine), single-delta ablations (value clipping, LR annealing, MSE scaling, timeout bootstrapping), and a dedicated empirical probe isolating Generalized Advantage Estimation (GAE) episode boundary indexing.
2. **Evolutionary Computation Taxonomy & JAX Benchmark** (`evolutionary-strategies`) — A formal taxonomy and comparative evaluation of three evolutionary computation paradigms (Direct Parameter Search, Symbolic Genetic Programming, and Probabilistic Models / Estimation of Distribution Algorithms like CMA-ES and PBIL) accelerated via JAX (`vmap` + `jit`) on canonical control tasks, with a 100-episode out-of-sample generalization protocol.
3. **Neuroevolution vs. Symbolic Programs in Complex Video Games** (`smw-evolutionary-agent`) — An empirical study comparing Neuroevolution of Augmenting Topologies (NEAT) against Genetic Programming (GP) playing *Super Mario World* via the BizHawk emulator, contrasting topological neural networks against interpretable symbolic decision trees under identical sensory receptors ($13 \times 13$ tile grid).

> [!NOTE]
> **Repository Curation & Deprecated Modules**:
> Early procedural 3D robotics and locomotion prototypes (`Chase`, `Race`, and `Parkour-Obstacles`) were permanently removed from the repository. Rigorous empirical auditing revealed fundamental physical and algorithmic discrepancies (such as non-physical kinematic chaser resets, unphysical lateral spring stabilizers $F_y = -80 \cdot y$, 0% goal completion rates, and severe policy collapse after $2.1\times 10^6$ steps). The monorepo has been streamlined to retain only scientifically validated, fully converging benchmarks.

---

## Sub-Project Directory

### 1. [`ppo-empirical-studies`](ppo-empirical-studies/README.md)
* **Objective**: Investigate variance, reproducibility, and mechanical attribution across PPO implementations and runtime stacks.
* **Phase 1: Multi-Library Ecosystem Benchmark** ([`01_ecosystem_benchmark`](ppo-empirical-studies/01_ecosystem_benchmark/README.md)):
  - Evaluates 7 distinct implementations: `cleanrl_original`, `cleanrl_style_pytorch`, `stable_baselines3`, `stable_baselines3_zoo`, `jax_pure`, `rllib`, `torchrl`, `tianshou`.
  - 10 seeds per configuration, formal hypothesis testing (Mann-Whitney U, Levene's variance homogeneity, Bonferroni corrections across 12 sequential empirical parts).
* **Phase 2: Factorial Attribution & Advantage Mechanics** ([`02_factorial_attribution`](ppo-empirical-studies/02_factorial_attribution/README.md)):
  - $2 \times 2$ factorial matrix (SB3 vs. CleanRL specification $\times$ PyTorch vs. JAX runtime).
  - Single-delta ablations (value clipping, LR annealing, MSE scaling, timeout bootstrapping).
  - GAE boundary indexing isolation probe ([`probe_gae`](ppo-empirical-studies/02_factorial_attribution/probe_gae/README.md)) isolating off-by-one advantage truncation.

### 2. [`evolutionary-strategies`](evolutionary-strategies/README.md)
* **Objective**: Comparative empirical evaluation of evolutionary algorithms classified by their mathematical object of variation:
  - **Family 1: Direct Solutions** (Direct Representation): Simple Genetic Algorithm (SimpleGA), Differential Evolution (DE), OpenAI Evolution Strategy (OpenAI-ES).
  - **Family 2: Symbolic Programs** (Genetic Programming): Linear Genetic Programming (LinearGP), Cartesian Genetic Programming (CartesianGP).
  - **Family 3: Probabilistic Models** (Estimation of Distribution Algorithms - EDA): Covariance Matrix Adaptation Evolution Strategy (CMA-ES), Population-Based Incremental Learning (PBIL).
* **Speedup & Evaluation**: Vectorized rollouts via JAX `vmap` and `jit` over `gymnax` environments (`CartPole-v1`, `Pendulum-v1`, `Acrobot-v1`, `MountainCar-v0`, `MountainCarContinuous-v0`). Includes a 100-episode out-of-sample evaluation protocol to measure the generalization gap.

### 3. [`smw-evolutionary-agent`](smw-evolutionary-agent/README.md)
* **Objective**: Compare topological neural network neuroevolution (NEAT) against symbolic tree genetic programming (GP) navigating *Super Mario World* (Yoshi's Island 1).
* **Architecture**: BizHawk emulator (Snes9x core) running custom Lua controllers (`MarIO.lua` and `MarIO_GP.lua`) interfacing with Python test harnesses.
* **Perception & Fitness**: Equivalent $13 \times 13$ tile perception window around Mario, with identical fitness metrics based on horizontal progress ($\Delta X$), timeout penalties, and velocity bonuses.

---

## Technology Stack

| Concern | Technologies | Active Projects |
|---|---|---|
| **Core DL & Tensors** | PyTorch (`torch`), JAX (`jax`, `jaxlib`, `optax`) | `ppo-empirical-studies`, `evolutionary-strategies` |
| **RL Baselines & Stacks** | Stable-Baselines3, CleanRL, Ray/RLlib, TorchRL, Tianshou | `ppo-empirical-studies` |
| **Vectorized RL Simulation** | `gymnax`, `lax.scan`, `vmap`, `jit` | `evolutionary-strategies`, `ppo-empirical-studies` |
| **Evolutionary Frameworks** | Custom JAX-accelerated ES/EDA/GA, Lua MarIO engine | `evolutionary-strategies`, `smw-evolutionary-agent` |
| **Emulation & Environment** | BizHawk 2.9.x, LuaJIT / Lua 5.1, Snes9x core | `smw-evolutionary-agent` |
| **Statistical Analysis** | SciPy, NumPy, Pandas (Mann-Whitney U, Levene, bootstrap confidence intervals, Cliff's delta) | `ppo-empirical-studies`, `evolutionary-strategies` |
| **Visualization** | Matplotlib, Seaborn | All projects |

---

## Repository Tree

```text
.
├── ppo-empirical-studies/        # Two-phase PPO empirical investigation
│   ├── README.md                 # Unified academic paper and synthesis
│   ├── 01_ecosystem_benchmark/   # Phase 1: 12-part multi-framework benchmark
│   │   ├── README.md             # Ecosystem benchmark report
│   │   ├── requirements.txt      # Multi-framework dependencies
│   │   ├── figures/              # High-resolution plots (.png)
│   │   ├── scripts/              # Implementation scripts and evaluation runners
│   │   │   ├── common.py         # Shared logging, evaluation, serialization
│   │   │   ├── train_*.py        # CleanRL, SB3, PyTorch, JAX, RLlib, TorchRL, Tianshou
│   │   │   ├── compare.py        # Multi-seed comparative curve plotting
│   │   │   ├── boxplot.py        # Return distribution boxplots
│   │   │   ├── eval_stats.py     # Evaluation statistics aggregation
│   │   │   └── stats_test.py     # Mann-Whitney U & Levene hypothesis tests
│   │   ├── results/              # Committed seed training curves (JSON)
│   │   └── results_eval/         # Committed evaluation distributions (JSON)
│   └── 02_factorial_attribution/ # Phase 2: 2x2 Factorial attribution & GAE probe
│       ├── README.md             # Factorial study documentation
│       ├── requirements-torch.txt# PyTorch stack requirements
│       ├── requirements-jax.txt  # JAX stack requirements
│       ├── config.py, specs.py   # Algorithmic specifications and hyperparameters
│       ├── variants.py           # Arm configurations and registry
│       ├── common.py, sb3_shim.py# Shared rollouts and NumPy 2 compatibility
│       ├── train_sb3.py          # Stable-Baselines3 driver
│       ├── train_cleanrl_torch.py# CleanRL PyTorch driver (cleanrl & sb3 modes)
│       ├── train_jax_ppo.py      # Pure JAX PPO driver (cleanrl, sb3, ablations)
│       ├── run_all.py            # Sequential benchmark runner
│       ├── run_full_background.py# Parallel multi-worker runner
│       ├── evaluate.py           # Deterministic multi-seed evaluation
│       ├── analyze.py            # Bootstrap resampling and attribution contrasts
│       ├── plot_results.py       # Training curves and attribution boxplots
│       ├── probe_gae/            # Empirical GAE boundary indexing isolation study
│       │   ├── analyze.py, run_probe.sh, README.md, data/
│       └── results/              # Committed tables (.csv, .md), curves (.npz), plots (.png)
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
| **`ppo-empirical-studies` (Phase 1)** | [`ppo-empirical-studies/01_ecosystem_benchmark/requirements.txt`](ppo-empirical-studies/01_ecosystem_benchmark/requirements.txt) | Includes PyTorch, JAX, Tianshou, TorchRL, RLlib, SB3. |
| **`ppo-empirical-studies` (Phase 2)** | [`ppo-empirical-studies/02_factorial_attribution/requirements-torch.txt`](ppo-empirical-studies/02_factorial_attribution/requirements-torch.txt)<br>[`ppo-empirical-studies/02_factorial_attribution/requirements-jax.txt`](ppo-empirical-studies/02_factorial_attribution/requirements-jax.txt) | Separate files allow isolated CPU/CUDA wheels for Torch and JAX. |
| **`evolutionary-strategies`** | [`evolutionary-strategies/requirements.txt`](evolutionary-strategies/requirements.txt) | Pure JAX, NumPy, SciPy, Matplotlib. |
| **`smw-evolutionary-agent`** | Python Standard Library + BizHawk 2.9+ | Cross-platform (Windows, Linux via Mono/native, macOS), SNES ROM. |

---

## Execution Guide

### 1. `ppo-empirical-studies` (Phase 1: Ecosystem Benchmark)
From inside `ppo-empirical-studies/01_ecosystem_benchmark/`:
```bash
# Execute CleanRL style on LunarLander-v3 with seed 42
python scripts/train_cleanrl.py --env-id LunarLander-v3 --seed 42

# Execute Stable-Baselines3 Zoo configuration
PPO_ENV=LunarLander-v3 PPO_SEED=42 python scripts/train_sb3_zoo.py

# Run hypothesis testing (Mann-Whitney U / Levene) across collected seeds
python scripts/stats_test.py
```

### 2. `ppo-empirical-studies` (Phase 2: Factorial Attribution)
From inside `ppo-empirical-studies/02_factorial_attribution/`:
```bash
# Smoke test (CartPole-v1, 1 seed, 1% timestep budget)
python run_all.py --env CartPole-v1 --seeds 0 --timesteps-scale 0.01

# Execute complete background run with multi-worker parallelism
python run_full_background.py --workers 4 --env LunarLander-v3

# Statistical evaluation, bootstrap resampling, and plot generation
python evaluate.py
python analyze.py
python plot_results.py

# Run GAE indexing probe analysis
python probe_gae/analyze.py
```

### 3. `evolutionary-strategies`
From inside `evolutionary-strategies/`:
```bash
# Run benchmark across all 3 evolutionary families (all vectorized in pure JAX)
python run_benchmark.py --quick

# Run Family 2 (LinearGP & CartesianGP) targeted benchmark in pure JAX
python run_gp_only.py

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
