# ML-Games: Open Source Machine Learning Experiments in Games

A comprehensive repository for open source machine learning experiments in games, featuring 3D physics simulations (PyBullet). This project includes reinforcement learning training pipelines and experiment tracking with MLflow.

## Project Overview

This repository contains multiple ML experiments for games:

### RL Experiments (PyBullet + Stable Baselines3)

- **Chase (Pega-Pega)**: A humanoid agent must evade a floating capsule that chases it at increasing speed
- **Parkour Obstacles**: A humanoid agent must navigate procedurally generated platforms, gaps, ramps, and ledges
- **Race**: Procedural terrain generation for competitive multi-agent racing environments
- **PPO Implementation Comparison**: A controlled experiment comparing Stable Baselines3, CleanRL, Custom PyTorch (continuous + discrete), TorchRL, and RLlib-extracted implementations of PPO

## Technology Stack

| Component | Technology |
|-----------|-----------|
| 3D physics simulation | PyBullet |
| RL algorithm | PPO (Stable Baselines3) |
| Experiment tracking | MLflow |
| Containerisation | Docker / Docker Compose |

## Project Structure

```
ML-Games/
├── Chase/                          # PyBullet chase environment
│   ├── src/
│   │   ├── env/chase_env.py       # Gymnasium environment
│   │   ├── agent/callbacks.py     # MLflow + SB3 callback
│   │   ├── train.py               # Batch training
│   │   └── train_interactive.py   # Interactive training
│   └── models/                    # Saved checkpoints
│
├── Parkour-Obstacles/             # PyBullet parkour environment
│   ├── src/
│   │   ├── env/parkour_env.py     # Gymnasium environment
│   │   ├── env/level_generator.py # Procedural level builder
│   │   ├── agent/callbacks.py
│   │   ├── train.py
│   │   ├── train_interactive.py
│   │   └── play.py
│   └── models/
│
├── Race/                          # Procedural terrain racing
│   ├── src/
│   │   ├── env/race_env.py
│   │   ├── env/terrain_generator.py
│   │   ├── train.py
│   │   └── play.py
│   └── models/
│
├── PPO-Comparison/                # PPO implementation comparison
│   ├── config.py                  # Shared hyperparameters
│   ├── run_all.py                 # Orchestrator
│   └── train_*.py                 # One script per implementation
│
├── Dockerfile                    # Parkour training image
├── docker-compose.yml
├── requirements.txt              # Shared training deps (Parkour uses this file)
└── .env.example
```

## Quick Start

### 1. Install Dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env to adjust hyperparameters or MLflow URI
```

### 3. Training

**Chase (interactive):**
```bash
cd Chase
python src/train_interactive.py
```

**Parkour (interactive):**
```bash
cd Parkour-Obstacles
python src/train_interactive.py
```

**Race:**
```bash
cd Race
python -m src.train --timesteps 1000000
```

### 4. Batch Training

```bash
# Chase
cd Chase
python train.py --timesteps 2000000

# Parkour
cd Parkour-Obstacles
python -m src.train --timesteps 1000000
```

See `PPO-Comparison/README.md` for the PPO implementation comparison orchestrator.

## MLflow Tracking

Each project tracks its own runs locally. From the project dir:

```bash
cd Parkour-Obstacles  # or Race, Chase, PPO-Comparison
mlflow ui --backend-store-uri ./mlruns
# Navigate to http://localhost:5000
```

Each run logs:
- All hyperparameters and device
- `episode_reward` and `episode_length` per episode
- Model checkpoints every 50k steps (as artifacts)
- Final model + VecNormalize stats

> Checkpoints (`*.zip`/`*.pkl`/`*.pt`) are git-ignored on purpose — they are regenerable via training and would bloat the repo (~560 MB before this change).

## Docker

```bash
docker compose up chase-trainer   # Chase training
docker compose up parkour-trainer # Parkour training
docker compose up race-trainer    # Race training
docker compose up ppo-comparison  # PPO comparison orchestrator
# MLflow viewers: :5000 chase, :5003 parkour, :5001 race, :5002 ppo-comparison
docker compose up chase-mlflow parkour-mlflow race-mlflow ppo-comparison-mlflow
```

## Environment Details

### Chase Environment (42-dim observation)

| Range | Description |
|-------|-------------|
| 0-1 | sin/cos angle to chaser |
| 2 | normalized distance to chaser |
| 3-5 | agent linear velocity |
| 6-8 | agent orientation (roll, pitch, yaw) |
| 9-18 | joint angles (10 joints) |
| 19-28 | joint velocities (10 joints) |
| 29-41 | raycasts (16 rays) |

**Reward Function:**
- Survival per step: +0.01
- Increasing distance from chaser: +2.0 * delta
- Being within 3m of chaser: -0.5 * (3 - dist)
- Fall / caught: -50

### Parkour Environment (44-dim observation)

| Range | Description |
|-------|-------------|
| 0-2 | torso position (x, y, z) |
| 3-5 | linear velocity |
| 6-8 | orientation (roll, pitch, yaw) |
| 9-11 | angular velocity |
| 12-21 | 10 joint angles (normalized) |
| 22-31 | 10 joint velocities |
| 32-36 | 5 ground raycasts |
| 37-39 | unit vector toward goal |
| 40-41 | foot contacts (left, right) |
| 42 | torso height |
| 43 | time remaining fraction |

**Reward Function:**
- Progress toward goal: +5.0 * delta
- Forward speed (when progressing): +0.3 * forward_vel
- Reaching goal: +150
- Fall penalty: -5
- Lateral drift: penalty proportional to Y offset
- Per timestep: -0.02

## Requirements

- Python 3.10+
- PyBullet 3.2.6
- Stable Baselines3 2.7.1
- PyTorch 2.10.0
- MLflow 3.10.1

See `requirements.txt` for exact versions (`Chase`, `Race`, and `PPO-Comparison` ship their own requirements files; `Parkour-Obstacles` uses the root one).

## Notes

- All paths in the project are relative to the repository root
