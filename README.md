# Exo-Kuiper: 3D Parkour RL

A 3D parkour game where an AI agent trained with Reinforcement Learning (PPO) navigates platforms, gaps, ramps, and ledges to reach a goal zone.

![Architecture](docs/arch.png)

## Overview

| Component | Technology |
|-----------|-----------|
| 3D physics simulation | PyBullet |
| RL algorithm | PPO (Stable Baselines3) |
| Experiment tracking | MLflow |
| Containerisation | Docker / Docker Compose |

## How It Works

1. A capsule-shaped agent spawns at the start of a procedurally generated parkour course.
2. At each timestep it receives a 20-dimensional observation (position, velocity, orientation, raycasts, goal direction) and outputs 3 continuous actions (forward force, lateral force, jump impulse).
3. PPO learns to maximise cumulative reward: progress toward the goal, reaching the goal (+100), and a small time penalty to encourage speed.
4. Every run is tracked in MLflow with all hyperparameters, per-episode metrics, and model checkpoints.

## Project Structure

```
exo-kuiper/
├── src/
│   ├── env/
│   │   ├── parkour_env.py        # Gymnasium environment (PyBullet)
│   │   └── level_generator.py   # Procedural level builder
│   ├── agent/
│   │   └── callbacks.py         # MLflow + SB3 callback
│   ├── train.py                 # Training entrypoint
│   └── play.py                  # Watch the trained agent
├── models/                      # Saved checkpoints
├── mlruns/                      # MLflow tracking data
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Clone and install

```bash
git clone <repo-url>
cd exo-kuiper
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

### 3. Train

```bash
# Start MLflow UI in another terminal first (optional)
mlflow ui --backend-store-uri mlruns

# Train the agent (default: 1M timesteps, 4 parallel envs)
python src/train.py

# Or with explicit overrides
python src/train.py --timesteps 500000 --n-envs 8 --lr 0.0001
```

### 4. Watch the agent

```bash
python src/play.py --model models/ppo_parkour_final
```

## Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--timesteps` | 1 000 000 | Total env steps |
| `--lr` | 3e-4 | PPO learning rate |
| `--n-steps` | 2048 | Steps per rollout per env |
| `--batch-size` | 64 | Mini-batch size |
| `--gamma` | 0.99 | Discount factor |
| `--n-epochs` | 10 | PPO update epochs |
| `--n-envs` | 4 | Parallel environments |
| `--net-arch` | `256,256` | MLP hidden layers |
| `--experiment-name` | `parkour_ppo` | MLflow experiment name |

## MLflow Tracking

Open the UI after training:
```bash
mlflow ui --backend-store-uri mlruns
# Navigate to http://localhost:5000
```

Each run logs:
- All hyperparameters
- `episode_reward` and `episode_length` per episode
- Model checkpoints as artifacts (every 50k steps by default)
- Final model + VecNormalize stats

## Docker

```bash
# Build and train inside Docker
docker compose up trainer

# Start MLflow UI only
docker compose up mlflow-ui
```

## Environment Details

### Observation Space (20-dim)

| Index | Description |
|-------|-------------|
| 0-2 | Agent position (x, y, z) |
| 3-5 | Linear velocity |
| 6-8 | Orientation (roll, pitch, yaw) |
| 9-11 | Angular velocity |
| 12-16 | Forward raycasts (5 rays, normalized [0, 1]) |
| 17-19 | Unit vector to goal |

### Action Space (3-dim continuous, [-1, 1])

| Index | Description |
|-------|-------------|
| 0 | Forward thrust |
| 1 | Lateral thrust |
| 2 | Jump impulse (applied only when on ground) |

### Reward Function

| Event | Reward |
|-------|--------|
| Progress toward goal | `+2 * delta_distance` |
| Reaching goal | `+100` |
| Falling off level | `-10` |
| Per timestep | `-0.005` |

## Requirements

- Python 3.10+
- PyBullet 3.2.6
- Stable Baselines3 2.3.2
- PyTorch 2.3.1
- MLflow 2.14.1

See `requirements.txt` for exact versions.
