# ML-Games: 2D/3D ML Experiment Engine

An experimentation engine for 2D and 3D ML game projects, with a shared PPO core,
live control dashboard, hot-reloadable reward shaping, and real-time viewport previews.

| Component | Technology |
|-----------|-----------|
| 3D physics simulation | PyBullet |
| 2D sandbox simulation | NumPy + Gymnasium |
| RL algorithm | PPO (Stable Baselines3) |
| Experiment tracking | MLflow |
| Live dashboard | Streamlit |
| Containerisation | Docker / Docker Compose |

## Games

### Chase (Pega-Pega)
A humanoid agent must evade a floating capsule that chases it at increasing speed. Survive as long as possible.

### Parkour Obstacles
A humanoid agent must navigate procedurally generated platforms, gaps, ramps, and ledges to reach a goal zone.

### Arena 2D
A lightweight top-down sandbox for rapid PPO iteration, reward shaping, and dashboard smoke tests.

## Project Structure

```
ML-Games/
├── Chase/
│   ├── src/
│   │   ├── env/chase_env.py           # Chase Gymnasium environment (PyBullet)
│   │   ├── agent/callbacks.py         # MLflow + SB3 callback
│   │   ├── train.py                   # Batch training entrypoint
│   │   └── train_interactive.py       # Interactive training with in-window controls
│   ├── models/                        # Saved checkpoints
│   └── play_chase.py                  # Watch the trained agent
│
├── Parkour-Obstacles/
│   ├── src/
│   │   ├── env/
│   │   │   ├── parkour_env.py         # Parkour Gymnasium environment (PyBullet)
│   │   │   └── level_generator.py     # Procedural level builder
│   │   ├── agent/callbacks.py         # MLflow + SB3 callback
│   │   ├── train.py                   # Batch training entrypoint
│   │   ├── train_interactive.py       # Interactive training with in-window controls
│   │   └── play.py                    # Watch the trained agent
│   └── models/                        # Saved checkpoints
│
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── requirements.txt
└── README.md
```

New engine pieces:
- `ml_games_engine/runner.py`: unified runtime with hot-reloadable control/state files
- `ml_games_engine/scenarios.py`: registry for the built-in 2D and 3D scenarios
- `ml_games_engine/brains.py`: live viewport brain switcher
- `ml_games_engine/catalog.py`: automatic discovery of checkpoints, vecnormalizers, and saved runs
- `ml_games_engine/envs/arena2d_env.py`: built-in 2D sandbox
- `streamlit_app.py`: dashboard for hyperparameters, reward shaping, metrics, and control

## Quick Start

### 1. Install

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

### 3. Engine Dashboard (recommended)

Start the unified dashboard:

```bash
streamlit run streamlit_app.py
```

From the dashboard you can:
- choose a 2D or 3D scenario before hitting play
- browse discovered checkpoints and VecNormalize files without typing paths
- set PPO hyperparameters
- inspect world and physics settings such as gravity, timestep, frame skip, and camera setup
- pause and resume training
- hot-reload reward weights without closing the run
- hot-reload physics and world settings while the run is alive
- switch the live viewport brain between the in-training policy, a random baseline, or a PPO checkpoint
- disable the viewport brain entirely when you only want raw training throughput
- inspect episode reward and training loss curves in real time
- open a saved-runs tab that reads archived configs, state history, metrics, charts, checkpoints, and preview snapshots

Notes:
- 3D scenarios open a separate PyBullet window for the viewport, with the default debug panes and shadow noise disabled.
- The 2D scenario streams frames directly into the dashboard.
- Runtime state is written to `runtime/`, and each run is archived under `runtime/runs/<run_id>/`.

### 4. Legacy Interactive Training

Both games have an interactive training script that opens a PyBullet window with **live controls embedded directly in the simulation window**:

**Chase:**
```bash
cd Chase
python src/train_interactive.py
# Continue from checkpoint:
python src/train_interactive.py --model models/ppo_chase_final.zip
```

**Parkour:**
```bash
cd Parkour-Obstacles
python src/train_interactive.py
# Continue from checkpoint:
python src/train_interactive.py --model models/ppo_parkour_final.zip
```

#### In-Window Controls (PyBullet sliders & buttons)

| Control | Description |
|---------|-------------|
| **Speed** slider (0.1 - 4.0) | Training speed multiplier |
| **Timestep Limit** slider (0 = unlimited) | Stop training after N timesteps |
| **Pause / Resume** button | Toggle training on/off |
| **Save Checkpoint** button | Save model immediately |

#### Live HUD (rendered in the 3D scene)

The top of the scene shows a yellow HUD with:
- Status (running / PAUSED)
- Current step vs. limit
- Episode count
- Last and best episode reward
- Speed multiplier
- Elapsed wall-clock time

### 5. Batch Training

```bash
# Chase
cd Chase
python src/train.py --timesteps 2000000

# Parkour
cd Parkour-Obstacles
python src/train.py --timesteps 1000000
```

### 6. Watch the Agent

```bash
# Chase
cd Chase
python play_chase.py --model models/ppo_chase_final.zip

# Parkour
cd Parkour-Obstacles
python src/play.py --model models/ppo_parkour_final
```

## Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | None | Load pretrained model to continue training |
| `--timesteps` | 0 (unlimited) | Timestep limit (interactive) / 1M (batch) |
| `--lr` | 3e-4 | PPO learning rate |
| `--n-steps` | 2048 | Steps per rollout per env |
| `--batch-size` | 64 | Mini-batch size |
| `--gamma` | 0.99 | Discount factor |
| `--n-epochs` | 10 | PPO update epochs |
| `--n-envs` | 4 | Parallel training environments |
| `--net-arch` | `256,256` | MLP hidden layers |
| `--experiment-name` | `chase_ppo` / `parkour_ppo` | MLflow experiment name |

## MLflow Tracking

```bash
mlflow ui --backend-store-uri Chase/mlruns      # Chase
mlflow ui --backend-store-uri Parkour-Obstacles/mlruns  # Parkour
# Navigate to http://localhost:5000
```

Each run logs:
- All hyperparameters and device
- `episode_reward` and `episode_length` per episode
- Model checkpoints every 50k steps (as artifacts)
- Final model + VecNormalize stats

## Docker

```bash
docker compose up trainer      # Build and train
docker compose up mlflow-ui    # Start MLflow UI only
```

## Chase Environment

### Observation Space (42-dim)
| Range | Description |
|-------|-------------|
| 0-1 | sin/cos angle to chaser |
| 2 | normalized distance to chaser |
| 3-5 | agent linear velocity |
| 6-8 | agent orientation (roll, pitch, yaw) |
| 9-18 | joint angles (10 joints) |
| 19-28 | joint velocities (10 joints) |
| 29-41 | raycasts (16 rays) |

### Reward Function
| Event | Reward |
|-------|--------|
| Survival per step | +0.01 |
| Increasing distance from chaser | +2.0 * delta |
| Being within 3m of chaser | -0.5 * (3 - dist) |
| Fall / caught | -50 |

## Parkour Environment

### Observation Space (44-dim)
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

### Reward Function
| Event | Reward |
|-------|--------|
| Progress toward goal | +5.0 * delta |
| Forward speed (when progressing) | +0.3 * forward_vel |
| Reaching goal | +150 |
| Fall penalty | -5 |
| Lateral drift | penalty proportional to Y offset |
| Per timestep | -0.02 |

## Requirements

- Python 3.10+
- PyBullet 3.2.6
- Stable Baselines3 2.7.1
- PyTorch 2.10.0
- MLflow 3.10.1

See `requirements.txt` for exact versions.
