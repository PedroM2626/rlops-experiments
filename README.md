# ML-Games: Open Source Machine Learning Experiments in Games

A comprehensive repository for open source machine learning experiments in games, featuring both 3D physics simulations (PyBullet) and game engine integration (Godot). This project includes reinforcement learning training pipelines, experiment tracking with MLflow, and ONNX model export for deployment in game engines.

## Project Overview

This repository contains multiple ML experiments for games:

### Python-based RL Experiments (PyBullet + Stable Baselines3)

- **Chase (Pega-Pega)**: A humanoid agent must evade a floating capsule that chases it at increasing speed
- **Parkour Obstacles**: A humanoid agent must navigate procedurally generated platforms, gaps, ramps, and ledges
- **Race**: Procedural terrain generation for competitive racing environments
- **Arena 2D**: A lightweight top-down sandbox for rapid PPO iteration and reward shaping
- **PPO Implementation Comparison**: A controlled experiment comparing Stable Baselines3, CleanRL, Custom PyTorch, and TorchRL implementations of PPO on LunarLander-v3

### Godot-based ML Experiments

- **ML Survival Arena**: 3D game in Godot where a humanoid agent must survive as long as possible from a pursuer
- Full-body physics locomotion with RigidBody3D joints
- ONNX inference server integration for real-time policy execution
- In-game training interface with live metrics

## Technology Stack

| Component | Technology |
|-----------|-----------|
| 3D physics simulation | PyBullet |
| 2D sandbox simulation | NumPy + Gymnasium |
| RL algorithm | PPO (Stable Baselines3) |
| Experiment tracking | MLflow |
| Live dashboard | Streamlit |
| Game engine | Godot 4.5+ |
| Deployment/export | ONNX + ONNX Runtime |
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
│   │   ├── train.py
│   │   └── play.py
│   └── models/
│
├── ml_games_engine/               # Unified ML experiment engine
│   ├── runner.py                  # Unified runtime
│   ├── scenarios.py               # Scenario registry
│   ├── brains.py                  # Live viewport brain switcher
│   ├── catalog.py                 # Checkpoint discovery
│   ├── exporters.py               # ONNX exporter
│   └── envs/arena2d_env.py        # 2D sandbox
│
├── ml/                             # Godot ML integration
│   ├── train_agent.py              # Training pipeline
│   ├── onnx_inference_server.py   # ONNX inference server
│   ├── game_env.py                 # Game environment wrapper
│   ├── policy.py                   # Policy network
│   └── requirements.txt
│
├── scenes/                         # Godot scenes
│   ├── Main.tscn                   # Main game scene
│   ├── FullBodyLocomotion.tscn     # Full-body physics
│   ├── HumanoidAgent.tscn
│   ├── Pursuer.tscn
│   └── HUD.tscn
│
├── scripts/                        # Godot scripts
├── Models/                         # 3D models
├── streamlit_app.py                # Unified dashboard
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── project.godot                   # Godot project file
```

## Quick Start

### Python-based Experiments

#### 1. Install Dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

#### 2. Configure

```bash
cp .env.example .env
# Edit .env to adjust hyperparameters or MLflow URI
```

#### 3. Engine Dashboard (Recommended)

Start the unified dashboard:

```bash
streamlit run streamlit_app.py
```

From the dashboard you can:
- Choose a 2D or 3D scenario
- Browse discovered checkpoints and VecNormalize files
- Set PPO hyperparameters
- Inspect world and physics settings
- Pause and resume training
- Hot-reload reward weights
- Switch between live viewport brains
- View real-time metrics (utilizing non-blocking Streamlit fragments for smooth updates)
- Access the local/remote MLflow UI via the sidebar tracking shortcut link
- Export ONNX models for Unity, Godot, Unreal
- Run reliably on Windows with concurrent file access retries

#### 4. Interactive Training

**Chase:**
```bash
cd Chase
python src/train_interactive.py
```

**Parkour:**
```bash
cd Parkour-Obstacles
python src/train_interactive.py
```

**Race:**
```bash
cd Race
python src/train.py --timesteps 1000000
```

#### 5. Batch Training

```bash
# Chase
cd Chase
python src/train.py --timesteps 2000000

# Parkour
cd Parkour-Obstacles
python src/train.py --timesteps 1000000
```

### Godot-based Experiments

#### 1. Install Dependencies

- Godot 4.5+
- Python 3.10+

```bash
pip install -r ml/requirements.txt
```

#### 2. Run the Game

1. Open the project in Godot
2. Run the main scene (configured as main scene)
3. If ONNX model exists in `ml/models/agent_policy.onnx` and ONNX toggle is active, the game starts the inference server automatically
4. If no ONNX exists, the agent uses heuristic policy until you train

#### 3. Train via In-Game Interface

1. Adjust Generations, Population, Episode Time, and Seed
2. Check/uncheck Reuse checkpoint
3. Click Start Training
4. Monitor metrics in the Training panel

Generated files:
- `ml/models/agent_policy.pt` (checkpoint for continuing training)
- `ml/models/agent_policy.onnx` (model for inference)
- `ml/models/training_status.json` (real-time status)
- `ml/models/training_history.jsonl` (generation history)

#### 4. Manual Training

```bash
python ml/train_agent.py --model-dir ml/models --generations 60 --population 28 --episode-seconds 45 --reuse true
```

#### 5. Manual Inference Server

```bash
python ml/onnx_inference_server.py --model ml/models/agent_policy.onnx --host 127.0.0.1 --port 8765
```

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

## ONNX Export

The engine exports deployment-friendly ONNX bundles for:
- Python projects with Stable Baselines3
- Pure ONNX Runtime without training stack
- Unity, Godot, or Unreal integration

Automatic flow:
- Every final run writes an ONNX export bundle to `runtime/runs/<run_id>/engine_export/`
- The Streamlit dashboard exposes bundles in Saved Runs and Assets tabs
- Exported ONNX includes VecNormalize observation normalization

Manual CLI export:

```bash
python -m ml_games_engine.exporters ^
  --scenario arena2d ^
  --model runtime/runs/<run_id>/models/ppo_arena2d_final.zip ^
  --vecnorm runtime/runs/<run_id>/models/vec_normalize.pkl ^
  --out runtime/runs/<run_id>/engine_export
```

Bundle contents:
- `policy.onnx`: deterministic clipped inference graph
- `manifest.json`: observation/action contract, bounds, and scenario metadata
- `normalization.json`: exported normalization stats
- `validation.json`: PyTorch vs ONNXRuntime parity check
- `python/run_policy.py`: standalone ONNX Runtime example
- `unity/MLGamesPolicyRunner.cs`: Unity integration starter
- `godot/ml_games_policy_runner.gd`: Godot integration starter
- `unreal/MLGamesPolicyRunner.h/.cpp`: Unreal integration starter

## Requirements

- Python 3.10+
- Godot 4.5+ (for Godot experiments)
- PyBullet 3.2.6
- Stable Baselines3 2.7.1
- PyTorch 2.10.0
- MLflow 3.10.1
- ONNX 1.20.1
- ONNX Runtime 1.24.3

See `requirements.txt` for exact versions.

## Notes

- The FBX model in `Models/RobotKyle.fbx` is loaded in the agent when available
- The procedural rig (capsules/spheres) remains active for separate body part movement
- If Python is not in PATH, adjust `python_command` in `OnnxPolicyClient.gd` and `TrainingManager.gd`
- All paths in the project are relative to the repository root
