# Chase - Humanoid RL Environment

## Overview
This project contains an interactive reinforcement learning environment `ChaseEnv` using PyBullet, where an AI agent humanoid learns to survive and outrun another rule-based humanoid chaser mimicking human gait in a TAG game.

The system utilizes Stable Baselines3 (PPO), PyBullet for realistic human locomotion physics, and MLflow for modern MLOps tracking of models and metrics.

## Features
- **Humanoid Physics**: Custom PyBullet physics setup. Ensure accurate representation of bipedal locomotion.
- **Rule-based Chaser AI**: The enemy chaser possesses identical physical attributes (gravity and shape) as the agent, moving via a robust programmed sinusoidal walk controller to ensure perfect balancing while pursuing the agent.
- **Intervention UI**: Provides sliders and real-time visualization tweaks to control RL behavior and save ML checkpoints dynamically.
- **MLOps Integrated**: Directly exports all hyperparams, charts, and `.zip` model files to an MLflow interface for strict model versioning and metrics tracking.

## Getting Started

### Local Setup
Using Python 3.10+:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Docker
To run in an isolated container instance easily deployable across environments:
```bash
docker build -t chase-ml .
docker run -it --rm chase-ml python test_chase.py
```

## Running the Project

### Train Interactive
Trains the RL agent dynamically:
```bash
export PYTHONPATH=.
python src/train_interactive.py --timesteps 500000
```
Use the `--model` flag to resume from an existing stable_baselines3 checkpoint. 

### Testing
To examine the environment simply and watch the chaser approach without training loop overheads:
```bash
export PYTHONPATH=.
python test_chase.py
```
