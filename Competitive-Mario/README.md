# Competitive Mario -- Multi-Agent Competition System

A reinforcement learning system where multiple Mario agents are trained independently
and then compete against each other on the same procedurally generated side-scrolling
level. Each agent uses PPO with a CNN policy, and all training runs are tracked with MLflow.

## Architecture

```
Competitive-Mario/
├── src/
│   ├── env/
│   │   ├── mario_env.py            # Custom Gymnasium environment (pure Python)
│   │   └── level_generator.py      # Procedural level builder
│   ├── agent/
│   │   └── callbacks.py            # MLflow + SB3 callback
│   ├── train.py                    # Train individual agents
│   ├── compete.py                  # Run competition between trained agents
│   └── watch_race.py               # Pygame visualization of races
├── configs/
│   └── agents.yaml                 # Agent hyperparameter profiles
├── models/                         # Saved checkpoints (per agent)
├── results/                        # Competition results and charts
├── gym-super-mario-bros/           # Reference submodule (unused at runtime)
├── requirements.txt
├── Dockerfile
├── test_competitive_mario.py
└── README.md
```

## How It Works

1. **Level Generation**: A procedural generator creates a Mario-style level with ground,
   gaps, pipes, platforms, enemies, coins, and a flag pole. Levels are seeded for
   reproducibility so all agents compete on the exact same level.

2. **Training**: Each agent trains independently on the level using PPO with a CNN policy.
   The observation is an 84x84 RGB image of the viewport around Mario, stacked 4 frames
   deep for temporal context.

3. **Competition**: Trained agents are loaded and each runs through N episodes on the
   same level. Metrics are collected (x position, flag completion, score, time) and a
   composite ranking is computed.

4. **Visualization**: A Pygame window shows all agents playing simultaneously in a grid
   layout with a live scoreboard.

## Quick Start

### 1. Install

```bash
cd Competitive-Mario
pip install -r requirements.txt
```

### 2. Train Agents

Train each agent with its configured profile:

```bash
# Train with YAML config (recommended)
python src/train.py --agent-name mario_speedster --config configs/agents.yaml
python src/train.py --agent-name mario_careful --config configs/agents.yaml
python src/train.py --agent-name mario_balanced --config configs/agents.yaml
python src/train.py --agent-name mario_explorer --config configs/agents.yaml

# Or train with custom hyperparameters
python src/train.py --agent-name custom_mario --lr 0.0005 --gamma 0.95 --timesteps 300000
```

### 3. Run Competition

```bash
# Auto-discover all trained agents
python src/compete.py

# Specify agents explicitly
python src/compete.py --agents mario_speedster mario_careful mario_balanced

# Custom level
python src/compete.py --level-seed 123 --difficulty 0.7 --n-episodes 10
```

### 4. Watch the Race

```bash
# Watch all trained agents compete
python src/watch_race.py

# Specific agents
python src/watch_race.py --agents mario_speedster mario_careful
```

Controls:
- **R**: Restart the race
- **ESC**: Exit

### 5. Run Tests

```bash
pytest test_competitive_mario.py -v
```

## Agent Profiles

| Agent | Style | LR | Gamma | Entropy | Timesteps |
|-------|-------|----|-------|---------|-----------|
| mario_speedster | Aggressive, short-term | 5e-4 | 0.95 | 0.02 | 500K |
| mario_careful | Conservative, long-term | 1e-4 | 0.999 | 0.005 | 750K |
| mario_balanced | Middle ground | 3e-4 | 0.99 | 0.01 | 500K |
| mario_explorer | High exploration | 3e-4 | 0.98 | 0.05 | 500K |

## Environment Details

### Observation Space
- Shape: (84, 84, 3) uint8 RGB
- 4-frame stack for temporal context (final shape: 84x84x12)
- Viewport follows Mario horizontally

### Action Space
Discrete(5):

| Action | Description |
|--------|-------------|
| 0 | NOOP |
| 1 | RIGHT |
| 2 | RIGHT + JUMP |
| 3 | JUMP |
| 4 | LEFT |

### Reward Function

| Event | Reward |
|-------|--------|
| Move right (per tile) | +1.0 |
| Time penalty (per frame) | -0.01 |
| Death | -15.0 |
| Reach flag | +50.0 |
| Collect coin | Score +50 |

### Level Features
- Flat ground with gaps of varying width
- Pipes of varying height
- Floating brick and question-block platforms
- Goomba-style enemies (killable by jumping on them)
- Collectible coins
- Flag pole at the end of the level

## Competition Scoring

Agents are ranked by a composite score:
- 50% flag completion rate
- 30% average x position (normalized)
- 10% average game score (normalized)
- 10% average time remaining (normalized)

## MLflow Tracking

```bash
mlflow ui --backend-store-uri Competitive-Mario/mlruns
# Navigate to http://localhost:5000
```

Each training run logs:
- All hyperparameters (agent name, algorithm, policy, LR, gamma, etc.)
- Per-episode metrics (reward, length, x_pos, flag completions)
- Model checkpoints as artifacts
- Final model

Competition runs log:
- Per-agent rank and composite score
- Comparison charts (bar, radar)

## Docker

```bash
# Build and train
docker build -t competitive-mario .
docker run -v ./models:/app/models -v ./mlruns:/app/mlruns competitive-mario

# Train a specific agent
docker run competitive-mario python src/train.py --agent-name mario_speedster
```

## Requirements

- Python 3.10+
- Stable Baselines3 2.7.1
- Gymnasium 1.1.1
- PyTorch 2.10.0
- MLflow 3.10.1
- Pygame CE 2.5.7 (for visualization)

See `requirements.txt` for exact versions.
