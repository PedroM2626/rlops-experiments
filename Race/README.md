# Race - Competitive Multi-Agent Humanoid Walking Race

A 3D reinforcement learning project where multiple humanoid agents with ragdoll
physics compete to reach a goal as fast as possible without falling.

Agents interact physically with each other and navigate procedurally generated
terrain with hills, rocks, ramps, and pillars. Training uses PPO with parameter
sharing (one shared policy updated by all agent trajectories simultaneously).

## Features

- **N competing humanoids** (default: 4) in a single PyBullet physics world
- **Ragdoll physics**: full 10-DOF articulated bodies with joint torque control
- **Dual-Phase Stability System**:
    - *Phase 1 (Forces)*: Upright torque (Stabilizer.cs style) and critical angular damping (150.0) for organic swaying.
    - *Phase 2 (Constraints)*: Hard-clamping of Roll/Pitch/Yaw (±30° max) and vertical/horizontal velocity caps (prevents explosive kinetics).
- **Physical interaction**: agents can collide and interfere with each other
- **Procedural terrain**: every episode generates a fresh map with rocks, hills,
  pillars, and ramps seeded randomly
- **Competitive rewards**: rank bonus, goal bonus by finishing position,
  progress-based rewards, upright/stability rewards
- **Parameter sharing training**: one PPO policy trained from all agent
  trajectories simultaneously (self-play)
- **Full MLflow tracking**: hyperparameters, episode metrics, charts, checkpoints, and model typing
- **Docker support**: containerized training environment

## Project Structure

```
Race/
  assets/
    humanoid.urdf             10-DOF articulated humanoid model
  src/
    env/
      race_env.py             Multi-agent Gymnasium environment
      terrain_generator.py    Procedural terrain with obstacles
    agent/
      callbacks.py            MLflow SB3 callback
    train.py                  PPO training entrypoint
    play.py                   Visualization / inference script
  tests/
    test_race_env.py          Automated test suite
  Dockerfile
  requirements.txt
  .env.example
```

## Quick Start

### 1. Setup environment

```bash
cd Race
pip install -r requirements.txt
cp .env.example .env
```

### 2. Run tests

```bash
python -m pytest tests/test_race_env.py -v
```

### 3. Watch random agents (no training needed)

```bash
python -m src.play --random --n-agents 4
```

### 4. Train agents

```bash
# Default: 4 agents, 3M timesteps
python -m src.train

# Custom settings
python -m src.train --n-agents 4 --n-envs 4 --timesteps 5000000

# Continue from checkpoint
python -m src.train --model models/ppo_race_0001000000.zip
```

### 5. Watch trained agents race

```bash
python -m src.play --model models/ppo_race_final.zip
```

### 6. View MLflow training dashboard

```bash
mlflow ui --backend-store-uri ./mlruns
# Open http://localhost:5000
```

## Docker

```bash
# Train with Docker (from project root)
docker compose up race-trainer

# MLflow dashboard
docker compose up race-mlflow
# Open http://localhost:5001
```

## Observation Space (56-dim per agent)

| Range     | Description                                       |
|-----------|---------------------------------------------------|
| [0:3]     | Torso position (x, y, z)                         |
| [3:6]     | Torso linear velocity                             |
| [6:9]     | Torso orientation (roll, pitch, yaw)              |
| [9:12]    | Torso angular velocity                            |
| [12:22]   | 10 joint angles (normalized -1..1)                |
| [22:32]   | 10 joint velocities                               |
| [32:37]   | 5 downward raycasts (ground probe)                |
| [37:40]   | Unit vector toward goal                           |
| [40:42]   | Foot contacts: left, right                        |
| [42]      | Torso height (Z)                                  |
| [43]      | Time remaining (fraction)                         |
| [44:47]   | Closest opponent relative position (normalized)   |
| [47:50]   | 2nd closest opponent relative position            |
| [50]      | Race rank normalized (0=1st, 1=last)              |
| [51]      | Being-pushed flag                                 |
| [52:56]   | Reserved / padding                                |

## Action Space (10-dim continuous [-1, 1])

Joint position targets mapped to physical joint limits:

```
[l_shoulder, l_elbow, r_shoulder, r_elbow,
 l_hip, l_knee, l_ankle, r_hip, r_knee, r_ankle]
```

## Reward Structure

| Signal              | Value     | Description                                    |
|---------------------|-----------|------------------------------------------------|
| step_penalty        | -0.01     | Per-step cost (encourages speed)               |
| alive_bonus         | +0.005    | Per-step reward for staying upright            |
| progress_scale      | 6.0 x delta | Reward for closing distance to goal          |
| speed_scale         | 0.4 x vel | Bonus for forward speed when progressing       |
| upright_scale       | -0.3      | Penalty per radian of roll+pitch tilt          |
| energy_scale        | -0.001    | Penalty for joint torque usage                 |
| rank_bonus_step     | +0.03     | Per-step bonus when race leader                |
| goal_bonus          | +200/120/60/20 | Finishing 1st/2nd/3rd/4th              |
| fall_penalty        | -8.0      | Terminal penalty for falling                   |
| collision_penalty   | -0.05     | Per-step penalty while being pushed            |

## Terrain Features

Generated fresh every episode:
- Flat ground base (60 m long x 20 m wide corridor)
- Random rolling hills (8-15 per episode)
- Random rocks / box obstacles (12-20)
- Cylindrical pillars to navigate around (6-12)
- Inclined ramps at various angles (4-8)
- Clear start zone (first 8 m) and goal zone (last 8 m)
- Golden goal arch at finish line

## Training Details

Uses **PPO** from Stable Baselines3 with:
- Parameter sharing: one policy, N_AGENTS x N_ENVS trajectories
- VecNormalize for observation and reward normalization
- Larger network (512, 512, 256) to handle 56-dim competitive obs
- Tanh activation
