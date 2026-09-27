# Parkour Obstacles - 3D Humanoid Parkour RL Environment

## Overview

This project contains a Gymnasium reinforcement learning environment, `ParkourEnv`, backed by
PyBullet, where an articulated humanoid agent learns to run a procedurally generated parkour
course: platforms, gaps, steps up and down, narrow ledges, ramps, stairs, stepping stones and
sinusoidally moving platforms, ending on a goal platform marked by a golden sphere.

The system uses Stable Baselines3 (PPO), PyBullet for rigid-body physics, and MLflow for
tracking of hyperparameters, episode metrics and model checkpoints.

Every episode builds a fresh course (`level_generator.py`), so the policy is trained on an open
distribution of layouts rather than a single map.

## Features

- **Humanoid physics**: 10-DOF articulated URDF agent driven by per-joint PD position control.
- **Procedural levels**: 10 random sections per episode, sampled from 8 section types, with gaps
  and occasional low-friction "icy" platforms.
- **Terrain perception**: 5 downward raycasts fan out in front of the torso to probe the ground.
- **Lateral stabilisation**: an invisible spring keeps the agent from drifting off the course axis.
- **Hot-reloadable physics**: gravity, timestep, frame skip and camera can be changed at runtime
  through `world_config`, and every reward term through `reward_config`.
- **Interactive training UI**: PyBullet sliders and buttons for speed, timestep limit, pause and
  save, plus a live HUD drawn inside the 3D scene.
- **MLOps integrated**: hyperparameters, per-episode reward/length, checkpoints and
  `VecNormalize` stats are exported to MLflow.

## Project Structure

```
Parkour-Obstacles/
  assets/
    humanoid.urdf               10-DOF articulated humanoid model
  src/
    env/
      parkour_env.py            Gymnasium environment (44-dim obs, 10-dim actions)
      level_generator.py        Procedural parkour course builder
    agent/
      callbacks.py              MLflow SB3 callback (metrics + checkpoints)
    train.py                    Batch PPO training entrypoint
    train_interactive.py        Interactive training with PyBullet UI controls
    play.py                     Visualization / inference script
  train.py                      Root wrapper: adds the project dir to sys.path, calls src.train
  play.py                       Root wrapper: calls src.play
  test_env.py                   Headless environment smoke test
  models/                       Checkpoints and vec_normalize.pkl (git-ignored)
  mlruns/                       Local MLflow tracking store (git-ignored)
```

Dependencies, `Dockerfile` and `docker-compose.yml` live at the repository root and are shared
with the other experiments; this project has no local requirements file.

## Quick Start

All commands run from `Parkour-Obstacles/`. Prefer the `-m src.xxx` form: the modules import
each other as `src.env...`, so `python src/train_interactive.py` fails with
`ModuleNotFoundError: No module named 'src'` unless you export `PYTHONPATH=.` first.

### 1. Setup environment

```bash
cd Parkour-Obstacles
pip install -r ../requirements.txt      # shared deps: SB3, Gymnasium, PyBullet, MLflow, Torch
cp ../.env.example ../.env              # optional: overrides read as CLI defaults (root .env)
```

`../.env` supplies the defaults for `EXPERIMENT_NAME`, `MLFLOW_TRACKING_URI`, `TOTAL_TIMESTEPS`,
`LEARNING_RATE`, `N_STEPS`, `BATCH_SIZE`, `GAMMA`, `N_EPOCHS`, `GAE_LAMBDA`, `CLIP_RANGE`,
`ENT_COEF`, `NET_ARCH`, `N_ENVS` and `CHECKPOINT_FREQ`. Command-line flags always win.

### 2. Smoke-test the environment

Resets a level, prints the torso state and steps a few zero actions in `DIRECT` mode (no window):

```bash
python test_env.py
```

### 3. Train

```bash
# Defaults: 1M timesteps, 4 parallel envs, MLP [256, 256], device "auto"
python -m src.train

# Custom volume / parallelism / network
python -m src.train --timesteps 3000000 --n-envs 8 --net-arch 512,512

# Tune PPO
python -m src.train --lr 1e-4 --n-steps 1024 --batch-size 128 --clip-range 0.1

# Resume from a checkpoint (also works with the root wrapper: python train.py ...)
python -m src.train --model models/ppo_parkour_0001000000.zip

# Custom MLflow destination and checkpoint cadence
python -m src.train --experiment-name parkour_ppo_v2 --tracking-uri mlruns --checkpoint-freq 25000
```

Other supported flags: `--gamma --n-epochs --gae-lambda --ent-coef --model-dir --test --episodes`.

### 4. Train interactively (GUI controls)

```bash
python -m src.train_interactive --timesteps 3000000
python -m src.train_interactive --model models/ppo_parkour_final.zip --seed 42
python -m src.train_interactive --timesteps 0 --n-envs 8      # 0 = unlimited
```

Training runs in 4 headless envs while a separate GUI window plays the current policy live. The
PyBullet side panel exposes:

| Control | Range | Effect |
|---|---|---|
| `Speed` | 0.1 - 4.0 | Throttles the training loop (lower = slower, watchable) |
| `Timestep Limit (0=unlimited)` | 0 - 5,000,000 | Stops training when this step count is reached |
| `Pause / Resume` | button | Freezes the loop; the HUD shows `PAUSED` |
| `Save Checkpoint` | button | Saves the policy + `VecNormalize` stats into `models/` immediately |

The in-scene HUD shows status, step/limit, episode count, last and best reward, speed and elapsed
time. Other flags: `--load-model` (alias of `--model`), `--lr --n-steps --batch-size --gamma
--n-epochs --gae-lambda --clip-range --ent-coef --net-arch --n-envs --model-dir --seed
--checkpoint-freq --experiment-name --tracking-uri`.

### 5. Watch a trained agent

```bash
# GUI playback of the default checkpoint; runs until you close the PyBullet window
python -m src.play

# A specific checkpoint, a fixed number of episodes
python -m src.play --model models/ppo_parkour_0000500000 --n-episodes 3

# Headless benchmark on a fixed level layout
python -m src.play --no-render --n-episodes 20 --level-seed 42

# Same level, deterministic policy vs random actions (env sanity check)
python -m src.play --random --n-episodes 1 --level-seed 7

# Long episodes on a custom normalization file
python -m src.play --model models/ppo_parkour_final --vecnorm models/vec_normalize.pkl --steps 1500
```

`--model` is given without the `.zip` extension and defaults to `models/ppo_parkour_final`;
`--random` still loads the checkpoint, so the file must exist. Playback prints a per-episode line
and a final mean/std/max summary.

There is also a lightweight test mode in the trainer, which renders the GUI directly:

```bash
python -m src.train --test --model models/ppo_parkour_final.zip --episodes 5
```

Note that `--test` builds a plain env without `VecNormalize`, so its rewards read lower than
`src.play`, which restores the saved observation statistics.

### 6. View the MLflow dashboard

```bash
mlflow ui --backend-store-uri ./mlruns
# Open http://localhost:5000
```

## Docker

From the repository root (the root `Dockerfile` installs `requirements.txt` and sets the working
directory to `Parkour-Obstacles`):

```bash
docker compose up parkour-trainer      # python -m src.train, models/ and mlruns/ bind-mounted
docker compose up parkour-mlflow       # MLflow UI -> http://localhost:5003

docker build -t parkour-ml .           # manual image
docker run -it --rm parkour-ml python test_env.py
```

## Observation Space (44-dim)

PyBullet convention: X = forward (direction of travel), Y = lateral, Z = up.

| Range     | Description                                              |
|-----------|----------------------------------------------------------|
| [0:3]     | Torso position (x, y, z)                                  |
| [3:6]     | Torso linear velocity                                     |
| [6:9]     | Torso orientation (roll, pitch, yaw)                      |
| [9:12]    | Torso angular velocity                                    |
| [12:22]   | 10 joint angles, normalized -1..1 inside each joint limit  |
| [22:32]   | 10 joint velocities, clipped to [-20, 20]                 |
| [32:37]   | 5 ground raycasts, hit fraction 0..1                      |
| [37:40]   | Unit vector from torso toward the goal                    |
| [40]      | Left foot contact (0/1)                                   |
| [41]      | Right foot contact (0/1)                                  |
| [42]      | Torso height (world Z)                                    |
| [43]      | Time remaining fraction (1 - steps/2000)                  |

The raycasts fan downward at 15, 30, 45, 60 and 75 degrees below horizontal, 5 m long. Only the
ray, goal-direction, foot-contact and time slots have finite bounds; everything else is unbounded
and the vector is clipped to the space before `VecNormalize` (clip_obs = 10) normalizes it.

## Action Space (10-dim continuous [-1, 1])

Each dimension is a PD position target, mapped linearly onto that joint's limits and applied with
`POSITION_CONTROL` at `maxVelocity = 10 rad/s`, capped by the per-joint torque limit:

```
[l_shoulder, l_elbow, r_shoulder, r_elbow,
 l_hip, l_knee, l_ankle, r_hip, r_knee, r_ankle]
```

| Joint           | Limits (rad) | Max torque (Nm) |
|-----------------|--------------|-----------------|
| left/right_shoulder | -1.57 .. 1.57 | 60 |
| left/right_elbow    |  0.00 .. 2.27 | 40 |
| left/right_hip      | -1.05 .. 1.05 | 120 |
| left/right_knee     |  0.00 .. 2.09 | 100 |
| left/right_ankle    | -0.70 .. 0.70 | 60 |

Feet get extra friction so the agent can push off; default velocity motors are disabled so only
torque-driven position control moves the body.

## Reward Structure

Per step, summed in this order:

| Signal              | Value                    | Description                                   |
|---------------------|--------------------------|-----------------------------------------------|
| step_penalty        | -0.02                    | Per-step cost (encourages speed)              |
| progress_scale      | +5.0 x delta             | Reward for closing distance to the goal       |
| speed_scale         | +0.3 x forward_vel       | Only while progressing (delta > 0.01 m)       |
| upright_scale       | -0.4 x (&#124;roll&#124; + &#124;pitch&#124;) | Posture penalty for leaning                  |
| lateral_vel_scale   | -0.5 x &#124;lateral velocity&#124; | Penalty for side-to-side motion            |
| lateral_pos_scale   | -0.3 x &#124;y - spawn_y&#124;   | Penalty for drifting off the course axis      |
| energy_scale        | -0.001 x sum(action^2)   | Penalty for large joint commands              |
| goal_reward         | +150.0                   | Within 1.5 m of the goal (terminates episode) |
| fall_penalty        | -5.0                     | Terminal penalty for falling                  |

Every term can be overridden at runtime through the `reward_config` dict
(`goal_reward`, `fall_penalty`, `step_penalty`, `progress_scale`, `speed_scale`, `upright_scale`,
`lateral_vel_scale`, `lateral_pos_scale`, `energy_scale`). `ALIVE_BONUS`, `HEIGHT_SCALE` and
`STANDING_ABOVE` exist as class constants but are currently not part of the reward sum.

## Episode Termination

- **Terminated** when the goal is reached (distance < 1.5 m), or when the agent has fallen: any
  link other than the feet touching a surface (thigh, shin, torso, arms), or the torso dropping
  below Z = -2 after falling off the course.
- **Truncated** after `MAX_STEPS = 2000` policy steps.
- **Truncated by the stagnation cutoff**: past step 150, if the agent has advanced less than
  0.5 m along +X the episode ends. This stops the policy from farming safe standing states.

## Procedural Level Generation

`LevelGenerator` lays boxes along +X: a 5 x 4 m start platform at Z = 0, then `num_sections = 10`
sections separated by 0.1-0.5 m gaps, then a 5 x 6 m goal platform with a golden marker sphere
1 m above it. Each section is sampled from a weighted pool:

| Section          | Weight | Shape                                             |
|------------------|--------|---------------------------------------------------|
| flat             | 3/13   | length 3-6 m, width 2-4 m, same height            |
| raised           | 2/13   | step up 0.2-0.5 m, length 2-4 m                   |
| dropped          | 2/13   | step down 0.15-0.4 m (never below Z = 0)          |
| moving           | 2/13   | slides on Y or Z, amplitude 0.5-1.5 m, speed 0.8-1.5 |
| narrow           | 1/13   | ledge 1.0-1.5 m wide, offset up to 0.3 m laterally |
| ramp             | 1/13   | 8-15 degree incline over 2-4 m                    |
| stairs           | 1/13   | 3-5 steps, riser 0.1-0.2 m, tread 0.4-0.7 m       |
| stepping_stones  | 1/13   | 3-4 small blocks, zigzagging 0.15-0.4 m apart      |

About 20% of platforms are generated "icy": pale blue, with lateral friction 0.05 instead of the
usual 0.9-1.0. Moving platforms are animated each step as `amplitude * sin(speed * t)`, driven
from the `mover` user-data string attached at build time. A `complex_slope` layout is implemented
in the generator but is not in the sampling pool, so it is never produced right now.

Pass `level_seed` (or `--level-seed` in `src/play.py`) to get the same course; otherwise the
course seed is drawn from the env's RNG in 0..9998.

## Physics and Control

- Simulation at 480 Hz (`TIME_STEP = 1/480`), `FRAME_SKIP = 8` → the policy runs at 60 Hz.
- Episode budget of 2000 steps ≈ 33 s of simulated time.
- Gravity -9.81 m/s²; the agent spawns 1.4 m above the start platform and the world settles for
  20 sim steps before the first observation.
- An invisible lateral spring (`LATERAL_SPRING_K = 80`, clamped at ±200 N) pulls the torso back to
  the spawn Y line each step.
- `world_config` keys: `gravity_z`, `time_step`, `frame_skip`, `camera_distance`, `camera_yaw`,
  `camera_pitch`, `show_gui_panels`, `show_shadows`; applied via
  `env.set_runtime_config(reward_config=..., world_config=...)`.

## MLflow Tracking

Both entrypoints call `mlflow.set_tracking_uri(args.tracking_uri)` and
`mlflow.set_experiment(args.experiment_name)`, defaulting to `MLFLOW_TRACKING_URI` (else `mlruns`,
relative to the current directory) and `EXPERIMENT_NAME` (else `parkour_ppo`). Inside the run:

- **Params**: `algorithm`, `learning_rate`, `n_steps`, `batch_size`, `gamma`, `n_epochs`,
  `gae_lambda`, `clip_range`, `ent_coef`, `net_arch`, `n_envs`, `device`, plus `total_timesteps`
  (`src/train.py`) or `seed` and `timestep_limit` (`src/train_interactive.py`).
- **Metrics**: `episode_reward` and `episode_length`, logged by `MLflowCallback` each time a
  wrapped episode finishes, stepped against the SB3 timestep counter.
- **Artifacts**: checkpoints under `checkpoints/` (`ppo_parkour_<step>.zip` every
  `--checkpoint-freq` steps, plus a `_final` copy at training end); `src/train.py` also logs the
  final policy and `vec_normalize.pkl` under `model/`.

## Artifacts on Disk

| Path | Contents |
|------|----------|
| `models/ppo_parkour_<10-digit step>.zip` | Periodic checkpoints (every `--checkpoint-freq`, default 50k steps; the interactive trainer also saves one every 60 s) |
| `models/ppo_parkour_<step>_final.zip` | Training-end checkpoint written by `MLflowCallback` |
| `models/ppo_parkour_final.zip` | Final policy (`src/train.py` and the interactive trainer) |
| `models/vec_normalize.pkl` | Observation/reward normalization stats, required by `src/play.py` |
| `mlruns/` | Local MLflow file store |

`*.zip`, `*.pkl` and `mlruns/` are git-ignored on purpose: they are regenerable by training and
would bloat the repository.

## Training Details

- **PPO** with `MlpPolicy`, separate actor/critic backbones of shape `--net-arch` (default
  `256,256`) and `Tanh` activations.
- `DummyVecEnv` over `--n-envs` monitored parkour envs, wrapped in `VecNormalize`
  (`norm_obs`, `norm_reward`, `clip_obs = 10.0`).
- Device `auto` (uses the GPU if Torch finds one); `src/train.py` trains with a progress bar, the
  interactive trainer runs until the timestep limit, the pause button, or Ctrl+C and then saves
  the final model.
