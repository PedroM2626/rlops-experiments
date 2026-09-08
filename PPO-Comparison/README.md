# PPO Implementation Comparison

This folder contains a controlled experiment comparing 6 different implementations of the Proximal Policy Optimization (PPO) algorithm. The default environment is `LunarLander-v3` (override with `PPO_ENV_ID` or `python run_all.py --env <EnvId>`; results are stored under `results/<EnvId>/`).

The goal is to demonstrate how implementation details affect learning performance and final evaluation scores, even when hyperparameters are strictly identical.

## Implementations Compared

1. **Stable Baselines3 (`train_sb3.py`)**: The standard high-level reinforcement learning library.
2. **CleanRL (`train_cleanrl.py`)**: A single-file, clean implementation of PPO using PyTorch. Highly readable and reproducible.
3. **Custom PyTorch Continuous (`train_custom_continuous.py`)**: A minimal, from-scratch PyTorch implementation of PPO for continuous action spaces.
4. **Custom PyTorch Discrete (`train_custom_discrete.py`)**: A minimal, from-scratch PyTorch implementation of PPO for discrete action spaces. Used automatically when the selected env has a discrete action space.
5. **TorchRL (`train_torchrl.py`)**: The official PyTorch RL library.
6. **RLlib Extracted (`train_rllib_extracted.py`)**: A standalone PPO implementation that extracts Ray RLlib's specific algorithmic choices (dynamic KL penalty, value function clipping, independent networks) without being dependent on the `ray` package itself.

## Setup

All hyperparameters are fixed and shared via `config.py` to ensure a fair comparison.
These include:
- `TOTAL_TIMESTEPS = 1,000,000`
- `LEARNING_RATE = 2.5e-4`
- `N_STEPS = 2048`
- `BATCH_SIZE = 64`
- `N_EPOCHS = 4`
- `GAE_LAMBDA = 0.95`, `CLIP_RANGE = 0.2`, `ENT_COEF = 0.01`, `VF_COEF = 0.5`, `MAX_GRAD_NORM = 0.5`
- Network Architecture: `[64, 64]` with `Tanh` activation.
- `SEED = 42`

## Running the Experiment

You can run the entire experiment (training all 6 models, evaluating them for 100 episodes each, and generating plots) with the orchestrator script:

```bash
python run_all.py
```

### Options

- `python run_all.py --skip-train`: Skip training and only run evaluation/plots on existing models.
- `python run_all.py --skip-eval`: Skip the evaluation phase.
- `python run_all.py --skip-plot`: Skip plot generation.
- `python run_all.py --only cleanrl`: Train only the CleanRL implementation.
- `python run_all.py --env CartPole-v1`: Run the comparison on another environment.

## Results

After running the orchestrator, you will find (under `results/<EnvId>/`):
- **Trained Models**: Saved in `results/<EnvId>/models/` (weights `*.pt`/`*.zip` are git-ignored; retrain with `run_all.py` to regenerate)
- **Training Curves**: CSVs in `results/<EnvId>/curves/`
- **Evaluation Scores**: Raw data and statistics in `results/eval_rewards.csv` and `results/eval_stats.csv`
- **Plots**: Visualizations in `results/<EnvId>/plots/` (Training curves and evaluation boxplot)
- **MLflow Tracking**: All metrics, parameters, and artifacts are logged to `mlruns/` (view with `mlflow ui --backend-store-uri ./mlruns`)
