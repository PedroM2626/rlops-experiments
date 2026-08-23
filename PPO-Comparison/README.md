# PPO Implementation Comparison

This folder contains a controlled experiment comparing 4 different implementations of the Proximal Policy Optimization (PPO) algorithm on the `LunarLander-v3` environment.

The goal is to demonstrate how implementation details affect learning performance and final evaluation scores, even when hyperparameters are strictly identical.

## Implementations Compared

1. **Stable Baselines3 (`train_sb3.py`)**: The standard high-level reinforcement learning library.
2. **CleanRL (`train_cleanrl.py`)**: A single-file, clean implementation of PPO using PyTorch. Highly readable and reproducible.
3. **Custom PyTorch (`train_custom.py`)**: A minimal, from-scratch PyTorch implementation of PPO.
4. **TorchRL (`train_torchrl.py`)**: The official PyTorch RL library.
5. **RLlib Extracted (`train_rllib_extracted.py`)**: A standalone PPO implementation that extracts Ray RLlib's specific algorithmic choices (dynamic KL penalty, value function clipping, independent networks) without being dependent on the `ray` package itself.

## Setup

All hyperparameters are fixed and shared via `config.py` to ensure a fair comparison. 
These include:
- `TOTAL_TIMESTEPS = 500,000`
- `LEARNING_RATE = 2.5e-4`
- `N_STEPS = 2048`
- `BATCH_SIZE = 64`
- `N_EPOCHS = 4`
- Network Architecture: `[64, 64]` with `Tanh` activation.
- `SEED = 42`

## Running the Experiment

You can run the entire experiment (training all 4 models, evaluating them for 100 episodes each, and generating plots) with the orchestrator script:

```bash
python run_all.py
```

### Options

- `python run_all.py --skip-train`: Skip training and only run evaluation/plots on existing models.
- `python run_all.py --only cleanrl`: Train only the CleanRL implementation.

## Results

After running the orchestrator, you will find:
- **Trained Models**: Saved in `results/models/`
- **Training Curves**: CSVs in `results/curves/`
- **Evaluation Scores**: Raw data and statistics in `results/eval_rewards.csv` and `results/eval_stats.csv`
- **Plots**: Visualizations in `results/plots/` (Training curves and evaluation boxplot)
- **MLflow Tracking**: All metrics, parameters, and artifacts are logged to `mlruns/` (view with `mlflow ui`).
