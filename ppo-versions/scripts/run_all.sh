#!/bin/bash
cd /home/claude/ppo-benchmark
export PPO_TIMESTEPS=150000
SEEDS="1 2 3 4 5"

for SEED in $SEEDS; do
  export PPO_SEED=$SEED
  echo "=== seed $SEED: sb3 ===" 
  python3 scripts/train_sb3.py
  echo "=== seed $SEED: cleanrl ===" 
  python3 scripts/train_cleanrl.py
  echo "=== seed $SEED: jax ===" 
  python3 scripts/train_jax.py
  echo "=== seed $SEED: rllib ===" 
  python3 scripts/train_rllib.py
done
echo "ALL DONE"
