#!/bin/bash
# Reproduce the compute_gae off-by-one probe.
#
# The probe never writes into the study's results/: it copies the study sources to
# a scratch workspace, toggles one line in common.compute_gae there, and brings
# back only CSVs into probe_gae/data/.
#
# usage: run_probe.sh <cartpole|lunarlander> [python]
#   needs the study's own deps (torch, gymnasium; box2d for LunarLander)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY="$(dirname "$HERE")"
PY="${2:-python}"
ENVIRONMENT="${1:-cartpole}"

case "$ENVIRONMENT" in
  cartpole)    ENV_ID=CartPole-v1;     SCALE=0.1; SEEDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14" ;;
  lunarlander) ENV_ID=LunarLander-v3;  SCALE=0.2; SEEDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14" ;;
  *) echo "environment must be cartpole|lunarlander" >&2; exit 2 ;;
esac

WORK="$HERE/workspace"
BUGGY='next_non_term = 1.0 - float(dones\[t + 1\])'
FIXED='next_non_term = 1.0 - float(dones\[t\])'

run_arm () {                        # $1 = A|B
  local arm="$1"
  rm -rf "$WORK/results"
  for seed in $SEEDS; do
    "$PY" train_cleanrl_torch.py --env "$ENV_ID" --seed "$seed" --mode cleanrl \
        --timesteps-scale "$SCALE" >"$WORK/train_${arm}_${seed}.log" 2>&1 \
      && echo "ok   $arm seed=$seed" || echo "FAIL $arm seed=$seed"
  done
  if [ "$ENV_ID" = "LunarLander-v3" ]; then
    "$PY" evaluate.py --env "$ENV_ID" >"$WORK/eval_${arm}.log" 2>&1
    cp "$WORK/results/tables/eval_rewards_${ENV_ID}.csv" \
       "$HERE/data/ll_eval_rewards_${arm}.csv"
  fi
  for f in "$WORK"/results/curves/cleanrl_torch_${ENV_ID}_seed*.csv; do
    seed=$(basename "$f" | sed -E "s/.*_seed([0-9]+)\.csv/\1/")
    tail -n 1 "$f" | awk -v a="$arm" -v s="$seed" -F, \
      '{print a","s","$2","NR}'
  done > "$WORK/finals_${arm}.csv"
}

mkdir -p "$WORK" "$HERE/data"
cp "$STUDY"/*.py "$WORK/"

if [ "$ENV_ID" = "LunarLander-v3" ]; then
  sed -i "s/^N_EVAL_EPISODES = 100/N_EVAL_EPISODES = 200/" "$WORK/config.py"
fi

echo "=== arm A: committed indexing (dones[t + 1]) ==="
sed -i "s/$FIXED/$BUGGY/" "$WORK/common.py"
grep -q "float(dones\[t + 1\])" "$WORK/common.py" || { echo "toggle to A failed"; exit 1; }
run_arm A

echo "=== arm B: SB3/CleanRL indexing (dones[t]) ==="
sed -i "s/$BUGGY/$FIXED/" "$WORK/common.py"
grep -q "1.0 - float(dones\[t\])" "$WORK/common.py" || { echo "toggle to B failed"; exit 1; }
run_arm B

echo "finals per arm (arm,seed,final_curve,1):"
cat "$WORK"/finals_*.csv
echo "now run: python analyze.py"
