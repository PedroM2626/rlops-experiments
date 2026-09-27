# PPO: SB3 vs CleanRL — Implementation or Library?

2×2 factorial experiment plus a bridge arm, asking whether the SB3–CleanRL
gap comes from the **algorithmic details** or from the **software stack**.
It runs on **CartPole-v1** and **LunarLander-v3** with identical hyperparameters.

## 1. Question and design

| arm | code | algorithm spec | stack |
|---|---|---|---|
| `sb3_torch` | SB3 `PPO` | SB3 | SB3 + Torch |
| `cleanrl_torch` | `train_cleanrl_torch.py --mode cleanrl` | CleanRL | manual Torch |
| `cleanrl_sb3mode_torch` | `train_cleanrl_torch.py --mode sb3` | **SB3** | manual Torch (bridge) |
| `jax_sb3` | `train_jax_ppo.py --mode sb3` | SB3 | JAX + Optax |
| `jax_cleanrl` | `train_jax_ppo.py --mode cleanrl` | CleanRL | JAX + Optax |
| `jax_abl_*` (×4) | `train_jax_ppo.py --mode cleanrl --abl <delta>` | CleanRL + 1 SB3 delta | JAX + Optax (ablation) |

Ablation arms (§8): `novclip` (no value clipping), `noanneal` (constant LR),
`fullmse` (no 0.5 factor on the MSE), `tboot` (with timeout bootstrapping).

Planned contrasts: Q1 total gap; Q2/Q3 stack effect with the spec held fixed;
Q4 does the bridge close the gap? Q5 spec effect inside JAX;
Q6 Torch bridge vs JAX under the same spec (sanity check).

## 2. What is "the same" and what differs (see `specs.py`)

Identical: LR 2.5e-4, γ 0.99, λ 0.95, ε 0.2, N_STEPS 2048, batch 64,
epochs 4, ent 0.01, vf 0.5, grad-clip 0.5, Adam eps 1e-5, MLP [64,64]
Tanh with separate actor/critic backbones, ortho-init (hidden √2, heads 0.01/1.0),
advantage normalized per minibatch, n_envs=1, seeds {0..4}, CPU.

Spec deltas (the only permitted algorithmic differences):
SB3 = no value clipping, constant LR, full MSE, timeout bootstrapping;
CleanRL = value clipping, LR with linear annealing, 0.5×MSE, no bootstrapping.
Everything documented and referenced in `specs.py`.

## 3. Rigor

5 training seeds per (arm, env); deterministic evaluation over 100
episodes with fixed seeds; curves = mean ± SD across seeds; seed-paired
contrasts with 95% bootstrap (N=5000) + Cliff's delta; every run
saves a `meta.json` with the spec, hyperparameters and package versions.

## 4. How to run

Prerequisite: Python 3.10–3.11 with the packages from both requirements
files (this study's setup uses ONE interpreter with torch+sb3+jax installed
together; if you prefer to split them, `run_all.py --python-torch X --python-jax Y`
accepts two interpreters):

```bash
pip install -r requirements-torch.txt   # SB3 + Torch + Gymnasium
pip install -r requirements-jax.txt     # JAX + Optax
```

Notes for this repo on Windows:
- `sb3_shim.py` is loaded automatically and works around the
  `cv2/_ARRAY_API not found` bug (SB3 + NumPy 2) — no manual workaround needed.
- All training is on CPU and single-threaded per process
  (`torch.set_num_threads(1)`, `OMP_NUM_THREADS=1`); throughput comes from
  running jobs in parallel, not from many threads.

Runs:

```bash
# smoke test (1 env, 1 seed, 1% of the budget; validates the pipeline)
python run_all.py --env CartPole-v1 --seeds 0 --timesteps-scale 0.01

# full experiment, sequential (50 runs; slow: ~1 day on a single CPU)
python run_all.py

# recommended: resume-safe parallel runner (skips jobs whose model already exists)
python run_full_background.py --workers 8                 # everything still missing
python run_full_background.py --workers 8 --env LunarLander-v3
python run_full_background.py --workers 8 --only jax_sb3,jax_cleanrl
python run_full_background.py --workers 8 --env LunarLander-v3 \
    --only jax_abl_novclip,jax_abl_noanneal,jax_abl_fullmse,jax_abl_tboot
python run_full_background.py --workers 8 --seeds 5,6,7,8,9  # extra seeds

# post-processing (evaluation + statistics + figures); this already runs at the
# end of run_all.py, but it can be re-run on its own:
python evaluate.py      # 100 deterministic episodes per model -> tables/
python analyze.py       # seed-paired bootstrap -> contrasts_*.md
python plot_results.py  # training curves + boxplots -> plots/
```

Costs measured on this hardware (i9-14900HX, CPU): SB3 ≈ 236s (CartPole) /
572s (LunarLander) per run; manual Torch ≈ 190–200s / 838s; JAX ≈ 500s /
740–1010s. Outputs: `results/models`, `results/curves`, `results/tables`,
`results/plots`, `results/meta`, `results/logs`.

## 5. Reading the results

If Q4/Q6 ≈ 0 and Q2/Q3 ≈ 0 but Q1/Q5 ≠ 0 → the difference is implementation-level.
If Q2/Q3 are large even with the spec held fixed → the stack/library carries weight.
The bridge (`cleanrl_sb3mode_torch`) is the crucial test: the same manual Torch
code as CleanRL, but wearing the SB3 spec.

## 6. Files

`config.py` hyperparameters; `specs.py` deltas; `variants.py` arms;
`train_*.py` training scripts; `jax_run_fast.py` (the default JAX backend of
`train_jax_ppo.py`: the entire update jitted per epoch via `lax.scan`; same
math as `jax_run.py`, which stays in the repo as a readable/legacy reference,
~3× slower); `sb3_shim.py` cv2/NumPy2 compatibility; `evaluate.py`,
`analyze*.py`, `plot_*.py` evaluation/statistics/figures; `run_all.py`
sequential orchestrator; `run_full_background.py` resume-safe parallel
runner (skips jobs whose model already exists).
Weights (`results/models/*`) are git-ignored and regenerable.

## 7. Results obtained (10 seeds × 100 deterministic episodes)

**CartPole-v1**: saturates at the 500 ceiling in 4 arms (including both "pure"
specs); only the manual bridge sits at ~440 ± 62 (small non-zero Q4/Q6,
a stack residue at saturation; the env remains barely informative).

**LunarLander-v3** (the discriminating env, 10 seeds; ablations with 5):

| arm | return (mean ± SD) |
|---|---|
| CleanRL-code/SB3-spec (bridge) | 151.4 ± 20.2 |
| JAX/SB3-spec | 142.7 ± 29.0 |
| SB3 (Torch lib) | 135.6 ± 72.4 (1 collapsed seed: 8.8) |
| CleanRL (Torch) | 108.1 ± 46.7 |
| JAX/CleanRL-spec | 92.6 ± 44.8 |

Main contrasts (paired bootstrap, 95% CI): Q2/Q3/Q4/Q6 ≈ 0 and NS —
the **stack produces no difference**; Q5 (spec inside JAX) = **+50.3
[25.5, 78.1], p<1e-4, δ=0.72** — the SB3 spec beats the CleanRL spec. Q1 = +27.1
[−28.1, 79.5], NS (variance inflated by the collapsed seed).

**Ablation of the 4 deltas (JAX, CleanRL + 1 SB3 delta):** vs `jax_cleanrl`:

| delta swapped | gain vs CleanRL | 95% CI | p | vs jax_sb3 |
|---|---|---|---|---|
| − value clipping | **+41.9** | [20.6, 57.8] | <1e-4 | +4.4 (NS — the gap closes) |
| − LR annealing | +20.4 | [4.3, 41.8] | <1e-4 | −17.0 (NS marginal, p=.076) |
| − 0.5 factor on the MSE | +0.3 | [−18, 18] | NS | −37.5 |
| + timeout bootstrapping | −8.6 | [−27, 12] | NS | −46.2 |

Answer: the dominant delta is **value-loss clipping** (removing the clip
recovered the SB3 level on its own); LR annealing is a secondary contributor;
MSE scaling and timeout bootstrapping do not change the picture in this setup.
Tables: `results/tables/contrasts_*.md` and `contrasts_ablation_*.md`.

## 8. Limitations / errata applied

- **Bug found and fixed in this study (errata)**: `terminal_observation`
  does not exist on a gymnasium single env, so the timeout bootstrap was
  *inactive* (dead branch) in the manual arms; and the 0.5 MSE factor was
  swallowed by the clipped branch. Both were fixed (bootstrap from the
  truncation observation, `vhalf` applied in both branches) and every arm
  with the manual SB3 spec + the relevant ablations were **retrained from
  scratch**; the numbers above already reflect the fix. The ablation shows the
  delta had an effect ≈ 0 on these envs (NS), so the Q2/Q4/Q6 conclusions held.
- Power: 10 seeds in the main arms and 5 in the ablations; Q1 still has a wide
  CI because of SB3's collapsed seed. `solved` (≥200 on LunarLander)
  requires extra seeds or per-env tuning, out of scope for this comparison.
- CartPole remains saturated; a smaller budget (~50–100k steps) would make it
  discriminative again.
