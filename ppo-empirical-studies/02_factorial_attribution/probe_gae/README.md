# GAE indexing probe

Measures what the deviation recorded in the project README (§8, *Known deviation,
deliberately unfixed*) is actually worth. `common.compute_gae` gates the bootstrap
of step *t* with `dones[t + 1]`, while SB3 and CleanRL gate it with `dones[t]`, so
every episode boundary in the manual arms is shifted one step. The question this
probe answers is not whether that is wrong — it is — but whether it moves the
results enough to justify retraining the manual arms.

## Design

Two arms, identical in every respect except the one toggled line:

| arm | indexing | meaning |
|---|---|---|
| A | `dones[t + 1]` | as committed in this study |
| B | `dones[t]` | SB3 / CleanRL convention |

Same interpreter and libraries as the committed runs (Python 3.11.9, torch 2.5.1,
numpy 2.4.6, gymnasium 1.3.0), same seeds paired across arms, reduced budget so the
curves are still in their rising part rather than pinned at the ceiling. The arm is
`cleanrl_torch`, one of the two manual implementations; both manual arms share the
same `compute_gae`, so the effect measured here is expected to carry to the JAX arm
as well, though only the Torch arm was actually run.

`run_probe.sh` copies the study sources into `probe_gae/workspace/` and toggles the
line there. It never writes into this project's `results/`, which stay as the record
of the study.

## Results

`python analyze.py` (needs scipy) reproduces every number below from `data/`.

| comparison | metric | A | B | paired B−A | 95% CI | p (paired t) |
|---|---|---|---|---|---|---|
| LunarLander-v3, n=41 | eval mean over 200 episodes | 85.3 ± 27.8 | 64.1 ± 48.2 | **−21.1** | [−36.9, −5.4] | **0.0098** |
| — first 15 seeds only | same | 78.6 ± 34.0 | 53.4 ± 54.2 | −25.2 | [−58.2, +7.8] | 0.124 |
| — seeds 15-40 only | same | — | — | −18.8 | — | 0.041 |
| CartPole-v1, n=15, budget 0.1 | final training curve | 323.3 ± 73.5 | 302.8 ± 48.8 | −20.5 | [−70.8, +29.8] | 0.397 |
| CartPole-v1, n=5, budget 0.05 | final training curve | 238.7 ± 38.5 | 227.9 ± 37.2 | −10.8 | [−99.2, +77.5] | 0.751 |

On LunarLander the paired difference is significant (Wilcoxon p = 0.022), B is worse
on 27 of 41 seeds, and the spread differs too: **Levene p = 0.0025**, with seed-to-seed
sd 27.8 for A against 48.2 for B. CartPole shows the same sign but never reaches
significance.

## Robustness

The 41 seeds came from two separate runs of the harness: the first 15, then 26 more
produced in a later session with no seed overlap. The effect replicates in the
independent block on its own (−18.8, p = 0.041), so it is not an artifact of one batch
or of the index toggle being applied to only one arm. Median paired difference is
−12.3 and the quartiles are [−41.8, −12.3, +12.0]: a shift in the whole distribution,
not three unlucky seeds.

## Reading

- **The deviation is material on the discriminating environment, and in the
  unexpected direction.** The committed indexing scores ~21 points (25%) *higher*
  than the SB3/CleanRL convention on LunarLander and is significantly more
  consistent across seeds. CartPole shows the same sign without reaching
  significance.
- **Higher is not the same as correct.** Gating the bootstrap with `dones[t + 1]`
  cuts every episode boundary one step early, so the terminal step bootstraps from
  the first observation of the next episode. That this happens to help on this
  environment at this budget is an empirical accident, not evidence that the
  convention is wrong. What it does establish is that the manual arms are not
  running a textbook GAE, so every manual-vs-SB3 contrast in the study carries this
  as a confound.
- **What correcting it would cost.** Retraining the manual arms (~17.6 h) would
  lower their reported LunarLander scores by roughly a quarter and widen their
  seed spread, which changes the contrasts and could change the conclusions. That is
  a research decision, so it is recorded here rather than taken.
- **Seeds, not episodes, are the lever.** Within a LunarLander policy the
  per-episode sd is 120.4, so a 200-episode mean carries ±8.5 of measurement error
  against a between-seed sd of 34.0. Doubling evaluation episodes again would cut
  the contrast's SE by ~1%; tripling seeds cuts it by 42%.

## Files

- `run_probe.sh <cartpole|lunarlander> [python]` — drives both arms; `SEEDS="..."`
  overrides the seed list to extend the sample.
- `analyze.py` — paired t, Wilcoxon, Levene, Mann-Whitney, CI and the seeds-needed
  figure, from `data/`. It unions every `ll_eval_rewards_<arm>*.csv` present.
- `data/ll_eval_rewards_{A,B}.csv` — seeds 0-14, and `..._ext.csv` — seeds 15-40.
  200 episodes per cell; this is the primary evidence for the LunarLander result.
- `data/ll_seed_means.csv` — one row per arm and seed (all 41), with mean, sd and
  episode count, matching the matrices above.
- `data/ll_curve_finals.csv` — final training-curve value for the first 15 seeds;
  the extension was kept as evaluation data only, since each arm's models directory
  is reused and the curve files are not part of the headline metric.
- `data/cartpole_finals.csv` — arm, budget scale, seed, final curve value.
