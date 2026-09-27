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
same `compute_gae`, so a null here applies to both.

`run_probe.sh` copies the study sources into `probe_gae/workspace/` and toggles the
line there. It never writes into this project's `results/`, which stay as the record
of the study.

## Results

`python analyze.py` (needs scipy) reproduces every number below from `data/`.

| comparison | metric | A | B | paired B−A | 95% CI | p (paired t) |
|---|---|---|---|---|---|---|
| LunarLander-v3, n=15 | eval mean over 200 episodes | 78.6 ± 34.0 | 53.4 ± 54.2 | **−25.2** | [−58.2, +7.8] | 0.124 |
| CartPole-v1, n=15, budget 0.1 | final training curve | 323.3 ± 73.5 | 302.8 ± 48.8 | **−20.5** | [−70.8, +29.8] | 0.397 |
| CartPole-v1, n=5, budget 0.05 | final training curve | 238.7 ± 38.5 | 227.9 ± 37.2 | −10.8 | [−79.1, +57.4] | 0.751 |

Wilcoxon, Mann-Whitney and Levene agree: nothing is significant in any comparison,
including on the spread (Levene p = 0.136 on LunarLander, 0.210 on CartPole).

## Reading

- **No detectable effect.** Fixing the index did not improve either arm on either
  environment. On LunarLander the point estimate goes the *other* way (B 32% below
  A) but the interval reaches past zero, so that is not evidence either.
- **The bound is the useful output.** On CartPole any true mean shift is under
  ~22% of the arm's score; on LunarLander the data cannot exclude a −58 shift,
  which is why the LunarLander number is the one worth chasing if this is revisited.
- **Resolving LunarLander needs ~41 seeds per arm** (80% power to detect the
  observed −25.2 at σ=57.6), roughly 4 h of CPU. The full-study retrain the
  deviation would imply is ~17.6 h, and this probe gives no reason to spend it.
- **Seeds, not episodes, are the lever.** Within a LunarLander policy the
  per-episode sd is 120.4, so a 200-episode mean carries ±8.5 of measurement error
  against a between-seed sd of 34.0. Doubling evaluation episodes again would cut
  the contrast's SE by ~1%; tripling seeds cuts it by 42%.

## Files

- `run_probe.sh <cartpole|lunarlander> [python]` — drives both arms.
- `analyze.py` — paired t, Wilcoxon, Levene, Mann-Whitney, CI, from `data/`.
- `data/ll_eval_rewards_{A,B}.csv` — 200 episodes × 15 seeds, the primary evidence.
- `data/ll_seed_means.csv`, `data/ll_curve_finals.csv` — tidy summaries.
- `data/cartpole_finals.csv` — arm, budget scale, seed, final curve value.
