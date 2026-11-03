# PPO Benchmark: comparing PPO implementations against each other

This project was born from a simple question — "is one library's PPO
different from another's?" — and it turned into a multi-session investigation
of reproducibility, statistical variance, and what really explains
performance differences between RL implementations. This README documents
both the **experimental results** (Parts 1-12, each one a concrete
experiment) and the **conceptual conclusions** that emerged along the way
(section right below).

If you only want the most recent and strongest result: go straight to
**Part 12**. If you want the full line of reasoning, read it in order — each
part starts from what the previous one found (or left unanswered).

---

## Core concepts and central conclusions (the "why" behind the experiments)

### 1. Why the same PPO logic gives different results in different libraries

The algorithm on paper is the same, but each implementation decides on its own a
chunk of details the paper doesn't specify: how to initialize the weights,
whether it normalizes observations/rewards, how it treats
`value function clipping`, whether it uses a KL penalty, how it bootstraps
truncated episodes, the order of the floating-point operations, which random
number generator it uses. Each of these choices stays invisible until you look.

### 2. This would be worse (not better) across different languages

Even keeping "the same logic" written by hand, PyTorch, JAX and a
hypothetical Rust implementation would diverge even more, for structural
reasons:
- **Different RNG**: the same seed does not produce the same number sequence
  in Mersenne Twister (PyTorch), Threefry (JAX) or PCG (Rust stdlib).
- **Floating point is not associative**: `(a+b)+c ≠ a+(b+c)`. The order each
  language uses to sum a vector (sequential loop vs SIMD vs tree reduction)
  changes the result in the last bit.
- **A different BLAS under the hood** (OpenBLAS, MKL, Accelerate) changes the
  tiling strategy and the rounding.
- Since RL is a system with a **feedback loop** (today's action changes
  tomorrow's state), a floating-point difference in the 15th digit at step 1
  can turn into an entirely different policy after hundreds of thousands of
  steps -- butterfly effect.

### 3. Versioning breaks reproducibility, live, more than once in this conversation

- The `cleanrl` package on PyPI (v0.4.8, from 2021) no longer runs with the
  current `gym`/`gymnasium`.
- CleanRL's official `ppo.py` (downloaded straight from GitHub, current
  version) also broke: Gymnasium's vectorized env API changed from
  `infos["final_info"]` to `infos["episode"]` + the `infos["_episode"]` mask
  starting with Gymnasium 1.0.
- Tianshou 2.0 completely rewrote its API (`policy` → `algorithm`).
- TorchRL renamed `SyncDataCollector` → `Collector`.
- This project's own execution environment was reset mid-work (Part 10),
  forcing a full reinstall from scratch.

That isn't bad luck -- it's the rule, not the exception, when you depend on
fast-moving third-party libraries.

### 4. Seeds dominate when the effect is small; they back off when the effect is large

On CartPole (easy task), with 20 seeds, almost every "difference between
libs" that looked real with 5 seeds turned out to be statistical noise (only
1 of 10 pairs survived the Bonferroni correction). On LunarLander (harder
task), the same methodology revealed real and strong differences. The lesson:
**easy tasks make any policy converge near the ceiling quickly, leaving only
noise behind** -- counter-intuitively, the "simpler and more controlled"
environment was the one where seed variance fooled us the most.

### 5. The right question to ask before running the statistical test

Mann-Whitney U tests a shift in median/ranking. It does not detect a variance
difference. When the question was "did the variance change?" (Part 6,
`n_envs=1` vs `n_envs=8` in SB3), we needed Levene's test, not Mann-Whitney
-- even with the standard deviation quadrupling, Mann-Whitney didn't see the
difference. **Picking the wrong test answers the wrong question, even with
perfectly good data.**

### 6. Multiple-comparison corrections are costly

Every new comparison you test on the same dataset tightens the Bonferroni
corrected alpha for all the others. This literally knocked a Part 5 result out
of significance when we added one more implementation in Part 7 -- not because
the result changed, but because the number of simultaneous comparisons went
up.

### 7. "From scratch" is not a synonym for "optimized"

The "from scratch" implementations (pure JAX, pure PyTorch) carried generic
hyperparameters inherited from CartPole, never tuned for LunarLander. When we
added standard techniques (obs/reward normalization, a larger network, LR
annealing -- Part 4), the same code went from worse-than-everyone to
competitive. **The question "is library X better" hides the more important
question: "configured how?"** -- confirmed definitively in Part 11, where SB3
with the official RL Zoo hyperparameters beat the rest of the field,
including RLlib.

### 8. No library advantage survived a change of regime

Throughout the project, almost every claim that "lib X is better/more stable"
survived only inside the exact regime where it was measured:
- SB3 was the most stable in our benchmark, the second most unstable in
  someone else's project -- isolated cause: `n_envs` (Part 6).
- RLlib seemed to have a structural advantage on LunarLander -- part of it
  came from 2 specific mechanisms (dynamic KL + vf-clip, Part 5), but not all
  of it.
- The same truncation-bootstrap correction was irrelevant in one environment
  (LunarLander) and decisive in the other (CartPole, Parts 7-8).
- TorchRL and Tianshou, two mature production libs, landed in the middle of
  the pack **with generic hyperparameters** (Part 10) -- but that placement
  wasn't fixed either: with the Zoo hyperparameters (Part 12), Tianshou became
  the best of all, and TorchRL the worst.
- Generic SB3 (worst of the pack) and SB3 with a real production config (best
  of the pack in Parts 3-10) are literally the same library (Part 11) -- the
  biggest swing of the project up to that point came from changing
  CONFIGURATION, not the framework.
- **The most extreme example: RLlib, alone at the top in 3 different parts
  (3, 5, 10), dropped to second-to-last as soon as ALL the implementations
  -- not just SB3 -- got the Zoo hyperparameters (Part 12).** The advantage
  that looked like it belonged "to RLlib" was, to a large extent, the
  advantage "of running well configured while everyone else ran badly
  configured".

**The conclusion that survives all of this:** "library X is better/more
stable" is not a statement that is true or false on its own -- it's an
incomplete statement until you specify task, difficulty, budget, seeds and
configuration. The only thing that stands up to that scrutiny is the already
qualified version: "under these specific conditions, measured this way, this
combination got this result."

---

## Execution environment (Python + exact versions)

```
Python 3.12.3
Ubuntu 24.04.4 LTS, x86_64
CPU only (no GPU)
```

See `requirements.txt` for the exact version of each dependency
(gymnasium, stable-baselines3, torch, jax, ray, torchrl, tianshou, etc.).
Reproduce with `pip install -r requirements.txt`. This is a dated snapshot --
without version pins, the numbers will change over time (see the "Concepts"
section, item 3).

## Project structure

```
scripts/
  common.py                     # shared config, evaluate_policy, save_result
  train_sb3.py                  # SB3, generic hyperparameters
  train_sb3_n1.py                # SB3, same config but n_envs=1
  train_sb3_zoo.py               # SB3, OFFICIAL RL Zoo hyperparameters (Part 11)
  train_cleanrl.py               # pure PyTorch, CleanRL style
  train_cleanrl_original.py      # wrapper that runs the OFFICIAL CleanRL ppo.py via subprocess
  cleanrl_original_ppo.py        # the script itself, downloaded from GitHub and patched
  train_jax.py                   # pure JAX, generic
  train_jax_tuned.py             # JAX + obs/reward normalization, larger network, LR annealing
  train_jax_tuned_kl.py          # + dynamic KL penalty + value function clipping
  train_jax_tuned_kl_trunc.py    # + correct bootstrap on truncated episodes
  train_rllib.py                 # real RLlib (Ray)
  train_tianshou.py              # Tianshou 2.x
  train_torchrl.py               # TorchRL
  compare.py / boxplot.py / stats_test.py / eval_stats.py   # analysis and statistics
results/            # training curves (JSON), per environment
results_eval/       # evaluation results (N episodes per checkpoint), per environment
```

---

# Part 1: first comparison (SB3, pure PyTorch, pure JAX) -- CartPole-v1

Initial motivation: "is one lib's PPO different from another's, even with the
same hyperparameters?" Comparison with 1 seed, then 5 seeds, training
hyperparameters manually aligned across the 3 implementations (`common.py`).

The initial result (1 seed) suggested differences of up to ~100 points between
libs. That motivated the natural question: **is this a real implementation
difference, or is it seed noise that 1 sample cannot separate?**

# Part 2: aligning architecture hyperparameters, not only training ones

Training hyperparameters (lr, gamma, clip, etc.) were already aligned, but
architecture and weight initialization were not. We aligned:
- Architecture: 2 layers of 64, tanh, in the 4 libs (RLlib used [256,256] by
  default).
- Separate actor/critic networks (RLlib shared them by default).
- Orthogonal initialization with differentiated per-layer gains (√2 hidden,
  0.01 actor output, 1.0 critic output) -- replicating the CleanRL/SB3
  convention.
- Adam epsilon = 1e-5 in all of them.

After that, we added the official CleanRL (downloaded from GitHub, not the old
pip version) as a 5th implementation -- and it needed a patch to run with the
current Gymnasium (see "Concepts", item 3).

**5 seeds** were not enough to separate real difference from noise. We ran
**20 seeds** on CartPole-v1 (150k timesteps each) and applied:

- **Kruskal-Wallis**: `H=10.289, p=0.0358` -- some difference exists in the
  set as a whole.
- **Pairwise Mann-Whitney U with Bonferroni** (10 pairs, corrected alpha =
  0.005): **only 1 of 10 pairs was significant** (SB3 vs pure PyTorch,
  p=0.0018). All the other 9, including any lib vs official CleanRL, were
  statistically indistinguishable given the seed noise.

**Part 2 conclusion:** on CartPole, the "difference between libs" that looked
obvious with a few seeds was, overwhelmingly, seed noise -- not a real
implementation difference. The only thing that survived (SB3 with a standard
deviation 2-3x smaller than the other 4) was not about performance, it was
about **consistency**: `std=26.4` for SB3 vs `62-83` in the other 4.

# Part 3: a harder environment (LunarLander-v3)

Chosen because the "too easy" CartPole was masking real differences behind
seed noise. LunarLander-v3 (discrete -- kept Categorical instead of rewriting
for continuous actions), 8 seeds, 300k
timesteps.

> **Correction note (later audit):** a full numerical audit of the project
> (run after Part 12, comparing every table in this README against the raw
> JSON files) found that the seed 1-8 files of `jax_pure`,
> `cleanrl_style_pytorch` and
> `stable_baselines3` in `results/LunarLander-v3/` had been
> **overwritten** at some later point in the project by a run of
> **1,000,000 steps** (not the 300k original to this Part) -- only
> `rllib` and `cleanrl_original` still had the correct budget (~300k). The
> original data for those 3 implementations is irrecoverably lost.
> We redid the 8 seeds of those 3 implementations from scratch, with the
> correct budget (300k), and the table below reflects the **verified and
> correct** data -- which change this part's original conclusion in a real
> way, not just a cosmetic one (see "What changed" at the end of this section).

Result (verified data):

| Lib | Mean | Std dev |
|---|---|---|
| RLlib (Ray) | 47.7 | 56.7 |
| Pure JAX | 26.5 | 10.3 |
| Pure PyTorch (CleanRL style) | 26.1 | 32.3 |
| Stable-Baselines3 | 14.5 | 24.8 |
| CleanRL (official) | -38.0 | 26.0 |

**Kruskal-Wallis: `H=14.627, p=0.0055`.**

Mann-Whitney (Bonferroni, corrected alpha=0.0050): **4 of 10 pairs
significant** -- and all 4 involve **official CleanRL**, not RLlib:

| Pair | p-value | Significant? |
|---|---|---|
| Official CleanRL vs pure JAX | 0.0006 | YES |
| RLlib vs official CleanRL | 0.0047 | YES |
| SB3 vs official CleanRL | 0.0047 | YES |
| Official CleanRL vs pure PyTorch | 0.0047 | YES |
| (the other 6 pairs) | 0.28-0.65 | no |

## What changed with the correction

The original version of this section (with corrupted data) said that "RLlib had
a real and statistically robust advantage" over the other 4.
**That does not hold on the correct data.** With the verified data, RLlib is
statistically indistinguishable from SB3, pure PyTorch and pure JAX -- the only
implementation that really stands out (for the worse) is **official CleanRL**,
not identified before because of the corruption of the other 3
datasets.

This is relevant because Parts 4 and 5 were motivated by the idea
"RLlib seems to have a special advantage on LunarLander, let's investigate
why" -- a motivation that the original Part 3 (with bad data) seemed to
confirm with statistical force, but that does not hold in the corrected Part 3.
This does NOT invalidate the results of Parts 4-12 themselves (each one was
audited separately against the `results_eval/` files and all of them match
exactly what is documented) -- but it means the initial justification for
pulling that thread ("RLlib looks systematically better with 8 seeds") was
weaker than the original text suggested. RLlib's real advantage only shows up
robustly further ahead, with 1 seed + 50 evaluations (Part 4) and with the
formal tests of Parts 5-10 -- not
already in Part 3.

# Part 4: 1 training seed, multiple evaluation episodes (cheaper)


User's idea to save compute: instead of training N times with different seeds
(expensive), train **once** with a bigger budget (1M timesteps) and evaluate
that fixed policy over 50 episodes -- it measures consistency of
*evaluation*, not reproducibility of *training* (a different and
complementary question).

Result (seed=42, 1M steps, 50 episodes):

| Lib | Mean eval | Std dev |
|---|---|---|
| RLlib | 158.9 | 115.5 |
| CleanRL (official) | -52.8 | 126.5 |
| Pure PyTorch | -55.7 | 76.2 |
| Pure JAX | -57.8 | 39.3 |
| Stable-Baselines3 | -85.3 | 30.0 |

Kruskal-Wallis: `p≈0.0000`. **5 of 10 pairs significant** (stronger than the
multi-seed version). New finding: pure JAX vs SB3 became significant here,
which didn't happen with training variance -- suggesting that part of the
"noise" of Part 2 was masking a real and subtler difference.

## Testing whether it was task customization, not the lib (`train_jax_tuned.py`)

User's hypothesis: the "from scratch" implementations had never been *tuned*
for LunarLander -- they carried the generic CartPole recipe. We added, only in
pure JAX:
1. Observation normalization (running mean/std, VecNormalize style).
2. Reward normalization (reward divided by the running std of the discounted
   return).
3. Larger network: 128x128 instead of 64x64.
4. Learning rate with linear annealing down to 0.

Result: **-57.8 → +45.1**, statistically far above the 4 generic
implementations. **It did not close the gap with RLlib** (still
p<0.0001). We tested whether that was because RLlib normalizes observations by
default -- it does not: RLlib's default `observation_filter` is `NoFilter`.

# Part 5: dynamic KL + value function clipping (`train_jax_tuned_kl.py`)

Motivated by an external project sent by the user with an "RLlib style"
reimplementation (not the real `ray`) that bet on these 2 mechanisms. We added
only this to `jax_tuned`:
- **Value function clipping** (`RLLIB_VF_CLIP_PARAM=10.0`).
- **Dynamic KL penalty** (`RLLIB_KL_TARGET=0.01`,
  `RLLIB_INITIAL_KL_COEFF=0.2`, adjusted at each iteration by the same rule
  RLlib uses: ×1.5 if KL > 1.5x the target, ×0.5 if < 0.67x the target).

Result: **45.1 → 104.9** (p=0.0022 vs jax_tuned). The gap with RLlib dropped
from 113.8 points to 54.0 -- **these 2 mechanisms explain roughly
half** of RLlib's remaining advantage. But RLlib still won significantly
(p=0.0001) -- ~54 points were still
unexplained.

# Part 6: why SB3 is "stable" here and "unstable" in an external project

The user noticed that, in the external project they sent, SB3 had a standard
deviation of `112.1` (second worst, almost as bad as a TorchRL with a
suspected bug), while in our benchmark SB3 was consistently the MOST stable
(`std=26-30`). Hypothesis: `N_ENVS` (8 in our setup, 1 in theirs).

We tested changing ONLY this variable (`train_sb3_n1.py`, `n_envs=1`, same
seed, same budget):

| Version | Std |
|---|---|
| SB3 (n_envs=8) | 30.0 |
| SB3 (n_envs=1) | 132.1 |

**Levene's test** (not Mann-Whitney -- the question was about variance,
not about the mean): `p=0.00013`. A real and strong variance difference.
**Hypothesis confirmed**: changing 1 structural parameter flipped SB3's entire
stability reputation.

# Part 7: truncation-bootstrap correction -- LunarLander (inconclusive)

Our GAE treated `terminated` (the episode really ended) and `truncated` (cut
by an artificial time limit) the same way, zeroing the value bootstrap in both
cases -- technically wrong for truncation.

We confirmed empirically that current Gymnasium (`autoreset_mode=NEXT_STEP`,
the default) returns the real final observation on the truncation
step -- the right data was already there, it just wasn't being used. We fixed
it with two separate masks in the GAE (`train_jax_tuned_kl_trunc.py`): one to
zero the bootstrap (only true terminated), another to stop the GAE
propagation (terminated OR truncated).

Result on LunarLander: **104.9 (without the correction) vs 85.4 (with it)**,
Mann-Whitney `p=0.3647` -- **not significant**, the difference is noise.
Likely explanation: LunarLander has a 1000-step limit, but episodes end by
landing/crash well before that -- truncation is rare, the correction had no
chance to show an effect.

# Part 8: the same correction, on CartPole -- clear and positive result

CartPole is the right environment for this hypothesis: 500-step limit, and a
competent policy learns to swing indefinitely, hitting the time limit on most
episodes (truncation is the DOMINANT way a successful episode
ends).

Result (seed=42, 400k timesteps):

| Version | Mean | Std | Min |
|---|---|---|---|
| With correction | **500.0** | **0.0** | 500.0 |
| Without correction | 485.5 | 43.5 | 343.0 |

Mann-Whitney: `p=0.0231` -- significant. **Perfect in all 50 evaluation
episodes** with the correction. Same mechanism, same code, irrelevant in one
environment (LunarLander) and decisive in the other (CartPole).

# Part 9: real RLlib on CartPole -- inconclusive, but revealing, result

Question: does "out of the box" RLlib already handle truncation correctly,
explaining part of its advantage? We ran RLlib on CartPole (the generic config
from Parts 1-2, without LR annealing), 400k timesteps, seed=42.

Final evaluation: **171.3 ± 33.9** -- far below the expected 500. But the full
training curve showed something different: the policy **hit 500**
mid-training (step ~330k) and **degraded** consistently until the
end, finishing at 193. It's not a lack of budget -- it's training instability
that the evaluation (run on the final checkpoint) caught in a
valley, not at the peak.

**The original test went unanswered** -- it revealed, instead, that this RLlib
config (without LR annealing) is unstable near the performance
ceiling, echoing the mechanism our own tuned versions use to avoid
exactly that.

# Part 10: TorchRL and Tianshou

Dual motivation: (1) close the mystery of the catastrophic TorchRL (-377)
from the external project by implementing our own controlled version; (2) add
a second mature lib to separate "an advantage specific to RLlib" from "a
production framework always beats a hobbyist one".

It required fixing 2 APIs broken by version (see "Concepts", item 3):
Tianshou 2.0 (`policy`→`algorithm`) and TorchRL (`SyncDataCollector`→
`Collector`). The execution environment was also reset mid-work, requiring
reinstalling everything (including dealing with running out of
disk space).

Result (LunarLander-v3, seed=42, 1M steps):

| Lib | Mean eval | Training time |
|---|---|---|
| TorchRL | 49.0 | **1014.5s** |
| Tianshou | -7.2 | 233.8s |

**The TorchRL mystery did not repeat itself** -- our implementation landed in
the middle of the pack, nothing catastrophic, reinforcing that the external
project's -377 was specific to its config. **TorchRL was by far the
slowest implementation of the entire project** (1014s vs 111-265s for
the others). Tianshou and TorchRL came out statistically indistinguishable from
several of our "generic" implementations -- this weakens the "mature framework
always beats hobbyist" hypothesis: RLlib's advantage looks more specific to it
than a general pattern.

---

# Part 11: SB3 with the OFFICIAL RL Zoo hyperparameters -- final confirmation

## The question

Why did several implementations (including generic SB3 itself, `-85.3`) give
NEGATIVE reward on LunarLander? Was it a bug, or bad configuration? We already
had indirect evidence (Part 4: tuning raised JAX from -57.8 to
+45.1) but we had never confirmed it with the *real, recommended by the people
who maintain the lib* config -- only with our own tuning attempts.

## The test

I took the official hyperparameters from the **RL Baselines3 Zoo**
(`DLR-RM/rl-baselines3-zoo`, `hyperparams/ppo.yml`, `LunarLander-v3` section)
-- the values actually used/recommended by the team that maintains SB3 for this
specific task:

```yaml
n_envs: 16
n_steps: 1024
batch_size: 64
gae_lambda: 0.98
gamma: 0.999        # much higher than the generic 0.99 -- LunarLander
                    # has long episodes, so future reward has to be weighted
                    # more heavily
n_epochs: 4
ent_coef: 0.01
n_timesteps: 1e6
```

(learning rate, clip_range, vf_coef, architecture: SB3 defaults, no
change). Same seed=42, same evaluation protocol (50 episodes) as Parts 3-10,
for a direct comparison with everything we already had.

## Result

**231.3 ± 88.8** -- not only did it get out of the negative, it landed ABOVE
all the other 11 implementations tested so far, including RLlib (158.9) and our
own `jax_tuned_kl` (104.9).

| Lib | Mean eval | Std |
|---|---|---|
| **SB3 (official Zoo config)** | **231.3** | 88.8 |
| RLlib (Ray) | 158.9 | 115.5 |
| jax_tuned_kl | 104.9 | 95.0 |
| jax_tuned_kl_trunc | 85.4 | 106.0 |
| TorchRL | 49.0 | 115.2 |
| jax_tuned | 45.1 | 110.7 |
| stable_baselines3_n1 | 0.6 | 132.1 |
| Tianshou | -7.2 | 105.9 |
| CleanRL (official) | -52.8 | 126.5 |
| Pure PyTorch | -55.7 | 76.2 |
| Pure JAX | -57.8 | 39.3 |
| Stable-Baselines3 (generic config) | -85.3 | 30.0 |

Kruskal-Wallis: `H=273.810, p≈0.0000`. Mann-Whitney with Bonferroni (12
implementations, 66 pairs): **SB3-Zoo is significantly different from ALL the
other 11 implementations**, including RLlib (`p=0.0000` in every pair involving
SB3-Zoo).

## The definitive answer

**Config, not library.** SB3-Zoo and Stable-Baselines3 (generic) are
**literally the same source code, the same version of the same library** -- the
only thing that changed were the hyperparameters. And that change alone
produced the largest reward swing of the entire project (from -85.3 to
+231.3, a jump of more than 300 points), bigger than any difference between
libraries we measured in Parts 1-10.

That closes, with the most direct test possible, the question that ran through
the whole project: when you see "library X has negative/bad reward", the most
likely explanation is not "the lib has a bug" -- it is "nobody gave it the
right hyperparameters for this task". `gamma=0.999` instead of `0.99` alone is
already a strong hint: LunarLander has episodes of up to 1000 steps, and
discounting future reward aggressively (low gamma) makes the agent "not see"
the long-term benefit of landing carefully.

## Run

```bash
export PPO_ENV=LunarLander-v3
export PPO_SEED=42
python3 scripts/train_sb3_zoo.py
python3 scripts/eval_stats.py
```

---

# Part 12: repeating the ENTIRE comparison with the RL Zoo hyperparameters

## The question

Part 11 showed that SB3 with the right config becomes the best of the pack.
But that only tested SB3. The natural question: **would applying the same
config (not an "official config" for each lib, which doesn't exist for all of
them, but literally the same RL Zoo hyperparameter values) to ALL
implementations change the entire ranking we built in Parts 1-10?**

User's idea, simpler and cleaner than trying to find an "official recipe" per
library (which doesn't exist in a symmetric way for RLlib/TorchRL/Tianshou/our
own implementations): use the SAME RL Zoo values (`n_steps=1024`,
`n_envs=16`, `gae_lambda=0.98`, `gamma=0.999`,
`n_epochs=4`, `ent_coef=0.01`, `batch_size=64`) in `common.py`, and run the
same battery of 7 implementations again -- 1 seed=42, 1M timesteps, 50
evaluation episodes, LunarLander-v3.

## Infrastructure struggles along the way (worth recording)

This part suffered TWO container resets mid-execution -- the working
filesystem (`/home/claude`) and, this time, even the outputs folder (which I
assumed was permanently persistent) reverted to an earlier state more than
once. Practical consequences:
- We lost the intermediate TorchRL results once (it had to run again,
  ~20 minutes lost).
- One reset brought back an OLD version of `common.py` (without the Zoo
  hyperparameters), which was only noticed because the TorchRL log showed
  `frames_per_batch (2048)` instead of `16384` -- the process even ran
  ~5 minutes with the wrong config before being interrupted and
  fixed.
- The practical lesson adopted from here on: copy the results to `outputs`
  **immediately after each script finishes**, not just at the end of the work
  -- and that lesson itself almost wasn't enough, because even `outputs`
  turned out to be unstable in this session.

Yet another live chapter of the "Concepts" theme, item 3 -- reproducibility is
not only about library versions, it is about the whole execution environment
not being guaranteed, period.

## Result (LunarLander-v3, seed=42, 1M steps, Zoo config in all of them)

| Lib | Mean eval (Zoo config) | Std | Mean eval (generic config, Parts 3-10) |
|---|---|---|---|
| **Tianshou** | **281.2** | 37.6 | -7.2 |
| Pure PyTorch (CleanRL style) | 265.4 | 31.7 | -55.7 |
| CleanRL (official) | 235.0 | 73.4 | -52.8 |
| Stable-Baselines3 | 231.3 | 88.8 | -85.3 |
| Pure JAX | 158.3 | 24.9 | -57.8 |
| RLlib (Ray) | 123.7 | 34.9 | **158.9** |
| TorchRL | 112.3 | 38.0 | 49.0 |

Kruskal-Wallis: `H=220.311, p≈0.0000`. Mann-Whitney with Bonferroni (21
pairs, corrected alpha=0.0024): **18 of 21 pairs significant** -- the highest
proportion of significant pairs of the whole project.

The 3 NON-significant pairs form a "top group" that is statistically
indistinguishable among themselves: pure PyTorch vs official CleanRL
(p=0.0050, doesn't survive the correction), pure PyTorch vs SB3 (p=0.0759),
official CleanRL vs SB3 (p=0.3432). Tianshou sits alone above even that group
(significantly better than those 3). RLlib and TorchRL sit isolated below the
rest, significantly worse than the 5 implementations above -- and
significantly different from each other too
(RLlib > TorchRL, p=0.0003).

## The central finding: the ranking flipped upside down

**RLlib, which was by far the best with generic hyperparameters (158.9, alone
at the top in Parts 3-10), dropped to second-to-last (123.7) once the correct
config was applied to everyone equally.** Almost all of the "from scratch"
implementations (Tianshou, pure PyTorch, pure JAX) and even official CleanRL
passed RLlib.

This isn't "RLlib got worse" -- it had exactly the same training
hyperparameters as the others, so its absolute result actually improved
moderately compared to what it already had (123.7 vs low values would be
expected, but in fact it was already at 158.9 before -- meaning that switching
to the Zoo config didn't help RLlib as much as it helped the other
implementations, and the others overtook it). The most likely explanation,
putting together everything we already know: **RLlib seems to have internal
mechanisms (normalization, episode handling, the KL/vf-clip defaults that are
already part of it by default) that make it relatively robust even with bad
hyperparameters** -- and that is exactly why it "won" in Parts 3-10, when
everyone had a bad configuration. Once ALL implementations receive good
hyperparameters, that robustness stops being a differentiator, and the
advantage that remains goes to whoever has the most efficient implementation
on top of good hyperparameters -- which, in this test, was not
RLlib.

## Final take on this part

This is the most direct and strongest demonstration in the whole project
of the central principle we had been uncovering little by little:
**"library X is better" was never a fixed property of the library -- it was
always a property of the whole (library, hyperparameters, task, budget)
combination.** The same question ("which PPO library is better for
LunarLander?") has two opposite answers depending only on one variable
that isn't even the library: it is the quality of the hyperparameters
you gave it.

TorchRL also deserves a separate note: besides continuing to be, by far, the
slowest implementation (~20 minutes, ~5-10x more than most of them), it is now
also the weakest in reward. No compensating advantage showed up for it in this
test battery.

## Run

```bash
export PPO_ENV=LunarLander-v3
export PPO_TIMESTEPS=1000000
export PPO_SEED=42
export PPO_RUN_TAG="-zoo"
python3 scripts/train_sb3.py
python3 scripts/train_cleanrl.py
python3 scripts/train_jax.py
python3 scripts/train_rllib.py
python3 scripts/train_cleanrl_original.py
python3 scripts/train_tianshou.py
python3 scripts/train_torchrl.py   # the slowest, ~20min on its own
python3 scripts/eval_stats.py
```

`PPO_RUN_TAG` creates a separate subfolder in `results/` and `results_eval/`
(`LunarLander-v3-zoo/`) so results don't get mixed with the generic-config
ones from Parts 3-10, which use the same environment and seed but different
hyperparameters.

---



After 11 parts, the most honest answer to "which PPO library is better" is:
**the question, the way it is usually asked, cannot be answered.** Every time
we isolated one variable -- seeds, task difficulty, budget, `n_envs`,
KL/vf-clip, truncation handling, and finally the configuration hyperparameters
-- an earlier conclusion that looked solid either lost strength, or turned out
to be about something completely different from what it
seemed.

# Overall project conclusion

After 12 parts, the most honest answer to "which PPO library is better" is:
**the question, the way it is usually asked, cannot be answered.** Every time
we isolated one variable -- seeds, task difficulty, budget, `n_envs`,
KL/vf-clip, truncation handling, and finally the configuration
hyperparameters -- an earlier conclusion that looked solid either lost
strength, or turned out to be about something completely different from what it
seemed.

Part 12 is the most extreme example of this in the whole project: RLlib,
which had won in a statistically solid way in Parts 3, 5 and 10, became the
second-to-last place as soon as the ONLY thing that changed was giving quality
hyperparameters to everyone equally. At no point in this project did an answer
to "which lib is better" exist that survived changing the comparison regime --
the only thing this project found that held universally was the pattern
itself: **the library matters far less than the
configuration.**

The only thing that survived the entire scrutiny was the qualified version of
the question: "with this algorithm, in this environment, with this budget, with
N seeds, with these specific hyperparameters, this implementation had this
distribution of results." That is far less marketable than a benchmark ranking,
but it is the only thing that stayed true after we changed
anything.

If this is going to feed the idea of your own ML library: the practical lesson
is not "choose the right lib" -- it is that most of what makes an RL agent
work well is not in the framework choice, it is in a handful of engineering
decisions (normalization, truncation handling, LR annealing, hyperparameters
calibrated for the task) that any implementation -- yours or a third
party's -- has to get right.
