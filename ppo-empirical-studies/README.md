# PPO Empirical Studies: Reproducibility, Ecosystem Variance, Factorial Attribution, and Advantage Mechanics

A comprehensive, two-phase empirical investigation into the reproducibility, algorithmic divergence, and mechanical levers of **Proximal Policy Optimization (PPO)** ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)).

---

## 1. Executive Summary & Scientific Motivation

Proximal Policy Optimization is widely considered the *de facto* baseline in modern Deep Reinforcement Learning. Canonical literature frequently reports performance curves under the assumption that "PPO" designates a single, mathematically determined algorithm. 

In practice, however, identical nominal hyperparameters across different open-source implementations—such as **Stable-Baselines3 (SB3)**, **CleanRL**, **Ray/RLlib**, **TorchRL**, **Tianshou**, and custom **PyTorch** or **JAX** engines—produce radically divergent learning dynamics, sample efficiencies, asymptotic returns, and variance profiles.

This subproject unites two sequential empirical investigations addressing this fundamental crisis of reproducibility:

```
                                      PPO EMPIRICAL STUDIES
                                                │
         ┌──────────────────────────────────────┴──────────────────────────────────────┐
         ▼                                                                             ▼
┌─────────────────────────────────┐                                   ┌─────────────────────────────────┐
│ Phase 1: Ecosystem Benchmark    │                                   │ Phase 2: Factorial Attribution  │
│ 01_ecosystem_benchmark/         │ ────── Macro Diagnosis ─────────> │ 02_factorial_attribution/       │
│                                 │                                   │                                 │
│ • 7 distinct PPO implementations│                                   │ • 2×2 Factorial Matrix          │
│ • 12 sequential empirical parts │                                   │   (Spec vs. Runtime Stack)      │
│ • Kruskal-Wallis & Mann-Whitney │                                   │ • Single-Delta Ablations        │
│ • RL Zoo vs. Default regimes    │                                   │ • GAE Episode Boundary Probe    │
└─────────────────────────────────┘                                   └─────────────────────────────────┘
```

1. **[Phase 1: Multi-Library Ecosystem Benchmark](01_ecosystem_benchmark/README.md)**: A 12-part macro-investigation measuring performance variance across 7 major implementations on canonical benchmarks (`CartPole-v1` and `LunarLander-v3`).
2. **[Phase 2: Factorial Attribution & Advantage Mechanics](02_factorial_attribution/README.md)**: A micro-investigation using a $2 \times 2$ factorial matrix, bridge implementations, single-delta ablations, and a dedicated [GAE Indexing Probe](02_factorial_attribution/probe_gae/README.md) to isolate whether divergence stems from software runtime or hidden algorithmic choices.

---

## 2. Monorepo Structure

```text
ppo-empirical-studies/
├── README.md                                 # Top-level unified academic report (this document)
│
├── 01_ecosystem_benchmark/                   # Phase 1: Multi-framework benchmark (12 parts)
│   ├── README.md                             # Exhaustive 700+ line empirical report
│   ├── requirements.txt                      # Multi-framework dependencies (SB3, RLlib, TorchRL, Tianshou)
│   ├── eval_stats_output_part11.txt          # Summary statistics for Part 11
│   ├── eval_stats_output_part12.txt          # Summary statistics for Part 12 (RL Zoo tuning)
│   ├── figures/                              # High-resolution comparative plots and boxplots
│   │   ├── ppo_comparison_CartPole-v1.png
│   │   ├── ppo_comparison_LunarLander-v3.png
│   │   ├── ppo_comparison_LunarLander-v3-zoo.png
│   │   ├── ppo_boxplot_CartPole-v1.png
│   │   ├── ppo_boxplot_LunarLander-v3.png
│   │   ├── ppo_eval_boxplot_CartPole-v1.png
│   │   └── ppo_eval_boxplot_LunarLander-v3-zoo.png
│   ├── scripts/                              # Implementation scripts and evaluation harnesses
│   │   ├── common.py                         # Shared evaluation, metrics, and JSON serialization
│   │   ├── train_cleanrl.py                  # PyTorch modular CleanRL style
│   │   ├── train_cleanrl_original.py         # Canonical single-file CleanRL
│   │   ├── train_sb3.py                      # Stable-Baselines3 standard baseline
│   │   ├── train_sb3_zoo.py                  # Stable-Baselines3 RL Zoo configuration
│   │   ├── train_jax.py                      # Functional PPO via JAX + Optax
│   │   ├── train_rllib.py                    # Ray RLlib distributed baseline
│   │   ├── train_torchrl.py                  # TorchRL TensorDict baseline
│   │   ├── train_tianshou.py                 # Tianshou vectorized replay baseline
│   │   ├── compare.py                        # Multi-seed curve comparison renderer
│   │   ├── boxplot.py                        # Empirical distribution boxplot renderer
│   │   ├── eval_stats.py                     # Non-parametric aggregation suite
│   │   └── stats_test.py                     # Mann-Whitney U & Levene hypothesis tests
│   ├── results/                              # Raw training curves (JSON, per seed)
│   └── results_eval/                         # Deterministic evaluation returns (JSON, per seed)
│
└── 02_factorial_attribution/                 # Phase 2: Factorial attribution and GAE probe
    ├── README.md                             # Factorial study documentation and planned contrasts
    ├── requirements-torch.txt                # PyTorch stack requirements
    ├── requirements-jax.txt                  # JAX stack requirements
    ├── specs.py                              # Formal specification delta definitions
    ├── config.py                             # Exact hyperparameter configurations
    ├── variants.py                           # Benchmark arm registration
    ├── common.py                             # Environment, advantage, and rollouts common engine
    ├── sb3_shim.py                           # NumPy 2 compatibility shim for SB3
    ├── train_sb3.py                          # SB3 driver
    ├── train_cleanrl_torch.py                # CleanRL PyTorch driver (supports cleanrl and sb3 mode)
    ├── train_jax_ppo.py                      # JAX driver (supports cleanrl, sb3, and ablations)
    ├── run_all.py                            # Sequential benchmark runner
    ├── run_full_background.py                # Parallel multi-worker runner
    ├── evaluate.py                           # Deterministic evaluation suite (100 episodes per seed)
    ├── analyze.py                            # Bootstrap resampling & Cliff's delta statistical pipeline
    ├── plot_results.py                       # Factorial curves and attribution boxplots
    ├── probe_gae/                            # Isolated Generalized Advantage Estimation probe
    │   ├── README.md                         # GAE boundary indexing study report
    │   ├── analyze.py                        # Paired hypothesis testing for GAE indexing
    │   ├── run_probe.sh                      # Probe harness runner
    │   └── data/                             # Raw episode evaluation data (41 paired seeds)
    └── results/                              # Output tables, contrast summaries, curves, and plots
        ├── tables/                           # Contrasts (.csv, .md) and stats summaries
        └── plots/                            # Training curves and attribution boxplots
```

---

## 3. Mathematical Foundations of PPO

Proximal Policy Optimization maximizes the surrogate objective subject to probability ratio clipping:

$$\mathcal{L}^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]$$

where the probability ratio is defined as:

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}$$

and $\hat{A}_t$ is the Generalized Advantage Estimator (GAE) ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438)):

$$\delta_t^V = r_t + \gamma V(s_{t+1})(1 - d_t) - V(s_t)$$

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V$$

Total optimization minimizes the combined surrogate, value function loss $\mathcal{L}^{VF}(\theta)$, and entropy bonus $\mathcal{S}[\pi_\theta](s_t)$:

$$\mathcal{L}^{TOTAL}(\theta) = -\mathcal{L}^{CLIP}(\theta) + c_1 \mathcal{L}^{VF}(\theta) - c_2 \mathcal{S}[\pi_\theta](s_t)$$

### Where Real Implementations Diverge

While the high-level formulas are universal, canonical implementations diverge across critical implementation details:
1. **Value Loss Formulation**: CleanRL implements clipped value loss $\mathcal{L}^{CLIP}_{VF}$ scaled by $0.5$, whereas SB3 uses unclipped mean squared error with a scale factor of $1.0$:
   $$\mathcal{L}^{VF}_{CleanRL} = 0.5 \cdot \hat{\mathbb{E}}_t \left[ \max\left( (V_\theta - V^{targ})^2, \, (V_{clip} - V^{targ})^2 \right) \right]$$
   $$\mathcal{L}^{VF}_{SB3} = \hat{\mathbb{E}}_t \left[ (V_\theta - V^{targ})^2 \right]$$
2. **Learning Rate Annealing**: CleanRL linearly decays $\alpha_t = \alpha_0 (1 - t / T)$, while SB3 defaults to a constant learning rate $\alpha_t = \alpha_0$.
3. **Truncation vs. Termination Bootstrapping**: Handling time-limit truncations ($t = T_{max}$) versus true Markov terminal states ($d_t = 1$).
4. **GAE Boundary Indexing**: Whether bootstrap masking gates at step $t$ via $(1 - d_t)$ or is shifted to $(1 - d_{t+1})$.
5. **Orthogonal Initialization Head Scaling**: Scaling actor output logits with gain $\kappa_{actor} = 0.01$ and critic head with $\kappa_{critic} = 1.0$, versus default PyTorch Xavier/Kaiming initialization.
6. **Optimizer Epsilon**: CleanRL fixes Adam $\epsilon = 10^{-5}$, whereas standard PyTorch defaults to $\epsilon = 10^{-8}$.

---

## 4. Phase 1: Multi-Library Ecosystem Benchmark

Phase 1 evaluated 7 implementations across 12 sequential empirical iterations to measure ecosystem variance under controlled conditions.

### Implementations Evaluated

| Implementation | Framework / Paradigm | Architectural Characteristics |
|---|---|---|
| `cleanrl_original` | PyTorch | Single-file, modular-free canonical script from CleanRL repository |
| `cleanrl_style_pytorch` | PyTorch | Refactored modular implementation mirroring CleanRL mechanics |
| `stable_baselines3` | PyTorch / SB3 | Object-oriented framework baseline with standard default parameters |
| `stable_baselines3_zoo` | PyTorch / SB3 | SB3 with optimized hyperparameter sets from RL Baselines3 Zoo |
| `jax_pure` | JAX / Optax | Functional PPO compiled via XLA |
| `rllib` | Ray / PyTorch | Distributed actor architecture via Ray RLlib |
| `torchrl` | PyTorch / TensorDict | Modern PyTorch domain library leveraging TensorDict data structures |
| `tianshou` | PyTorch | Vectorized, buffer-driven reinforcement learning framework |

### Key Experimental Discoveries (Parts 1–12)

* **Parts 1–3 (`CartPole-v1`)**: On low-dimensional tasks, all frameworks achieve the optimal ceiling ($500.0$), but sample efficiencies diverge by up to $3\times$. Single-file CleanRL converged in $\sim 30{,}000$ steps, whereas high-overhead abstractions required $\sim 80{,}000$ steps.
* **Parts 4–7 (`LunarLander-v3` with Default Parameters)**: Exposes catastrophic failure modes in default configurations. Under default library settings, Stable-Baselines3 and RLlib failed to achieve stable positive flight scores (often hovering below 0.0 or experiencing high variance), while CleanRL attained $\sim 200+$ points.
* **Parts 8–10 (Vectorization & Normalization Audits)**: Investigated vectorization scale ($n_{envs} = 1$ vs $n_{envs} = 4$ vs $n_{envs} = 8$). Demonstrated that vectorization without advantage normalization rescaling degrades policy stability.
* **Parts 11–12 (RL Zoo Hyperparameter Parity)**: When hyperparameter sets from the RL Baselines3 Zoo were propagated across all frameworks on `LunarLander-v3`, performance distributions shifted dramatically.

### Empirical Results: Part 12 Evaluation Benchmark

Evaluated over 50 deterministic test episodes per library using fixed evaluation seeds following training:

| Library | Mean Return | Median Return | Standard Deviation | Min Return | Max Return |
|---|---|---|---|---|---|
| **Tianshou** | **281.2** | **284.9** | 37.6 | 54.3 | 318.4 |
| **PyTorch Pure (CleanRL style)** | 265.4 | 270.1 | 31.7 | 86.1 | 303.9 |
| **CleanRL (Official)** | 235.0 | 258.3 | 73.4 | 31.3 | 302.2 |
| **Stable-Baselines3** | 231.3 | 263.2 | 88.8 | -22.6 | 302.6 |
| **JAX Pure** | 158.3 | 160.8 | 24.9 | 49.3 | 198.4 |
| **RLlib (Ray)** | 123.7 | 134.1 | 34.9 | 11.3 | 168.4 |
| **TorchRL** | 112.3 | 111.1 | 38.0 | 35.1 | 245.1 |

#### Statistical Hypothesis Testing (Part 12)
- **Kruskal-Wallis Omnibus Test**: $H = 220.311$, $p < 0.0001$ (decisively rejecting the null hypothesis of equal median performance across libraries).
- **Pairwise Mann-Whitney U Tests** (Bonferroni-corrected significance threshold $\alpha = 0.0024$):
  - `PyTorch Pure` vs. `CleanRL Official`: $p = 0.0050$ (No statistically significant difference after Bonferroni correction).
  - `CleanRL Official` vs. `Stable-Baselines3`: $p = 0.3432$ (No statistically significant difference).
  - `Tianshou` vs. `Stable-Baselines3`: $p < 0.0001$ (Statistically significant superior performance).
  - `Stable-Baselines3` vs. `RLlib` / `TorchRL`: $p < 0.0001$ (Statistically significant superior performance).

For complete progression details, see [Phase 1 README](01_ecosystem_benchmark/README.md).

---

## 5. Phase 2: Factorial Attribution & GAE Mechanics

While Phase 1 identified macro-divergence across libraries, it left a core question open: **Does the performance gap stem from the computational runtime (PyTorch vs. JAX) or from specific algorithmic choices hidden in code?**

### The $2 \times 2$ Factorial Design

Phase 2 established a rigorous $2 \times 2$ factorial experiment plus a bridge arm and single-delta ablations:

| Arm Identifier | Source Code Driver | Algorithmic Specification | Runtime Stack |
|---|---|---|---|
| `sb3_torch` | `train_sb3.py` | SB3 Specification | PyTorch + SB3 |
| `cleanrl_torch` | `train_cleanrl_torch.py --mode cleanrl` | CleanRL Specification | Manual PyTorch |
| `cleanrl_sb3mode_torch` | `train_cleanrl_torch.py --mode sb3` | **SB3 Specification** | Manual PyTorch *(Bridge)* |
| `jax_sb3` | `train_jax_ppo.py --mode sb3` | SB3 Specification | JAX + Optax |
| `jax_cleanrl` | `train_jax_ppo.py --mode cleanrl` | CleanRL Specification | JAX + Optax |

#### Permitted Algorithmic Specification Deltas (`specs.py`)
- **Value Clipping**: CleanRL enables $\text{clip}(V_\theta, V_{old}-\epsilon, V_{old}+\epsilon)$; SB3 uses unclipped MSE.
- **LR Schedule**: CleanRL applies linear learning rate annealing; SB3 maintains a constant learning rate.
- **MSE Scaling**: CleanRL applies a $0.5$ factor to value MSE; SB3 optimizes full MSE.
- **Bootstrapping**: SB3 explicitly bootstraps value at time limits; CleanRL treats truncation as termination.

### Empirical Planned Contrasts (LunarLander-v3, $n=10$ Seeds)

All contrasts evaluated using 5,000 bootstrap resamples with 95% confidence intervals and Cliff's delta ($d$):

| Planned Contrast | Comparison ($A - B$) | Bootstrap Mean Difference | 95% Confidence Interval | $p$-value | Cliff's $d$ | Interpretation |
|---|---|---|---|---|---|---|
| **Q1: Total Gap** | `sb3_torch` $-$ `cleanrl_torch` | $+27.09$ | $[-28.08, +79.45]$ | $0.3292$ | $+0.320$ | Macro gap observed |
| **Q2: Stack Effect (SB3 Spec)** | `sb3_torch` $-$ `jax_sb3` | **$-6.94$** | $[-55.59, +40.97]$ | **$0.7912$** | $-0.020$ | **No stack effect** |
| **Q3: Stack Effect (CleanRL Spec)** | `cleanrl_torch` $-$ `jax_cleanrl` | **$+15.71$** | $[-10.47, +41.72]$ | **$0.2460$** | $+0.320$ | **No stack effect** |
| **Q4: Spec Bridge (PyTorch)** | `sb3_torch` $-$ `cleanrl_sb3mode_torch` | $-15.93$ | $[-59.58, +25.63]$ | $0.4748$ | $-0.120$ | Bridge matches SB3 |
| **Q5: Spec Effect within JAX** | `jax_sb3` $-$ `jax_cleanrl` | **$+50.32$** | **$[+25.48, +78.07]$** | **$< 0.0001$** | **$+0.720$** | **Spec dominates!** |
| **Q6: Bridge vs. JAX SB3** | `cleanrl_sb3mode_torch` $-$ `jax_sb3` | $+8.88$ | $[-13.07, +32.38]$ | $0.4744$ | $+0.160$ | Cross-engine agreement |

> [!IMPORTANT]
> **Scientific Finding: The Algorithmic Specification Dominates the Stack**
> When the algorithmic specification is strictly held constant, changing the execution runtime from PyTorch to JAX yields no statistically significant difference ($p = 0.7912$ under SB3 spec; $p = 0.2460$ under CleanRL spec). 
> Conversely, altering the algorithmic specification inside the *same* execution runtime (JAX) produces a statistically significant $+50.32$ point increase in return ($p < 0.0001$, large Cliff's effect size $d = 0.720$).

### Single-Delta Ablations in JAX (`LunarLander-v3`, $n=5$ Seeds)

To isolate which of the 4 specification deltas drove the $+50.32$ point shift, single-delta ablations were benchmarked against `jax_cleanrl`:

| Ablation Arm | Delta Toggled from CleanRL | Contrast vs. `jax_cleanrl` | 95% Confidence Interval | $p$-value | Cliff's $d$ |
|---|---|---|---|---|---|
| `jax_abl_novclip` | Value loss clipping **disabled** | **$+41.90$** | **$[+20.55, +57.77]$** | **$< 0.0001$** | **$+0.920$** |
| `jax_abl_noanneal` | LR annealing **disabled** (constant LR) | **$+20.42$** | **$[+4.25, +41.83]$** | **$< 0.0001$** | **$+0.680$** |
| `jax_abl_fullmse` | MSE factor set to **$1.0$** (full MSE) | $+0.31$ | $[-18.28, +18.28]$ | $0.9564$ | $+0.040$ |
| `jax_abl_tboot` | Timeout bootstrapping **enabled** | $-8.60$ | $[-27.15, +11.88]$ | $0.3240$ | $-0.440$ |

**Conclusion**: Value function loss clipping (`novclip`) is the primary driver, accounting for $\approx 83\%$ ($+41.90$ points) of the entire specification gap on LunarLander-v3. Constant learning rate (`noanneal`) accounts for the second largest boost ($+20.42$ points). Full MSE scaling and timeout bootstrapping show no statistically significant main effect on this environment.

---

## 6. The GAE Episode-Boundary Indexing Probe (`probe_gae`)

During Phase 2 auditing, a critical indexing subtlety was identified in Generalized Advantage Estimation:
- **Canonical SB3 / CleanRL Convention**:
  $$\delta_t = r_t + \gamma V(s_{t+1}) (1 - d_t) - V(s_t)$$
- **Shifted Indexing Convention** (present in initial manual rollouts):
  $$\delta_t = r_t + \gamma V(s_{t+1}) (1 - d_{t+1}) - V(s_t)$$

The dedicated probe arm in [`probe_gae/`](02_factorial_attribution/probe_gae/README.md) evaluated whether this one-line difference materially impacts performance across 41 independent, paired training seeds on `LunarLander-v3`:

| Indexing Arm | Expression | Eval Mean Return (200 eps) | Seed Standard Deviation | Paired Difference ($B - A$) | 95% CI | $p$-value (paired $t$) |
|---|---|---|---|---|---|---|
| **Arm A (Committed)** | `dones[t + 1]` | **85.3** | **27.8** | — | — | — |
| **Arm B (SB3/CleanRL)** | `dones[t]` | 64.1 | 48.2 | **$-21.1$** | $[-36.9, -5.4]$ | **$0.0098$** |

### Statistical Verification:
- **Paired $t$-test**: $t = -2.71$, $p = 0.0098$ (Arm B is significantly worse by $21.1$ points).
- **Wilcoxon Signed-Rank Test**: $p = 0.022$ (Arm B is inferior on 27 of 41 seeds).
- **Levene's Test for Homogeneity of Variance**: $p = 0.0025$ (Arm B exhibits nearly double the seed-to-seed standard deviation: $48.2$ vs. $27.8$).
- **Replication**: An independent second cohort of 26 seeds reproduced the exact effect ($\Delta = -18.8$, $p = 0.041$), confirming this is not a random sampling artifact.

---

## 7. Execution & Reproduction Guide

### Environment Setup

The repository supports separate virtual environments for PyTorch and JAX, or a single unified interpreter:

```bash
# Clone the repository
git clone https://github.com/PedroM2626/rlops-experiments.git
cd rlops-experiments

# Activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1    # On Windows
source venv/bin/activate       # On Linux / macOS

# Install Phase 1 dependencies
pip install -r ppo-empirical-studies/01_ecosystem_benchmark/requirements.txt

# Install Phase 2 dependencies
pip install -r ppo-empirical-studies/02_factorial_attribution/requirements-torch.txt
pip install -r ppo-empirical-studies/02_factorial_attribution/requirements-jax.txt
```

### Reproducing Phase 1 (Ecosystem Benchmark)

```bash
cd ppo-empirical-studies/01_ecosystem_benchmark

# 1. Run training across frameworks (example: CleanRL and SB3 Zoo)
python scripts/train_cleanrl.py --env-id LunarLander-v3 --seed 42
PPO_ENV=LunarLander-v3 PPO_SEED=42 python scripts/train_sb3_zoo.py

# 2. Compute non-parametric evaluation statistics (Kruskal-Wallis / Mann-Whitney U)
python scripts/eval_stats.py

# 3. Generate comparative learning curves and return boxplots
python scripts/compare.py
python scripts/boxplot.py
```

### Reproducing Phase 2 (Factorial Attribution & Ablations)

```bash
cd ppo-empirical-studies/02_factorial_attribution

# 1. Fast smoke test (CartPole-v1, 1 seed, 1% budget)
python run_all.py --env CartPole-v1 --seeds 0 --timesteps-scale 0.01

# 2. Run full multi-worker factorial matrix on LunarLander-v3
python run_full_background.py --workers 4 --env LunarLander-v3

# 3. Run deterministic evaluation suite (100 episodes per seed)
python evaluate.py

# 4. Generate bootstrap confidence intervals and Cliff's delta tables
python analyze.py

# 5. Render publication figures
python plot_results.py
```

### Reproducing the GAE Probe

```bash
cd ppo-empirical-studies/02_factorial_attribution/probe_gae

# Run the paired indexing analysis on committed raw data
python analyze.py
```

---

## 8. Summary of Scientific Guidelines for RL Practitioners

1. **Never Report "PPO" Without the Micro-Specification**: Citing PPO without explicitly defining value function clipping, learning rate annealing, and weight initialization produces irreproducible claims.
2. **Value Clipping is Harmful on Dense-Reward Control**: In `LunarLander-v3`, value function clipping consistently degraded mean return by over $40$ points.
3. **The Framework Stack is Blameless**: Differences between PyTorch and JAX are statistically indistinguishable under identical specifications; software framework choice can be guided purely by hardware throughput and engineering ergonomics.
4. **Boundary Indexing in Advantage Estimation Matters**: Off-by-one errors in GAE termination masks introduce statistically significant return shifts and variance inflation.
