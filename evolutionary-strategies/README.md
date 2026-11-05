# Scientific Benchmark: 3 Families of Evolutionary Algorithms for Reinforcement Learning with JAX

A rigorous experimental, conceptual, and algorithmic comparison of **3 families of evolutionary algorithms** applied to canonical Reinforcement Learning (RL) problems, with evaluation accelerated through vectorized compilation in **JAX** (`vmap` + `jit`).

---

## 1. Theoretical Foundations and Structural Taxonomy

A formal taxonomy of evolutionary computation must be defined by the **mathematical object that undergoes variation and selection over time**, not by historical nomenclature:

```
                                  ┌─────────────────────────────┐
                                  │   Evolutionary Algorithms   │
                                  └──────────────┬──────────────┘
            ┌────────────────────────────────────┼────────────────────────────────────┐
            ▼                                    ▼                                    ▼
┌──────────────────────────────┐   ┌──────────────────────────────┐   ┌──────────────────────────────┐
│  Family 1: Direct Solutions  │   │  Family 2: Programs          │   │  Family 3: Models (EDA)      │
│  (Direct Representations)    │   │   (Genetic Programming)      │   │  (Estimation of Distribution)│
├──────────────────────────────┤   ├──────────────────────────────┤   ├──────────────────────────────┤
│ • Object: parameter vector   │   │ • Object: sequence/graph of  │   │ • Object: probability        │
│   θ or vector population     │   │   computational instructions │   │   distribution P(x; Θ)       │
│ • Individuals are persisted, │   │ • Phenotype executed as a    │   │ • Individuals are temporary  │
│   mutated and recombined     │   │   symbolic function directly │   │   disposable samples         │
│ • Methods:                   │   │ • Methods:                   │   │ • Methods:                   │
│   - SimpleGA (Goldberg 1989) │   │   - LinearGP (Banzhaf 2007)  │   │   - CMA-ES (Hansen 2001/2016)│
│   - DE (Price & Storn 1997)  │   │   - CartesianGP (Miller 2000)│   │   - PBIL (Baluja 1994)       │
│   - OpenAI-ES (Salimans 2017)│   │                              │   │                              │
└──────────────────────────────┘   └──────────────────────────────┘   └──────────────────────────────┘
```

### 1.1 Why Is CMA-ES Placed in Family 3 (Models / EDA)?
**CMA-ES** (*Covariance Matrix Adaptation Evolution Strategy*, Hansen & Ostermeier 2001; Hansen 2016) can be read as a continuous second-order **Estimation of Distribution Algorithm (EDA)** (the classification is conventional; CMA-ES is also commonly called an evolution strategy):
1. **No Individual-to-Individual Genomic Heredity**: Parent individuals are never preserved, point-mutated, or directly recombined. At each generation $g$, an entire population of $\lambda$ individuals is produced by independent sampling from a multivariate Gaussian distribution:
   $$x_k \sim \mathcal{N}\left(\mu^{(g)}, (\sigma^{(g)})^2 \Sigma^{(g)}\right), \quad k = 1, \dots, \lambda$$
2. **100% Disposable Population**: Once the fitness returns are computed, **every individual is discarded**. No genome carries over into the next generation (the implementation also stores the best sampled individual, but only for reporting; it is never used to generate the next samples).
3. **Parametric Evolution of the Distribution**: What evolves iteratively are the hyperparameters of the probabilistic model:
   - **Mean vector $\mu$**: Shift driven by the weighted intermediate recombination of the $\mu_{\text{eff}}$ best samples.
   - **Global step-size $\sigma$ (CSA)**: Cumulative step-length adaptation based on the conjugate evolution path $p_\sigma$.
   - **Covariance matrix $\Sigma$ (CMA)**: Second-order adaptation via a *rank-one update* (accumulating the anisotropic path $p_c$) and a *rank-$\mu$ update* (empirical estimator of the elite dispersion).
4. On convex-quadratic objectives the adapted covariance approaches the inverse Hessian (Hansen 2016), which is the sense in which it "learns the curvature"; no such property is claimed or tested here for RL fitness landscapes. For EDAs in general see Larrañaga & Lozano (2002).

### 1.2 Why Does OpenAI-ES Belong to Family 1 (Direct Solutions)?
The algorithm proposed by Salimans et al. (2017) (**OpenAI-ES**), despite the historical term "Evolution Strategy", **neither learns nor maintains any probabilistic distribution model**:
1. There is no adaptive covariance matrix and no parametric distribution being learned. The perturbation is isotropic Gaussian noise $\epsilon \sim \mathcal{N}(0, I)$ scaled by a scalar $\sigma$ that decays geometrically ($\times 0.999$ per generation, floor $0.001$) and is not adapted from the fitness.
2. The algorithm maintains a **single parameter vector $\theta \in \mathbb{R}^d$** holding the neural network weights. (In this benchmark the evaluated policy is the best sampled candidate, not $\theta$; see Section 6.9.)
3. The random perturbations act exclusively as a **stochastic finite-difference / score-function (NES) gradient estimator**:
   $$\nabla_\theta \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)} [f(\theta + \sigma \epsilon)] = \frac{1}{\sigma} \mathbb{E}_{\epsilon} [f(\theta + \sigma \epsilon) \epsilon] \approx \frac{1}{2 n \sigma} \sum_{i=1}^n \left( f(\theta + \sigma \epsilon_i) - f(\theta - \sigma \epsilon_i) \right) \epsilon_i$$
4. The parameter vector is updated directly by gradient ascent with a deterministic optimizer (plain SGD with a geometrically decaying learning rate in this implementation): $\theta \leftarrow \theta + \alpha \widehat{\nabla} f$. The implemented estimator uses rank-shaped fitness and differs from the expression above by a constant factor absorbed in $\alpha$.
5. It is therefore a direct single-solution search in parameter space (analogous to hill-climbing and direct-perturbation algorithms), with no probabilistic density modeling.

---

## 2. Mathematical Formulation of the Evaluated Algorithms

### 2.1 Family 1: Direct Solutions (Direct Representations)
* **SimpleGA** (Goldberg 1989; Deb & Agrawal 1995):
  * Maintains an explicit population $P = \{x_1, \dots, x_N\} \subset \mathbb{R}^d$.
  * Selection by stochastic binary tournament ($k=2$).
  * Simulated binary crossover (**SBX**):
    $$\beta = \begin{cases} (2u)^{\frac{1}{\eta+1}}, & \text{if } u \le 0.5 \\ \left(\frac{1}{2(1-u)}\right)^{\frac{1}{\eta+1}}, & \text{otherwise} \end{cases}$$
    $$c_1 = \frac{1}{2}[(1+\beta)p_1 + (1-\beta)p_2], \quad c_2 = \frac{1}{2}[(1-\beta)p_1 + (1+\beta)p_2]$$
  * Gaussian mutation with exponential variance decay ($\sigma \times 0.999$ per generation, floor $0.01$) and strict elitism of 1 individual.
* **Differential Evolution (DE)** (Storn & Price 1997):
  * Classic `rand/1/bin` strategy:
    $$v_i = x_{r1} + F \cdot (x_{r2} - x_{r3}), \quad r_1 \ne r_2 \ne r_3 \ne i$$
  * Binomial crossover with probability $CR$ and a guarantee that at least 1 gene is mutated.
  * Greedy *one-to-one* selection: the trial individual replaces the parent if and only if $f(u_i) > f(x_i)$ (strict). The parent's fitness is **not** re-evaluated, so a parent keeps the noisy fitness it had when it was accepted.
* **OpenAI-ES** (Salimans et al. 2017):
  * Mirrored antithetic perturbations ($+\epsilon_i, -\epsilon_i$) for first-order variance cancellation.
  * *Fitness shaping*: zero-centered rank-based normalization, yielding invariance to monotonic reward transformations.
  * Direct weight-vector update via SGD with geometric learning-rate decay ($0.9999$ per generation, floor $10^{-5}$) and geometric decay of $\sigma$ ($0.999$ per generation, floor $0.001$).

### 2.2 Family 2: Symbolic Programs (Genetic Programming)
* **LinearGP (LGP)** (Brameier & Banzhaf 2007):
  * Individual represented as a linear sequence of register instructions: `[op, dst, src1, src2]`.
  * Register set: $R = R_{\text{obs}} \cup R_{\text{extra}} \cup R_{\text{act}}$.
  * Primitive function set (10 operators, as implemented in `apply_op_jax`): $\{+, -, \times, \tanh, \sin, \cos, |\cdot|, \text{noop}, \max, \min\}$. There is no protected division.
  * Phenotype executed as a sequential imperative program, which allows reuse of intermediate variables and the presence of intrinsically neutral code (*introns*).
  * Binary tournament selection, single-point crossover, point mutation and elitism of 1. The reported best program is the best of the current generation, which is carried over and re-evaluated on fresh episodes (see Section 6.8 for why the earlier best-so-far rule was replaced).
* **CartesianGP (CGP)** (Miller & Thomson 2000):
  * Individual encoded as a positional 2D directed acyclic graph (DAG) ($1 \times N_{\text{cols}}$).
  * Intermediate nodes receive connections only from inputs or from nodes in earlier columns.
  * Pointwise structural mutation on connections and functions.
  * $(1+\lambda)$ strategy with $\lambda=7$: the parent is re-evaluated on fresh episodes every generation together with its mutated offspring and is replaced by the best individual of the generation. On ties `argmax` keeps the parent, so there is no neutral drift.

### 2.3 Family 3: Probabilistic Models (EDA)
* **CMA-ES** (Hansen & Ostermeier 2001; Hansen 2016):
  * Multivariate continuous model $\mathcal{N}(\mu, \sigma^2 C)$.
  * Eigendecomposition of the covariance $C = B D^2 B^T$ (where $B$ is the orthonormal eigenbasis and $D$ the diagonal matrix of principal standard deviations).
  * Step-length adaptation via CSA (*Cumulative Step-length Adaptation*):
    $$p_\sigma \leftarrow (1-c_\sigma) p_\sigma + \sqrt{c_\sigma(2-c_\sigma)\mu_{\text{eff}}} \, C^{-1/2} \frac{\mu^{(g+1)}-\mu^{(g)}}{\sigma^{(g)}}$$
    $$\sigma^{(g+1)} = \sigma^{(g)} \exp \left( \frac{c_\sigma}{d_\sigma} \left( \frac{\|p_\sigma\|}{E[\|\mathcal{N}(0, I)\|]} - 1 \right) \right)$$
  * Covariance matrix adaptation:
    $$C^{(g+1)} = (1 - c_1 - c_\mu) C^{(g)} + c_1 \left( p_c p_c^T + \delta(h_\sigma) C^{(g)} \right) + c_\mu \sum_{i=1}^\mu w_i y_{i:\lambda} y_{i:\lambda}^T$$
* **Continuous PBIL** (Baluja 1994; Sebag & Ducoulombier 1998):
  * Independent univariate Gaussian model: $\Theta = \{(\mu_1, \sigma_1), \dots, (\mu_d, \sigma_d)\}$.
  * Incremental update based on the centroid and variance of the top-$k$ solutions:
    $$\mu \leftarrow (1-\alpha) \mu + \alpha \, \bar{x}_{\text{top-k}}$$
    $$\sigma \leftarrow (1-\alpha_\sigma) \sigma + \alpha_\sigma \, \text{std}(x_{\text{top-k}})$$
  * Stochastic mutation applied directly to the parameters of the probabilistic model itself.

---

## 3. Policy Architecture and Engineering in JAX

### 3.1 Unified Neural Policy (Families 1 and 3)
To guarantee homogeneous and unbiased comparisons, all methods in Families 1 and 3 optimize the same multilayer perceptron (MLP) architecture:
$$\text{obs} \xrightarrow{\quad} \text{Dense}(32) \xrightarrow{\tanh} \text{Dense}(32) \xrightarrow{\tanh} \text{Dense}(\text{act})$$

* **Discrete actions**: $a = \arg\max(\text{logits})$.
* **Continuous actions**: $a = \tanh(\text{logits}) \times \text{action\_scale}$.

Total number of parameters per environment:
* **CartPole-v1** ($4 \to 32 \to 32 \to 2$): $1{,}282$ parameters
* **Acrobot-v1** ($6 \to 32 \to 32 \to 3$): $1{,}379$ parameters
* **Pendulum-v1** ($3 \to 32 \to 32 \to 1$): $1{,}217$ parameters
* **MountainCarContinuous-v0** ($2 \to 32 \to 32 \to 1$): $1{,}185$ parameters

### 3.2 Vectorization and XLA Acceleration in Pure JAX (All 3 Families)
* **Families 1 and 3 (Neural Policies)**:
  * Evaluating one individual in one episode is modeled as a pure function `rollout(flat_params, rng) -> float`.
  * The whole population of size $P$ is parallelized through native vectorized compilation:
    $$\text{eval\_pop} = \text{jax.jit}(\text{jax.vmap}(\text{rollout\_multi}, \text{in\_axes}=(0, \text{None})))$$
* **Family 2 (Pure Vectorized Symbolic Programs in JAX)**:
  * Previously, symbolic programs in LinearGP and CartesianGP were evaluated through interpreted Python loops.
  * In the current implementation, **Family 2 is 100% compiled and vectorized in pure JAX**:
    1. **Branchless Operator Dispatch (`apply_op_jax`)**: The 10 primitives of Section 2.2 are dispatched via `jax.lax.switch`, compiling directly to branchless XLA tables; results are clipped to $[-100, 100]$ and nan/inf are replaced by 0.
    2. **LinearGP Forward Pass (`forward_lgp_jax`)**: Instructions `[op, dst, s1, s2]` iterate over register arrays via `jax.lax.scan` inside XLA.
    3. **CartesianGP Forward Pass (`forward_cgp_jax`)**: DAG column nodes iterate over execution buffers via `jax.lax.fori_loop` inside XLA.
    4. **Nested Parallelism (`make_program_pop_eval_fn`)**: The entire population of programs is parallelized across multiple rollout episodes and time horizons via nested `jax.vmap(rollout_fn)(programs, rngs)`, evaluating the whole population in one compiled call. No speedup figure over the earlier interpreted-Python implementation is reported: those timings were not preserved in a reproducible form (the legacy code is no longer in the repository).
* In accordance with the `gymnax 1.0.0` specification, the environment transition returns 6 elements:
  `obs, state, reward, terminated, truncated, info = env.step(...)`

### 3.3 Training and Evaluation Compute Budget (Episodes and Steps)

To ensure reproducibility and formal clarity, the exact interaction budget with the environments is itemized below:

#### A. Temporal Horizon per Episode ($H = \text{max\_steps}$)
In each environment an episode has a maximum duration bounded by the horizon $H$:
* **CartPole-v1**: $H = 500$ steps (success criterion: stay balanced for 500 steps, cumulative return = +500).
* **Acrobot-v1**: $H = 500$ steps (cost of $-1.0$ per step until the tip reaches the target height; optimum around $-64$ to $-70$ steps).
* **Pendulum-v1**: $H = 200$ steps (continuous penalty on normalized angle, angular velocity and torque effort; optimum around $-150$ to $-200$).
* **MountainCarContinuous-v0**: $H = 999$ steps (per-step action cost, plus $+100$ upon reaching the top of the right hill; optimum around $+90$ to $+95$).

#### B. Evaluation Episodes per Individual ($N_{\text{rollouts}}$)
To mitigate the stochastic variance inherent in the environment initial conditions (such as the initial angular perturbation in CartPole and Pendulum):
* **Families 1 and 3 (SimpleGA, DE, OpenAI-ES, CMA-ES, PBIL)**: Each individual/sample is evaluated over **$N_{\text{rollouts}} = 2$ independent episodes** per generation in the standard regime of all reported tables (Pendulum: 8, Section 3.3.C). The default non-quick mode of `run_benchmark.py` ($G=150$, $N_{\text{rollouts}}=4$; GP: $G=60$) was not used for the reported results. The fitness assigned is the mean of the returns obtained:
  $$f(x) = \frac{1}{N_{\text{rollouts}}} \sum_{e=1}^{N_{\text{rollouts}}} R_e$$
  At each generation a new stochastic seed (`rng_eval`) is spawned via `jax.random.split`, ensuring that individuals are tested against different initial conditions and cannot memorize a single trajectory.
* **Family 2 (LinearGP and CartesianGP)**: Each symbolic program is evaluated over **$N_{\text{rollouts}} = 3$ independent episodes** per generation.

#### C. Total Training Budget Matrix (Simulated Episodes and Steps)

To guarantee academic transparency and prevent misleading aggregation, the benchmark employs two clearly separated budget regimes:
1. **Standard Regime (`CartPole-v1`, `Acrobot-v1`, `MountainCarContinuous-v0`)**:
   * Families 1 & 3: $G = 30$ generations, $N_{\text{rollouts}} = 2$ episodes per individual.
   * Family 2 (GP): $G = 15$ generations, $N_{\text{rollouts}} = 3$ episodes per individual.
2. **Extended Regime (`Pendulum-v1`)**:
   * Adopted specifically to mitigate the extreme stochastic variance of the initial continuous angle $\theta_0 \sim \mathcal{U}[-\pi, \pi]$ (*Winner's Curse*).
   * Families 1 & 3: $G = 100$ generations, $N_{\text{rollouts}} = 8$ episodes per individual.
   * Family 2 (GP): $G = 40$ generations, $N_{\text{rollouts}} = 8$ episodes per individual.

The detailed matrix of simulated episodes and maximum step ceilings is presented below:

| Algorithm | Family | Standard Regime ($G$, $N$) | Pendulum Regime ($G$, $N$) | Total Episodes (Std) | Total Episodes (Pendulum) | Step Ceiling CartPole ($H=500$) | Step Ceiling Acrobot ($H=500$) | Step Ceiling MountainCar ($H=999$) | Step Ceiling Pendulum ($H=200$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **SimpleGA** | F1 | 30 gen, $N=2$ | 100 gen, $N=8$ | **1,920** | **25,600** | 960,000 | 960,000 | 1,918,080 | 5,120,000 |
| **DE** | F1 | 30 gen, $N=2$ | 100 gen, $N=8$ | **1,920** | **25,600** | 960,000 | 960,000 | 1,918,080 | 5,120,000 |
| **OpenAI-ES** | F1 | 30 gen, $N=2$ | 100 gen, $N=8$ | **1,920** | **25,600** | 960,000 | 960,000 | 1,918,080 | 5,120,000 |
| **LinearGP** | F2 | 15 gen, $N=3$ | 40 gen, $N=8$ | **720** | **5,120** | 360,000 | 360,000 | 719,280 | 1,024,000 |
| **CartesianGP**| F2 | 15 gen, $N=3$ | 40 gen, $N=8$ | **360** | **2,560** | 180,000 | 180,000 | 359,640 | 512,000 |
| **CMA-ES** | F3 | 30 gen, $N=2$ | 100 gen, $N=8$ | **1,500** | **20,000** | 750,000 | 750,000 | 1,498,500 | 4,000,000 |
| **PBIL** | F3 | 30 gen, $N=2$ | 100 gen, $N=8$ | **1,920** | **25,600** | 960,000 | 960,000 | 1,918,080 | 5,120,000 |

*\*Note on Family 2 (GP)*: In pure JAX vectorized rollouts (`lax.scan` over max steps), episodes step until the environment horizon $H$ with early termination masking or bounding, maintaining strictly identical step ceilings across all evaluation batches.

---

## 4. Consolidated Experimental Results

All data below was obtained empirically through the automated benchmark (no synthetic or extrapolated data):

### 4.1 Discrete Environments (All 3 Families in Pure JAX)

Data corresponding to the benchmark evaluation runs saved in `results/results_summary.csv` and `results/validation_100_summary.csv`:

<!-- BEGIN:table_4_1 -->
| Family | Algorithm | Paradigm | CartPole-v1 (Best Train) | Acrobot-v1 (Best Train) | CartPole Time (s) |
|:---|:---|:---|:---:|:---:|:---:|
| **F1: Direct Solutions** | **SimpleGA** | Tournament + SBX crossover | 318.50 | −64.00 | 4.7s |
| **F1: Direct Solutions** | **DE** | rand/1/bin trial vectors | 500.00 | −78.00 | 3.9s |
| **F1: Direct Solutions** | **OpenAI-ES** | NES gradient ascent | 419.50 | −64.00 | 4.0s |
| **F2: Programs** | **LinearGP** | Register sequence | 53.67 | −76.67 | 1.6s |
| **F2: Programs** | **CartesianGP** | Positional 2D DAG (1+7)ES | 500.00 | −84.33 | 1.5s |
| **F3: Models (EDA)** | **CMA-ES** | Covariance $\mathcal{N}(\mu, \sigma^2 \Sigma)$ | 500.00 | −64.00 | 17.6s |
| **F3: Models (EDA)** | **PBIL** | Univariate marginal model | 464.50 | −75.00 | 4.5s |
<!-- END:table_4_1 -->

*Timing note*: times are wall-clock seconds measured on one machine (CPU, JAX, other jobs not running at the same time), include JIT compilation, and are indicative only. For Family 2 the per-environment training budget is $G=15$ generations and $N_{\text{rollouts}}=3$ (Section 3.3), so its times are not comparable with Families 1 and 3 at equal cost.

*Meaning of "Best Train" for Family 2*: since the update described in Section 2.2, the reported program is the best of the **last** generation (re-evaluated on fresh episodes), not the best fitness ever observed. The original best-so-far rule is kept as an option (`best_so_far=True`) and its multi-seed results are stored in `results/multiseed_study_S5_best_so_far_gp.json`.

### 4.2 Continuous Environments (All 3 Families in Pure JAX)

<!-- BEGIN:table_4_2 -->
| Family | Algorithm | Pendulum-v1 (Best) | Pendulum-v1 (Mean Final) | MountainCarCont (Best) | MountainCarCont (Mean Final) | Pendulum Time (s) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **F1: Direct Solutions** | **SimpleGA** | −157.84 | −944.37 | 94.17 | −56.30 | 6.5s |
| **F1: Direct Solutions** | **DE** | −572.58 | −1378.27 | 85.50 | −90.91 | 5.7s |
| **F1: Direct Solutions** | **OpenAI-ES** | −885.38 | −1228.99 | 0.00 | −1.12 | 5.8s |
| **F2: Programs** | **LinearGP**† | −965.75 | −1256.17 | 0.00 | −3.78 | 3.6s |
| **F2: Programs** | **CartesianGP** | −762.83 | −1011.80 | 0.00 | −2.14 | 3.0s |
| **F3: Models (EDA)** | **CMA-ES** | −229.74 | −924.98 | 95.83 | −59.37 | 56.2s |
| **F3: Models (EDA)** | **PBIL** | −614.96 | −1520.16 | 94.22 | −74.45 | 8.2s |
<!-- END:table_4_2 -->

†*LinearGP on Pendulum-v1 is the degenerate zero-torque policy (Section 6.6.3 item 6): its "Best" is the best of noisy $N$-episode evaluations of a constant-zero-output program, not evidence of control. Do not use it in performance comparisons on this environment.*

---

## 5. Out-of-Sample Statistical Validation (Rigorous 100-Episode Protocol)

In line with modern methodological guidelines for reproducibility and evaluation in Reinforcement Learning (Henderson et al. 2018; Machado et al. 2018; Agarwal et al. 2021), **a metric obtained during training with few rollouts ($N=2$ or $3$) is not sufficient evidence of robust convergence**. Evolutionary policies can overfit the stochastic conditions seen during selection (*Winner's Curse*).

To quantify the true generalization capability and epistemic uncertainty, the best controller of each algorithm was subjected to a battery of **$N=100$ independent test episodes** with a pseudo-random seed unseen during optimization (`seed = 99999`):
* **Families 1 and 3 (Neural Networks)**: Evaluated through pure vectorized compilation in JAX (`jax.vmap` over 100 seeds with `jax.lax.scan`), guaranteeing identical parallelism and no temporal bias.
* **Family 2 (Symbolic Programs)**: Evaluated through pure vectorized compilation in JAX (`make_program_test_eval_fn` via `jax.vmap` over 100 seeds with `jax.lax.scan`), executing symbolic register sequences and Cartesian DAGs with continuous `tanh` action bounding under exact out-of-sample seeds.

### 5.1 Computed Statistical Metrics
* **Central Tendency**: Sample mean ($\bar{R}_{\text{test}}$) and median.
* **Sampling Uncertainty**: Standard error of the mean ($\text{SEM} = s / \sqrt{N}$) and **non-parametric 95% bootstrap confidence interval** ($B = 2{,}000$ resamples).
* **Dispersion and Robustness**: Standard deviation ($s$), interquartile range ($\text{IQR} = Q_3 - Q_1$) and coefficient of variation ($\text{CV} = s / |\bar{R}|$).
* **Reliability**: *Signal-to-Noise Ratio* ($\text{SNR} = |\bar{R}| / s$).
* **Optimism Gap / Winner Bias (*Winner's Curse*)**:
  $$\Delta_{\text{optimism}} = f_{\text{train\_best}} - \bar{R}_{\text{test}}$$
  Measures the degree of spurious overfitting of the policy to the few training episodes.
* **Success Rate ($\%$)**: Proportion of episodes that reached the canonical task-resolution thresholds:
  * CartPole-v1: $R \ge 475.0$
  * Acrobot-v1: $R \ge -100.0$
  * Pendulum-v1: $R \ge -200.0$
  * MountainCarContinuous-v0: $R \ge 90.0$

### 5.2 Consolidated Validation Table ($N=100$ Out-of-Sample Episodes Across All 28 Configurations)

Raw empirical data extracted directly from `results/validation_100_summary.csv`:

<!-- BEGIN:table_5_2 -->
| Family | Algorithm | Environment | Train Best ($N \le 3$) | Test Mean $\pm$ SEM | 95% Bootstrap CI | Median | IQR | Std Dev | SNR | Optimism Gap ($\Delta$) | Success Rate (%) |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **F1** | **SimpleGA** | CartPole-v1 | 318.50 | 318.82 $\pm$ 19.43 | [281.37, 356.55] | 500.00 | 395.50 | 194.29 | 1.64 | −0.32 | 51.0% |
| **F1** | **DE** | CartPole-v1 | 500.00 | 123.84 $\pm$ 8.48 | [108.68, 141.80] | 101.00 | 40.00 | 84.79 | 1.46 | +376.16 | 2.0% |
| **F1** | **OpenAI-ES** | CartPole-v1 | 419.50 | 138.27 $\pm$ 10.66 | [118.93, 161.17] | 96.00 | 66.75 | 106.60 | 1.30 | +281.23 | 4.0% |
| **F2** | **LinearGP** | CartPole-v1 | 53.67 | 42.69 $\pm$ 1.97 | [39.09, 46.68] | 40.00 | 21.00 | 19.70 | 2.17 | +10.98 | 0.0% |
| **F2** | **CartesianGP** | CartPole-v1 | 500.00 | 482.68 $\pm$ 4.95 | [472.51, 492.04] | 500.00 | 0.00 | 49.48 | 9.76 | +17.32 | 88.0% |
| **F3** | **CMA-ES** | CartPole-v1 | 500.00 | 500.00 $\pm$ 0.00 | [500.00, 500.00] | 500.00 | 0.00 | 0.00 | $\infty$ | 0.00 | 100.0% |
| **F3** | **PBIL** | CartPole-v1 | 464.50 | 412.90 $\pm$ 7.64 | [397.39, 427.45] | 406.50 | 139.00 | 76.36 | 5.41 | +51.60 | 28.0% |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **F1** | **SimpleGA** | Acrobot-v1 | −64.00 | −85.55 $\pm$ 4.55 | [−95.12, −78.73] | −76.50 | 19.00 | 45.46 | 1.88 | +21.55 | 88.0% |
| **F1** | **DE** | Acrobot-v1 | −78.00 | −93.68 $\pm$ 3.61 | [−101.43, −87.34] | −85.50 | 20.25 | 36.09 | 2.60 | +15.68 | 82.0% |
| **F1** | **OpenAI-ES** | Acrobot-v1 | −64.00 | −85.85 $\pm$ 2.22 | [−90.36, −81.58] | −81.50 | 19.25 | 22.19 | 3.87 | +21.85 | 88.0% |
| **F2** | **LinearGP** | Acrobot-v1 | −76.67 | −81.66 $\pm$ 1.07 | [−83.77, −79.73] | −77.00 | 10.25 | 10.70 | 7.63 | +4.99 | 92.0% |
| **F2** | **CartesianGP** | Acrobot-v1 | −84.33 | −93.56 $\pm$ 1.60 | [−96.70, −90.54] | −91.00 | 20.50 | 16.01 | 5.84 | +9.23 | 73.0% |
| **F3** | **CMA-ES** | Acrobot-v1 | −64.00 | −77.79 $\pm$ 1.22 | [−80.30, −75.46] | −73.00 | 12.50 | 12.22 | 6.37 | +13.79 | 96.0% |
| **F3** | **PBIL** | Acrobot-v1 | −75.00 | −101.90 $\pm$ 4.50 | [−112.47, −94.93] | −95.50 | 20.25 | 44.96 | 2.27 | +26.90 | 55.0% |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **F1** | **SimpleGA** | Pendulum-v1 | −157.84 | −352.97 $\pm$ 23.94 | [−399.54, −307.80] | −272.60 | 436.69 | 239.42 | 1.47 | +195.12 | 35.0% |
| **F1** | **DE** | Pendulum-v1 | −572.58 | −714.67 $\pm$ 56.98 | [−827.75, −603.07] | −599.67 | 1041.44 | 569.79 | 1.25 | +142.10 | 25.0% |
| **F1** | **OpenAI-ES** | Pendulum-v1 | −885.38 | −1075.51 $\pm$ 24.89 | [−1125.43, −1027.93] | −998.93 | 384.17 | 248.94 | 4.32 | +190.14 | 0.0% |
| **F2** | **LinearGP**† | Pendulum-v1 | −965.75 | −1247.90 $\pm$ 39.37 | [−1323.54, −1170.53] | −1254.86 | 593.47 | 393.71 | 3.17 | +282.15 | 0.0% |
| **F2** | **CartesianGP** | Pendulum-v1 | −762.83 | −852.61 $\pm$ 20.13 | [−892.01, −813.49] | −854.11 | 367.20 | 201.35 | 4.23 | +89.78 | 0.0% |
| **F3** | **CMA-ES** | Pendulum-v1 | −229.74 | −268.67 $\pm$ 18.59 | [−303.78, −231.68] | −261.94 | 252.75 | 185.90 | 1.45 | +38.92 | 35.0% |
| **F3** | **PBIL** | Pendulum-v1 | −614.96 | −641.46 $\pm$ 23.08 | [−688.33, −598.69] | −542.80 | 227.23 | 230.80 | 2.78 | +26.49 | 0.0% |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **F1** | **SimpleGA** | MountainCarCont | 94.17 | 91.84 $\pm$ 0.24 | [91.35, 92.31] | 90.71 | 3.65 | 2.40 | 38.25 | +2.33 | 84.0% |
| **F1** | **DE** | MountainCarCont | 85.50 | 85.68 $\pm$ 0.36 | [84.96, 86.36] | 87.74 | 5.19 | 3.61 | 23.75 | −0.18 | 0.0% |
| **F1** | **OpenAI-ES** | MountainCarCont | 0.00 | 0.00 $\pm$ 0.00 | [0.00, 0.00] | 0.00 | 0.00 | 0.00 | 1.02 | 0.00 | 0.0% |
| **F2** | **LinearGP** | MountainCarCont | 0.00 | 0.00 $\pm$ 0.00 | [0.00, 0.00] | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0% |
| **F2** | **CartesianGP** | MountainCarCont | 0.00 | 0.00 $\pm$ 0.00 | [0.00, 0.00] | 0.00 | 0.00 | 0.00 | 1.02 | 0.00 | 0.0% |
| **F3** | **CMA-ES** | MountainCarCont | 95.83 | 96.43 $\pm$ 0.11 | [96.22, 96.63] | 96.72 | 1.76 | 1.06 | 90.99 | −0.60 | 100.0% |
| **F3** | **PBIL** | MountainCarCont | 94.22 | 93.87 $\pm$ 0.12 | [93.64, 94.10] | 94.06 | 1.80 | 1.15 | 81.28 | +0.35 | 100.0% |
<!-- END:table_5_2 -->

*\*Note on DE in MountainCar*: with seed 42, DE converged to $\approx 85.7$, below the formal success criterion $R \ge 90.0$ (multi-seed behavior differs, see Section 6.6.3 item 2).

†*LinearGP on Pendulum-v1 (seed 42) is the zero-torque baseline: constant zero action, mean $-1247.903$, std $393.71$, identical to the reported row. It says nothing about GP's ability to learn control. See Sections 6.6.3 item 6 and 6.8.*

---

## 6. Analysis and Scientific Discussion of the Results

### 6.1 The Empirical Manifestation of the Winner's Curse (Spurious Optimism in Fast Evaluations)
Validation over 100 episodes exposed a crucial phenomenon in evolutionary RL:
1. **The Collapse of Differential Evolution (DE) on CartPole (seed 42)**:
   * During training ($N_{\text{rollouts}}=2$), DE reported an apparent optimal fitness of **500.00**.
   * In testing ($N=100$), the mean collapsed to **123.84** with a **success rate of only 2.0%** ($\Delta = +376.16$).
   * Across 10 seeds (Section 6.6.2) DE obtains $314.62 \pm 46.29$ with $43.7\%$ success, so seed 42 is a bad case for DE, not its typical outcome.
   * *Interpretation (hypothesis, not tested)*: the large gap is consistent with selection on a noisy fitness estimate (best-of-run chosen on 2 episodes). The causal ablation of Section 6.7 tests this on CartPole; the specific individual selected here was not inspected.
2. **CMA-ES on CartPole (seed 42)**:
   * **CMA-ES** reached **500.00 $\pm$ 0.00** over the 100 out-of-sample episodes, with a **standard deviation of strictly 0.00** and **100% success**.
   * *Caveat (see Section 6.6)*: this is the seed-42 run only. Across 10 training seeds, CMA-ES obtained $479.56 \pm 7.94$ with $88.6\%$ success.
   * A possible explanation (hypothesis, untested; no flatness or curvature measurement was made) is that covariance adaptation steers the search toward a broad basin that tolerates variation in the initial state. What is measured is the outcome: the lowest $\sigma_{\text{train}}$ across seeds on CartPole (Section 6.6.3 item 1).

### 6.2 Generalization of Symbolic Programs (Family 2)
Single-seed (seed 42) results, Section 5.2:
* On **CartPole-v1**, **CartesianGP** achieved a **88.0% success rate** with a test mean of **482.68 $\pm$ 4.95** and median 500.00. LinearGP reached only 42.69 (0.0% success).
* On **Acrobot-v1**, **LinearGP** achieved a **92.0% success rate** with a test mean of **-81.66 $\pm$ 1.07** and an **SNR of 7.63** (the highest SNR among the seven algorithms on Acrobot in this run).
* On **Pendulum-v1**, CartesianGP obtained -852.61 $\pm$ 20.13 with 0% success, and LinearGP $-1247.90$, which is the zero-torque baseline (Section 6.8). Neither is a successful swing-up controller.

Multi-seed (S=10, Section 6.6.2):
* CartPole: CartesianGP 228.98 (21.5% success), LinearGP 205.06 (11.1%).
* Acrobot: LinearGP -93.14 (76.9% success) is statistically in line with DE (-98.09, 79.1%) and SimpleGA (-89.67, 79.6%), whereas CartesianGP is the worst of all (-181.07, 39.8%, $\rho=3.27$).
* Pendulum: CartesianGP -1097.95 and LinearGP -1189.09, both with 0% success.
* **Conclusion supported by the data**: symbolic programs do well in some single-seed runs (CartesianGP on CartPole, LinearGP on Acrobot) but not robustly across seeds, and not on Pendulum or MountainCar. The idea that compact programs avoid the overfitting of neural weights is a hypothesis; the data do not support a general claim of robust generalization.

### 6.3 Failure of OpenAI-ES and GP vs. Other Methods on MountainCarContinuous
* **OpenAI-ES, LinearGP, and CartesianGP (Score: 0.00, Success: 0.0%)**:
  All three methods collapsed on MountainCarContinuous. The measured fact is a return of exactly $0.00$ in all 10 seeds (Section 6.6.3 item 5). The mechanism is a **hypothesis, not tested**: a policy that outputs near-zero torque incurs ~zero action cost, so if the population never reaches the goal the fitness landscape is flat and gives no selection or gradient signal ($\nabla_\theta \mathbb{E}[f] \approx \mathbf{0}$ for OpenAI-ES; a very low chance for GP mutations to find the energy-pumping oscillation).
* **CMA-ES and PBIL, seed 42 (Success: 100.0%, SNR: 91.0 and 81.3)**: both solved the environment with $\text{SEM} \approx 0.11$ and $\text{Std} \le 1.15$.
* **Across 10 seeds (Section 6.6.2)** the methods that escape the valley are less uniform: PBIL 90.36 (84.2% success), DE 91.01 (62.7%), SimpleGA 88.82 (70.2%), CMA-ES 80.28 (63.2%, with some seeds ending near 50). The data support that these four methods escape the valley while OpenAI-ES and GP did not; they do not support a "decisive superiority" of covariance adaptation, and the reason was not isolated.

### 6.4 The Stochastic Sensitivity of Pendulum-v1 and the Extended Rollout Budget
* **Why the budget differs on Pendulum**: the uniform initial angle $\theta_0 \sim \mathcal{U}[-\pi, \pi]$ makes a fitness based on very few episodes extremely noisy (an individual that happens to start near the upright position scores well without swinging up). Pendulum is therefore run with $N_{\text{rollouts}}=8$ and $G=100$ (Families 1 and 3) or $G=40$ (GP), instead of $N=2$ and $G=30$. A quick $N=2$ run on Pendulum was not preserved, so no claim is made about its optimism gap.
* **Seed-42 results ($G=100$, $N_{\text{rollouts}}=8$)**:
  * **CMA-ES**: mean test return **$-268.67 \pm 18.59$** (median **$-261.94$**, **35.0% success**), optimism gap **$+38.92$**. This is *partial* stabilization: $35\%$ of the 100 test episodes reach $R \ge -200$ and the distribution is wide (IQR $252.75$). The breakdown by initial angle is measured in Section 6.5.3: 68% success for $|\theta_0| < \pi/3$, 37% for $[\pi/3, 2\pi/3)$ and 0% for $|\theta_0| \ge 2\pi/3$.
  * **SimpleGA**: test mean **$-352.97 \pm 23.94$** (median **$-272.60$**, **35.0% success**).
* **Across 10 seeds (Section 6.6.2)**: CMA-ES -440.96 (37.6% success), SimpleGA -555.98 (30.7%), DE -709.68 (24.8%), PBIL -830.37 (6.6%), OpenAI-ES -1015.65 (0.0%). Partial swing-up is reproducible for CMA-ES and SimpleGA, but no evolutionary method solves the task consistently, and the role of the rollout budget was not isolated by an ablation on Pendulum.

### 6.5 Plateau Test (CMA-ES, 250 Generations) and Black-Box ES vs. SAC on Pendulum-v1

#### 6.5.1 Empirical Measurement: 250-Generation Stress Test (Seed 42)
To rigorously determine whether the remaining performance deficit on `Pendulum-v1` was simply a matter of insufficient optimization budget, a dedicated long-horizon stress test was executed and saved to `results/pendulum_stress_test_250.json` (reproducible via `run_pendulum_stress_test.py`):
* **Optimization Budget** (seed 42): $G=250$ generations, $\lambda=25$ candidates, $N_{\text{rollouts}}=8$, $H=200$ steps per rollout ($\approx 10{,}000{,}000$ simulated environment steps).
* **Training Convergence**: Apparent train best fitness reached **$-70.13$** (apparent generation mean: $-455.71$).
* **Out-of-Sample Test Evaluation ($N=100$ independent episodes, seed $99999$)**:
  * Mean Test Return: **$-242.43 \pm 15.37$** (Stratified 95% Bootstrap CI: `[-271.51, -212.91]`)
  * Median Test Return: **$-248.35$** (Interquartile Range IQR: $227.03$)
  * Success Rate ($R \ge -200$): **$37.0\%$**
  * Optimism Gap ($\Delta$): **$+172.29$**
* **Empirical Finding**: Increasing the computational budget by $2.5\times$ (from $G=100$ to $G=250$) yielded only a marginal gain of $+26.24$ return points and an increase of only $+2.0\%$ in success rate (from $35.0\%$ to $37.0\%$).
* **Consistency check**: this run uses the same seed, algorithm and hyperparameters as the $G=100$ validation. Its training best at generation 100 is $-229.74$, identical to the $G=100$ value of Section 5.2, so the two runs follow the same deterministic trajectory and the train-fitness difference is only the extra 150 generations ($-229.74 \to -70.13$).
* **Interpretation of the seed-42 run alone (hypothesis)**: the test-return gain ($+26.24$) is of the order of the individual SEMs ($15.37$ and $18.59$), and the train-best fitness kept improving while the test return barely moved (optimism gap $+38.92 \to +172.29$). A single seed cannot establish a plateau, so it was repeated with 10 seeds (Section 6.5.1b).

#### 6.5.1b Multi-Seed Repeat of the 250-Generation Test (S=10)
`run_pendulum_stress_multiseed.py` repeats the same CMA-ES configuration ($N_{\text{rollouts}}=8$, test seed $99999$, 100 test episodes) at $G=250$ and, for the extra seeds, at $G=100$. Seeds $101$–$404$ were run first ($S=5$) and seeds $505, 606, 707, 808, 909$ were added afterwards to reach $S=10$ (`results/pendulum_stress_multiseed.json`, `results/pendulum_cmaes_G250_extra_seeds.json`, `results/pendulum_cmaes_G100_extra_seeds.json`). Seed 42 is the run above; the $G=100$ reference for seeds $42$–$404$ is the CMA-ES row of Section 6.6.2.

| Seed | Test mean $G=100$ | Test mean $G=250$ | Change | Success $G=250$ | Train best $G=250$ |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 42 | −268.67 | −242.43 | +26.2 | 37% | −70.13 |
| 101 | −701.44 | −247.97 | +453.5 | 34% | −97.37 |
| 202 | −221.47 | −317.54 | −96.1 | 38% | −70.79 |
| 303 | −274.77 | −210.85 | +63.9 | 44% | −65.10 |
| 404 | −796.35 | −336.59 | +459.8 | 53% | −66.49 |
| 505 | −849.62 | −849.62 | 0.0 | 35% | −65.69 |
| 606 | −248.00 | −244.08 | +3.9 | 41% | −94.32 |
| 707 | −430.69 | −215.29 | +215.4 | 44% | −67.74 |
| 808 | −234.28 | −338.47 | −104.2 | 33% | −80.60 |
| 909 | −384.32 | −843.40 | −459.1 | 36% | −71.50 |
| **Mean (SD)** | **−441.0 (247.2)** | **−384.6 (247.9)** | **+56.3 (272.3)** | **39.5%** | |
| **Median** | −329.5 | −282.8 | | | |

* **Measured**: over 10 seeds, $G=250$ is better than $G=100$ in 6 seeds, equal in 1 and worse in 3. The mean paired change is $+56.3$ with standard deviation $272.3$ ($t \approx 0.65$, Wilcoxon signed-rank $p \approx 0.57$), i.e. **no detectable difference** between $G=100$ and $G=250$. Between-seed dispersion is unchanged ($\approx 247$ at both budgets). Mean success is $37.6\%$ at $G=100$ and $39.5\%$ at $G=250$.
* **Correction of an earlier reading**: with the first $S=5$ seeds the gain looked large ($+181$ mean, $\sigma_{\text{train}}$ from $273$ to $53$). The five additional seeds removed that effect (seeds 505 and 909 end near $-850$ at $G=250$), so the $S=5$ result was a small-sample artifact. This is an example of the imprecision warned about in Section 6.6.
* **Reading**: the data are compatible with a plateau in mean return from $G=100$ to $G=250$ for CMA-ES, but the dominant feature is large seed-to-seed variability (final test means from $-211$ to $-850$ at $G=250$), not a systematic improvement with budget. Two seeds (505, and 101 at $G=100$) remain stuck near $-800$ to $-850$, which looks like convergence to a poor local solution; this was not analyzed further. The optimism gap stays large (train best $\approx -65$ to $-97$ versus test $-211$ to $-850$).
* **Scope**: CMA-ES only, $S=10$; the other algorithms were not run at $G=250$.

#### 6.5.1c Budget Test for the Other Parametric Algorithms (S=10)
`run_pendulum_g250_all.py` repeats the $G=100$ vs. $G=250$ comparison for SimpleGA, DE, OpenAI-ES and PBIL ($N_{\text{rollouts}}=8$, 10 seeds, same test protocol; `results/pendulum_budget_other_algos.json`; $G=100$ for seeds $42$–$404$ from Section 6.6.2).

| Algorithm | Mean test $G=100$ (SD over seeds) | Mean test $G=250$ (SD) | Paired change (SD) | Seeds improved | Wilcoxon $p$ | Success $G=100$ → $G=250$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| SimpleGA | −556.0 (275.7) | −439.8 (254.8) | +116.2 (236.2) | 5/10 | 0.156 | 30.7% → 32.8% |
| DE | −709.7 (145.0) | −550.0 (266.0) | +159.7 (264.7) | 7/10 | 0.129 | 24.8% → 29.2% |
| OpenAI-ES | −1015.7 (28.2) | −662.4 (157.0) | +353.2 (147.9) | 10/10 | 0.002 | 0.0% → 5.2% |
| PBIL | −830.4 (182.0) | −776.5 (248.2) | +53.9 (309.3) | 5/10 | 0.570 | 6.6%* → 17.7% |
| CMA-ES (Section 6.5.1b) | −441.0 (247.2) | −384.6 (247.9) | +56.3 (272.3) | 6/10 | 0.570 | 37.6% → 39.5% |

*\*Success at $G=100$ is the mean of the 5-seed value of Section 6.6.2 and the 5 extra seeds. PBIL's success rises while its mean return barely changes, so success and mean return should not be conflated.*

* **Measured**: only OpenAI-ES shows a statistically detectable gain with budget (all 10 seeds improve, $p=0.002$), but it still reaches only $5.2\%$ success. SimpleGA and DE improve on average by $116$ and $160$ points but not detectably with $S=10$ ($p \approx 0.13$–$0.16$). PBIL and CMA-ES show no detectable change. Between-seed SD stays $\approx 250$ for SimpleGA, DE, PBIL and CMA-ES.
* **Reading**: none of the algorithms comes close to the SAC/TD3 level ($-156$, Section 6.5.3) at $G=250$; the best evolutionary mean is CMA-ES at $-384.6$. For OpenAI-ES the budget matters, for the others the data neither show a plateau nor a clear gain.

#### 6.5.2 Theoretical Hypotheses (External Context, Not Tested Here)
> [!NOTE]
> SAC and TD3 were run in this repository (Section 6.5.3). The figure sometimes quoted for them on `Pendulum-v1` (return around $-150$) is not reported in the SAC or TD3 papers (neither evaluates Pendulum); the values measured here are the ones to cite.

#### 6.5.3 Measured SAC and TD3 Baselines (Same Out-of-Sample Protocol)
`run_sac_pendulum.py` and `run_td3_pendulum.py` implement SAC (Haarnoja et al., 2018) and TD3 (Fujimoto et al., 2018) in JAX with standard defaults, **untuned**: $2\times64$ ReLU networks, Adam $3\times10^{-4}$, $\gamma=0.99$, $\tau=0.005$, batch 256, 1,000 random warm-up steps, one update per environment step; SAC with twin critics and automatic entropy tuning (target $-1$); TD3 with exploration noise $0.1$, target-policy noise $0.2$ clipped at $0.5$ and delayed policy updates (2). Both are evaluated with their deterministic policy on the same 100 test episodes (test seed $99999$, $H=200$, success $R \ge -200$), 5 training seeds (`results/sac_pendulum.json`, `results/td3_pendulum.json`). The episode reset keys are identical to those used for the evolutionary methods, so every method sees the same 100 initial states.

| Env. steps | SAC mean test return (SD over seeds) | SAC success | TD3 mean test return (SD) | TD3 success |
|:---:|:---:|:---:|:---:|:---:|
| 10,000 | −322.0 (254.2) | 37.8% | −676.4 (532.5) | 28.6% |
| 20,000 | −159.7 (3.1) | 63.0% | −172.3 (32.0) | 57.6% |
| **30,000** | **−156.4 (2.2)** | **63.8%** | **−156.4 (3.0)** | **64.0%** |

* **Measured**: after 30,000 environment steps SAC and TD3 reach the same mean ($-156.4$), with $\approx 64\%$ success and SD over seeds of $2$–$3$. Compared with CMA-ES at $G=100$ ($-441.0$, 10 seeds, $\approx 4\times10^{6}$ simulated steps) and $G=250$ ($-384.6$, $\approx 10^{7}$ steps), they are better by $\approx 230$ to $285$ points, far more reproducible across seeds, and use roughly $130\times$ to $330\times$ fewer environment steps. This comparison is untuned for every method.
* **Why success saturates near 64% (measured, `run_pendulum_angle_analysis.py`, `results/pendulum_angle_analysis.json`)**: the 100 test episodes were binned by the initial angle $|\theta_0|$ (mean return and success over seeds):

| $\lvert\theta_0\rvert$ | Episodes | SAC mean / success | TD3 mean / success | CMA-ES $G=250$ mean / success |
|:---:|:---:|:---:|:---:|:---:|
| $[0, \pi/3)$ | 34 | −70.4 / 100% | −73.1 / 100% | −188.7 / 68% |
| $[\pi/3, 2\pi/3)$ | 27 | −118.8 / 100% | −120.9 / 100% | −342.9 / 37% |
| $[2\pi/3, 5\pi/6)$ | 18 | −220.4 / 17% | −223.5 / 17% | −511.0 / 0% |
| $[5\pi/6, \pi]$ | 21 | −289.3 / 0% | −279.4 / 0% | −647.1 / 0% |

  The correlation between $|\theta_0|$ and return is $-0.89$ (SAC), $-0.88$ (TD3) and $-0.80$ (CMA-ES). SAC and TD3 solve every episode with $|\theta_0| < 2\pi/3$ and essentially none with $|\theta_0| \ge 5\pi/6$: with the 200-step horizon and the torque limit $|u| \le 2$, starts near the bottom need a swing-up whose cost exceeds $-200$, so the $-200$ criterion caps success near $64\%$ for any of these policies. This also confirms for CMA-ES what Section 6.4 had stated as an expectation: it handles favorable and intermediate angles (68% and 37% success) but 0% from $|\theta_0| \ge 2\pi/3$, and it is worse than SAC/TD3 in every bin, including the easiest.
* **Not isolated**: why the evolutionary methods are worse in every bin (credit assignment, sample reuse, population size, Section 6.5.2) was not tested by any experiment here.

Section 6.5.3 measures a performance gap between SAC/TD3 and the evolutionary methods. Three hypotheses that could explain it, none tested in this repository:

1. **The Epistemic Credit Assignment Problem (Bellman Recursion vs. Episodic Sum)**:
   * Temporal Difference RL optimizes individual transitions via recursive dynamic programming:
     $$Q(s_t, a_t) = r(s_t, a_t) + \gamma \max_{a'} Q(s_{t+1}, a')$$
   * In contrast, ES operates as an episodic black-box optimizer evaluating only the cumulative scalar sum:
     $$f(\theta) = \sum_{t=0}^{H-1} r(s_t, a_t)$$
   * In `Pendulum-v1`, control is non-linearly bipartite:
     * **Phase 1 (Bottom energy pumping)**: Positive feedback (torque aligned with angular velocity $\dot{\theta}$ to amplify kinetic resonance and escape the gravitational potential well);
     * **Phase 2 (Top upright balance)**: Damped negative feedback (torque opposing angular velocity to arrest overshoot as $\theta \to 0$).
   * Under a single episode scalar $f(\theta)$, a parameter perturbation improving Phase 1 frequently degrades Phase 2. The optimizer lacks temporal credit information to distinguish which specific time-steps caused trajectory failure.
2. **Initial Condition Sensitivity at Rest**:
   * When $\theta_0 \approx \pm \pi$ and $\dot{\theta}_0 \approx 0$, the torque limit $|u| \le 2.0$ requires precise multi-cycle resonance. A phase error of mere centiseconds dissipates kinetic energy, forcing another oscillation cycle and which would incur extra penalty over the 200-step horizon (the magnitude was not measured here).
3. **Population Scaling and Sample Reuse**:
   * Salimans et al. (2017) demonstrated that ES in continuous control requires populations of $P = 1{,}440$ to $4{,}000$ to maintain gradient signal-to-noise ratio in high dimensions. Our budget ($P = 32$, $\lambda \approx 25$) maintains computational tractability on local CPU but limits exploration variance.
   * Unlike off-policy RL (e.g., SAC, TD3) which caches transitions in a Replay Buffer for thousands of gradient updates, ES discards all rollouts after each generation, achieving lower sample efficiency.

#### 6.5.4 Equal Environment-Step Budget: CMA-ES vs. SAC/TD3
To separate sample efficiency from the earlier budget differences, `run_equal_budget.py` (`results/pendulum_equal_budget.json`) trains CMA-ES with a given number of simulated environment steps (steps per generation $=\lambda \cdot N_{\text{rollouts}} \cdot 200$ with $\lambda=25$), 10 seeds per row, same 100-episode test protocol.

| $N_{\text{rollouts}}$ | Generations | Env. steps | Mean test return (SD over seeds) | Median | Best seed | Mean success |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 6 | 30,000 | −1131.8 (132.2) | −1113.8 | −951.8 | 2.2% |
| 1 | 20 | 100,000 | −1295.2 (170.2) | −1295.7 | −1026.3 | 9.0% |
| 1 | 60 | 300,000 | −1151.3 (199.0) | −1228.4 | −647.4 | 15.4% |
| 1 | 200 | 1,000,000 | −920.4 (288.2) | −947.4 | −516.4 | 21.8% |
| 4 | 5 | 100,000 | −1134.3 (149.5) | −1140.0 | −813.6 | 2.3% |
| 4 | 15 | 300,000 | −1010.6 (166.5) | −1071.2 | −727.0 | 6.7% |
| 4 | 50 | 1,000,000 | −719.2 (352.4) | −683.3 | −286.8 | 22.1% |
| *SAC* | | 30,000 | −156.4 (2.2) | | −153.8 | 63.8% |
| *TD3* | | 30,000 | −156.4 (3.0) | | −153.0 | 64.0% |

* **Measured**: with the same 30,000 environment steps, CMA-ES is far behind SAC and TD3 (−1131.8 against −156.4). Even with $33\times$ more steps (1,000,000), CMA-ES reaches a mean of $-720$ to $-920$ and its best seed ($-286.8$) is still worse than the worst SAC/TD3 seed at 30,000 steps. The earlier gap therefore is not an artifact of unequal budgets.
* **Not isolated**: why the evolutionary methods need so many more steps (credit assignment, sample reuse, population size, Section 6.5.2) was not tested. $N=1$ and $N=4$ are only two points of the noise/step trade-off, and CMA-ES hyperparameters were not tuned.

### 6.6 Multi-Seed Scientific Evaluation Protocol (Agarwal et al., NeurIPS 2021)

#### 6.6.1 Methodological Distinction: Aleatoric vs. Epistemic Uncertainty
In standard single-seed evaluations ($S=1$, typically $\text{seed}=42$), test metrics measure solely **aleatoric evaluation variance** ($\sigma_{\text{eval}}$): the stochasticity introduced by the environment's randomized initial state distribution $s_0 \sim p(s_0)$ given a fixed, frozen policy $\pi_\theta$.

However, as emphasized by Henderson et al. (2018) and formalized by Agarwal et al. (NeurIPS 2021), a rigorous scientific study must disentangle:
1. **Aleatoric Evaluation Variance ($\sigma_{\text{eval}}$)**: within-policy variance across out-of-sample test episodes.
2. **Epistemic Optimization Variance ($\sigma_{\text{train}}$)**: between-policy variance across independent training runs initialized from distinct pseudo-random seeds ($S=10$: `[42, 101, 202, 303, 404, 505, 606, 707, 808, 909]`).

To quantify the balance between these two sources of uncertainty, we compute the **Variance Ratio**:
$$\rho = \frac{\sigma_{\text{train}}}{\sigma_{\text{eval}}}$$
* **$\rho < 1.0$ (Evaluation-Dominated)**: The optimization trajectory is reproducible across seeds; residual performance dispersion is dominated by environmental stochasticity.
* **$\rho > 1.0$ (Training-Dominated / Unstable)**: The optimization trajectory is fragile; the choice of initial random seed creates larger disparities in final policy capability than testing stochasticity.

Furthermore, following the NeurIPS 2021 recommendations of Agarwal et al., we report:
* **Stratified 95% Bootstrap Confidence Intervals**: computed via $B=2{,}000$ hierarchical resamplings over both seeds and episodes.
* **Interquartile Mean (IQM)**: the 25% trimmed mean across all seed-episode samples, discarding outliers to ensure robust central tendency estimation.

#### 6.6.2 Consolidated Multi-Seed Results ($S=10$ Seeds $\times$ $100$ Test Episodes = $1{,}000$ Episodes per Algorithm)

Data extracted from `results/multiseed_study.json` / `results/multiseed_summary.csv` (generated by `run_multiseed_study.py`; table regenerated by `generate_readme_tables.py`). Family 2 uses the current elite rule (best of the last generation). The original best-so-far GP results with $S=5$ are in `results/multiseed_study_S5_best_so_far_gp.json`. $\rho$ is shown as "—" when $\sigma_{\text{eval}}=0$.

<!-- BEGIN:table_6_6_2 -->
| Family | Algorithm | Environment | Mean across Seeds $\pm$ SEM | Stratified 95% CI | IQM | $\sigma_{\text{train}}$ | $\sigma_{\text{eval}}$ | Variance Ratio $\rho$ | Success Rate (%) | Mean Train Best | Mean $\Delta$ |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **F1** | **SimpleGA** | CartPole-v1 | 362.71 $\pm$ 53.33 | [254.79, 455.71] | 443.94 | 168.64 | 81.98 | 2.06 | 64.9% | 481.85 | +119.14 |
| **F1** | **DE** | CartPole-v1 | 314.62 $\pm$ 46.29 | [225.68, 400.19] | 336.48 | 146.39 | 107.39 | 1.36 | 43.7% | 472.15 | +157.53 |
| **F1** | **OpenAI-ES** | CartPole-v1 | 379.51 $\pm$ 44.09 | [290.61, 457.31] | 465.10 | 139.44 | 86.64 | 1.61 | 61.6% | 491.95 | +112.44 |
| **F2** | **LinearGP** | CartPole-v1 | 205.06 $\pm$ 42.00 | [129.28, 285.35] | 189.16 | 132.82 | 54.16 | 2.45 | 11.1% | 228.63 | +23.57 |
| **F2** | **CartesianGP** | CartPole-v1 | 228.98 $\pm$ 45.41 | [147.00, 319.37] | 187.88 | 143.59 | 88.95 | 1.61 | 21.5% | 286.00 | +57.02 |
| **F3** | **CMA-ES** | CartPole-v1 | 479.56 $\pm$ 7.94 | [462.74, 492.56] | 500.00 | 25.10 | 48.39 | 0.52 | 88.6% | 500.00 | +20.44 |
| **F3** | **PBIL** | CartPole-v1 | 345.38 $\pm$ 37.87 | [267.33, 410.22] | 383.47 | 119.74 | 96.93 | 1.24 | 38.7% | 388.90 | +43.52 |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **F1** | **SimpleGA** | Acrobot-v1 | −89.67 $\pm$ 1.66 | [−93.39, −86.32] | −85.04 | 5.24 | 26.74 | 0.20 | 79.6% | −66.20 | +23.47 |
| **F1** | **DE** | Acrobot-v1 | −98.09 $\pm$ 6.28 | [−112.27, −88.85] | −85.11 | 19.87 | 46.13 | 0.43 | 79.1% | −70.05 | +28.04 |
| **F1** | **OpenAI-ES** | Acrobot-v1 | −93.90 $\pm$ 5.02 | [−104.77, −85.44] | −82.81 | 15.88 | 36.53 | 0.43 | 80.7% | −68.35 | +25.55 |
| **F2** | **LinearGP** | Acrobot-v1 | −93.14 $\pm$ 5.67 | [−105.03, −84.94] | −81.97 | 17.93 | 27.37 | 0.66 | 76.9% | −92.57 | +0.58 |
| **F2** | **CartesianGP** | Acrobot-v1 | −181.07 $\pm$ 41.27 | [−265.09, −113.91] | −133.18 | 130.51 | 39.85 | 3.27 | 39.8% | −161.60 | +19.47 |
| **F3** | **CMA-ES** | Acrobot-v1 | −95.48 $\pm$ 9.25 | [−116.29, −82.66] | −81.79 | 29.26 | 38.57 | 0.76 | 84.0% | −64.50 | +30.98 |
| **F3** | **PBIL** | Acrobot-v1 | −114.35 $\pm$ 17.09 | [−149.40, −94.61] | −92.28 | 54.04 | 49.99 | 1.08 | 65.3% | −77.60 | +36.75 |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **F1** | **SimpleGA** | Pendulum-v1 | −555.98 $\pm$ 87.19 | [−718.77, −393.31] | −418.86 | 275.72 | 398.62 | 0.69 | 30.7% | −216.57 | +339.41 |
| **F1** | **DE** | Pendulum-v1 | −709.68 $\pm$ 45.86 | [−801.40, −620.85] | −641.56 | 145.02 | 522.63 | 0.28 | 24.8% | −407.51 | +302.17 |
| **F1** | **OpenAI-ES** | Pendulum-v1 | −1015.65 $\pm$ 8.93 | [−1038.00, −995.99] | −1011.41 | 28.23 | 184.24 | 0.15 | 0.0% | −864.59 | +151.06 |
| **F2** | **LinearGP**† | Pendulum-v1 | −1189.09 $\pm$ 58.82 | [−1266.30, −1057.82] | −1188.98 | 185.99 | 365.06 | 0.51 | 0.0% | −964.52 | +224.56 |
| **F2** | **CartesianGP** | Pendulum-v1 | −1097.95 $\pm$ 68.63 | [−1212.12, −951.95] | −1091.43 | 217.03 | 305.40 | 0.71 | 0.0% | −997.89 | +100.06 |
| **F3** | **CMA-ES** | Pendulum-v1 | −440.96 $\pm$ 78.17 | [−601.08, −306.52] | −288.25 | 247.18 | 339.74 | 0.73 | 37.6% | −156.07 | +284.89 |
| **F3** | **PBIL** | Pendulum-v1 | −830.37 $\pm$ 57.56 | [−941.65, −716.97] | −783.14 | 182.03 | 313.32 | 0.58 | 6.6% | −665.72 | +164.65 |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **F1** | **SimpleGA** | MountainCarCont | 88.82 $\pm$ 2.54 | [83.16, 92.18] | 92.19 | 8.03 | 8.89 | 0.90 | 70.2% | 93.03 | +4.20 |
| **F1** | **DE** | MountainCarCont | 91.01 $\pm$ 0.93 | [89.24, 92.61] | 91.29 | 2.93 | 1.48 | 1.98 | 62.7% | 91.41 | +0.40 |
| **F1** | **OpenAI-ES** | MountainCarCont | 0.00 $\pm$ 0.00 | [0.00, 0.00] | 0.00 | 0.00 | 0.00 | — | 0.0% | 0.00 | 0.00 |
| **F2** | **LinearGP** | MountainCarCont | 0.00 $\pm$ 0.00 | [0.00, 0.00] | 0.00 | 0.00 | 0.00 | — | 0.0% | 0.00 | 0.00 |
| **F2** | **CartesianGP** | MountainCarCont | 0.00 $\pm$ 0.00 | [0.00, 0.00] | 0.00 | 0.00 | 0.00 | — | 0.0% | 0.00 | 0.00 |
| **F3** | **CMA-ES** | MountainCarCont | 80.28 $\pm$ 5.90 | [68.57, 90.54] | 92.86 | 18.66 | 22.05 | 0.85 | 63.2% | 92.21 | +11.93 |
| **F3** | **PBIL** | MountainCarCont | 90.36 $\pm$ 2.33 | [84.76, 93.91] | 93.53 | 7.37 | 9.69 | 0.76 | 84.2% | 93.54 | +3.19 |
<!-- END:table_6_6_2 -->

†*LinearGP on Pendulum-v1: in 9 of 10 seeds the selected program is the zero-torque policy, see item 6 below.*

> [!NOTE]
> **Statistical scope.** $S=10$ seeds still give imprecise estimates of $\sigma_{	ext{train}}$ and $ho$ (a standard deviation estimated from 10 values has a relative error of roughly 24%), so small differences between $ho$ values should not be over-read. The causal ablation (Section 6.7) covers only CartPole. The $S=5$ analyses of the 250-generation test were revised once $S=10$ was available (Section 6.5.1b).

#### 6.6.3 Observations from the Multi-Seed Variance Decomposition
> [!NOTE]
> This subsection reports what was measured. Causal explanations are deliberately limited to Section 6.7, where they were tested by controlled ablation, and to clearly labeled hypotheses.

1. **CartPole-v1 (Measured)**: the other six algorithms have $\sigma_{\text{train}}$ between $119.7$ and $168.6$ ($\rho$ between $1.24$ and $2.45$) under the standard budget ($G=30$, $N=2$; GP: $G=15$, $N=3$). CMA-ES has $\sigma_{\text{train}} = 25.10$, $\rho = 0.52$, IQM $= 500.00$ and $88.6\%$ aggregate success. CMA-ES is not "perfect" over seeds: its mean is $479.56$, and only the seed-42 run in Section 5.2 shows $500.00 \pm 0.00$.
2. **Differential Evolution on MountainCarContinuous-v0 (Measured)**: with seed 42, DE reached $85.68$ (below the $90.0$ success threshold). Across 10 seeds the aggregate mean is $91.01 \pm 0.93$ (stratified CI `[89.24, 92.61]`) with $62.7\%$ success, the highest mean in this environment. The seed-42 result is not representative of DE's average behavior, and the exact cause of that seed's lower result was not determined.
3. **PBIL and CMA-ES on MountainCarContinuous-v0 (Measured)**: PBIL has mean $90.36 \pm 2.33$, IQM $93.53$, $\sigma_{\text{train}} = 7.37$ and the highest success ($84.2\%$). CMA-ES has IQM $92.86$ but mean $80.28$, $\sigma_{\text{train}} = 18.66$ and $63.2\%$ success: five seeds end between 48 and 83.
4. **Pendulum-v1 (Measured)**: CMA-ES has the best mean ($-440.96$), IQM ($-288.25$) and success ($37.6\%$), followed by SimpleGA ($-555.98$, IQM $-418.86$, $30.7\%$). All values come with $\sigma_{\text{train}}$ between 28 and 276 and $\sigma_{\text{eval}}$ above 180. The effect of the generation budget is tested in Sections 6.5.1b and 6.5.1c.
5. **Zero returns on MountainCarContinuous-v0 (Measured)**: OpenAI-ES, LinearGP and CartesianGP obtained exactly $0.00$ in all 10 seeds ($1{,}000$ test episodes each). The failure is not specific to seed 42. The mechanism (flat reward landscape giving no selection signal) is a **hypothesis** that was not tested here.
6. **LinearGP on Pendulum-v1 is mostly a degenerate policy (Measured)**: in 9 of 10 seeds LinearGP obtained exactly $-1247.90$, the mean of a constant zero-torque policy on the same 100 test episodes (std $393.71$, $0\%$ success). The selected programs output a raw action of exactly $0$ at every step. The one non-degenerate seed reached $-659.75$, still with $0\%$ success. The aggregate ($-1189.09$) and $\sigma_{\text{train}}$ (186.0) therefore do not describe a learned policy. Section 6.8 analyzes why.
7. **Acrobot-v1 (Measured)**: SimpleGA has the best mean ($-89.67 \pm 1.66$, $\rho=0.20$); CMA-ES the highest success ($84.0\%$) with a lower mean ($-95.48$); CartesianGP is clearly the worst ($-181.07$, $\rho=3.27$).
8. **CartesianGP on Pendulum-v1 (Measured)**: mean $-1097.95$, $0.0\%$ success; the 10 per-seed means range from $-598$ to $-1248$.

### 6.7 Causal Ablation on CartPole-v1: Evaluation Noise ($N$) vs. Optimization Budget ($G$)

**Question.** Is the large between-seed variance on CartPole caused by (a) noisy fitness estimation (few rollouts $N$ per evaluation) or (b) insufficient optimization budget (few generations $G$)?

**Design** (`run_causal_ablation.py`, results in `results/causal_ablation_cartpole.json/csv`): 5 parametric algorithms (SimpleGA, DE, OpenAI-ES, CMA-ES, PBIL), $S=10$ training seeds `[42, 101, 202, 303, 404, 505, 606, 707, 808, 909]`, configurations $(G, N) \in \{(30,2), (30,8), (30,16), (100,2), (100,8)\}$, each policy evaluated on the same 100 test episodes (test seed 99999). The baseline $(30,2)$ reproduces the rows of Section 6.6.2. A determinism check (SimpleGA, seed 42, run twice) gave identical train and test values, so the variance across seeds is not run-to-run non-determinism. The policy at test time is deterministic (`argmax` of logits), so within-seed variance comes only from the initial states. The evaluated policy is the best sampled individual (Section 6.9). The earlier $S=5$ version of this table is kept in `results/causal_ablation_cartpole_S5.*`.

**Results** (mean test return across 10 seeds; in parentheses $\sigma_{\text{train}}$ / optimism gap $\Delta$):

| Algorithm | $G{=}30,N{=}2$ | $G{=}30,N{=}8$ | $G{=}30,N{=}16$ | $G{=}100,N{=}2$ | $G{=}100,N{=}8$ |
|:---|:---:|:---:|:---:|:---:|:---:|
| SimpleGA | 362.7 (168.6 / 119.1) | 477.7 (32.8 / 13.1) | 450.8 (51.4 / 28.1) | 362.7 (168.6 / 137.3) | 489.6 (16.8 / 10.4) |
| DE | 314.6 (146.4 / 157.5) | 405.5 (99.9 / 31.0) | 383.1 (74.0 / 1.9) | 378.0 (116.7 / 122.0) | 487.0 (15.9 / 12.5) |
| OpenAI-ES | 379.5 (139.4 / 112.4) | 335.8 (124.8 / 68.8) | 390.8 (132.4 / 8.6) | 412.4 (112.3 / 87.7) | 469.1 (80.0 / 30.9) |
| CMA-ES | 479.6 (25.1 / 20.4) | 483.0 (32.5 / 17.0) | 495.5 (7.9 / 3.1) | 479.6 (25.1 / 20.4) | 483.0 (32.5 / 17.0) |
| PBIL | 345.4 (119.7 / 43.5) | 382.1 (102.2 / 14.3) | 424.8 (83.9 / −2.5) | 394.0 (130.2 / 103.0) | 472.0 (31.3 / 7.5) |

**What the data show (measured):**
1. **Evaluation noise ($N$) matters most for SimpleGA and, partly, DE and PBIL.** At $G=30$, raising $N$ from 2 to 8 lowers $\sigma_{\text{train}}$ from 168.6 to 32.8 for SimpleGA (mean 362.7 to 477.7) and the DE gap from 157.5 to 31.0 (mean 314.6 to 405.5). OpenAI-ES does **not** benefit: its mean goes from 379.5 to 335.8.
2. **The generation budget ($G$) matters for DE, OpenAI-ES and PBIL, especially with $N=8$.** Going from $G=30$ to $G=100$ at $N=8$ raises the mean from 405.5 to 487.0 (DE), 335.8 to 469.1 (OpenAI-ES) and 382.1 to 472.0 (PBIL). At $N=2$ the gains are smaller (314.6 to 378.0 for DE, 379.5 to 412.4 for OpenAI-ES, 345.4 to 394.0 for PBIL).
3. **Both together fix most of the problem.** At $(G=100, N=8)$ SimpleGA, DE, OpenAI-ES, PBIL and CMA-ES reach 469.1 to 489.6 with success 79.2% to 94.8%.
4. **CMA-ES is already close to its ceiling at the standard budget** (479.6, $\sigma_{\text{train}}=25.1$); its $\sigma_{\text{train}}$ is lowest at $N=16$ (7.9) but is not monotonic in $N$ (32.5 at $N=8$).
5. **Methodological caveat on the $G$ axis.** For SimpleGA and CMA-ES at $N=2$, and for CMA-ES at $N=8$, the results at $G=100$ are identical to $G=30$ down to the second decimal. The implementation returns the best-so-far individual by training fitness; once a seed reaches the fitness ceiling of $500$ on its training rollouts, that individual can never be replaced, so extra generations cannot change the output. For these cases the $G$ axis is not informative; it does **not** show that more training is useless in general.
6. **The effect of $N$ is not monotonic:** at $G=30$, $N=16$ is worse than $N=8$ for SimpleGA (450.8 vs 477.7) and DE (383.1 vs 405.5) but better for OpenAI-ES, PBIL and CMA-ES. A plausible, **untested** hypothesis is that with larger $N$ the same $G=30$ budget leaves fewer effective selection steps for convergence on some algorithms.

**Answer to the question (CartPole only).** The between-seed variance is **not a single cause** and is **algorithm-dependent**: for SimpleGA it is mainly **noisy fitness evaluation** (small $N$); for OpenAI-ES, DE and PBIL, under-training also matters (the generation budget gives large gains at $N=8$); CMA-ES is already near its ceiling. The "seed" is not an independent cause: it changes the initial population and the sampled training start states, and the variance it induces is amplified when selection is made on a noisy fitness estimate or when the search has not converged.

**Limitations:** single environment (CartPole), $S=10$ seeds, only five $(G,N)$ points (not a full factorial), and the best-so-far ceiling effect of item 5. The ablation was not run for Pendulum-v1, MountainCarContinuous-v0, Acrobot-v1, or for the GP families.

---

### 6.8 Diagnosis of the Degenerate LinearGP Policy on Pendulum-v1 and the Elite Rule

#### 6.8.1 Diagnosis
`diagnose_linear_gp_pendulum.py` (results in `results/linear_gp_pendulum_diagnosis_*.json`) inspects the programs selected by LinearGP on Pendulum with the same test protocol (100 episodes, seed $99999$). Files without the `_reeval` suffix were produced with the **legacy best-so-far rule** (`--best-so-far`), files with it with the current rule.

| Configuration | Elite rule | Seeds | Selected programs with non-zero output | Test mean over seeds | Max success |
|:---|:---:|:---:|:---:|:---:|:---:|
| $G=40$, $N=8$, pop 16 | best-so-far | 5 | 0/5 | −1247.9 (all identical) | 0% |
| $G=200$, $N=8$, pop 16 | best-so-far | 5 | 0/5 | −1247.9 (all identical) | 0% |
| $G=200$, $N=8$, pop 16 | current-generation best | 5 | 2/5 | −1164.0, −1175.1 and 3× −1247.9 | 0% |
| $G=200$, $N=32$, pop 64 | current-generation best | 5 | 3/5 | −880.4, −502.4 (best) and 3× −1247.9 | 5% |
| $G=200$, $N=32$, pop 64 | current-generation best | 10 | 6/10 | **−849.2 (SD 379.9)**; best seed −217.2 (54% success), others 0–5% | 54% (1 seed) |
| $G=40$, $N=8$, pop 16 (main benchmark) | current-generation best | 10 | 1/10 | −1189.1 (SD 186.0) | 0% |

* **Measured**: the output register is written in every selected program, yet the raw output is exactly $0$ on 2,000 sampled observations. At initialization $50$–$62.5\%$ of random programs produce a non-zero output, so zero output is not a consequence of the initial population.
* **What changes the outcome**: the legacy loop keeps the program with the best single noisy evaluation ($N=8$ episodes) ever seen. The zero-torque program, whose noisy training fitness reaches $\approx -770$ to $-915$ on favorable draws (its true mean is $-1247.9$), is retained. Using the current-generation best (re-evaluated each generation), and increasing $N$ and population, lets non-zero programs survive in some seeds, but useful policies remain rare: with $S=10$ and the larger configuration 6/10 seeds are non-trivial, 1 seed reaches $54\%$ success ($-217.2$), and the mean is $-849.2$ (still far below SAC/TD3 and CMA-ES).
* **Interpretation (hypothesis, consistent with the data)**: selection on a noisy fitness with best-so-far retention (a winner's-curse effect) plus the fact that zero torque is a competitive local attractor in this landscape. The experiment shows the retention rule matters but does not prove it is the only cause.

#### 6.8.2 Effect of the elite rule in the main benchmark
Since this analysis the main benchmark (Sections 4, 5.2, 6.6.2) uses the current-generation best for both GP algorithms (`best_so_far=False`, the default). The table compares the legacy rule (5 seeds, `results/multiseed_study_S5_best_so_far_gp.json`) with the current rule on the same 5 seeds and on all 10 seeds (mean test return, and success over 10 seeds). Same budgets as before.

| Algorithm | Environment | Legacy rule, $S=5$ (success) | Current rule, same 5 seeds | Current rule, $S=10$ (success) |
|:---|:---|:---:|:---:|:---:|
| LinearGP | CartPole | 174.60 (0.4%) | 180.05 | 205.06 (11.1%) |
| LinearGP | Acrobot | −103.64 (64.0%) | −87.80 | −93.14 (76.9%) |
| LinearGP | Pendulum | −1247.90 (0.0%) | −1247.90 | −1189.09 (0.0%) |
| LinearGP | MountainCarCont | 0.00 (0.0%) | 0.00 | 0.00 (0.0%) |
| CartesianGP | CartPole | 306.65 (31.0%) | 306.65 | 228.98 (21.5%) |
| CartesianGP | Acrobot | −215.16 (31.6%) | −217.76 | −181.07 (39.8%) |
| CartesianGP | Pendulum | −1016.59 (0.0%) | −986.58 | −1097.95 (0.0%) |
| CartesianGP | MountainCarCont | 0.00 (0.0%) | 0.00 | 0.00 (0.0%) |

* **Reading**: on the same 5 seeds the two rules give similar results (largest differences: LinearGP on Acrobot $+15.8$ and CartesianGP on Pendulum $+30.0$ in mean return), because in most seeds the best program of the last generation is also the best-so-far one. The larger differences against the $S=10$ columns come mainly from the five extra seeds. The rule change alone does not remove the LinearGP degeneracy on Pendulum (9 of 10 seeds still output zero), and neither rule makes GP competitive on Pendulum.
* **Consequence**: LinearGP as configured in this benchmark is not a valid competitor on Pendulum. Best-so-far elitism on a noisy fitness is also a plausible factor in the optimism gaps of the other algorithms (Section 6.7 item 5), but that was only tested for CartPole.

---

### 6.9 Which Policy Is Evaluated: Best Sampled Individual vs. Distribution Mean

Every Family 1 and 3 runner in this repository returns the **best sampled individual** (the candidate with the highest single, noisy training fitness seen during the run), not the mean of the search distribution. For the three model-based algorithms the distribution has a mean, so `run_mean_vs_best_eval.py` (`results/mean_vs_best_eval_*.json`) tests both on the same 100 test episodes, standard budgets, 10 seeds (mean test return over seeds and mean success in parentheses).

| Environment | Algorithm | Best individual (success) | Distribution mean (success) | Mean difference (mean − best) | Seeds where mean is better | Wilcoxon $p$ |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| CartPole-v1 | OpenAI-ES | 379.5 (61.6%) | 387.9 (70.0%) | 8.4 | 5/10 | 0.844 |
| CartPole-v1 | CMA-ES | 479.6 (88.6%) | 406.8 (61.2%) | −72.7 | 4/10 | 0.301 |
| CartPole-v1 | PBIL | 345.4 (38.7%) | 15.8 (0.0%) | −329.6 | 0/10 | 0.002 |
| Acrobot-v1 | OpenAI-ES | −93.9 (80.7%) | −90.9 (82.0%) | 3.0 | 7/10 | 0.301 |
| Acrobot-v1 | CMA-ES | −95.5 (84.0%) | −84.5 (89.6%) | 11.0 | 6/10 | 0.432 |
| Acrobot-v1 | PBIL | −114.3 (65.3%) | −447.2 (1.5%) | −332.8 | 0/10 | 0.002 |
| Pendulum-v1 | OpenAI-ES | −1015.7 (0.0%) | −1180.6 (0.0%) | −165.0 | 0/10 | 0.002 |
| Pendulum-v1 | CMA-ES | −441.0 (37.6%) | −431.5 (37.4%) | 9.4 | 7/10 | 0.770 |
| Pendulum-v1 | PBIL | −830.4 (6.6%) | −1507.2 (0.0%) | −676.8 | 0/10 | 0.002 |
| MountainCarCont | OpenAI-ES | 0.0 (0.0%) | −0.5 (0.0%) | −0.5 | 0/10 | 0.002 |
| MountainCarCont | CMA-ES | 80.3 (63.2%) | −19.7 (1.6%) | −100.0 | 0/10 | 0.002 |
| MountainCarCont | PBIL | 90.4 (84.2%) | −10.7 (0.0%) | −101.1 | 0/10 | 0.002 |

* **Measured**: the best sampled individual is the better policy in most algorithm/environment pairs at these budgets. The largest differences are PBIL (the mean of its distribution is a very poor policy in every environment, e.g. $15.8$ with 0% success on CartPole and $-1507$ against $-830$ on Pendulum) and CMA-ES on MountainCarContinuous ($-19.7$ against $80.3$), with $p\approx0.002$. OpenAI-ES is roughly equal on CartPole and Acrobot and worse with the mean on Pendulum. CMA-ES's mean is comparable on Pendulum, better on Acrobot (lower variance across seeds) and worse on CartPole.
* **Consequence**: all results in this README for Families 1 and 3 refer to the best sampled individual. They are exposed to selection on noisy fitness (the winner's-curse effect of Sections 6.1 and 6.7), but the distribution mean is not a better alternative at these budgets (it has not converged), so the optimism gaps should not be read as an artifact that a different reporting choice would remove. Whether the means would overtake the best individual at much larger budgets was not tested.

---

## 7. Generated Visualizations

All figures are rendered in a high-resolution academic dark-mode style and saved under `results/`:

### 7.1 Multi-Seed Uncertainty and Out-of-Sample Validation
* **`results/multiseed_variance_analysis.png`**: 4-panel multi-seed evaluation chart across all 4 environments displaying the aggregate mean return across 10 independent training seeds ($S=10$, 1,000 test episodes total per algorithm), error bars for the **stratified 95% bootstrap confidence interval**, and variance ratio annotations ($\rho = \sigma_{\text{train}}/\sigma_{\text{eval}}$) according to the Agarwal et al. (NeurIPS 2021) protocol. Caveats: the Pendulum LinearGP bar is mostly the zero-torque policy (Section 6.6.3 item 6), and the plot writes "ρ n/a" where $\sigma_{\text{eval}}=0$ (Section 6.6.2 shows "—").
* **`results/validation_100_ci95_bars.png`**: 4-panel comparison chart across all 4 environments (CartPole-v1, Acrobot-v1, Pendulum-v1, MountainCarContinuous-v0) showing the mean test return ($\bar{R}_{\text{test}}$), error bars for the **95% bootstrap confidence interval**, and **Signal-to-Noise Ratio (SNR)** annotations for all 7 algorithms across all 3 families.
* **`results/optimism_gap_analysis.png`**: Complete quantitative analysis of the **Optimism Gap (*Winner's Curse*)** ($\Delta = f_{\text{train}} - \bar{R}_{\text{test}}$) across all 28 experimental configurations ($7 \text{ algorithms} \times 4 \text{ environments}$), highlighting severely overfit policies in red and robust policies in green.

### 7.2 Training Curves and Benchmark Dynamics
* **`results/all_families_curves.png`**: 4-panel $2 \times 2$ grid comparison across all 4 environments displaying the training convergence trajectories of all 7 algorithms across the 3 families.
* **`results/all_families_bar.png`**: 4-panel $2 \times 2$ grouped bar chart comparing the best training fitness of all 7 algorithms across all 4 environments.
* **`results/family_comparison.png`**: 4-panel $2 \times 2$ grouped bar chart comparing the three algorithmic families (Direct Solutions, Symbolic Programs, Estimation of Distribution) across all 4 environments with independent Y-axis scaling.
* **`results/learning_curves.png`**: Multi-panel training dynamics across all evaluated environments from the unified benchmark run.
* **`results/time_vs_performance.png`**: Scatter plot relating computational runtime (seconds) to final fitness across all algorithms and environments.

---

## 8. Hyperparameter Table

Values used for all reported results (the non-quick mode of `run_benchmark.py`, not used here, doubles the Family 1/3 populations to 64 and enlarges the GP programs). No hyperparameter was tuned for any algorithm.

| Algorithm | Family | Key Hyperparameters |
|:---|:---:|:---|
| **SimpleGA** | F1 | Population: 32, $\sigma_0 = 0.5$, $\sigma$ decay: $0.999$ (floor $0.01$), SBX $\eta = 2.0$, $P_{\text{cx}} = 0.8$, Tournament $k=2$, Elitism: 1 |
| **DE** | F1 | Population: 32, $F = 0.8$, $CR = 0.9$, Strategy: `rand/1/bin`, Initial scale: 0.5, strict greedy selection |
| **OpenAI-ES** | F1 | Population: 32 (16 antithetic pairs), $\sigma = 0.05$ (decay $0.999$, floor $0.001$), Learning rate: 0.01 (decay $0.9999$), plain SGD, Rank-based fitness shaping |
| **LinearGP** | F2 | Population: 16, Instructions: 48, Registers: $\text{obs} + 8 + \text{act}$, Mutation rate: 0.15, Crossover prob. 0.7 (single point), Tournament $k=2$, Elitism: 1 |
| **CartesianGP**| F2 | Population: 8 (1 parent + 7 offspring), Columns: 30, Per-gene mutation rate: 0.05 |
| **CMA-ES** | F3 | $\sigma_0 = 0.5$, $\lambda = 4 + \lfloor 3 \ln d \rfloor$, CSA/CMA hyperparameters derived from Hansen (2016) via $\sqrt{d}$ |
| **PBIL** | F3 | Population: 32, $\alpha_\mu = 0.1$, $\alpha_\sigma = 0.05$, Elite: top 20%, $\sigma_{\text{init}} = 1.0$, $\sigma_{\text{min}} = 0.01$, model mutation prob. $0.02$ with shift scale $0.05$ |

---

## 9. Repository Structure

```
evolutionary-strategies/
├── src/
│   ├── environments.py       # Gymnax 1.0.0 wrapper + vectorized JAX MLP forward pass
│   ├── family1_direct.py     # Family 1: SimpleGA, DE, OpenAI-ES
│   ├── family2_programs.py   # Family 2: LinearGP, CartesianGP (best_so_far option for the elite rule)
│   ├── family3_eda.py        # Family 3: CMA-ES, PBIL (Estimation of Distribution)
│   ├── benchmark.py          # Unified ask/tell training loops (also returns the final distribution mean)
│   ├── evaluation.py         # Out-of-sample validation module (100 eps, bootstrap CI, SNR)
│   └── visualization.py      # Dark-mode figure and table generation
├── run_validation_100.py     # Seed-42 training + 100-episode validation of all 28 configurations (Sections 4, 5.2)
├── run_multiseed_study.py    # S=10 multi-seed study (aleatoric vs epistemic variance, Agarwal et al., 2021; Section 6.6)
├── run_causal_ablation.py    # CartPole ablation of N (rollouts) vs G (generations), S=10 (Section 6.7)
├── run_pendulum_stress_test.py        # CMA-ES G=250 on Pendulum, seed 42 (Section 6.5.1)
├── run_pendulum_stress_multiseed.py   # Other seeds / budgets of the same test (Section 6.5.1b)
├── run_pendulum_g250_all.py           # G=100 vs G=250 for SimpleGA, DE, OpenAI-ES, PBIL (Section 6.5.1c)
├── run_sac_pendulum.py, run_td3_pendulum.py   # SAC and TD3 baselines (Section 6.5.3)
├── run_pendulum_angle_analysis.py     # Per-initial-angle breakdown (Section 6.5.3)
├── run_equal_budget.py                # CMA-ES at equal environment-step budgets (Section 6.5.4)
├── diagnose_linear_gp_pendulum.py     # LinearGP degeneracy diagnosis (Section 6.8)
├── run_mean_vs_best_eval.py           # Best individual vs distribution mean (Section 6.9)
├── generate_readme_tables.py          # Regenerates the data tables of this README from results/
├── run_benchmark.py, run_gp_only.py   # Standalone training scripts (not used for the reported tables, see Section 10)
├── generate_final_plots.py            # Training-curve / bar figures from raw_results.json and gp_results.json
├── results/                  # Raw data and figures (JSON/CSV files named after the section that uses them):
│   ├── validation_100_*.{json,csv,png}, optimism_gap_analysis.png, results_summary.csv, raw_results.json, gp_results.json
│   ├── multiseed_study.json, multiseed_summary.csv, multiseed_variance_analysis.png
│   ├── multiseed_study_S5_best_so_far_gp.json/.csv      # legacy GP elite rule, S=5
│   ├── causal_ablation_cartpole.json/.csv, causal_ablation_cartpole_S5.json/.csv
│   ├── pendulum_stress_test_250.json, pendulum_stress_multiseed.json, pendulum_cmaes_G{100,250}_extra_seeds.json
│   ├── pendulum_budget_other_algos.json, pendulum_equal_budget.json, pendulum_angle_analysis.json
│   ├── sac_pendulum.json, td3_pendulum.json
│   ├── linear_gp_pendulum_diagnosis_*.json, mean_vs_best_eval_*.json
│   └── all_families_curves.png, all_families_bar.png, family_comparison.png, learning_curves.png, time_vs_performance.png
└── README.md
```

---

## 10. Reproducing the Experiments

Tested with Python 3.11, `jax`/`jaxlib` 0.6.2 on CPU, `gymnax` 1.0.0, `optax` 0.2.8, `numpy` 2.4, `scipy` 1.17, `matplotlib` 3.11. In our re-runs the same seeds reproduced the same results on the same machine and library versions.

```bash
# 1. Set up the virtual environment
py -3.11 -m venv C:\ev
C:\ev\Scripts\activate
pip install "jax[cpu]==0.6.2" gymnax==1.0.0 optax scipy matplotlib numpy tqdm "orbax-checkpoint==0.5.23" "flax==0.8.5"

# 2. Seed-42 training and 100-episode validation (Sections 4 and 5.2; rewrites results_summary.csv, raw_results.json, gp_results.json)
python run_validation_100.py
python generate_final_plots.py

# 3. Multi-seed study, S=10 (Section 6.6)
python run_multiseed_study.py

# 4. Pendulum budget tests (Sections 6.5.1 to 6.5.4)
python run_pendulum_stress_test.py --generations 250 --rollouts 8
python run_pendulum_stress_multiseed.py
python run_pendulum_stress_multiseed.py --seeds 505 606 707 808 909 --generations 250 --out results/pendulum_cmaes_G250_extra_seeds.json
python run_pendulum_stress_multiseed.py --seeds 505 606 707 808 909 --generations 100 --out results/pendulum_cmaes_G100_extra_seeds.json
python run_pendulum_g250_all.py
python run_sac_pendulum.py
python run_td3_pendulum.py
python run_pendulum_angle_analysis.py
python run_equal_budget.py

# 5. Causal ablation (Section 6.7), GP diagnosis (Section 6.8), best vs mean policy (Section 6.9)
python run_causal_ablation.py
python diagnose_linear_gp_pendulum.py --generations 40 --best-so-far
python diagnose_linear_gp_pendulum.py --generations 200 --rollouts 32 --pop 64 --seeds 42 101 202 303 404 505 606 707 808 909
python run_mean_vs_best_eval.py

# 6. Regenerate the data tables of this README from results/
python generate_readme_tables.py
```

`run_benchmark.py` (default mode: $G=150$, $N_{\text{rollouts}}=4$) and `run_gp_only.py` are earlier standalone training scripts; they write the same files under `results/` with different budgets, so running them overwrites the data behind the tables above and they are not part of the reproduction recipe.

---

## 11. References

1. **Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A. C., & Bellemare, M.** (2021). Deep reinforcement learning at the edge of the statistical precipice. *Advances in Neural Information Processing Systems (NeurIPS)*, 34, 29304–29320.
2. **Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., & Meger, D.** (2018). Deep reinforcement learning that matters. *AAAI Conference on Artificial Intelligence*, 32(1).
3. **Machado, M. C., et al.** (2018). Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents. *Journal of Artificial Intelligence Research*, 61, 523–562.
4. **Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S.** (2018). Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. *International Conference on Machine Learning (ICML)*, PMLR 80, 1861–1870.
5. **Fujimoto, S., van Hoof, H., & Meger, D.** (2018). Addressing Function Approximation Error in Actor-Critic Methods. *International Conference on Machine Learning (ICML)*, PMLR 80, 1587–1596.
6. **Goldberg, D. E.** (1989). *Genetic Algorithms in Search, Optimization and Machine Learning*. Addison-Wesley.
7. **Deb, K., & Agrawal, R. B.** (1995). Simulated binary crossover for continuous search space. *Complex Systems*, 9(2), 115–148.
8. **Storn, R., & Price, K.** (1997). Differential Evolution – A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces. *Journal of Global Optimization*, 11(4), 341–359.
9. **Salimans, T., Ho, J., Chen, X., Sidor, S., & Sutskever, I.** (2017). Evolution Strategies as a Scalable Alternative to Reinforcement Learning. *arXiv:1703.03864*.
10. **Brameier, M. F., & Banzhaf, W.** (2007). *Linear Genetic Programming*. Springer Science & Business Media.
11. **Miller, J. F., & Thomson, P.** (2000). Cartesian Genetic Programming. *European Conference on Genetic Programming (EuroGP)*, 121–132.
12. **Hansen, N., & Ostermeier, A.** (2001). Completely derandomized self-adaptation in evolution strategies. *Evolutionary Computation*, 9(2), 159–195.
13. **Hansen, N.** (2016). The CMA Evolution Strategy: A Tutorial. *arXiv:1604.00772*.
14. **Baluja, S.** (1994). Population-Based Incremental Learning. *Technical Report CMU-CS-94-163*, Carnegie Mellon University.
15. **Sebag, M., & Ducoulombier, A.** (1998). Extending Population-Based Incremental Learning to Continuous Search Spaces. *Parallel Problem Solving from Nature (PPSN V)*, LNCS 1498, 418–427. Springer.
16. **Larrañaga, P., & Lozano, J. A.** (2002). *Estimation of Distribution Algorithms: A New Tool for Evolutionary Computation*. Springer.
17. **Lange, R. T.** (2022). Gymnax: Standard Model Environments in JAX. *GitHub Repository*.

