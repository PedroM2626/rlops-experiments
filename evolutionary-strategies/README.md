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

### 1.1 Why Is CMA-ES Unambiguously Family 3 (Models / EDA)?
**CMA-ES** (*Covariance Matrix Adaptation Evolution Strategy*, Hansen & Ostermeier 2001; Hansen 2016) carries the canonical signature of a continuous second-order **Estimation of Distribution Algorithm (EDA)**:
1. **No Individual-to-Individual Genomic Heredity**: Parent individuals are never preserved, point-mutated, or directly recombined. At each generation $g$, an entire population of $\lambda$ individuals is produced by independent sampling from a multivariate Gaussian distribution:
   $$x_k \sim \mathcal{N}\left(\mu^{(g)}, (\sigma^{(g)})^2 \Sigma^{(g)}\right), \quad k = 1, \dots, \lambda$$
2. **100% Disposable Population**: Once the fitness returns are computed, **every individual is discarded**. No genome physically carries over into the next generation.
3. **Parametric Evolution of the Distribution**: What evolves iteratively are the hyperparameters of the probabilistic model:
   - **Mean vector $\mu$**: Shift driven by the weighted intermediate recombination of the $\mu_{\text{eff}}$ best samples.
   - **Global step-size $\sigma$ (CSA)**: Cumulative step-length adaptation based on the conjugate evolution path $p_\sigma$.
   - **Covariance matrix $\Sigma$ (CMA)**: Second-order adaptation via a *rank-one update* (accumulating the anisotropic path $p_c$) and a *rank-$\mu$ update* (empirical estimator of the elite dispersion).
4. Throughout the modern stochastic-optimization literature (Larrañaga & Lozano 2002), CMA-ES is recognized as the flagship of continuous EDAs, learning the Riemannian curvature metric / inverse Hessian of the fitness function.

### 1.2 Why Does OpenAI-ES Belong to Family 1 (Direct Solutions)?
The algorithm proposed by Salimans et al. (2017) (**OpenAI-ES**), despite the historical term "Evolution Strategy", **neither learns nor maintains any probabilistic distribution model**:
1. There is no adaptive covariance matrix and no parametric distribution being learned. The perturbation used is isotropic Gaussian noise with fixed variance $\epsilon \sim \mathcal{N}(0, I)$.
2. The algorithm maintains a **single parameter vector $\theta \in \mathbb{R}^d$** holding the neural network weights.
3. The random perturbations act exclusively as a **stochastic finite-difference / score-function (NES) gradient estimator**:
   $$\nabla_\theta \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)} [f(\theta + \sigma \epsilon)] = \frac{1}{\sigma} \mathbb{E}_{\epsilon} [f(\theta + \sigma \epsilon) \epsilon] \approx \frac{1}{2 n \sigma} \sum_{i=1}^n \left( f(\theta + \sigma \epsilon_i) - f(\theta - \sigma \epsilon_i) \right) \epsilon_i$$
4. The parameter vector is updated directly by gradient ascent with a deterministic optimizer (SGD/Adam): $\theta \leftarrow \theta + \alpha \widehat{\nabla} f$.
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
  * Gaussian mutation with exponential variance decay and strict elitism of 1 individual.
* **Differential Evolution (DE)** (Storn & Price 1997):
  * Classic `rand/1/bin` strategy:
    $$v_i = x_{r1} + F \cdot (x_{r2} - x_{r3}), \quad r_1 \ne r_2 \ne r_3 \ne i$$
  * Binomial crossover with probability $CR$ and a guarantee that at least 1 gene is mutated.
  * Greedy *one-to-one* selection: the trial individual replaces the parent if and only if $f(u_i) \ge f(x_i)$.
* **OpenAI-ES** (Salimans et al. 2017):
  * Mirrored antithetic perturbations ($+\epsilon_i, -\epsilon_i$) for first-order variance cancellation.
  * *Fitness shaping*: zero-centered rank-based normalization, yielding invariance to monotonic reward transformations.
  * Direct weight-vector update via SGD with geometric learning-rate decay.

### 2.2 Family 2: Symbolic Programs (Genetic Programming)
* **LinearGP (LGP)** (Brameier & Banzhaf 2007):
  * Individual represented as a linear sequence of register instructions: `[op, dst, src1, src2]`.
  * Register set: $R = R_{\text{obs}} \cup R_{\text{extra}} \cup R_{\text{act}}$.
  * Primitive function set (10 operators, as implemented in `apply_op_jax`): $\{+, -, \times, \tanh, \sin, \cos, |\cdot|, \text{noop}, \max, \min\}$. There is no protected division.
  * Phenotype executed as a sequential imperative program, which allows reuse of intermediate variables and the presence of intrinsically neutral code (*introns*).
* **CartesianGP (CGP)** (Miller & Thomson 2000):
  * Individual encoded as a positional 2D directed acyclic graph (DAG) ($1 \times N_{\text{cols}}$).
  * Intermediate nodes receive connections only from inputs or from nodes in earlier columns.
  * Pointwise structural mutation on connections and functions.
  * Evolutionary $(1+\lambda)$-ES with purely neutral selection (replacement occurs whenever the offspring fitness is greater than or equal to the parent fitness).

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
    4. **Nested Parallelism (`make_program_pop_eval_fn`)**: The entire population of programs is parallelized across multiple rollout episodes and time horizons via nested `jax.vmap(rollout_fn)(programs, rngs)`, evaluating the whole population in one compiled call. The speedup over the earlier Python implementation is reported only from the two recorded measurements in Section 4.1 (~$68\times$ and ~$73\times$); the legacy Python code is no longer in the repository, so those timings cannot be reproduced today.
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
* **Families 1 and 3 (SimpleGA, DE, OpenAI-ES, CMA-ES, PBIL)**: Each individual/sample is evaluated over **$N_{\text{rollouts}} = 2$ independent episodes** per generation (in the reported `--quick` mode). In the full mode (`--full`), **$N_{\text{rollouts}} = 4$ episodes** are used. The fitness assigned is the mean of the returns obtained:
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

| Family | Algorithm | Paradigm | CartPole-v1 (Best Train) | Acrobot-v1 (Best Train) | Pure JAX Time (s) | Legacy Python Time (s)* | JAX Speedup* |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **F1: Direct Solutions** | **DE** | rand/1/bin trial vectors | **500.00** | −78.00 | 4.0s | — | — |
| **F1: Direct Solutions** | **SimpleGA** | Tournament + SBX crossover | 318.50 | **−64.00** | 4.4s | — | — |
| **F1: Direct Solutions** | **OpenAI-ES** | NES gradient ascent | 419.50 | **−64.00** | 4.8s | — | — |
| **F2: Programs** | **CartesianGP** | Positional 2D DAG (1+7)ES | **500.00** | −76.67 | **0.9s** | 61.9s | **~68×** |
| **F2: Programs** | **LinearGP** | Register sequence | 162.67 | −75.00 | **1.3s** | 95.4s | **~73×** |
| **F3: Models (EDA)** | **CMA-ES** | Covariance $\mathcal{N}(\mu, \sigma^2 \Sigma)$ | **500.00** | **−64.00** | 18.8s | — | — |
| **F3: Models (EDA)** | **PBIL** | Univariate marginal model | 464.50 | −75.00 | 5.0s | — | — |

*\*Legacy Python Time*: Reference benchmark measurements recorded during the initial migration from unvectorized Python loops to branchless XLA dispatch (`apply_op_jax`, `lax.scan`, `lax.fori_loop`).

*Vectorization Impact*: By replacing interpreted Python rollout loops with `apply_op_jax` (`jax.lax.switch`), `forward_lgp_jax` (`jax.lax.scan`), and `forward_cgp_jax` (`jax.lax.fori_loop`), Family 2 ran **~68× (CartesianGP) and ~73× (LinearGP) faster** on CartPole than the earlier Python implementation, according to the recorded timings above (not reproducible today). Per-generation latency was not measured separately, so no sub-30 ms claim is made.

### 4.2 Continuous Environments (All 3 Families in Pure JAX)

| Family | Algorithm | Pendulum-v1 (Best) | Pendulum-v1 (Mean Final) | MountainCarCont (Best) | MountainCarCont (Mean Final) | Total Time (s) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **F1: Direct Solutions** | **SimpleGA** | **−157.84** | −944.37 | 94.17 | −56.30 | 6.6s |
| **F1: Direct Solutions** | **DE** | −572.58 | −1378.27 | 85.50 | −90.91 | 4.8s |
| **F1: Direct Solutions** | **OpenAI-ES** | −885.38 | −1228.99 | **−0.00** *(Gradient failure)* | −1.12 | 5.0s |
| **F2: Programs** | **LinearGP**† | −895.69 | −1243.22 | 0.00 | −14.03 | **2.0s** |
| **F2: Programs** | **CartesianGP** | −642.25 | −1011.80 | −0.00 | −0.00 | **1.7s** |
| **F3: Models (EDA)** | **CMA-ES** | **−229.74** | −924.98 | **95.83** *(Optimum)* | −59.37 | 46.0s |
| **F3: Models (EDA)** | **PBIL** | −614.96 | −1520.16 | 94.22 | −74.45 | 4.6s |

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

| Family | Algorithm | Environment | Train Best ($N \le 3$) | Test Mean $\pm$ SEM | 95% Bootstrap CI | Median | IQR | Std Dev | SNR | Optimism Gap ($\Delta$) | Success Rate (%) |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **F1** | **SimpleGA** | CartPole-v1 | 318.50 | 318.82 $\pm$ 19.43 | [281.37, 356.55] | 500.00 | 395.50 | 194.29 | 1.64 | **−0.32** | 51.0% |
| **F1** | **DE** | CartPole-v1 | **500.00** | 123.84 $\pm$ 8.48 | [108.68, 141.80] | 101.00 | 40.00 | 84.79 | 1.46 | **+376.16** | **2.0%** |
| **F1** | **OpenAI-ES** | CartPole-v1 | 419.50 | 138.27 $\pm$ 10.66 | [118.93, 161.17] | 96.00 | 66.75 | 106.60 | 1.30 | **+281.23** | 4.0% |
| **F2** | **LinearGP** | CartPole-v1 | 162.67 | 184.65 $\pm$ 5.14 | [174.63, 194.90] | 180.50 | 77.25 | 51.36 | 3.60 | **−21.98** | 0.0% |
| **F2** | **CartesianGP**| CartPole-v1 | **500.00** | **482.68 $\pm$ 4.95** | [472.51, 492.04] | 500.00 | 0.00 | 49.48 | 9.76 | +17.32 | **88.0%** |
| **F3** | **CMA-ES** | CartPole-v1 | **500.00** | **500.00 $\pm$ 0.00** | **[500.00, 500.00]** | **500.00** | **0.00** | **0.00** | **$\infty$** | **0.00** | **100.0%** |
| **F3** | **PBIL** | CartPole-v1 | 464.50 | 412.90 $\pm$ 7.64 | [397.39, 427.45] | 406.50 | 139.00 | 76.36 | 5.41 | +51.60 | 28.0% |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **F1** | **SimpleGA** | Acrobot-v1 | −64.00 | −85.55 $\pm$ 4.55 | [−95.12, −78.73] | −76.50 | 19.00 | 45.46 | 1.88 | +21.55 | 88.0% |
| **F1** | **DE** | Acrobot-v1 | −78.00 | −93.68 $\pm$ 3.61 | [−101.43, −87.34] | −85.50 | 20.25 | 36.09 | 2.60 | +15.68 | 82.0% |
| **F1** | **OpenAI-ES** | Acrobot-v1 | −64.00 | −85.85 $\pm$ 2.22 | [−90.36, −81.58] | −81.50 | 19.25 | 22.19 | 3.87 | +21.85 | 88.0% |
| **F2** | **LinearGP** | Acrobot-v1 | −75.00 | **−81.66 $\pm$ 1.07** | [−83.77, −79.73] | −77.00 | 10.25 | 10.70 | **7.63** | +6.66 | **92.0%** |
| **F2** | **CartesianGP**| Acrobot-v1 | −76.67 | −93.56 $\pm$ 1.60 | [−96.70, −90.54] | −91.00 | 20.50 | 16.01 | 5.84 | +16.89 | 73.0% |
| **F3** | **CMA-ES** | Acrobot-v1 | −64.00 | **−77.79 $\pm$ 1.22** | **[−80.30, −75.46]** | **−73.00** | **12.50** | **12.22** | **6.37** | **+13.79** | **96.0%** |
| **F3** | **PBIL** | Acrobot-v1 | −75.00 | −101.90 $\pm$ 4.50 | [−112.47, −94.93] | −95.50 | 20.25 | 44.96 | 2.27 | +26.90 | 55.0% |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **F1** | **SimpleGA** | Pendulum-v1 | −157.84 | **−352.97 $\pm$ 23.94** | [−399.54, −307.80] | −272.60 | 436.69 | 239.42 | 1.47 | +195.12 | **35.0%** |
| **F1** | **DE** | Pendulum-v1 | −572.58 | −714.67 $\pm$ 56.98 | [−827.75, −603.07] | −599.67 | 1041.44 | 569.79 | 1.25 | +142.10 | 25.0% |
| **F1** | **OpenAI-ES** | Pendulum-v1 | −885.38 | −1075.51 $\pm$ 24.89 | [−1125.43, −1027.93] | −998.93 | 384.17 | 248.94 | **4.32** | +190.14 | 0.0% |
| **F2** | **LinearGP**† | Pendulum-v1 | −895.69 | −1247.90 $\pm$ 39.37 | [−1323.54, −1170.53] | −1254.86 | 593.47 | 393.71 | 3.17 | +352.22 | 0.0% |
| **F2** | **CartesianGP**| Pendulum-v1 | −642.25 | −940.34 $\pm$ 38.81 | [−1015.10, −865.38] | −853.53 | 360.33 | 388.08 | 2.42 | +298.10 | 0.0% |
| **F3** | **CMA-ES** | Pendulum-v1 | −229.74 | **−268.67 $\pm$ 18.59** | **[−303.78, −231.68]** | **−261.94** | 252.75 | 185.90 | 1.45 | **+38.92** | **35.0%** |
| **F3** | **PBIL** | Pendulum-v1 | −614.96 | −641.46 $\pm$ 23.08 | [−688.33, −598.69] | −542.80 | 227.23 | 230.80 | 2.78 | **+26.49** | 0.0% |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **F1** | **SimpleGA** | MountainCarCont | 94.17 | 91.84 $\pm$ 0.24 | [91.35, 92.31] | 90.71 | 3.65 | 2.40 | 38.25 | +2.33 | 84.0% |
| **F1** | **DE** | MountainCarCont | 85.50 | 85.68 $\pm$ 0.36 | [84.96, 86.36] | 87.74 | 5.19 | 3.61 | 23.75 | −0.18 | 0.0%* |
| **F1** | **OpenAI-ES** | MountainCarCont | −0.00 | −0.00 $\pm$ 0.00 | [−0.00, −0.00] | −0.00 | 0.00 | 0.00 | 1.02 | 0.00 | 0.0% |
| **F2** | **LinearGP** | MountainCarCont | 0.00 | 0.00 $\pm$ 0.00 | [0.00, 0.00] | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0% |
| **F2** | **CartesianGP**| MountainCarCont | −0.00 | −0.00 $\pm$ 0.00 | [−0.00, −0.00] | −0.00 | 0.00 | 0.00 | 1.02 | 0.00 | 0.0% |
| **F3** | **CMA-ES** | MountainCarCont | **95.83** | **96.43 $\pm$ 0.11** | **[96.22, 96.63]** | **96.72** | **1.76** | **1.06** | **90.99** | **−0.60** | **100.0%** |
| **F3** | **PBIL** | MountainCarCont | 94.22 | **93.87 $\pm$ 0.12** | **[93.64, 94.10]** | **94.06** | **1.80** | **1.15** | **81.28** | **+0.35** | **100.0%** |

*\*Note on DE in MountainCar*: with seed 42, DE converged to $\approx 85.7$, below the formal success criterion $R \ge 90.0$ (multi-seed behavior differs, see Section 6.6.3 item 2).

†*LinearGP on Pendulum-v1 is the zero-torque baseline (constant zero action; mean $-1247.903$, std $393.71$, identical to the reported row). It says nothing about GP's ability to learn control. See Section 6.6.3 item 6.*

---

## 6. Analysis and Scientific Discussion of the Results

### 6.1 The Empirical Manifestation of the Winner's Curse (Spurious Optimism in Fast Evaluations)
Validation over 100 episodes exposed a crucial phenomenon in evolutionary RL:
1. **The Collapse of Differential Evolution (DE) on CartPole**:
   * During training ($N_{\text{rollouts}}=2$), DE reported an apparent optimal fitness of **500.00**.
   * In testing ($N=100$), the mean collapsed to **123.84** with a **success rate of only 2.0%** ($\Delta = +376.16$).
   * *Interpretation (hypothesis, not tested)*: the large gap is consistent with selection on a noisy fitness estimate (best-of-run chosen on 2 episodes). The causal ablation of Section 6.7 supports noisy evaluation as a main driver for DE on CartPole ($N=2 \to 8$ removes the gap), but the specific individual was not inspected.
2. **CMA-ES on CartPole (seed 42)**:
   * **CMA-ES** reached **500.00 $\pm$ 0.00** over the 100 out-of-sample episodes, with a **standard deviation of strictly 0.00** and **100% success**.
   * *Caveat (see Section 6.6)*: this is the seed-42 run only. Across 5 training seeds, CMA-ES obtained $483.98 \pm 6.81$ with $87.4\%$ success.
   * A possible explanation (hypothesis, untested; no flatness or curvature measurement was made) is that covariance adaptation steers the search toward a broad basin that tolerates variation in the initial state. What is measured is the outcome: low $\sigma_{\text{train}}$ across seeds (Section 6.6.3 item 1).

### 6.2 Generalization of Symbolic Programs (Family 2)
* On **CartPole-v1**, **CartesianGP** achieved an **88.0% success rate** with a test mean of **482.68 $\pm$ 4.95** and median of **500.00**, confirming that compact 2D DAGs create stable feedback loops around unstable equilibria.
* On **Acrobot-v1**, **LinearGP** achieved a **92.0% success rate** with a test mean of **−81.66 $\pm$ 1.07** and an **SNR of 7.63** (the highest SNR among all seven algorithms in this seed-42 run), outperforming PBIL (55%) and DE (82%) in the seed-42 run. Across 5 seeds (Section 6.6.2) the picture changes: LinearGP obtains $-103.64$ with $64.0\%$ success versus DE $-101.54$ ($80.2\%$) and SimpleGA $-86.58$ ($82.8\%$), so the seed-42 ranking is not robust.
* In **continuous control (Pendulum-v1)**, pure vectorized GP with continuous $u = \text{clip}(\tanh(\text{raw}) \cdot a_{\text{max}})$ reached competitive test returns under extended training (CartesianGP: −940.34 $\pm$ 38.81 with median −853.53; LinearGP: −1247.90, which we verified is the zero-torque baseline, see Section 6.6.3 item 6). CartesianGP's multi-seed mean is −1016.59 (0% success), so no claim of successful swing-up control by symbolic programs is supported by the data.
* **Conclusion supported by the data**: symbolic programs generalize well in some single-seed runs (CartesianGP on CartPole, LinearGP on Acrobot) but not robustly across seeds: CartesianGP gets $306.65$ on CartPole and $-215.16$ on Acrobot (the worst of all algorithms, $\rho = 4.90$) and LinearGP gets $174.60$ on CartPole with $0.4\%$ success. The idea that compact programs avoid the overfitting of neural weights is a hypothesis; the data do not support a general claim of robust generalization.

### 6.3 Failure of OpenAI-ES and GP vs. Other Methods on MountainCarContinuous
* **OpenAI-ES, LinearGP, and CartesianGP (Score: 0.00, Success: 0.0%)**:
  All three methods collapsed entirely on MountainCarContinuous. The measured fact is a return of exactly $0.00$ in all 5 seeds (Section 6.6.3 item 5). The mechanism is a **hypothesis, not tested**: a policy that outputs near-zero torque incurs ~zero action cost, so if the population never reaches the goal the fitness landscape is flat and gives no selection or gradient signal ($\nabla_\theta \mathbb{E}[f] \approx \mathbf{0}$ for OpenAI-ES; a very low chance for GP mutations to find the energy-pumping oscillation).
* **CMA-ES and PBIL (Success: 100.0%, SNR: 91.0 and 81.3)**:
  Both EDAs solved the environment with deterministic asymptotic stability ($\text{SEM} \approx 0.11$, $\text{Std} \le 1.15$), in the seed-42 run. Across 5 seeds (Section 6.6.2) the EDAs are less uniform: PBIL keeps $96.8\%$ success, but CMA-ES drops to $82.05 \pm 9.20$ with $76.0\%$ success, and DE ($91.44$) and SimpleGA ($86.36$) also solve the task. The data therefore support that these methods escape the valley while OpenAI-ES and GP did not; they do not support a "decisive superiority" of covariance adaptation, and the reason was not isolated.

### 6.4 The Stochastic Sensitivity of Pendulum-v1 and Resolution via Extended Rollouts
* **The Winner's Curse Diagnosis**: In the quick benchmark ($N_{\text{rollouts}}=2$), every method exhibited a large optimism gap ($\Delta > 500$), caused by the uniform angular perturbation $\theta_0 \sim \mathcal{U}[-\pi, \pi]$. With only 2 rollouts, individuals that merely spawned near upright equilibrium were spuriously selected as elites without possessing an actual swing-up policy.
* **Empirical Resolution ($G=100$, $N_{\text{rollouts}}=8$)**:
  * **CMA-ES achieves partial stabilization**: Mean test return of **$-268.67 \pm 18.59$** (median **$-261.94$**, **35.0% success rate**), with the optimism gap collapsing from $+500$ down to **$+38.92$**. Rather than "near-optimal" control across all initial states, CMA-ES achieves *partial stabilization*: $35\%$ of the 100 test episodes reach $R \ge -200$ and the distribution is wide (IQR $252.75$). The breakdown by initial angle ($|\theta_0| \le 2\pi/3$ vs. near-bottom starts) was **not measured**; the expectation that failures concentrate at near-bottom initial states is a hypothesis consistent with Section 6.5.2 item 2, not a result.
  * **SimpleGA** achieved a test mean of **$-352.97 \pm 23.94$** (median **$-272.60$**, **35.0% success rate**).
  * **CartesianGP** improved its test mean from $-1336.94$ to **$-940.34 \pm 38.81$** (median **$-853.53$**).
  * Larger $N$ and $G$ coincided with a much smaller optimism gap and a $35\%$ success rate for the best methods in the seed-42 run. In the multi-seed study (Section 6.6.2) SimpleGA and CMA-ES reach $37.2\%$ and $38.6\%$ success with $\rho<1$, so partial swing-up is reproducible, but no method solves the task consistently, and the role of the rollout budget was not isolated by ablation on Pendulum.

### 6.5 Plateau Test in Continuous Control (CMA-ES, 250 Generations) and Black-Box ES vs. TD RL

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
* **Interpretation of the seed-42 run alone (hypothesis)**: the test-return gain ($+26.24$) is of the order of the individual SEMs ($15.37$ and $18.59$), and the train-best fitness kept improving while the test return barely moved (optimism gap $+38.92 \to +172.29$). A single seed cannot establish a plateau, so it was repeated with 5 seeds (Section 6.5.1b).

#### 6.5.1b Multi-Seed Repeat of the 250-Generation Test (S=5)
`run_pendulum_stress_multiseed.py` repeats the same CMA-ES configuration ($G=250$, $N_{\text{rollouts}}=8$, test seed $99999$, 100 test episodes) for seeds $101, 202, 303, 404$ (`results/pendulum_stress_multiseed.json`); seed 42 is the run above. The $G=100$ reference is the CMA-ES Pendulum row of Section 6.6.2 (same seeds, same protocol).

| Seed | Test mean $G=100$ | Test mean $G=250$ (success) | Change | Train best $G=250$ |
|:---:|:---:|:---:|:---:|:---:|
| 42 | −268.67 | −242.43 (37%) | +26.2 | −70.13 |
| 101 | −701.44 | −247.97 (34%) | +453.5 | −97.37 |
| 202 | −221.47 | −317.54 (38%) | −96.1 | −70.79 |
| 303 | −274.77 | −210.85 (44%) | +63.9 | −65.10 |
| 404 | −796.35 | −336.59 (53%) | +459.8 | −66.49 |
| **Mean** | **−452.54** ($\sigma_{\text{train}}=273.39$) | **−271.08** ($\sigma_{\text{train}}=53.46$) | **+181.5** | −73.98 |

* **Measured**: with 5 seeds, going from $G=100$ to $G=250$ raises the mean test return from $-452.54$ to $-271.08$, cuts the between-seed standard deviation from $273$ to $53$, and raises mean success from $38.6\%$ to $41.2\%$. Four of five seeds improve; seed 202 gets worse ($-96.1$).
* **Therefore the single-seed "plateau" reading was not supported**: seed 42 happened to be the seed whose $G=100$ result was already good, and the multi-seed gain comes mostly from the seeds that were poor at $G=100$ (101 and 404). Under $G=250$ all five seeds end between $-211$ and $-337$.
* **Caveats**: the paired change has mean $+181$ with standard deviation $\approx 255$ across seeds, so with $S=5$ it is suggestive but not statistically conclusive ($t \approx 1.6$). The result is for CMA-ES only; the other algorithms were not run at $G=250$. The optimism gap remains large ($\approx +170$ to $+260$ for most seeds), so the training-best fitness ($\approx -70$) is not a reliable indicator of test performance.
* **Reading**: the remaining deficit at $G=100$ was largely an optimization-budget (and seed-fragility) effect, and returns at $G=250$ (≈ $-210$ to $-340$) are much more reproducible across seeds. Whether performance saturates beyond $G=250$ was not tested.

#### 6.5.2 Theoretical Hypotheses (External Context, Not Tested Here)
> [!NOTE]
> SAC (Haarnoja et al., 2018) and TD3 (Fujimoto et al., 2018) were **not executed in this repository**. The figure sometimes quoted for them on `Pendulum-v1` (return around $-150$, high success) is **not a result reported in those two papers** (neither evaluates Pendulum) and is **not verified here**; it should be treated as an unverified expectation until SAC/TD3 are run under the same 100-episode protocol. The references are cited only for the algorithms themselves (off-policy actor-critic with a replay buffer).

No TD baseline was run here, so no performance gap against TD methods was measured. Three hypotheses that could explain such a gap, none tested in this repository:

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

### 6.6 Multi-Seed Scientific Evaluation Protocol (Agarwal et al., NeurIPS 2021)

#### 6.6.1 Methodological Distinction: Aleatoric vs. Epistemic Uncertainty
In standard single-seed evaluations ($S=1$, typically $\text{seed}=42$), test metrics measure solely **aleatoric evaluation variance** ($\sigma_{\text{eval}}$): the stochasticity introduced by the environment's randomized initial state distribution $s_0 \sim p(s_0)$ given a fixed, frozen policy $\pi_\theta$.

However, as emphasized by Henderson et al. (2018) and formalized by Agarwal et al. (NeurIPS 2021), a rigorous scientific study must disentangle:
1. **Aleatoric Evaluation Variance ($\sigma_{\text{eval}}$)**: within-policy variance across out-of-sample test episodes.
2. **Epistemic Optimization Variance ($\sigma_{\text{train}}$)**: between-policy variance across independent training runs initialized from distinct pseudo-random seeds ($S=5$: `[42, 101, 202, 303, 404]`).

To quantify the balance between these two sources of uncertainty, we compute the **Variance Ratio**:
$$\rho = \frac{\sigma_{\text{train}}}{\sigma_{\text{eval}}}$$
* **$\rho < 1.0$ (Evaluation-Dominated)**: The optimization trajectory is reproducible across seeds; residual performance dispersion is dominated by environmental stochasticity.
* **$\rho > 1.0$ (Training-Dominated / Unstable)**: The optimization trajectory is fragile; the choice of initial random seed creates larger disparities in final policy capability than testing stochasticity.

Furthermore, following the NeurIPS 2021 recommendations of Agarwal et al., we report:
* **Stratified 95% Bootstrap Confidence Intervals**: computed via $B=2{,}000$ hierarchical resamplings over both seeds and episodes.
* **Interquartile Mean (IQM)**: the 25% trimmed mean across all seed-episode samples, discarding outliers to ensure robust central tendency estimation.

#### 6.6.2 Consolidated Multi-Seed Results ($S=5$ Seeds $\times$ $100$ Test Episodes = $500$ Episodes per Algorithm)

Raw empirical data extracted directly from `results/multiseed_summary.csv` (generated by `run_multiseed_study.py`):

| Family | Algorithm | Environment | Mean across 5 Seeds $\pm$ SEM | Stratified 95% CI | IQM | $\sigma_{\text{train}}$ | $\sigma_{\text{eval}}$ | Variance Ratio $\rho$ | Success Rate (%) | Mean Train Best | Mean $\Delta$ |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **F1** | **SimpleGA** | CartPole-v1 | 310.08 $\pm$ 82.74 | [167.12, 458.09] | 345.61 | 185.02 | 99.83 | **1.85** | 50.2% | 463.70 | +153.62 |
| **F1** | **DE** | CartPole-v1 | 283.30 $\pm$ 66.85 | [171.68, 402.00] | 272.51 | 149.48 | 104.73 | **1.43** | 38.8% | 474.40 | +191.10 |
| **F1** | **OpenAI-ES** | CartPole-v1 | 317.08 $\pm$ 72.92 | [186.77, 445.30] | 350.16 | 163.06 | 105.01 | **1.55** | 44.0% | 483.90 | +166.82 |
| **F3** | **CMA-ES** | CartPole-v1 | **483.98 $\pm$ 6.81** | **[470.55, 494.53]** | **500.00** | **15.22** | 42.96 | **0.35** | **87.4%** | **500.00** | **+16.02** |
| **F3** | **PBIL** | CartPole-v1 | 354.86 $\pm$ 74.56 | [210.58, 462.87] | 423.78 | 166.73 | 84.61 | **1.97** | 50.8% | 391.10 | +36.24 |
| **F2** | **LinearGP** | CartPole-v1 | 174.60 $\pm$ 22.00 | [128.81, 201.35] | 179.50 | 49.20 | 53.98 | **0.91** | 0.4% | 232.67 | +58.07 |
| **F2** | **CartesianGP**| CartPole-v1 | 306.65 $\pm$ 66.82 | [189.37, 415.51] | 322.76 | 149.41 | 77.25 | **1.93** | 31.0% | 375.20 | +68.55 |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **F1** | **SimpleGA** | Acrobot-v1 | **−86.58 $\pm$ 1.09** | **[−89.80, −83.60]** | **−81.34** | **2.44** | 27.80 | **0.09** | **82.8%** | −65.20 | +21.38 |
| **F1** | **DE** | Acrobot-v1 | −101.54 $\pm$ 12.74 | [−126.95, −85.50] | −84.68 | 28.49 | 54.33 | **0.52** | 80.2% | −71.20 | +30.34 |
| **F1** | **OpenAI-ES** | Acrobot-v1 | −92.47 $\pm$ 6.57 | [−105.69, −84.49] | −85.82 | 14.70 | 23.55 | **0.62** | 74.4% | −73.10 | +19.37 |
| **F3** | **CMA-ES** | Acrobot-v1 | −105.73 $\pm$ 18.15 | [−144.31, −80.39] | −82.86 | 40.59 | 56.52 | **0.72** | 79.0% | −64.30 | +41.43 |
| **F3** | **PBIL** | Acrobot-v1 | −131.86 $\pm$ 34.00 | [−200.21, −93.31] | −95.21 | 76.03 | 66.49 | **1.14** | 60.2% | −76.50 | +55.36 |
| **F2** | **LinearGP** | Acrobot-v1 | −103.64 $\pm$ 15.63 | [−134.41, −82.49] | −90.06 | 34.94 | 26.73 | **1.31** | 64.0% | −86.33 | +17.30 |
| **F2** | **CartesianGP**| Acrobot-v1 | −215.16 $\pm$ 75.79 | [−364.82, −106.05] | −154.97 | 169.48 | 34.58 | **4.90** | 31.6% | −194.27 | +20.89 |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **F1** | **SimpleGA** | Pendulum-v1 | **−444.26 $\pm$ 106.87**| [−664.40, −316.79] | **−288.52** | 238.98 | 387.93 | **0.62** | **37.2%** | −125.97 | +318.29 |
| **F1** | **DE** | Pendulum-v1 | −775.44 $\pm$ 48.79 | [−874.32, −677.94] | −748.06 | 109.11 | 565.39 | **0.19** | 26.0% | −406.31 | +369.13 |
| **F1** | **OpenAI-ES** | Pendulum-v1 | −1025.61 $\pm$ 14.86| [−1059.85, −998.08]| −1012.01 | 33.23 | 184.91 | **0.18** | 0.0% | −873.18 | +152.43 |
| **F3** | **CMA-ES** | Pendulum-v1 | **−452.54 $\pm$ 122.26**| [−666.12, −247.93] | **−288.35** | 273.39 | 345.45 | **0.79** | **38.6%** | −185.97 | +266.57 |
| **F3** | **PBIL** | Pendulum-v1 | −789.78 $\pm$ 91.63 | [−947.13, −625.84] | −761.61 | 204.89 | 276.89 | **0.74** | 6.2% | −676.19 | +113.60 |
| **F2** | **LinearGP**† | Pendulum-v1 | −1247.90 $\pm$ 0.00 | [−1281.84, −1213.84]| −1260.89 | 0.00 | 393.71 | **0.00** | 0.0% | −892.05 | +355.85 |
| **F2** | **CartesianGP**| Pendulum-v1 | −1016.59 $\pm$ 118.96| [−1207.92, −799.79]| −966.27 | 266.00 | 308.83 | **0.86** | 0.0% | −757.69 | +258.90 |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **F1** | **SimpleGA** | MountainCarCont | 86.36 $\pm$ 4.97 | [76.55, 91.98] | 91.99 | 11.11 | 14.39 | **0.77** | 69.4% | 93.39 | +7.04 |
| **F1** | **DE** | MountainCarCont | **91.44 $\pm$ 1.80** | [88.05, 94.40] | 92.06 | 4.03 | 1.61 | **2.51** | 60.2% | 91.42 | **−0.02** |
| **F1** | **OpenAI-ES** | MountainCarCont | 0.00 $\pm$ 0.00 | [0.00, 0.00] | 0.00 | 0.00 | 0.00 | — | 0.0% | 0.00 | 0.00 |
| **F3** | **CMA-ES** | MountainCarCont | 82.05 $\pm$ 9.20 | [65.14, 95.63] | **95.65** | 20.58 | 21.44 | **0.96** | 76.0% | 94.25 | +12.20 |
| **F3** | **PBIL** | MountainCarCont | **93.86 $\pm$ 0.88** | **[92.39, 95.71]** | **94.21** | **1.97** | **3.79** | **0.52** | **96.8%** | **94.63** | **+0.77** |
| **F2** | **LinearGP** | MountainCarCont | 0.00 $\pm$ 0.00 | [0.00, 0.00] | 0.00 | 0.00 | 0.00 | — | 0.0% | 0.00 | 0.00 |
| **F2** | **CartesianGP**| MountainCarCont | 0.00 $\pm$ 0.00 | [0.00, 0.00] | 0.00 | 0.00 | 0.00 | — | 0.0% | 0.00 | 0.00 |

†*LinearGP on Pendulum-v1: zero-torque baseline, see item 6 below. $\rho = 0$ reflects identical degenerate programs, not stable learning. For rows with $\sigma_{\text{eval}} = 0$ (MountainCar zeros) $\rho$ is undefined ("—"); the plot shows a nominal value there, which should be ignored.*

> [!NOTE]
> **Statistical scope.** $S=5$ seeds give imprecise estimates of $\sigma_{\text{train}}$ and $\rho$ (a standard deviation estimated from 5 values has a relative error of roughly 35%), so small differences between $\rho$ values should not be over-read. The causal ablation (Section 6.7) covers only CartPole, and the 250-generation test only one seed.

#### 6.6.3 Observations from the Multi-Seed Variance Decomposition
> [!NOTE]
> This subsection reports what was measured. Causal explanations are deliberately limited to Section 6.7, where they were tested by controlled ablation, and to clearly labeled hypotheses.

1. **CartPole-v1 (Measured)**: SimpleGA, DE, OpenAI-ES, PBIL and CartesianGP have $\sigma_{\text{train}} \approx 149 - 185$ ($\rho$ between 1.4 and 2.0) under the standard budget ($G=30$, $N=2$). CMA-ES has $\sigma_{\text{train}} = 15.22$, $\rho = 0.35$, IQM $= 500.00$ and $87.4\%$ aggregate success. Note that CMA-ES is not "perfect" over seeds: its mean is $483.98$, and only the seed-42 run in Section 5.2 shows $500.00 \pm 0.00$.
2. **Differential Evolution on MountainCarContinuous-v0 (Measured)**: With seed 42, DE reached $85.68$ (below the $90.0$ success threshold). Across 5 seeds the aggregate mean is $91.44 \pm 1.80$ with $60.2\%$ success. The seed-42 result is therefore not representative of DE's average behavior under this budget. With $S=5$ we cannot determine the exact cause of that seed's lower result.
3. **PBIL on MountainCarContinuous-v0 (Measured)**: Mean $93.86 \pm 0.88$, stratified 95% CI `[92.39, 95.71]`, IQM $94.21$, $\sigma_{\text{train}} = 1.97$, $96.8\%$ success: the highest aggregate performance and lowest between-seed variance among the 7 algorithms in this environment.
4. **Pendulum-v1 (Measured)**: SimpleGA and CMA-ES reach very similar IQM ($-288.52$ and $-288.35$) and success ($37.2\%$ and $38.6\%$), with large $\sigma_{\text{eval}} > 340$. The 5-seed $G=250$ repeat of Section 6.5.1b shows that, for CMA-ES, more generations improve and stabilize the result, so the $G=100$ numbers here should not be read as a ceiling. Whether the SimpleGA/CMA-ES similarity persists at $G=250$ was not tested. Whether this plateau is a property of the algorithms or of the budget was not isolated for Pendulum (the ablation of Section 6.7 covers CartPole only).
5. **Zero returns on MountainCarContinuous-v0 (Measured)**: OpenAI-ES, LinearGP and CartesianGP obtained exactly $0.00$ in all 5 seeds ($500$ test episodes each). This shows the failure is not specific to seed 42. The mechanism (flat reward landscape giving no selection signal) is a **hypothesis** that was not tested here.
6. **LinearGP on Pendulum-v1 is a degenerate policy (Measured, correction)**: LinearGP obtained exactly $-1247.90$ in all 5 seeds, hence $\sigma_{\text{train}} = 0.00$ and $\rho = 0.00$. We verified that the selected programs output a raw action of exactly $0$ at every step, and that a constant zero-torque policy on the same 100 test episodes yields mean $-1247.903$, std $393.71$ and $0\%$ success (identical to the reported values). The zero variance across seeds therefore does **not** indicate stable optimization: LinearGP never learned a non-trivial policy on Pendulum. Its apparent "train best" ($-860$ to $-896$) is the best of noisy $N=8$ evaluations of such zero-output programs (favorable initial angles), i.e., selection on noise. The LinearGP Pendulum numbers in Sections 5.2 and 6.2 must be read as the zero-torque baseline.

### 6.7 Causal Ablation on CartPole-v1: Evaluation Noise ($N$) vs. Optimization Budget ($G$)

**Question.** Is the large between-seed variance on CartPole caused by (a) noisy fitness estimation (few rollouts $N$ per evaluation) or (b) insufficient optimization budget (few generations $G$)?

**Design** (`run_causal_ablation.py`, results in `results/causal_ablation_cartpole.json/csv`): 5 parametric algorithms (SimpleGA, DE, OpenAI-ES, CMA-ES, PBIL), $S=5$ training seeds `[42, 101, 202, 303, 404]`, configurations $(G, N) \in \{(30,2), (30,8), (30,16), (100,2), (100,8)\}$, each policy evaluated on the same 100 test episodes (test seed 99999). The baseline $(30,2)$ exactly reproduces Section 6.6.2. A determinism check (SimpleGA, seed 42, run twice) gave identical train and test values, so the variance across seeds is not run-to-run non-determinism. The policy at test time is deterministic (`argmax` of logits), so within-seed variance comes only from the initial states.

**Results** (mean test return across 5 seeds; in parentheses $\sigma_{\text{train}}$ / optimism gap $\Delta$):

| Algorithm | $G{=}30,N{=}2$ | $G{=}30,N{=}8$ | $G{=}30,N{=}16$ | $G{=}100,N{=}2$ | $G{=}100,N{=}8$ |
|:---|:---:|:---:|:---:|:---:|:---:|
| SimpleGA | 310.1 (185.0 / 153.6) | 478.5 (31.4 / 13.6) | 425.4 (56.9 / 45.0) | 310.1 (185.0 / 189.9) | 484.9 (20.3 / 15.1) |
| DE | 283.3 (149.5 / 191.1) | 467.9 (31.3 / −0.9) | 423.6 (37.7 / 0.7) | 375.1 (105.0 / 124.9) | 494.0 (9.2 / 4.9) |
| OpenAI-ES | 317.1 (163.1 / 166.8) | 313.8 (112.5 / 99.5) | 344.0 (127.7 / 14.6) | 382.8 (137.1 / 117.2) | 439.9 (110.8 / 60.1) |
| CMA-ES | 484.0 (15.2 / 16.0) | 468.2 (42.6 / 31.8) | 496.6 (3.8 / 3.4) | 484.0 (15.2 / 16.0) | 468.2 (42.6 / 31.8) |
| PBIL | 354.9 (166.7 / 36.2) | 372.7 (101.6 / 3.6) | 438.7 (78.0 / −12.0) | 442.4 (80.0 / 51.6) | 476.7 (39.6 / 14.0) |

**What the data show (measured):**
1. **Verified: there is no run-to-run non-determinism**; identical seeds give identical results.
2. **Evaluation noise ($N$) is a major cause for SimpleGA and DE.** Raising $N$ from 2 to 8 at fixed $G=30$ lowers $\sigma_{\text{train}}$ from 185 to 31 (SimpleGA) and from 149 to 31 (DE), and the optimism gap from +154 / +191 to +14 / −1. More generations at $N=2$ did **not** fix this for SimpleGA (identical result, see item 5) and helped DE only partially ($\sigma_{\text{train}}$ 105, gap +125).
3. **Optimization budget ($G$) is a major cause for OpenAI-ES, PBIL (and partly DE).** For these, the training best fitness was still improving at $G=30$: at $N=2$, going from $G=30$ to $G=100$ raised the mean test return from 317 to 383 (OpenAI-ES) and from 355 to 442 (PBIL). In this sense these algorithms were under-trained at the standard budget.
4. **CMA-ES is already close to its ceiling at the standard budget.** Its $\sigma_{\text{train}}$ is lowest at $N=16$ (3.8), but it is not monotonic in $N$ (42.6 at $N=8$), so no clear $N$-trend can be claimed with $S=5$.
5. **Methodological caveat on the $G$ axis.** For SimpleGA and CMA-ES at $N=2$, and for CMA-ES at $N=8$, the results at $G=100$ are identical to $G=30$ down to the second decimal in every seed. The implementation returns the best-so-far individual by training fitness; once a seed reaches the fitness ceiling of $500$ on its training rollouts, that individual can never be replaced, so extra generations cannot change the output. For these cases, "more generations" is mechanically ineffective and the $G$ axis is not informative; it does **not** show that more training is useless in general.
6. **The effect of $N$ is not strictly monotonic:** for SimpleGA and DE, $N=16$ at $G=30$ is worse than $N=8$ ($425.4$ vs $478.5$, $423.6$ vs $467.9$). With $S=5$ seeds and per-seed differences of this size, we cannot distinguish this from sampling noise; a plausible, **untested** hypothesis is that with larger $N$ the same $G=30$ budget leaves fewer effective selection steps for convergence.

**Answer to the question (CartPole only).** The between-seed variance is **not a single cause** and is **algorithm-dependent**: for algorithms that rapidly find a lucky individual and keep it (SimpleGA, DE, best-so-far selection on noisy fitness) it is mainly **noisy fitness evaluation** (small $N$); for OpenAI-ES and PBIL it is mainly **insufficient generations**. The "seed" is therefore not an independent cause: it changes the initial population and the sampled training start states, and the variance it induces is amplified when selection is made on a noisy fitness estimate or when the search has not converged.

**Limitations:** single environment (CartPole), $S=5$ seeds, only five $(G,N)$ points (not a full factorial), and the best-so-far ceiling effect of item 5. The ablation was not run for Pendulum-v1, MountainCarContinuous-v0, Acrobot-v1, or for the GP families.

---

## 7. Generated Visualizations

All figures are rendered in a high-resolution academic dark-mode style and saved under `results/`:

### 7.1 Multi-Seed Uncertainty and Out-of-Sample Validation
* **`results/multiseed_variance_analysis.png`**: 4-panel multi-seed evaluation chart across all 4 environments displaying the aggregate mean return across 5 independent training seeds ($S=5$, 500 test episodes total per algorithm), error bars for the **stratified 95% bootstrap confidence interval**, and variance ratio annotations ($\rho = \sigma_{\text{train}}/\sigma_{\text{eval}}$) according to the Agarwal et al. (NeurIPS 2021) protocol. Caveats: the Pendulum LinearGP bar ($-1248$, $\rho=0.0$) is the zero-torque baseline (Section 6.6.3 item 6), and on MountainCar the plot annotates nominal $\rho$ values for the all-zero methods where Section 6.6.2 reports "—".
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

| Algorithm | Family | Key Hyperparameters |
|:---|:---:|:---|
| **SimpleGA** | F1 | Population: 32–64, $\sigma_0 = 0.5$, $\sigma$ decay: $0.999$, SBX $\eta = 2.0$, $P_{\text{cx}} = 0.8$, Elitism: 1 |
| **DE** | F1 | Population: 32–64, $F = 0.8$, $CR = 0.9$, Strategy: `rand/1/bin`, Initial scale: 0.5 |
| **OpenAI-ES** | F1 | Population: 32 (16 antithetic pairs), $\sigma = 0.05$, Learning rate: 0.01, Rank-based fitness shaping |
| **LinearGP** | F2 | Population: 16, Instructions: 48, Registers: $\text{obs} + 8 + \text{act}$, Mutation rate: 0.15 |
| **CartesianGP**| F2 | Population: 8 (1 parent + 7 offspring), Columns: 30, Per-gene mutation rate: 0.05 |
| **CMA-ES** | F3 | $\sigma_0 = 0.5$, $\lambda = 4 + \lfloor 3 \ln d \rfloor$, CSA/CMA hyperparameters derived from Hansen (2016) via $\sqrt{d}$ |
| **PBIL** | F3 | Population: 32, $\alpha_\mu = 0.1$, $\alpha_\sigma = 0.05$, Elite: top 20%, $\sigma_{\text{init}} = 1.0$, $\sigma_{\text{min}} = 0.01$ |

---

## 9. Repository Structure

```
evolutionary-strategies/
├── src/
│   ├── environments.py       # Gymnax 1.0.0 wrapper + vectorized JAX MLP forward pass
│   ├── family1_direct.py     # Family 1: SimpleGA, DE, OpenAI-ES
│   ├── family2_programs.py   # Family 2: LinearGP, CartesianGP (Genetic Programming)
│   ├── family3_eda.py        # Family 3: CMA-ES, PBIL (Estimation of Distribution)
│   ├── benchmark.py          # Unified ask/tell benchmark with JIT compilation
│   ├── evaluation.py         # Formal out-of-sample validation module (100 eps, bootstrap CI, SNR)
│   └── visualization.py      # Dark-mode figure and table generation
├── run_benchmark.py          # Accelerated training script (Families 1 and 3)
├── run_gp_only.py            # Dedicated training script for Family 2 (GP)
├── run_validation_100.py     # Master out-of-sample validation script (100 eps)
├── run_multiseed_study.py    # Multi-seed study disentangling aleatoric vs epistemic variance (Agarwal et al., 2021)
├── run_pendulum_stress_test.py # Long-horizon stress test (250 generations, 10M steps)
├── run_pendulum_stress_multiseed.py # 5-seed repeat of the 250-generation CMA-ES Pendulum test
├── run_causal_ablation.py    # CartPole ablation of N (rollouts) vs G (generations) over 5 seeds
├── generate_final_plots.py   # Consolidated visual compilation of training runs
├── results/                  # Raw empirical data and figures:
│   ├── validation_100_results.json
│   ├── validation_100_summary.csv
│   ├── validation_100_ci95_bars.png
│   ├── optimism_gap_analysis.png
│   ├── pendulum_stress_test_250.json
│   ├── pendulum_stress_multiseed.json
│   ├── multiseed_study.json
│   ├── multiseed_summary.csv
│   ├── multiseed_variance_analysis.png
│   ├── causal_ablation_cartpole.json
│   ├── causal_ablation_cartpole.csv
│   ├── all_families_curves.png
│   ├── all_families_bar.png
│   ├── family_comparison.png
│   ├── learning_curves.png
│   ├── time_vs_performance.png
│   ├── raw_results.json
│   ├── gp_results.json
│   └── results_summary.csv
└── README.md
```

---

## 10. Reproducing the Experiments

```bash
# 1. Set up the virtual environment (Python 3.11 recommended)
py -3.11 -m venv C:\ev
C:\ev\Scripts\activate
pip install "jax[cpu]" gymnax "orbax-checkpoint==0.5.23" "flax==0.8.5" matplotlib numpy tqdm

# 2. Run the complete out-of-sample validation protocol (100 episodes)
python run_validation_100.py

# 3. Run the Multi-Seed Scientific Study (Agarwal et al., 2021)
python run_multiseed_study.py

# 4. Run the 250-generation Pendulum asymptotic stress test
python run_pendulum_stress_test.py --generations 250 --rollouts 8

# 4b. Repeat the 250-generation test for seeds 101, 202, 303, 404
python run_pendulum_stress_multiseed.py

# 5. Run the CartPole causal ablation (N vs G)
python run_causal_ablation.py

# 6. (Optional) Run the separate training benchmark and regenerate all plots
python run_benchmark.py --quick --no-gp
python run_gp_only.py
python generate_final_plots.py
```

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

