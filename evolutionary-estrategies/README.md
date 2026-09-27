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
  * Primitive function set: $\{+, -, \times, \div_{\text{safe}}, \sin, \cos, \tanh\}$.
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

### 3.2 Vectorization and XLA Acceleration (JAX)
* Evaluating one individual in one episode is modeled as a pure function `rollout(flat_params, rng) -> float`.
* The whole population of size $P$ is parallelized through native vectorized compilation:
  $$\text{eval\_pop} = \text{jax.jit}(\text{jax.vmap}(\text{rollout\_multi}, \text{in\_axes}=(0, \text{None})))$$
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

The table below details the number of generations ($G$), population size ($P$), rollouts per individual, the **total number of simulated episodes**, and the **ceiling on environment interaction steps** over the whole training run in each environment:

| Algorithm | Family | Generations ($G$) | Population ($P$) | Rollouts/Individual | Total Episodes / Env | Step Ceiling CartPole ($H=500$) | Step Ceiling Acrobot ($H=500$) | Step Ceiling Pendulum ($H=200$) | Step Ceiling MountainCar ($H=999$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **SimpleGA** | F1 | 30 | 32 | 2 | **1,920** | 960,000 | 960,000 | 384,000 | 1,918,080 |
| **DE** | F1 | 30 | 32 | 2 | **1,920** | 960,000 | 960,000 | 384,000 | 1,918,080 |
| **OpenAI-ES** | F1 | 30 | 32 (16 pairs) | 2 | **1,920** | 960,000 | 960,000 | 384,000 | 1,918,080 |
| **LinearGP** | F2 | 15 | 16 | 3 | **720** | $\le 360{,}000$* | $\le 360{,}000$* | — | — |
| **CartesianGP**| F2 | 15 | 8 (1+7) | 3 | **360** | $\le 180{,}000$* | $\le 180{,}000$* | — | — |
| **CMA-ES** | F3 | 30 | 25 ($\lambda$) | 2 | **1,500** | 750,000 | 750,000 | 300,000 | 1,498,500 |
| **PBIL** | F3 | 30 | 32 | 2 | **1,920** | 960,000 | 960,000 | 384,000 | 1,918,080 |

*\*Note on Family 2 (GP)*: In the Python-interpreted implementation the loop exits as soon as `terminated` or `truncated` is raised (`break`). Therefore, when the program learns to stabilize the pendulum quickly (or fails early), the actual number of steps executed is significantly lower than the maximum ceiling $H$.

---

## 4. Consolidated Experimental Results

All data below was obtained empirically through the automated benchmark (no synthetic or extrapolated data):

### 4.1 Discrete Environments (All 3 Families)

| Family | Algorithm | Paradigm | CartPole-v1 (Best) | Acrobot-v1 (Best) | CartPole Time (s) | Acrobot Time (s) |
|:---|:---|:---|:---:|:---:|:---:|:---:|
| **F1: Direct Solutions** | **DE** | rand/1/bin trial vectors | **500.00** *(Gen 6)* | −78.00 | 2.7s | 2.5s |
| **F1: Direct Solutions** | **SimpleGA** | Tournament + SBX crossover | 318.50 | **−64.00** | 3.0s | 2.9s |
| **F1: Direct Solutions** | **OpenAI-ES** | NES gradient ascent | 419.50 | **−64.00** *(Mean: −84.78)* | 2.1s | 2.6s |
| **F2: Programs** | **CartesianGP** | Positional 2D DAG (1+7)ES | **500.00** *(Gen 5)* | −78.67 | 61.9s | 65.5s |
| **F2: Programs** | **LinearGP** | Register sequence | 453.33 | −74.00 | 95.4s | 239.4s |
| **F3: Models (EDA)** | **CMA-ES** | Covariance $\mathcal{N}(\mu, \sigma^2 \Sigma)$ | **500.00** | **−64.00** | 9.2s | 10.2s |
| **F3: Models (EDA)** | **PBIL** | Univariate marginal model | 464.50 | −75.00 | 2.1s | 3.0s |

### 4.2 Continuous Environments (Families 1 and 3)

| Family | Algorithm | Pendulum-v1 (Best) | Pendulum-v1 (Mean Final) | MountainCarCont (Best) | MountainCarCont (Mean Final) | Total Time (s) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **F1: Direct Solutions** | **SimpleGA** | **−9.13** | −1237.52 | 94.17 | −56.30 | 4.3s |
| **F1: Direct Solutions** | **DE** | −468.77 | −1412.38 | 85.50 | −90.91 | 3.6s |
| **F1: Direct Solutions** | **OpenAI-ES** | −691.96 | −1231.18 | **−0.00** *(Gradient failure)* | −1.12 | 3.9s |
| **F3: Models (EDA)** | **CMA-ES** | −410.60 | −1348.26 | **95.83** *(Optimum)* | −59.37 | 16.0s |
| **F3: Models (EDA)** | **PBIL** | −192.09 | −1524.69 | 94.22 | −74.45 | 3.2s |

---

## 5. Out-of-Sample Statistical Validation (Rigorous 100-Episode Protocol)

In line with modern methodological guidelines for reproducibility and evaluation in Reinforcement Learning (Henderson et al. 2018; Machado et al. 2018; Agarwal et al. 2021), **a metric obtained during training with few rollouts ($N=2$ or $3$) is not sufficient evidence of robust convergence**. Evolutionary policies can overfit the stochastic conditions seen during selection (*Winner's Curse*).

To quantify the true generalization capability and epistemic uncertainty, the best controller of each algorithm was subjected to a battery of **$N=100$ independent test episodes** with a pseudo-random seed unseen during optimization (`seed = 99999`):
* **Families 1 and 3 (Neural Networks)**: Evaluated through pure vectorized compilation in JAX (`jax.vmap` over 100 seeds with `jax.lax.scan`), guaranteeing identical parallelism and no temporal bias.
* **Family 2 (Symbolic Programs)**: Evaluated by direct iterative execution of the graphs/registers with transition steps compiled in XLA (`jax.jit`).

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

### 5.2 Consolidated Validation Table ($N=100$ Out-of-Sample Episodes)

Raw empirical data extracted directly from `results/validation_100_summary.csv`:

| Family | Algorithm | Environment | Train Best ($N \le 3$) | Test Mean $\pm$ SEM | 95% Bootstrap CI | Median | IQR | Std Dev | SNR | Optimism Gap ($\Delta$) | Success Rate (%) |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **F1** | **SimpleGA** | CartPole-v1 | 318.50 | 318.82 $\pm$ 19.43 | [281.37, 356.55] | 500.00 | 395.50 | 194.29 | 1.64 | **−0.32** | 51.0% |
| **F1** | **DE** | CartPole-v1 | **500.00** | 123.84 $\pm$ 8.48 | [108.68, 141.80] | 101.00 | 40.00 | 84.79 | 1.46 | **+376.16** | **2.0%** |
| **F1** | **OpenAI-ES** | CartPole-v1 | 419.50 | 138.27 $\pm$ 10.66 | [118.93, 161.17] | 96.00 | 66.75 | 106.60 | 1.30 | **+281.23** | 4.0% |
| **F3** | **CMA-ES** | CartPole-v1 | **500.00** | **500.00 $\pm$ 0.00** | **[500.00, 500.00]** | **500.00** | **0.00** | **0.00** | **$\infty$** | **0.00** | **100.0%** |
| **F3** | **PBIL** | CartPole-v1 | 464.50 | 412.90 $\pm$ 7.64 | [397.39, 427.45] | 406.50 | 139.00 | 76.36 | 5.41 | +51.60 | 28.0% |
| **F2** | **LinearGP** | CartPole-v1 | 453.33 | **487.53 $\pm$ 4.07** | [479.06, 494.89] | 500.00 | 0.00 | 40.73 | 11.97 | **−34.20** | **91.0%** |
| **F2** | **CartesianGP**| CartPole-v1 | **500.00** | **487.52 $\pm$ 4.07** | [479.04, 494.88] | 500.00 | 0.00 | 40.74 | 11.97 | +12.48 | **91.0%** |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **F1** | **SimpleGA** | Acrobot-v1 | −64.00 | −85.55 $\pm$ 4.55 | [−95.12, −78.73] | −76.50 | 19.00 | 45.46 | 1.88 | +21.55 | 88.0% |
| **F1** | **DE** | Acrobot-v1 | −78.00 | −93.68 $\pm$ 3.61 | [−101.43, −87.34] | −85.50 | 20.25 | 36.09 | 2.60 | +15.68 | 82.0% |
| **F1** | **OpenAI-ES** | Acrobot-v1 | −64.00 | −85.85 $\pm$ 2.22 | [−90.36, −81.58] | −81.50 | 19.25 | 22.19 | 3.87 | +21.85 | 88.0% |
| **F3** | **CMA-ES** | Acrobot-v1 | −64.00 | **−77.79 $\pm$ 1.22** | **[−80.30, −75.46]** | **−73.00** | **12.50** | **12.22** | **6.37** | **+13.79** | **96.0%** |
| **F3** | **PBIL** | Acrobot-v1 | −75.00 | −101.90 $\pm$ 4.50 | [−112.47, −94.93] | −95.50 | 20.25 | 44.96 | 2.27 | +26.90 | 55.0% |
| **F2** | **LinearGP** | Acrobot-v1 | −74.00 | **−85.31 $\pm$ 2.84** | [−91.04, −80.39] | −74.50 | 16.25 | 28.39 | 3.00 | +11.31 | **87.0%** |
| **F2** | **CartesianGP**| Acrobot-v1 | −78.67 | −90.50 $\pm$ 1.50 | [−93.52, −87.69] | −86.00 | 17.25 | 15.04 | 6.02 | +11.83 | **82.0%** |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **F1** | **SimpleGA** | Pendulum-v1 | −9.13 | −1073.68 $\pm$ 39.64 | [−1143.46, −989.30] | −1225.48 | 43.61 | 396.38 | 2.71 | +1064.56 | 12.0% |
| **F1** | **DE** | Pendulum-v1 | −468.77 | −1117.59 $\pm$ 41.93 | [−1197.90, −1033.62] | −1162.13 | 710.31 | 419.34 | 2.67 | +648.83 | 0.0% |
| **F1** | **OpenAI-ES** | Pendulum-v1 | −691.96 | −1258.51 $\pm$ 36.11 | [−1329.19, −1188.75] | −1253.45 | 576.31 | 361.06 | 3.49 | +566.55 | 0.0% |
| **F3** | **CMA-ES** | Pendulum-v1 | −410.60 | −1312.86 $\pm$ 41.49 | [−1389.52, −1228.50] | −1494.44 | 105.20 | 414.94 | 3.16 | +902.26 | 2.0% |
| **F3** | **PBIL** | Pendulum-v1 | −192.09 | **−804.32 $\pm$ 65.14** | **[−929.87, −675.68]** | **−387.26** | 1359.68 | 651.38 | 1.23 | +612.22 | **29.0%** |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **F1** | **SimpleGA** | MountainCarCont | 94.17 | 91.84 $\pm$ 0.24 | [91.35, 92.31] | 90.71 | 3.65 | 2.40 | 38.25 | +2.33 | 84.0% |
| **F1** | **DE** | MountainCarCont | 85.50 | 85.68 $\pm$ 0.36 | [84.96, 86.36] | 87.74 | 5.19 | 3.61 | 23.75 | −0.18 | 0.0%* |
| **F1** | **OpenAI-ES** | MountainCarCont | −0.00 | −0.00 $\pm$ 0.00 | [−0.00, −0.00] | −0.00 | 0.00 | 0.00 | 1.02 | 0.00 | 0.0% |
| **F3** | **CMA-ES** | MountainCarCont | **95.83** | **96.43 $\pm$ 0.11** | **[96.22, 96.63]** | **96.72** | **1.76** | **1.06** | **90.99** | **−0.60** | **100.0%** |
| **F3** | **PBIL** | MountainCarCont | 94.22 | **93.87 $\pm$ 0.12** | **[93.64, 94.10]** | **94.06** | **1.80** | **1.15** | **81.28** | **+0.35** | **100.0%** |

*\*Note on DE in MountainCar*: DE consistently converged to a score of $\approx 85.7$, but the formal success criterion requires $R \ge 90.0$.

---

## 6. Analysis and Scientific Discussion of the Results

### 6.1 The Empirical Manifestation of the Winner's Curse (Spurious Optimism in Fast Evaluations)
Validation over 100 episodes exposed a crucial phenomenon in evolutionary RL:
1. **The Collapse of Differential Evolution (DE) on CartPole**:
   * During training ($N_{\text{rollouts}}=2$), DE reported an apparent optimal fitness of **500.00**.
   * In testing ($N=100$), the mean collapsed to **123.84** with a **success rate of only 2.0%** ($\Delta = +376.16$).
   * *Mechanistic Diagnosis*: DE found a hypersensitive weight vector that worked exceptionally well for the pair of initial angles drawn in generation 29, but was unstable under the continuous distribution of initial states.
2. **The Perfect Robustness of CMA-ES**:
   * **CMA-ES** reached **500.00 $\pm$ 0.00** over the 100 out-of-sample episodes, with a **standard deviation of strictly 0.00** and **100% success**.
   * By modeling the curvature of the joint distribution through covariance adaptation $\Sigma$, CMA-ES found a broad attractor (*flat minimum*) in parameter space, immune to fluctuations in the initial conditions.

### 6.2 Generalization of Symbolic Programs (Family 2)
* Both **LinearGP** and **CartesianGP** achieved a **91.0% success rate** on CartPole (mean return $\approx 487.5$, $\text{IQR} = 0.0$).
* On Acrobot, LinearGP and CartesianGP reached **success rates of 87% and 82%**, outperforming neural algorithms such as PBIL (55%) and DE (82%).
* This demonstrates that **controllers expressed as graphs of discrete instructions have excellent viability and out-of-sample generalization**, producing compact decision surfaces that do not suffer from the overfitting typical of high-dimensional continuous vectors.

### 6.3 Gradient Collapse vs. EDAs on MountainCarContinuous
* **OpenAI-ES (Score: −0.00, SNR: 1.02, Success: 0.0%)**:
  The gradient estimator collapsed entirely because a sparse reward signal provides no gradient ($\nabla_\theta \mathbb{E}[f] = \mathbf{0}$).
* **CMA-ES and PBIL (Success: 100.0%, SNR: 91.0 and 81.3)**:
  Both EDAs solved the environment with deterministic asymptotic stability ($\text{SEM} \approx 0.11$, $\text{Std} \le 1.15$), confirming the superiority of second-order distribution-based methods for *exploration valley* problems.

### 6.4 The Stochastic Sensitivity of the Inverted Pendulum
* On **Pendulum-v1**, every method exhibited a large optimism gap ($\Delta > 500$), reflecting the high sensitivity of the initial angle $\theta_0 \sim \mathcal{U}[-\pi, \pi]$.
* The best algorithm in testing was **PBIL** ($\bar{R} = -804.3$, median = $-387.3$, success = $29\%$).
* For nonlinear continuous control with chaotic dynamics around the bottom pole, training with only $N_{\text{rollouts}} = 2$ is clearly insufficient to cover the support of the state space, requiring a training protocol with at least $N=8$ to $16$ rollouts.

---

## 7. Generated Visualizations

All figures are rendered in a high-resolution academic dark-mode style and saved under `results/`:

### 7.1 Out-of-Sample Statistical Validation (100 Episodes)
* **`results/validation_100_ci95_bars.png`**: 4-panel comparison chart showing the mean test return ($\bar{R}_{\text{test}}$), error bars for the **95% bootstrap confidence interval**, and **Signal-to-Noise Ratio (SNR)** labels per family.
* **`results/optimism_gap_analysis.png`**: Quantitative analysis of the **Optimism Gap (*Winner's Curse*)** ($\Delta = f_{\text{train}} - R_{\text{test}}$), highlighting severely overfit policies in red and robust policies in green.

### 7.2 Training Curves and Evolutionary Dynamics
* **`results/all_families_curves.png`**: Learning curves comparing the 3 families on the discrete environments.
* **`results/all_families_bar.png`**: Comparative performance of all 7 algorithms grouped by family.
* **`results/learning_curves.png`**: Learning curves of Families 1 and 3 on the 4 environments.
* **`results/family_comparison.png`**: Final fitness comparison on the 4 environments.
* **`results/time_vs_performance.png`**: Scatter plot relating computational cost (seconds) to the score achieved.

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
evolutionary-estrategies/
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
├── generate_final_plots.py   # Consolidated visual compilation of training runs
├── results/                  # Raw empirical data and figures:
│   ├── validation_100_results.json
│   ├── validation_100_summary.csv
│   ├── validation_100_ci95_bars.png
│   ├── optimism_gap_analysis.png
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

# 3. (Optional) Run the separate training benchmark
python run_benchmark.py --quick --no-gp
python run_gp_only.py
python generate_final_plots.py
```

---

## 11. References

1. **Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A. C., & Bellemare, M.** (2021). Deep reinforcement learning at the edge of the statistical precipice. *Advances in Neural Information Processing Systems (NeurIPS)*, 34, 29304–29320.
2. **Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., & Meger, D.** (2018). Deep reinforcement learning that matters. *AAAI Conference on Artificial Intelligence*, 32(1).
3. **Machado, M. C., et al.** (2018). Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents. *Journal of Artificial Intelligence Research*, 61, 523–562.
4. **Goldberg, D. E.** (1989). *Genetic Algorithms in Search, Optimization and Machine Learning*. Addison-Wesley.
5. **Deb, K., & Agrawal, R. B.** (1995). Simulated binary crossover for continuous search space. *Complex Systems*, 9(2), 115–148.
6. **Storn, R., & Price, K.** (1997). Differential Evolution – A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces. *Journal of Global Optimization*, 11(4), 341–359.
7. **Salimans, T., Ho, J., Chen, X., Sidor, S., & Sutskever, I.** (2017). Evolution Strategies as a Scalable Alternative to Reinforcement Learning. *arXiv:1703.03864*.
8. **Brameier, M. F., & Banzhaf, W.** (2007). *Linear Genetic Programming*. Springer Science & Business Media.
9. **Miller, J. F., & Thomson, P.** (2000). Cartesian Genetic Programming. *European Conference on Genetic Programming (EuroGP)*, 121–132.
10. **Hansen, N., & Ostermeier, A.** (2001). Completely derandomized self-adaptation in evolution strategies. *Evolutionary Computation*, 9(2), 159–195.
11. **Hansen, N.** (2016). The CMA Evolution Strategy: A Tutorial. *arXiv:1604.00772*.
12. **Baluja, S.** (1994). Population-Based Incremental Learning. *Technical Report CMU-CS-94-163*, Carnegie Mellon University.
13. **Larrañaga, P., & Lozano, J. A.** (2002). *Estimation of Distribution Algorithms: A New Tool for Evolutionary Computation*. Springer.
14. **Lange, R. T.** (2022). Gymnax: Standard Model Environments in JAX. *GitHub Repository*.

