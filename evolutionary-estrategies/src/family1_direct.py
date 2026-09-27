"""
Família 1 — Soluções Diretas (Direct Encoding / Parameter Perturbation)

Algoritmos que evoluem diretamente vetores de soluções/parâmetros sem modelar ou aprender
uma distribuição de probabilidade explícita sobre o espaço de busca:
  1. SimpleGA  — Algoritmo Genético clássico com seleção por torneio, SBX crossover e mutação gaussiana
  2. DE        — Differential Evolution (rand/1/bin, estratégia de Price & Storn 1997)
  3. OpenAI-ES — Natural Evolution Strategies com perturbações gaussianas isotrópicas de variância
                 fixa (Salimans et al. 2017). Não aprende distribuição nem covariância: utiliza o
                 ruído fixo como estimador estocástico de gradiente (NES/finite-difference) para
                 atualizar diretamente o vetor de parâmetros θ por ascensão de gradiente.

Todos seguem a interface comum:
    init(rng, n_params) -> state
    ask(state, rng)     -> (population [pop_size, n_params], state)
    tell(state, ...)    -> state
    best_params(state)  -> [n_params]
"""
from __future__ import annotations
from typing import NamedTuple
import jax
import jax.numpy as jnp


# ===========================================================================
# 1. SimpleGA — Algoritmo Genético Clássico
# ===========================================================================

class GAState(NamedTuple):
    population:   jnp.ndarray   # [pop_size, n_params]
    fitness:      jnp.ndarray   # [pop_size]
    best_params:  jnp.ndarray   # [n_params]
    best_fitness: jnp.ndarray   # escalar
    generation:   jnp.ndarray   # escalar int

class SimpleGA:
    """
    Algoritmo Genético com:
      - Seleção por torneio binário (tournament_size=2)
      - Crossover SBX (Simulated Binary Crossover, Deb & Agrawal 1995)
      - Mutação gaussiana adaptativa
      - Elitismo (1 elite preservada)

    Referência: Goldberg, D. E. (1989). Genetic algorithms in search,
    optimization and machine learning. Addison-Wesley.
    """

    def __init__(self, pop_size: int = 64, sigma_init: float = 0.5,
                 sigma_decay: float = 0.999, sigma_limit: float = 0.01,
                 cx_eta: float = 2.0, cx_prob: float = 0.9):
        self.pop_size   = pop_size
        self.sigma_init = sigma_init
        self.sigma_decay= sigma_decay
        self.sigma_limit= sigma_limit
        self.cx_eta     = cx_eta     # parâmetro de distribuição SBX
        self.cx_prob    = cx_prob    # probabilidade de crossover

    def init(self, rng: jax.Array, n_params: int) -> GAState:
        pop = jax.random.normal(rng, (self.pop_size, n_params)) * self.sigma_init
        return GAState(
            population   = pop,
            fitness      = jnp.full(self.pop_size, -jnp.inf),
            best_params  = jnp.zeros(n_params),
            best_fitness = jnp.array(-jnp.inf),
            generation   = jnp.array(0),
        )

    def ask(self, state: GAState, rng: jax.Array):
        return state.population, state

    def tell(self, state: GAState, fitness: jnp.ndarray, rng: jax.Array) -> GAState:
        pop_size, n_params = state.population.shape
        sigma = jnp.maximum(
            self.sigma_init * (self.sigma_decay ** state.generation.astype(float)),
            self.sigma_limit
        )

        # Atualiza elite
        best_idx     = jnp.argmax(fitness)
        best_fitness = jnp.where(fitness[best_idx] > state.best_fitness,
                                 fitness[best_idx], state.best_fitness)
        best_params  = jnp.where(fitness[best_idx] > state.best_fitness,
                                 state.population[best_idx], state.best_params)

        rng, r1, r2, r3, r4 = jax.random.split(rng, 5)

        # Seleção por torneio binário para criar pop_size pais (a e b)
        idx_a1 = jax.random.randint(r1, (pop_size,), 0, pop_size)
        idx_a2 = jax.random.randint(r2, (pop_size,), 0, pop_size)
        idx_b1 = jax.random.randint(r3, (pop_size,), 0, pop_size)
        idx_b2 = jax.random.randint(r4, (pop_size,), 0, pop_size)

        parent_a = jnp.where(
            (fitness[idx_a1] >= fitness[idx_a2])[:, None],
            state.population[idx_a1], state.population[idx_a2]
        )
        parent_b = jnp.where(
            (fitness[idx_b1] >= fitness[idx_b2])[:, None],
            state.population[idx_b1], state.population[idx_b2]
        )

        # SBX Crossover
        rng, r_cx, r_u, r_beta = jax.random.split(rng, 4)
        do_cx    = jax.random.uniform(r_cx, (pop_size,)) < self.cx_prob
        u        = jax.random.uniform(r_u,  (pop_size, n_params))
        eta      = self.cx_eta
        beta     = jnp.where(u <= 0.5,
                             (2 * u) ** (1 / (eta + 1)),
                             (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
        child    = 0.5 * ((1 + beta) * parent_a + (1 - beta) * parent_b)
        offspring = jnp.where(do_cx[:, None], child, parent_a)

        # Mutação gaussiana
        rng, r_mut = jax.random.split(rng)
        noise     = jax.random.normal(r_mut, offspring.shape) * sigma
        offspring = offspring + noise

        # Elitismo: substitui o pior por best_params
        worst_idx  = jnp.argmin(fitness)
        offspring  = offspring.at[worst_idx].set(best_params)

        return GAState(
            population   = offspring,
            fitness      = fitness,
            best_params  = best_params,
            best_fitness = best_fitness,
            generation   = state.generation + 1,
        )

    def best_params(self, state: GAState) -> jnp.ndarray:
        return state.best_params


# ===========================================================================
# 2. DE — Differential Evolution
# ===========================================================================

class DEState(NamedTuple):
    population:   jnp.ndarray
    fitness:      jnp.ndarray
    best_params:  jnp.ndarray
    best_fitness: jnp.ndarray
    generation:   jnp.ndarray

class DE:
    """
    Differential Evolution — estratégia rand/1/bin (Price, Storn & Lampinen 2005).

    Mutação:    v_i = x_r1 + F * (x_r2 - x_r3)
    Crossover:  u_ij = v_ij se U(0,1) < CR, senão x_ij
    Seleção:    greedy one-to-one (preserva indivíduo diretamente se trial for pior)

    Parâmetros padrão: F=0.8, CR=0.9 (recomendados em Storn & Price 1997).
    """

    def __init__(self, pop_size: int = 64, F: float = 0.8, CR: float = 0.9,
                 init_scale: float = 0.5):
        self.pop_size   = pop_size
        self.F          = F
        self.CR         = CR
        self.init_scale = init_scale

    def init(self, rng: jax.Array, n_params: int) -> DEState:
        pop = jax.random.normal(rng, (self.pop_size, n_params)) * self.init_scale
        return DEState(
            population   = pop,
            fitness      = jnp.full(self.pop_size, -jnp.inf),
            best_params  = jnp.zeros(n_params),
            best_fitness = jnp.array(-jnp.inf),
            generation   = jnp.array(0),
        )

    def ask(self, state: DEState, rng: jax.Array):
        """Gera população de vetores trial (mutação + crossover)."""
        pop_size, n_params = state.population.shape
        rng, r1, r2, r3, r4 = jax.random.split(rng, 5)

        # Amostragem de r1, r2, r3 distintos (aproximação via permutação)
        idx_r1 = jax.random.randint(r1, (pop_size,), 0, pop_size)
        idx_r2 = jax.random.randint(r2, (pop_size,), 0, pop_size)
        idx_r3 = jax.random.randint(r3, (pop_size,), 0, pop_size)

        x_r1 = state.population[idx_r1]
        x_r2 = state.population[idx_r2]
        x_r3 = state.population[idx_r3]

        # Vetor mutante
        v = x_r1 + self.F * (x_r2 - x_r3)

        # Crossover binomial
        mask  = jax.random.uniform(r4, (pop_size, n_params)) < self.CR
        # Garante ao menos 1 gene do mutante por indivíduo
        j_rand = jax.random.randint(rng, (pop_size,), 0, n_params)
        mask   = mask.at[jnp.arange(pop_size), j_rand].set(True)

        trial = jnp.where(mask, v, state.population)
        return trial, state

    def tell(self, state: DEState, trial_fitness: jnp.ndarray,
             trial: jnp.ndarray) -> DEState:
        """Seleção greedy: mantém trial se melhor que o pai."""
        improved = trial_fitness > state.fitness
        new_pop  = jnp.where(improved[:, None], trial, state.population)
        new_fit  = jnp.where(improved, trial_fitness, state.fitness)

        best_idx     = jnp.argmax(new_fit)
        best_fitness = new_fit[best_idx]
        best_params  = new_pop[best_idx]

        return DEState(
            population   = new_pop,
            fitness      = new_fit,
            best_params  = best_params,
            best_fitness = best_fitness,
            generation   = state.generation + 1,
        )

    def best_params(self, state: DEState) -> jnp.ndarray:
        return state.best_params


# ===========================================================================
# 3. OpenAI-ES — Natural Evolution Strategies (Solução Direta / Perturbação de Parâmetros)
# ===========================================================================

class OpenAIESState(NamedTuple):
    mean:         jnp.ndarray   # [n_params] centróide / vetor de parâmetros θ
    sigma:        jnp.ndarray   # escalar step-size atual (isotrópico fixo com decaimento)
    noise:        jnp.ndarray   # [pop_size//2, n_params] perturbações antitéticas
    best_params:  jnp.ndarray   # [n_params]
    best_fitness: jnp.ndarray   # escalar
    generation:   jnp.ndarray

class OpenAIES:
    """
    OpenAI-ES (Salimans et al. 2017).
    
    Classificação taxonômica: Família 1 (Soluções Diretas).
    Diferente de EDAs (como CMA-ES ou PBIL), o OpenAI-ES não aprende nem adapta
    uma matriz de covariância ou distribuição probabilística. Ele perturba um
    único vetor de parâmetros θ com ruído gaussiano isotrópico de escala fixa σ,
    calcula um estimador Monte Carlo de gradiente por diferenças finitas/score function:
        ∇_θ E[f] ≈ 1 / (σ N) Σ f(θ + σ ε_i) ε_i
    e atualiza diretamente θ via gradiente ascendente (SGD/Adam). O vetor de
    parâmetros em si é a solução direta que evolui no espaço de busca.

    Recursos implementados:
      - Perturbação antitética (mirrored sampling) para redução de variância
      - Normalização de fitness (rank-based fitness shaping)
      - Decaimento programado de learning rate e taxa de perturbação
    """

    def __init__(self, pop_size: int = 64, sigma: float = 0.05,
                 lr: float = 0.01, sigma_decay: float = 0.999,
                 sigma_limit: float = 0.001, lr_decay: float = 0.9999):
        assert pop_size % 2 == 0, "pop_size deve ser par (perturbação antitética)"
        self.pop_size    = pop_size
        self.sigma_init  = sigma
        self.lr          = lr
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit
        self.lr_decay    = lr_decay

    def init(self, rng: jax.Array, n_params: int) -> OpenAIESState:
        return OpenAIESState(
            mean         = jax.random.normal(rng, (n_params,)) * 0.1,
            sigma        = jnp.array(self.sigma_init),
            noise        = jnp.zeros((self.pop_size // 2, n_params)),
            best_params  = jnp.zeros(n_params),
            best_fitness = jnp.array(-jnp.inf),
            generation   = jnp.array(0),
        )

    def ask(self, state: OpenAIESState, rng: jax.Array):
        half = self.pop_size // 2
        n    = state.mean.shape[0]
        noise = jax.random.normal(rng, (half, n))
        # Perturbações antitéticas: [+ε₁, −ε₁, +ε₂, −ε₂, ...]
        pop_plus  = state.mean[None, :] + state.sigma * noise
        pop_minus = state.mean[None, :] - state.sigma * noise
        population = jnp.concatenate([pop_plus, pop_minus], axis=0)
        new_state  = OpenAIESState(
            mean=state.mean, sigma=state.sigma, noise=noise,
            best_params=state.best_params, best_fitness=state.best_fitness,
            generation=state.generation,
        )
        return population, new_state

    def tell(self, state: OpenAIESState, fitness: jnp.ndarray) -> OpenAIESState:
        half  = self.pop_size // 2
        noise = state.noise  # [half, n_params]

        # Fitness shaping: rank-based normalization (0-centrado)
        ranks    = jnp.argsort(jnp.argsort(fitness))  # rank 0..pop-1
        shaped   = (ranks.astype(float) / (self.pop_size - 1)) - 0.5

        # Gradiente NES: (1/n*σ) Σ shaped_i * ε_i
        # Para perturbações antitéticas: F(+ε) e F(−ε) → diferença
        f_plus  = shaped[:half]
        f_minus = shaped[half:]
        gradient = ((f_plus - f_minus)[:, None] * noise).mean(axis=0)

        # Learning rate com decaimento
        lr    = self.lr * (self.lr_decay ** state.generation.astype(float))
        lr    = jnp.maximum(lr, 1e-5)

        new_mean  = state.mean + lr / state.sigma * gradient
        new_sigma = jnp.maximum(
            state.sigma * (self.sigma_decay ** 1.0),
            self.sigma_limit
        )

        # Elite global
        best_idx  = jnp.argmax(fitness)
        pop_plus  = state.mean[None, :] + state.sigma * state.noise
        pop_minus = state.mean[None, :] - state.sigma * state.noise
        population = jnp.concatenate([pop_plus, pop_minus], axis=0)
        best_cand  = population[best_idx]
        best_fitness = jnp.where(fitness[best_idx] > state.best_fitness,
                                 fitness[best_idx], state.best_fitness)
        best_params  = jnp.where(fitness[best_idx] > state.best_fitness,
                                 best_cand, state.best_params)

        return OpenAIESState(
            mean         = new_mean,
            sigma        = new_sigma,
            noise        = state.noise,
            best_params  = best_params,
            best_fitness = best_fitness,
            generation   = state.generation + 1,
        )

    def best_params(self, state: OpenAIESState) -> jnp.ndarray:
        return state.best_params
