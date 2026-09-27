"""
Família 3 — Modelos / EDA (Estimation of Distribution Algorithms)

Algoritmos que aprendem e adaptam explicitamente distribuições de probabilidade sobre o
espaço de busca. Diferente das abordagens de soluções diretas (que mutam e cruzam indivíduos-pais
preservados ao longo das gerações), em EDAs os indivíduos são apenas amostras estocásticas
temporárias geradas pela distribuição atual e descartadas integralmente a cada geração.
O que evolui é a própria distribuição de probabilidade:

  1. CMA-ES — Covariance Matrix Adaptation Evolution Strategy (Hansen & Ostermeier 2001).
              Apesar do nome histórico de "Evolution Strategy", CMA-ES possui a assinatura
              estrutural canônica de um EDA contínuo de segunda ordem: mantém e adapta um vetor
              de média μ e uma matriz de covariância completa Σ (via rank-1 update com caminho
              de evolução p_c e rank-μ update ponderado), além do controle de step-size σ (CSA
              com caminho p_σ). A cada geração, toda a população é amostrada do zero a partir
              de N(μ, σ² Σ) e descartada após a avaliação. É reconhecido na literatura de EDA
              como um dos algoritmos de distribuição estimada mais sofisticados existentes.

  2. PBIL   — Population-Based Incremental Learning (Baluja 1994; Sebag & Ducoulombier 1998).
              EDA com modelo de distribuição marginal independente (univariada). Mantém um
              vetor de médias μ e desvios padrão σ por gene, atualizados incrementalmente
              a partir das estatísticas das top-k amostras mais bem sucedidas.

Interface comum:
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
# 1. CMA-ES — Covariance Matrix Adaptation Evolution Strategy (EDA de 2ª Ordem)
# ===========================================================================

class CMAState(NamedTuple):
    mean:         jnp.ndarray  # [n_params] vetor de média μ
    sigma:        jnp.ndarray  # escalar step-size global σ
    C:            jnp.ndarray  # [n_params, n_params] matriz de covariância adaptativa Σ
    p_sigma:      jnp.ndarray  # [n_params] caminho de evolução conjugado (step-size)
    p_c:          jnp.ndarray  # [n_params] caminho de evolução anisotrópico (covariância)
    eigenvalues:  jnp.ndarray  # [n_params] autovalores de C (eixos principais)
    eigenvectors: jnp.ndarray  # [n_params, n_params] autovetores de C (matriz de rotação B)
    best_params:  jnp.ndarray  # [n_params] melhor solução global observada
    best_fitness: jnp.ndarray  # escalar
    generation:   jnp.ndarray
    count_eval:   jnp.ndarray

class CMAES:
    """
    CMA-ES completo com adaptação de matriz de covariância e controle de passo acumulativo (CSA).

    Referência:
        Hansen, N., & Ostermeier, A. (2001). Completely derandomized self-adaptation
        in evolution strategies. Evolutionary Computation, 9(2), 159-195.

        Hansen, N. (2016). The CMA Evolution Strategy: A Tutorial. arXiv:1604.00772.

        Larrañaga, P., & Lozano, J. A. (2002). Estimation of Distribution Algorithms:
        A New Tool for Evolutionary Computation. Springer.
    """

    def __init__(self, pop_size: int | None = None, sigma0: float = 0.5):
        self.pop_size_override = pop_size
        self.sigma0 = sigma0

    def _hyperparams(self, n: int):
        """Calcula hiperparâmetros canônicos dependentes da dimensão n (Hansen 2016, §3)."""
        lam   = self.pop_size_override or (4 + int(3 * jnp.log(n)))
        mu    = lam // 2
        w_raw = jnp.log(mu + 0.5) - jnp.log(jnp.arange(1, mu + 1))
        w     = w_raw / w_raw.sum()
        mueff = 1.0 / (w ** 2).sum()

        # Adaptação de sigma (CSA)
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, jnp.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma

        # Adaptação de C (CMA)
        c_c  = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1  = 2 / ((n + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

        return dict(lam=lam, mu=mu, w=w, mueff=float(mueff),
                    c_sigma=float(c_sigma), d_sigma=float(d_sigma),
                    c_c=float(c_c), c_1=float(c_1), c_mu=float(c_mu))

    def init(self, rng: jax.Array, n_params: int) -> CMAState:
        hp = self._hyperparams(n_params)
        self._hp = hp
        self._n  = n_params
        return CMAState(
            mean         = jnp.zeros(n_params),
            sigma        = jnp.array(self.sigma0),
            C            = jnp.eye(n_params),
            p_sigma      = jnp.zeros(n_params),
            p_c          = jnp.zeros(n_params),
            eigenvalues  = jnp.ones(n_params),
            eigenvectors = jnp.eye(n_params),
            best_params  = jnp.zeros(n_params),
            best_fitness = jnp.array(-jnp.inf),
            generation   = jnp.array(0),
            count_eval   = jnp.array(0),
        )

    def ask(self, state: CMAState, rng: jax.Array):
        hp  = self._hp
        lam = hp["lam"]
        n   = self._n
        # Amostragem da distribuição Gaussiana multivariada N(μ, σ² C):
        # x_k = mean + sigma * B * D * z_k, onde C = B * D² * B^T
        z   = jax.random.normal(rng, (lam, n))
        D   = jnp.sqrt(jnp.maximum(state.eigenvalues, 1e-10))
        y   = z * D[None, :]               # escala pelos autovalores (eixos elipsoidais)
        y   = (state.eigenvectors @ y.T).T  # rotaciona pelos autovetores
        pop = state.mean[None, :] + state.sigma * y
        return pop, state

    def tell(self, state: CMAState, population: jnp.ndarray,
             fitness: jnp.ndarray) -> CMAState:
        hp  = self._hp
        n   = self._n
        mu  = hp["mu"]
        w   = hp["w"]

        # Atualiza melhor indivíduo observado
        best_idx     = jnp.argmax(fitness)
        best_fitness = jnp.where(fitness[best_idx] > state.best_fitness,
                                 fitness[best_idx], state.best_fitness)
        best_params  = jnp.where(fitness[best_idx] > state.best_fitness,
                                 population[best_idx], state.best_params)

        # Ordena por fitness (descendente) e seleciona top-mu
        order    = jnp.argsort(-fitness)
        selected = population[order[:mu]]  # [mu, n]

        # Atualização da média da distribuição (recombinação intermediária ponderada)
        new_mean = (w[:, None] * selected).sum(axis=0)
        step     = (new_mean - state.mean) / state.sigma

        # Adaptação cumulativa do comprimento de passo (CSA — Cumulative Step-length Adaptation)
        c_sigma = hp["c_sigma"]
        mueff   = hp["mueff"]
        chi_n   = jnp.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        D_inv   = 1.0 / jnp.sqrt(jnp.maximum(state.eigenvalues, 1e-10))
        invsqrtC_step = (state.eigenvectors * D_inv[None, :]) @ state.eigenvectors.T @ step
        new_p_sigma = ((1 - c_sigma) * state.p_sigma
                       + jnp.sqrt(c_sigma * (2 - c_sigma) * mueff) * invsqrtC_step)

        # Atualização do step-size σ
        new_sigma = state.sigma * jnp.exp(
            (c_sigma / hp["d_sigma"]) * (jnp.linalg.norm(new_p_sigma) / chi_n - 1)
        )
        new_sigma = jnp.clip(new_sigma, 1e-6, 10.0)

        # Adaptação da matriz de covariância (CMA — Covariance Matrix Adaptation)
        c_c     = hp["c_c"]
        h_sigma = (jnp.linalg.norm(new_p_sigma) / jnp.sqrt(1 - (1 - c_sigma) ** (2 * (state.count_eval + 1)))
                   < (1.4 + 2 / (n + 1)) * chi_n).astype(float)
        new_p_c = ((1 - c_c) * state.p_c
                   + h_sigma * jnp.sqrt(c_c * (2 - c_c) * mueff) * step)

        c_1  = hp["c_1"]
        c_mu = hp["c_mu"]
        y_mu = (selected - state.mean[None, :]) / state.sigma  # [mu, n]
        rank_mu = (w[:, None, None] * (y_mu[:, :, None] * y_mu[:, None, :])).sum(axis=0)
        
        # Atualização combinada: memória prévia + rank-one update (p_c) + rank-mu update (amostras)
        new_C = ((1 - c_1 - c_mu) * state.C
                 + c_1 * (new_p_c[:, None] * new_p_c[None, :] + (1 - h_sigma) * c_c * (2 - c_c) * state.C)
                 + c_mu * rank_mu)
        new_C = (new_C + new_C.T) / 2  # garante simetria numérica estrita

        # Eigendecomposição da matriz de covariância C = B D² B^T
        eigenvalues, eigenvectors = jnp.linalg.eigh(new_C)
        eigenvalues  = jnp.maximum(eigenvalues, 1e-10)

        return CMAState(
            mean         = new_mean,
            sigma        = new_sigma,
            C            = new_C,
            p_sigma      = new_p_sigma,
            p_c          = new_p_c,
            eigenvalues  = eigenvalues,
            eigenvectors = eigenvectors,
            best_params  = best_params,
            best_fitness = best_fitness,
            generation   = state.generation + 1,
            count_eval   = state.count_eval + hp["lam"],
        )

    def best_params(self, state: CMAState) -> jnp.ndarray:
        return state.best_params

    @property
    def pop_size(self):
        return self.pop_size_override or 64


# ===========================================================================
# 2. PBIL Contínuo — Population-Based Incremental Learning (EDA Univariado)
# ===========================================================================

class PBILState(NamedTuple):
    mu:           jnp.ndarray   # [n_params] média da distribuição Gaussiana por parâmetro
    sigma:        jnp.ndarray   # [n_params] desvio padrão por parâmetro
    best_params:  jnp.ndarray
    best_fitness: jnp.ndarray
    generation:   jnp.ndarray

class PBIL:
    """
    Population-Based Incremental Learning para espaço contínuo.

    O PBIL original (Baluja 1994) operava em espaço binário mantendo
    um vetor de probabilidades marginais P atualizado incrementalmente:
        P ← (1−lr)*P + lr * best_solution

    Esta versão para domínio contínuo mantém e adapta um vetor de médias μ
    e desvios padrão σ por gene (distribuição Gaussiana com covariância diagonal):
        μ ← (1−lr)*μ + lr * média_ponderada(top-k soluções)
        σ ← (1−lr_σ)*σ + lr_σ * std(top-k soluções) + σ_min

    com perturbação de mutação no próprio modelo de probabilidade.

    Referências:
        Baluja, S. (1994). Population-based incremental learning.
        Technical Report CMU-CS-94-163, Carnegie Mellon University.

        Sebag, M., & Ducoulombier, A. (1998). Extending population-based
        incremental learning to continuous search spaces. PPSN V.
    """

    def __init__(self, pop_size: int = 64, lr: float = 0.1, lr_sigma: float = 0.05,
                 top_k_frac: float = 0.2, sigma_init: float = 1.0,
                 sigma_min: float = 0.01, mut_prob: float = 0.02,
                 mut_shift: float = 0.05):
        self.pop_size    = pop_size
        self.lr          = lr
        self.lr_sigma    = lr_sigma
        self.top_k       = max(1, int(pop_size * top_k_frac))
        self.sigma_init  = sigma_init
        self.sigma_min   = sigma_min
        self.mut_prob    = mut_prob    # prob. de mutação no modelo de probabilidade
        self.mut_shift   = mut_shift   # magnitude da perturbação

    def init(self, rng: jax.Array, n_params: int) -> PBILState:
        return PBILState(
            mu           = jnp.zeros(n_params),
            sigma        = jnp.full(n_params, self.sigma_init),
            best_params  = jnp.zeros(n_params),
            best_fitness = jnp.array(-jnp.inf),
            generation   = jnp.array(0),
        )

    def ask(self, state: PBILState, rng: jax.Array):
        n   = state.mu.shape[0]
        eps = jax.random.normal(rng, (self.pop_size, n))
        population = state.mu[None, :] + state.sigma[None, :] * eps
        return population, state

    def tell(self, state: PBILState, fitness: jnp.ndarray,
             population: jnp.ndarray, rng: jax.Array) -> PBILState:
        # Seleciona top-k amostras por fitness
        top_indices = jnp.argsort(-fitness)[:self.top_k]
        top_pop     = population[top_indices]           # [top_k, n]

        top_mu    = top_pop.mean(axis=0)
        top_sigma = top_pop.std(axis=0)

        # Atualização do modelo paramétrico
        new_mu    = (1 - self.lr) * state.mu + self.lr * top_mu
        new_sigma = jnp.maximum(
            (1 - self.lr_sigma) * state.sigma + self.lr_sigma * top_sigma,
            self.sigma_min
        )

        # Mutação no modelo (perturbação estocástica da distribuição)
        rng, r_mask, r_shift = jax.random.split(rng, 3)
        mask     = jax.random.uniform(r_mask, new_mu.shape) < self.mut_prob
        shift    = jax.random.normal(r_shift, new_mu.shape) * self.mut_shift
        new_mu   = jnp.where(mask, new_mu + shift, new_mu)

        # Elite global
        best_idx     = jnp.argmax(fitness)
        best_fitness = jnp.where(fitness[best_idx] > state.best_fitness,
                                 fitness[best_idx], state.best_fitness)
        best_params  = jnp.where(fitness[best_idx] > state.best_fitness,
                                 population[best_idx], state.best_params)

        return PBILState(
            mu           = new_mu,
            sigma        = new_sigma,
            best_params  = best_params,
            best_fitness = best_fitness,
            generation   = state.generation + 1,
        )

    def best_params(self, state: PBILState) -> jnp.ndarray:
        return state.best_params
