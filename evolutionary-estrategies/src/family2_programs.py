"""
Família 2 — Programas (Program Synthesis / Genetic Programming)

Dois algoritmos que evoluem programas (sequências de instruções) em vez de
vetores de parâmetros reais:

  1. LinearGP — Genetic Programming Linear (Brameier & Banzhaf 2007)
     Cada indivíduo é uma sequência de instrções do tipo:
         r[dst] = op(r[src1], r[src2])
     onde op ∈ {add, sub, mul, tanh, sin, cos, abs, noop, cond_gt, max}
     Os registradores de entrada são inicializados com a observação.
     Os registradores de saída são lidos para produzir a ação.

  2. CartesianGP — Genetic Programming Cartesiano (Miller & Thomson 2000)
     O programa é representado como um grafo acíclico dirigido (DAG)
     de largura C_cols e altura 1 (CGP clássico unidimensional).
     Cada nó tem: [op, conn1, conn2].
     Apenas os nós conectados à saída são avaliados (genes neutros ignorados).

Ambos os algoritmos usam GA clássico (seleção por torneio + mutação) como
mecanismo evolutivo, pois crossover estruturado em programas é delicado.

IMPORTANTE: Para integração com gymnax, o "phenotype" é sempre um vetor
de parâmetros reais que controla a ação — os programas evoluem funções que
mapeiam observações em ações, mas internamente usam instruções discretas.
A aptidão é avaliada diretamente nos ambientes gymnax.

Interface:
    init(rng)     -> state
    ask(state, rng) -> (programs [pop_size, prog_len, ...], state)
    tell(state, fitness) -> state
    get_policy_fn(program) -> Callable[[obs], action]
    best_program(state)    -> program (array)
"""
from __future__ import annotations
from typing import NamedTuple, Callable
import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Operações disponíveis para os programas (operadores primitivos)
# ---------------------------------------------------------------------------

# IDs de operadores (inteiros)
OP_ADD  = 0
OP_SUB  = 1
OP_MUL  = 2
OP_TANH = 3
OP_SIN  = 4
OP_COS  = 5
OP_ABS  = 6
OP_NOOP = 7     # identidade: r[dst] = r[src1]
OP_MAX  = 8
OP_MIN  = 9
N_OPS   = 10

def apply_op(op_id: int, a: float, b: float) -> float:
    """Aplica operador binário/unário."""
    ops = [
        a + b, a - b, a * b, np.tanh(a), np.sin(a), np.cos(a),
        np.abs(a), a, np.maximum(a, b), np.minimum(a, b)
    ]
    return ops[int(op_id)]


# ===========================================================================
# 1. Linear GP (LGP)
# ===========================================================================

class LGPState(NamedTuple):
    programs:     jnp.ndarray  # [pop_size, prog_len, 4]  (op, dst, src1, src2)
    fitness:      jnp.ndarray  # [pop_size]
    best_program: jnp.ndarray  # [prog_len, 4]
    best_fitness: jnp.ndarray
    generation:   jnp.ndarray

class LinearGP:
    """
    Genetic Programming Linear (Brameier & Banzhaf, 2007).

    Representação de instrução: (op, dst, src1, src2) — 4 inteiros.
    - op   : operador em {0..N_OPS-1}
    - dst  : registrador destino em {0..n_regs-1}
    - src1, src2: registradores fonte em {0..n_regs-1}

    Os primeiros obs_dim registradores são inicializados com a observação.
    Os últimos act_dim registradores são lidos como saída.

    Evolução: torneio binário + mutação pontual de instruções.
    Crossover: single-point no nível de instrução.

    Referência:
        Brameier, M. F., & Banzhaf, W. (2007). Linear genetic programming.
        Springer Science & Business Media.
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 pop_size: int = 64,
                 prog_len: int = 64,       # número de instruções por programa
                 n_extra_regs: int = 16,   # registradores auxiliares
                 mut_rate: float = 0.15,
                 cx_prob: float = 0.7):
        self.obs_dim      = obs_dim
        self.act_dim      = act_dim
        self.pop_size     = pop_size
        self.prog_len     = prog_len
        self.n_regs       = obs_dim + n_extra_regs + act_dim
        self.n_extra_regs = n_extra_regs
        self.mut_rate     = mut_rate
        self.cx_prob      = cx_prob
        # Índices dos registradores de saída
        self.out_start    = obs_dim + n_extra_regs

    def _rand_program(self, rng: jax.Array) -> np.ndarray:
        """Gera um programa aleatório [prog_len, 4] como ndarray NumPy."""
        ops  = np.random.randint(0, N_OPS,     self.prog_len)
        dst  = np.random.randint(0, self.n_regs, self.prog_len)
        src1 = np.random.randint(0, self.n_regs, self.prog_len)
        src2 = np.random.randint(0, self.n_regs, self.prog_len)
        return np.stack([ops, dst, src1, src2], axis=1).astype(np.int32)

    def init(self, rng: jax.Array, **kwargs) -> LGPState:
        # Programas como arrays NumPy (execução interpretada)
        np.random.seed(int(jax.random.randint(rng, (), 0, 2**30, dtype=jnp.int32)))
        programs = np.stack([self._rand_program(rng) for _ in range(self.pop_size)])
        return LGPState(
            programs     = jnp.array(programs),
            fitness      = jnp.full(self.pop_size, -jnp.inf),
            best_program = jnp.zeros((self.prog_len, 4), dtype=jnp.int32),
            best_fitness = jnp.array(-jnp.inf),
            generation   = jnp.array(0),
        )

    def execute_program(self, program: np.ndarray, obs: np.ndarray) -> np.ndarray:
        """
        Executa um programa LGP dado uma observação.
        Retorna os registradores de saída.
        """
        regs = np.zeros(self.n_regs, dtype=np.float32)
        regs[:self.obs_dim] = obs[:self.obs_dim]
        regs = np.clip(regs, -10.0, 10.0)
        for instr in program:
            op, dst, s1, s2 = int(instr[0]), int(instr[1]), int(instr[2]), int(instr[3])
            result = apply_op(op, float(regs[s1]), float(regs[s2]))
            result = np.clip(result, -100.0, 100.0)
            if not np.isnan(result) and not np.isinf(result):
                regs[dst] = result
        return regs[self.out_start: self.out_start + self.act_dim]

    def get_policy_fn(self, program: np.ndarray) -> Callable:
        """Retorna função política: obs -> ação (discreta ou contínua)."""
        def policy(obs: np.ndarray) -> np.ndarray:
            return self.execute_program(program, obs)
        return policy

    def _mutate_program(self, prog: np.ndarray) -> np.ndarray:
        """Mutação pontual: cada instrução muta com probabilidade mut_rate."""
        prog = prog.copy()
        for i in range(len(prog)):
            if np.random.random() < self.mut_rate:
                gene = np.random.randint(0, 4)  # qual gene da instrução mutar
                if gene == 0:
                    prog[i, 0] = np.random.randint(0, N_OPS)
                elif gene == 1:
                    prog[i, 1] = np.random.randint(0, self.n_regs)
                elif gene == 2:
                    prog[i, 2] = np.random.randint(0, self.n_regs)
                else:
                    prog[i, 3] = np.random.randint(0, self.n_regs)
        return prog

    def _crossover(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Single-point crossover entre dois programas."""
        if np.random.random() < self.cx_prob:
            pt = np.random.randint(1, self.prog_len)
            return np.concatenate([a[:pt], b[pt:]], axis=0)
        return a.copy()

    def tell(self, state: LGPState, fitness: jnp.ndarray) -> LGPState:
        """Seleção por torneio + reprodução + mutação."""
        fitness_np = np.array(fitness)
        programs   = np.array(state.programs)
        pop_size   = self.pop_size

        # Elite
        best_idx   = int(np.argmax(fitness_np))
        best_fit   = float(fitness_np[best_idx])
        if best_fit > float(state.best_fitness):
            best_prog = programs[best_idx].copy()
            best_fit_ = jnp.array(best_fit)
        else:
            best_prog = np.array(state.best_program)
            best_fit_ = state.best_fitness

        # Torneio binário
        def tournament(k=2):
            idxs = np.random.choice(pop_size, k, replace=False)
            return idxs[np.argmax(fitness_np[idxs])]

        new_programs = []
        for _ in range(pop_size - 1):
            pa = programs[tournament()]
            pb = programs[tournament()]
            child = self._crossover(pa, pb)
            child = self._mutate_program(child)
            new_programs.append(child)
        new_programs.append(best_prog)  # elitismo

        return LGPState(
            programs     = jnp.array(np.stack(new_programs)),
            fitness      = fitness,
            best_program = jnp.array(best_prog),
            best_fitness = best_fit_,
            generation   = state.generation + 1,
        )

    def best_program(self, state: LGPState) -> np.ndarray:
        return np.array(state.best_program)


# ===========================================================================
# 2. Cartesian GP (CGP)
# ===========================================================================

class CGPState(NamedTuple):
    programs:     jnp.ndarray  # [pop_size, n_nodes+act_dim, 3]  (op, c1, c2) por nó; últimos act_dim são índices de saída
    fitness:      jnp.ndarray
    best_program: jnp.ndarray
    best_fitness: jnp.ndarray
    generation:   jnp.ndarray

class CartesianGP:
    """
    Cartesian Genetic Programming (Miller & Thomson 2000).

    Grafo DAG de C_cols colunas (1 linha). Cada nó computacional tem
    codificação [op, conn1, conn2] onde conn ∈ {0..obs_dim+col-1}
    (só pode conectar a nós anteriores — garantia de aciclicidade).

    Os últimos act_dim genes codificam os índices de saída.

    Evolução: (1+λ) ES — um pai gera λ filhos mutados; sobrevive o melhor.

    Referência:
        Miller, J. F., & Thomson, P. (2000). Cartesian genetic programming.
        In European Conference on Genetic Programming (pp. 121–132). Springer.
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 pop_size: int = 8,   # (1+λ): 1 pai + 7 filhos típico
                 n_cols: int = 40,    # número de nós computacionais
                 mut_rate: float = 0.05):
        self.obs_dim  = obs_dim
        self.act_dim  = act_dim
        self.pop_size = pop_size
        self.n_cols   = n_cols
        self.mut_rate = mut_rate
        self.n_nodes  = n_cols  # 1 linha, n_cols colunas
        # Total de genes: n_nodes * 3  +  act_dim (saídas)
        self.prog_genes = n_cols * 3 + act_dim

    def _rand_program(self) -> np.ndarray:
        """Programa CGP: [n_nodes*3 + act_dim] inteiros."""
        prog = np.zeros(self.prog_genes, dtype=np.int32)
        obs = self.obs_dim
        for col in range(self.n_cols):
            max_conn = obs + col  # só conexões para trás
            prog[col*3 + 0] = np.random.randint(0, N_OPS)
            prog[col*3 + 1] = np.random.randint(0, max(1, max_conn))
            prog[col*3 + 2] = np.random.randint(0, max(1, max_conn))
        # Genes de saída: índice de qualquer nó (input ou computacional)
        for i in range(self.act_dim):
            prog[self.n_cols*3 + i] = np.random.randint(0, obs + self.n_cols)
        return prog

    def init(self, rng: jax.Array, **kwargs) -> CGPState:
        np.random.seed(int(jax.random.randint(rng, (), 0, 2**30, dtype=jnp.int32)))
        programs = np.stack([self._rand_program() for _ in range(self.pop_size)])
        return CGPState(
            programs     = jnp.array(programs),
            fitness      = jnp.full(self.pop_size, -jnp.inf),
            best_program = jnp.zeros(self.prog_genes, dtype=jnp.int32),
            best_fitness = jnp.array(-jnp.inf),
            generation   = jnp.array(0),
        )

    def execute_program(self, prog: np.ndarray, obs: np.ndarray) -> np.ndarray:
        """Avalia o grafo CGP para uma observação."""
        obs = obs.astype(np.float32)
        # Buffer: inputs primeiro, depois nós computacionais
        buf = np.zeros(self.obs_dim + self.n_cols, dtype=np.float32)
        buf[:self.obs_dim] = np.clip(obs, -10.0, 10.0)

        for col in range(self.n_cols):
            op   = int(prog[col*3 + 0])
            c1   = int(prog[col*3 + 1])
            c2   = int(prog[col*3 + 2])
            c1   = min(c1, self.obs_dim + col - 1) if self.obs_dim + col > 0 else 0
            c2   = min(c2, self.obs_dim + col - 1) if self.obs_dim + col > 0 else 0
            a, b = float(buf[c1]), float(buf[c2])
            result = apply_op(op, a, b)
            result = np.clip(result, -100.0, 100.0)
            if not np.isnan(result) and not np.isinf(result):
                buf[self.obs_dim + col] = result

        # Lê saídas
        out_idxs = prog[self.n_cols*3: self.n_cols*3 + self.act_dim]
        out_idxs = np.clip(out_idxs, 0, self.obs_dim + self.n_cols - 1)
        return buf[out_idxs]

    def get_policy_fn(self, prog: np.ndarray) -> Callable:
        def policy(obs: np.ndarray) -> np.ndarray:
            return self.execute_program(prog, obs)
        return policy

    def _mutate(self, prog: np.ndarray) -> np.ndarray:
        """Mutação pontual respeitando a restrição de conectividade."""
        prog = prog.copy()
        obs = self.obs_dim
        for col in range(self.n_cols):
            if np.random.random() < self.mut_rate:
                gene = np.random.randint(0, 3)
                max_conn = max(1, obs + col)
                if gene == 0:
                    prog[col*3 + 0] = np.random.randint(0, N_OPS)
                else:
                    prog[col*3 + gene] = np.random.randint(0, max_conn)
        for i in range(self.act_dim):
            if np.random.random() < self.mut_rate:
                prog[self.n_cols*3 + i] = np.random.randint(0, obs + self.n_cols)
        return prog

    def tell(self, state: CGPState, fitness: jnp.ndarray) -> CGPState:
        """(1+λ) ES: o melhor sobrevive e gera pop_size-1 filhos."""
        fitness_np = np.array(fitness)
        programs   = np.array(state.programs)

        best_idx = int(np.argmax(fitness_np))
        best_fit = float(fitness_np[best_idx])
        parent   = programs[best_idx].copy()

        if best_fit > float(state.best_fitness):
            best_prog = parent.copy()
            best_fit_ = jnp.array(best_fit)
        else:
            best_prog = np.array(state.best_program)
            best_fit_ = state.best_fitness

        new_programs = [parent]
        for _ in range(self.pop_size - 1):
            new_programs.append(self._mutate(parent))

        return CGPState(
            programs     = jnp.array(np.stack(new_programs)),
            fitness      = fitness,
            best_program = jnp.array(best_prog),
            best_fitness = best_fit_,
            generation   = state.generation + 1,
        )

    def best_program(self, state: CGPState) -> np.ndarray:
        return np.array(state.best_program)
