"""
Family 2 — Programs (Program Synthesis / Genetic Programming)

Two algorithms that evolve programs (sequences of instructions) instead of
vectors of real-valued parameters:

  1. LinearGP — Linear Genetic Programming (Brameier & Banzhaf 2007)
     Each individual is a sequence of instructions of the form:
         r[dst] = op(r[src1], r[src2])
     where op ∈ {add, sub, mul, tanh, sin, cos, abs, noop, cond_gt, max}
     The input registers are initialized with the observation.
     The output registers are read to produce the action.

  2. CartesianGP — Cartesian Genetic Programming (Miller & Thomson 2000)
     The program is represented as a directed acyclic graph (DAG)
     of width C_cols and height 1 (classical one-dimensional CGP).
     Each node has: [op, conn1, conn2].
     Only the nodes connected to the output are evaluated (neutral genes are ignored).

Both algorithms use a classic GA (tournament selection + mutation) as the
evolutionary mechanism, since structured crossover over programs is delicate.

IMPORTANT: to integrate with gymnax, the "phenotype" is always a vector of
real parameters that controls the action — the programs evolve functions that
map observations into actions, but internally they use discrete instructions.
Fitness is evaluated directly on the gymnax environments.

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
# Operations available to the programs (primitive operators)
# ---------------------------------------------------------------------------

# Operator IDs (integers)
OP_ADD  = 0
OP_SUB  = 1
OP_MUL  = 2
OP_TANH = 3
OP_SIN  = 4
OP_COS  = 5
OP_ABS  = 6
OP_NOOP = 7     # identity: r[dst] = r[src1]
OP_MAX  = 8
OP_MIN  = 9
N_OPS   = 10

def apply_op(op_id: int, a: float, b: float) -> float:
    """Applies a binary/unary operator (Python/NumPy fallback)."""
    ops = [
        a + b, a - b, a * b, np.tanh(a), np.sin(a), np.cos(a),
        np.abs(a), a, np.maximum(a, b), np.minimum(a, b)
    ]
    return ops[int(op_id)]


def apply_op_jax(op_id: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Pure JAX evaluation of primitive operations without Python branching."""
    branches = (
        lambda a, b: a + b,                     # 0: ADD
        lambda a, b: a - b,                     # 1: SUB
        lambda a, b: a * b,                     # 2: MUL
        lambda a, b: jnp.tanh(a),               # 3: TANH
        lambda a, b: jnp.sin(a),                # 4: SIN
        lambda a, b: jnp.cos(a),                # 5: COS
        lambda a, b: jnp.abs(a),                # 6: ABS
        lambda a, b: a,                         # 7: NOOP
        lambda a, b: jnp.maximum(a, b),         # 8: MAX
        lambda a, b: jnp.minimum(a, b),         # 9: MIN
    )
    res = jax.lax.switch(op_id, branches, a, b)
    res = jnp.clip(res, -100.0, 100.0)
    return jnp.where(jnp.isnan(res) | jnp.isinf(res), 0.0, res)


def forward_lgp_jax(
    program: jnp.ndarray,
    obs: jnp.ndarray,
    obs_dim: int,
    act_dim: int,
    n_regs: int,
    out_start: int,
) -> jnp.ndarray:
    """Pure JAX execution of an LGP instruction sequence via jax.lax.scan."""
    regs = jnp.zeros(n_regs, dtype=jnp.float32)
    regs = regs.at[:obs_dim].set(jnp.clip(obs[:obs_dim], -10.0, 10.0))

    def step_fn(regs, instr):
        op, dst, s1, s2 = instr[0], instr[1], instr[2], instr[3]
        a = regs[s1]
        b = regs[s2]
        res = apply_op_jax(op, a, b)
        regs = regs.at[dst].set(res)
        return regs, None

    regs, _ = jax.lax.scan(step_fn, regs, program)
    return regs[out_start : out_start + act_dim]


def forward_cgp_jax(
    prog: jnp.ndarray,
    obs: jnp.ndarray,
    obs_dim: int,
    act_dim: int,
    n_cols: int,
) -> jnp.ndarray:
    """Pure JAX execution of a Cartesian GP DAG via jax.lax.fori_loop."""
    buf = jnp.zeros(obs_dim + n_cols, dtype=jnp.float32)
    buf = buf.at[:obs_dim].set(jnp.clip(obs[:obs_dim], -10.0, 10.0))

    # Split prog into node genes [n_cols, 3] and output genes [act_dim]
    node_genes = prog[: n_cols * 3].reshape(n_cols, 3)
    out_genes = prog[n_cols * 3 : n_cols * 3 + act_dim]

    def col_fn(col, buf):
        op = node_genes[col, 0]
        c1 = node_genes[col, 1]
        c2 = node_genes[col, 2]

        max_conn = jnp.maximum(1, obs_dim + col)
        c1 = jnp.clip(c1, 0, max_conn - 1)
        c2 = jnp.clip(c2, 0, max_conn - 1)

        a = buf[c1]
        b = buf[c2]
        res = apply_op_jax(op, a, b)
        buf = buf.at[obs_dim + col].set(res)
        return buf

    buf = jax.lax.fori_loop(0, n_cols, col_fn, buf)
    out_idxs = jnp.clip(out_genes, 0, obs_dim + n_cols - 1)
    return buf[out_idxs]


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
    Linear Genetic Programming (Brameier & Banzhaf, 2007).

    Instruction representation: (op, dst, src1, src2) — 4 integers.
    - op   : operator in {0..N_OPS-1}
    - dst  : destination register in {0..n_regs-1}
    - src1, src2: source registers in {0..n_regs-1}

    The first obs_dim registers are initialized with the observation.
    The last act_dim registers are read as the output.

    Evolution: binary tournament + point mutation of instructions.
    Crossover: single-point at the instruction level.

    Reference:
        Brameier, M. F., & Banzhaf, W. (2007). Linear genetic programming.
        Springer Science & Business Media.
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 pop_size: int = 64,
                 prog_len: int = 64,       # number of instructions per program
                 n_extra_regs: int = 16,   # auxiliary registers
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
        # Indices of the output registers
        self.out_start    = obs_dim + n_extra_regs
        self.algo_type    = "lgp"

    def _rand_program(self, rng: jax.Array) -> np.ndarray:
        """Generates a random program [prog_len, 4] as a NumPy ndarray."""
        ops  = np.random.randint(0, N_OPS,     self.prog_len)
        dst  = np.random.randint(0, self.n_regs, self.prog_len)
        src1 = np.random.randint(0, self.n_regs, self.prog_len)
        src2 = np.random.randint(0, self.n_regs, self.prog_len)
        return np.stack([ops, dst, src1, src2], axis=1).astype(np.int32)

    def init(self, rng: jax.Array, **kwargs) -> LGPState:
        # Programs initialized as JAX arrays
        np.random.seed(int(jax.random.randint(rng, (), 0, 2**30, dtype=jnp.int32)))
        programs = np.stack([self._rand_program(rng) for _ in range(self.pop_size)])
        return LGPState(
            programs     = jnp.array(programs, dtype=jnp.int32),
            fitness      = jnp.full(self.pop_size, -jnp.inf),
            best_program = jnp.zeros((self.prog_len, 4), dtype=jnp.int32),
            best_fitness = jnp.array(-jnp.inf),
            generation   = jnp.array(0),
        )

    def execute_program(self, program: np.ndarray | jnp.ndarray, obs: np.ndarray | jnp.ndarray) -> np.ndarray:
        """
        Executes an LGP program given an observation via pure JAX forward execution.
        Returns the output action vector.
        """
        prog_jnp = jnp.asarray(program, dtype=jnp.int32)
        obs_jnp = jnp.asarray(obs, dtype=jnp.float32)
        act = forward_lgp_jax(prog_jnp, obs_jnp, self.obs_dim, self.act_dim, self.n_regs, self.out_start)
        return np.asarray(act)

    def get_policy_fn(self, program: np.ndarray) -> Callable:
        """Returns the policy function: obs -> action (discrete or continuous)."""
        def policy(obs: np.ndarray) -> np.ndarray:
            return self.execute_program(program, obs)
        return policy

    def _mutate_program(self, prog: np.ndarray) -> np.ndarray:
        """Point mutation: each instruction mutates with probability mut_rate."""
        prog = prog.copy()
        for i in range(len(prog)):
            if np.random.random() < self.mut_rate:
                gene = np.random.randint(0, 4)  # which gene of the instruction to mutate
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
        """Single-point crossover between two programs."""
        if np.random.random() < self.cx_prob:
            pt = np.random.randint(1, self.prog_len)
            return np.concatenate([a[:pt], b[pt:]], axis=0)
        return a.copy()

    def tell(self, state: LGPState, fitness: jnp.ndarray) -> LGPState:
        """Tournament selection + reproduction + mutation."""
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

        # Binary tournament
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
        new_programs.append(best_prog)  # elitism

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
    programs:     jnp.ndarray  # [pop_size, n_nodes+act_dim, 3]  (op, c1, c2) per node; the last act_dim entries are output indices
    fitness:      jnp.ndarray
    best_program: jnp.ndarray
    best_fitness: jnp.ndarray
    generation:   jnp.ndarray

class CartesianGP:
    """
    Cartesian Genetic Programming (Miller & Thomson 2000).

    DAG graph with C_cols columns (1 row). Each computational node has
    encoding [op, conn1, conn2] where conn ∈ {0..obs_dim+col-1}
    (it may only connect to earlier nodes — acyclicity is guaranteed).

    The last act_dim genes encode the output indices.

    Evolution: (1+λ) ES — one parent produces λ mutated offspring; the best one survives.

    Reference:
        Miller, J. F., & Thomson, P. (2000). Cartesian genetic programming.
        In European Conference on Genetic Programming (pp. 121–132). Springer.
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 pop_size: int = 8,   # (1+λ): typically 1 parent + 7 offspring
                 n_cols: int = 40,    # number of computational nodes
                 mut_rate: float = 0.05):
        self.obs_dim  = obs_dim
        self.act_dim  = act_dim
        self.pop_size = pop_size
        self.n_cols   = n_cols
        self.mut_rate = mut_rate
        self.n_nodes  = n_cols  # 1 linha, n_cols colunas
        # Total number of genes: n_nodes * 3  +  act_dim (outputs)
        self.prog_genes = n_cols * 3 + act_dim
        self.algo_type  = "cgp"

    def _rand_program(self) -> np.ndarray:
        """CGP program: [n_nodes*3 + act_dim] integers."""
        prog = np.zeros(self.prog_genes, dtype=np.int32)
        obs = self.obs_dim
        for col in range(self.n_cols):
            max_conn = obs + col  # only backward connections
            prog[col*3 + 0] = np.random.randint(0, N_OPS)
            prog[col*3 + 1] = np.random.randint(0, max(1, max_conn))
            prog[col*3 + 2] = np.random.randint(0, max(1, max_conn))
        # Output genes: index of any node (input or computational)
        for i in range(self.act_dim):
            prog[self.n_cols*3 + i] = np.random.randint(0, obs + self.n_cols)
        return prog

    def init(self, rng: jax.Array, **kwargs) -> CGPState:
        np.random.seed(int(jax.random.randint(rng, (), 0, 2**30, dtype=jnp.int32)))
        programs = np.stack([self._rand_program() for _ in range(self.pop_size)])
        return CGPState(
            programs     = jnp.array(programs, dtype=jnp.int32),
            fitness      = jnp.full(self.pop_size, -jnp.inf),
            best_program = jnp.zeros(self.prog_genes, dtype=jnp.int32),
            best_fitness = jnp.array(-jnp.inf),
            generation   = jnp.array(0),
        )

    def execute_program(self, prog: np.ndarray | jnp.ndarray, obs: np.ndarray | jnp.ndarray) -> np.ndarray:
        """Evaluates the CGP graph for one observation via pure JAX forward execution."""
        prog_jnp = jnp.asarray(prog, dtype=jnp.int32)
        obs_jnp = jnp.asarray(obs, dtype=jnp.float32)
        act = forward_cgp_jax(prog_jnp, obs_jnp, self.obs_dim, self.act_dim, self.n_cols)
        return np.asarray(act)

    def get_policy_fn(self, prog: np.ndarray) -> Callable:
        def policy(obs: np.ndarray) -> np.ndarray:
            return self.execute_program(prog, obs)
        return policy

    def _mutate(self, prog: np.ndarray) -> np.ndarray:
        """Point mutation that respects the connectivity constraint."""
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
        """(1+λ) ES: the best individual survives and produces pop_size-1 offspring."""
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
