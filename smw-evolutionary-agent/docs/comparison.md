# NEAT vs. Genetic Programming — Detailed Comparative Analysis

## 1. Comparison of Theoretical Foundations

### 1.1 Search Space

#### NEAT
The NEAT search space is the Cartesian product of:
- The space of directed graph topologies (not guaranteed to be acyclic)
- The space of real weights R^n

The dimensionality of R^n grows dynamically as connections are added.
The search is guided by an implicit gradient through weight mutation (Gaussian perturbation).

#### GP
The search space is the set of all valid trees over (F, T):
- F = function set (internal nodes)
- T = terminal set (leaves)

The space is discrete and infinite. There is no gradient — the search is purely stochastic.
Size estimate: |Trees of depth <= d| grows exponentially in d.

### 1.2 Variation Operators

| Operator | NEAT | GP |
|----------|------|----|
| Crossover | Alignment by innovation number | Subtree swap |
| Structural mutation | Add node / connection | Hoist, expansion, collapse, subtree |
| Parametric mutation | Weight perturbation | Point mutation (swap function/terminal) |
| Diversity preservation | Speciation by genetic distance | None (in this project); alternative: crowding |

### 1.3 Fitness and Selection Pressure

Both approaches use the same raw fitness function:

    F_raw = rightmost_x - T/2 + completion_bonus

NEAT uses fitness adjusted per species (implicit fitness sharing through speciation).
GP uses explicit parsimony:

    F_gp = F_raw - λ * max(0, |tree| - θ)

where λ = GP_ParsePenaltyRate and θ = GP_ParsePenaltyStart.

## 2. Complexity Analysis

### 2.1 Evaluation Complexity

#### NEAT
- Network construction: O(|genes|)
- Forward pass evaluation: O(|neurons| * |average_connections|)
- For InputSize=169 + Outputs=8 and ~50 hidden neurons: ~O(10^3) ops/frame

#### GP
- Evaluating one tree: O(|nodes|) — a simple pre-order traversal
- For trees of ~100 nodes and 8 buttons: ~O(800) ops/frame
- Simpler than NEAT, but with no inherent parallelism

### 2.2 Reproduction Complexity

#### NEAT
- Crossover: O(|genes1| + |genes2|) — alignment by innovation
- Speciation: O(|population| * |species|)
- Per generation: O(P * S) where P = population, S = number of species

#### GP
- Subtree crossover: O(|tree|) — enumeration + copy
- Per generation: O(P * |average_tree|)
- Generally slower than NEAT for large populations with deep trees

## 3. Interpretability

### 3.1 What does NEAT reveal?

The NEAT neural network is a **black box**:
- The weights have no direct interpretation
- The topology is hard to analyze manually for networks with many neurons
- You can inspect which inputs (tiles) carry the largest weights, but that is tedious
- Techniques such as SHAP or saliency maps would be needed for post-hoc interpretability

### 3.2 What does GP reveal?

The GP program is **directly readable**:

Example of a real program (hypothetical, after convergence) for the "B" button (jump):

    IF(
      OR(
        GT(tile(16,0), 0),          -- solid tile ahead on the ground
        sprite_near(16,0)           -- enemy in front
      ),
      C_1,                          -- presses B (jumps)
      AND(
        NOT(sprite_near(0,-16)),    -- no enemy above
        C_N1                        -- does not press B
      )
    )

Reading: "Jump if there is an obstacle or an enemy ahead;
          otherwise don't jump (especially if there is an enemy above)."

This is **interpretable science**: we can extract the decision rules of the agent.

### 3.3 Interpretability Metrics

| Metric | NEAT | GP |
|--------|------|----|
| Model complexity | Number of connections | Number of nodes |
| Direct readability | No | Yes |
| Number of active features | Hard to compute | Count of distinct terminals |
| Reasoning depth | Implicit | Depth of the tree |
| Extractable rules | No (without extra techniques) | Yes (straight from the S-expression) |

## 4. Known Problems and Mitigations

### 4.1 Bloat in GP

**Problem**: trees grow indefinitely with no fitness benefit (Poli et al. 2008).
**Cause**: subtree crossover frequently creates dead code (introns) that does not affect fitness but increases size.

**Mitigations implemented**:
1. GP_MaxDepth = 12: hard depth limit
2. GP_MaxNodes = 300: node limit per individual
3. Explicit parsimony: F = F_raw - 0.5 * max(0, size - 50)
4. Hoist mutation: reduces size by replacing subtrees with one of their descendants

### 4.2 Premature Convergence in GP

**Problem**: early loss of genetic diversity.
**Mitigations**:
1. Tournament selection (not proportional to fitness) — higher, controllable selection pressure
2. Ramped Half-and-Half initialization — initial structural diversity
3. Multiple mutation operators — continuous exploration of the space

### 4.3 Deception in NEAT

**Problem**: intermediate solutions (stepping stones) can have low fitness.
**Mitigation**: speciation protects structural innovations for StaleSpecies generations.

## 5. Comparative Experiment Protocol

### 5.1 Controlled Conditions

For a fair comparison:
- **Same game, same level**: Yoshi's Island 1 (SMW)
- **Same initial save state**: DP1.state
- **Same fitness function**: rightmost_x - frames/2 + 1000 (completion)
- **Same BoxRadius**: 6 (13x13 grid = 169 inputs)
- **Same hardware**: run sequentially on the same computer

### 5.2 Parameters to Record

Each generation, record in CSV:
- generation, max_fitness, mean_fitness, std_fitness
- best_size, mean_size, n_evals_total
- wall_clock_seconds, completions

### 5.3 Statistical Analysis

For each algorithm, run N=5 independent seeds.
Compare learning curves using:
- Wilcoxon test (non-parametric) for the final comparison
- Area Under Curve (AUC) of the fitness x generations curve as a convergence metric

### 5.4 Actual Experimental Results (180s Paired Benchmark)

Run under identical conditions on BizHawk 2.9.1 (Super Mario World USA, Yoshi's Island 1, `DP1.state`, `speedmode(600)`, fixed total duration of 180 seconds):

| Evaluated Metric | NEAT (`MarIO.lua`) | Genetic Programming (`MarIO_GP.lua`) |
| :--- | :---: | :---: |
| **Execution Time** | 180s | 180s |
| **Individuals Evaluated** | 1,362 evaluations | 880 evaluations |
| **Generations Completed** | 8 generations (Gen 0 to 7, Pop=300) | 22 generations (Gen 0 to 21, Pop=40) |
| **Initial Fitness ($F_{\max}$ Gen 0)** | 127.0 | 123.0 |
| **Maximum Fitness Reached** | **145.0** (reached at Gen 1, 40s in) | **145.0** (reached at Gen 6, 54s in) |
| **Final Mean Fitness ($\bar{F}$)** | 114.5 – 127.0 (dispersed population) | **144.1** (converged population) |
| **Final Standard Deviation ($\sigma_F$)** | High (wide inter-species variance) | **3.9** (homogeneous stability) |
| **Policy Complexity** | Non-interpretable neural graph (real weights) | Symbolic trees of 46 nodes (readable S-expressions) |

#### Empirical Conclusion of the Paired Test:
1. **In-game performance:** **strict tie** ($145.0$ vs $145.0$). Both agents converged on the same early local barrier (the first obstacle/enemy before the pit) within the 180-second training window.
2. **Sample efficiency and convergence:** GP with $N=40$ showed much faster population convergence ($\sigma = 3.9$ and $\bar{F} = 144.1$), reaching the plateau with fewer total evaluations (880 versus 1,362).
3. **Interpretability:** clear superiority of GP, which allowed direct inspection of the decision rules in symbolic form (`Y: NOT(tile(48, -16))`).

## 6. Specific References

### On GP and Bloat
- Poli, R., Langdon, W.B., McPhee, N.F. (2008). A Field Guide to Genetic Programming. §4.3.
- Soule, T., Foster, J.A. (1998). Effects of Code Growth and Parsimony Pressure on
  Populations in Genetic Programming. Evolutionary Computation 6(4), 293-309.

### On NEAT and Speciation
- Stanley, K.O., Miikkulainen, R. (2002). Evolving Neural Networks through Augmenting
  Topologies. Evolutionary Computation 10(2), 99-127.
- Stanley, K.O. (2004). Efficient Evolution of Neural Networks through Complexification.
  PhD Dissertation, University of Texas at Austin.

### On Interpretability in Evolutionary RL
- Hein, D., Udluft, S., Runkler, T.A. (2018). Interpretable policies for reinforcement
  learning by genetic programming. Engineering Applications of AI, 76, 158-169.
- Custode, L.L., Iacca, G. (2021). Interpretable policies for reinforcement learning by
  evolutionary optimization. Expert Systems with Applications, 113999.
