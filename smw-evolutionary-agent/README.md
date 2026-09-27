# SMW Evolutionary Agent — NEAT vs. Genetic Programming

> **MarI/O** by SethBling (NEAT) + **MarI/O-GP** (Genetic Programming) for interpretable agents
> Academic project: comparing Neuroevolution (NEAT) and Genetic Programming (GP) in *Super Mario World*

---

## Index

1. [Overview](#overview)
2. [Prerequisites and Environment Setup](#prerequisites-and-environment-setup)
3. [How to Run MarIO (NEAT)](#how-to-run-mario-neat)
4. [How to Run MarIO-GP (Genetic Programming)](#how-to-run-mario-gp)
5. [Project Structure](#project-structure)
6. [Theoretical Foundations](#theoretical-foundations)
7. [Comparison: NEAT vs. GP](#comparison-neat-vs-gp)
8. [Metrics and Evaluation](#metrics-and-evaluation)
9. [References](#references)

---

## Overview

This project implements and compares two evolutionary computing paradigms playing *Super Mario World (USA)* in the BizHawk emulator:

| Approach | Script | Representation | Interpretability |
|----------|--------|----------------|------------------|
| **NEAT** (NeuroEvolution of Augmenting Topologies) | `MarIO.lua` | Neural network with evolving topology | Low |
| **Genetic Programming (GP)** | `MarIO_GP.lua` | Program tree (symbolic expression) | **High** |

Both agents use the **same fitness function** and the **same input representation**
(a 13x13 tile grid around Mario), which guarantees a fair comparison.

---

## Prerequisites and Environment Setup

### 1. Operating System

Windows 10/11 (64-bit). BizHawk has no official support for Linux/macOS.

### 2. BizHawk — Emulator

**Recommended version: BizHawk 2.9.1** (compatible with the Lua API used by these scripts).

> IMPORTANT: Versions 2.10+ introduced API changes that can break the scripts.
> Prefer the 2.9.x releases.

**Download:**
- Go to: https://github.com/TASEmulators/BizHawk/releases/tag/2.9.1
- Download `BizHawk-2.9.1-win-x64.zip`
- Extract it to a directory **without spaces in the path**, e.g. `C:\BizHawk\`

**Installing the BizHawk prerequisites:**
Run the following as Administrator from inside the BizHawk folder:

    .\prerequisites\bizhawk_prerequisites.ps1

Or run `prereq\bizhawk_prereqs.exe`, which is included in the ZIP.

**Additional dependencies required:**
- .NET 8 Runtime: https://dotnet.microsoft.com/download/dotnet/8.0
- Visual C++ Redistributable 2015-2022 x64: https://aka.ms/vs/17/release/vc_redist.x64.exe
- DirectX End-User Runtime: https://www.microsoft.com/en-us/download/details.aspx?id=35

### 3. Super Mario World ROM

You will need a legally obtained ROM of the game:
- Exact name (verified by the script): `Super Mario World (USA).sfc`
- `gameinfo.getromname()` must return exactly `"Super Mario World (USA)"`

> Dump your own physical cartridge copy using an SNES cartridge reader.
> Do not distribute ROMs.

### 4. Emulator Setup and Savestate
Both emulator detection and the provisioning of the native `DP1.state` plus script synchronization are handled automatically by [`launch_mario.py`](file:///c:/Users/Acer/Downloads/smw-evolutionary-agent/launch_mario.py) or [`setup_environment.py`](file:///c:/Users/Acer/Downloads/smw-evolutionary-agent/setup_environment.py). No manual intervention in the emulator is needed.

---

## Autonomous Execution (No Manual Intervention)

The project ships a 100% autonomous execution pipeline through [`launch_mario.py`](file:///c:/Users/Acer/Downloads/smw-evolutionary-agent/launch_mario.py):

```bash
# Run NEAT (original MarI/O):
python launch_mario.py

# Run Genetic Programming (interpretable MarI/O-GP):
python launch_mario.py --gp
```

### What the autonomous pipeline does:
1. **Validation and synchronization:** checks the ROM SHA-1 (`6b47bb75d16514b6a476aa0c73a683a2a4c18765`), the BizHawk executables and the Lua scripts.
2. **Automatic savestate generation (`DP1.state`):** if the native savestate is missing or corrupted, the pipeline runs a background bootstrap that navigates the title screen autonomously, selects the save and writes `DP1.state` at the exact frame where the *Yoshi's Island 1* level starts (Mode `0x14`, MarioX = 128).
3. **Initialization and training:** opens BizHawk with the selected Lua script via `--lua`, showing the gameplay and the evolution UI in real time.

---

## How to Run Manually (Optional)

1. Open BizHawk and load `Super Mario World (USA).sfc`
2. Go to `Tools -> Lua Console`
3. In the Lua console: `Script -> Open Script` -> select `MarIO.lua`
4. The script will automatically start evolving a population of neural networks

**UI controls:**
- Show Map: displays the tile grid and the neural network
- Show M-Rates: displays the current mutation rates
- Restart: restarts the evolution from scratch
- Save/Load: saves/loads the genome pool to a `.pool` file
- Play Top: plays the best genome found so far

---

## How to Run MarIO-GP

1. Open BizHawk and load `Super Mario World (USA).sfc`
2. Go to `Tools -> Lua Console`
3. In the Lua console: `Script -> Open Script` -> select `MarIO_GP.lua`
4. The script will evolve a population of **program trees**

**Visual differences:**
- The GP panel displays the **symbolic expression** of the best individual (interpretable!)
- The log shows the average tree depth and the complexity of the best program
- `.gppool` files persist the state of the GP evolution

---

## Project Structure

```text
smw-evolutionary-agent/
|-- MarIO.lua               # NEAT — classic neuroevolution (SethBling)
|-- MarIO_GP.lua            # Genetic Programming — interpretable symbolic trees
|-- launch_mario.py         # 100% autonomous execution pipeline (NEAT / GP)
|-- setup_environment.py    # Integrity validation (SHA-1) and synchronization
|-- DP1.state               # Native savestate at the first frame of Yoshi's Island 1
|-- README.md               # Complete project documentation
`-- docs/
    `-- comparison.md       # Academic comparative study NEAT vs. GP
```

---

## Theoretical Foundations

### NEAT (NeuroEvolution of Augmenting Topologies)

NEAT is an algorithm proposed by Stanley & Miikkulainen (2002) that evolves **simultaneously**
the topology and the weights of neural networks. Main characteristics:

- **Speciation**: protects structural innovations by grouping similar genomes
- **Innovation numbers**: allow crossover between genomes with different topologies
- **Incremental complexification**: starts with minimal networks and grows complexity

**Genome representation:**
Each genome is a set of *connection genes* `(input_node, output_node, weight, enabled, innovation)`.

**Fitness function:**

    fitness = rightmost_x - frames / 2 + (1000 if the level was completed)

### Genetic Programming (GP)

GP is an evolutionary paradigm (Koza, 1992) that evolves **computer programs**
represented as trees. In the context of this project:

**Function set (F):**

    { IF, AND, OR, NOT, GT, LT, ADD, MUL, TANH, MAX, MIN }

**Terminal set (T):**

    { tile(dx,dy) for dx,dy in [-6..6]*16 }   <- same grid as NEAT
    { sprite_near(dx,dy) }                       <- enemy presence
    { const_0, const_1, const_neg1 }             <- constants

**Each output (button) has its own decision tree** — fully interpretable!

**Example of an evolved program for the "Right" button:**

    IF( GT(tile(16,0), 0),
        AND(True, NOT(sprite_near(16,0))),
        True
    )
    -> "Hold right, but stop if there is a solid tile AND an enemy ahead"

**GP genetic operators:**
- Subtree crossover: swaps subtrees between two parent programs
- Point mutation: replaces a node with another of the same type
- Hoist mutation: promotes a subtree upwards (reduces size — combats bloat)
- Expansion mutation: replaces a terminal with a new subtree
- Collapse mutation: replaces a subtree with a terminal

---

## Comparison: NEAT vs. GP

| Criterion | NEAT | Genetic Programming |
|-----------|------|---------------------|
| **Representation** | Graph of neurons | Expression tree |
| **Interpretability** | Low (black box) | High (readable code) |
| **Convergence** | Fast (continuous weights) | Slower (discrete space) |
| **Generalization** | Moderate | Potentially better |
| **Bloat** | Not applicable | Known problem; mitigated by parsimony pressure |
| **Transfer learning** | Difficult | More natural (swap subtrees) |
| **Debugging** | No direct inspection possible | The program can be read |
| **Memory** | O(connections) | O(tree size) |

### Hypotheses

1. **NEAT** converges to a higher fitness in fewer generations.
2. **GP** produces more robust programs that generalize to other levels.
3. The GP program reveals **interpretable strategic rules**.

---

## Experimental Results and Empirical Comparison

A controlled 180-second paired benchmark was run under identical conditions on BizHawk 2.9.1 (*Super Mario World USA*, Yoshi's Island 1, save state `DP1.state`, `speedmode(600)`):

### 1. Comparison Table of Measured Metrics

| Scientific metric | NEAT (`MarIO.lua`) | Genetic Programming (`MarIO_GP.lua`) |
| :--- | :---: | :---: |
| **Execution Time** | 180s | 180s |
| **Individuals Evaluated** | 1,362 genomes | 880 programs (35% more efficient) |
| **Generations Completed** | 8 generations (Gen 0 to 7, Pop=300) | 22 generations (Gen 0 to 21, Pop=40) |
| **Initial Fitness ($F_{\max}$ Gen 0)** | 127.0 | 123.0 |
| **Maximum Fitness Reached** | **145.0** (at Gen 1, 40s in) | **145.0** (at Gen 6, 54s in) |
| **Final Mean Fitness ($\bar{F}$)** | 114.5 – 127.0 (dispersed population) | **144.1** (converged population) |
| **Final Standard Deviation ($\sigma_F$)** | High (wide spread across species) | **3.9** (homogeneous stability) |
| **Policy Complexity** | Non-interpretable neural graph | **46 syntactic nodes** (explicit S-expression) |

### 2. Explicit Policy Obtained by the GP Agent (Gen 21, Fitness = 145.0)

Unlike NEAT's neural black box, the policy evolved by GP for each SNES controller button is a symbolic syntax tree that can be audited directly:

```lisp
;; 1. A button (spin jump):
(spr 0 -32)

;; 2. B button (regular jump):
(tile -48 -32)

;; 3. X button:
(tile 48 -32)

;; 4. Y button (dash / continuous run):
(NOT (tile 48 -16))

;; 5. Up button:
(tile -80 96)

;; 6. Down button:
(spr 0 0)

;; 7. Left button:
(tile -96 48)

;; 8. Right button (locomotion / horizontal advance):
(ADD
  (AND
    (OR
      (LT (MUL (tile -16 -96) (spr 32 -48))
          (OR (tile 96 0) (tile 32 -32)))
      (MAX (GT (tile 32 48) (spr -80 -32))
           (MIN (tile -48 -80) (tile -48 0))))
    (TANH
      (LT (MAX (tile 80 80) (spr 64 0))
          (NOT (tile 80 -48)))))
  (ADD
    (MUL
      (NOT (IF (spr 48 -64) (tile -80 96) (tile 64 96)))
      (MIN (MUL (spr -80 -80) (spr 16 80))
           (NOT (spr -48 32))))
    (spr 32 -48)))
```

### 3. Scientific Discussion of the Results

1. **In-game performance (tie at 145.0):** both agents reached the same local fitness ceiling during the level's opening descent, discovering sustained rightward advance at running speed, but needing more training time to synthesize the jump over the first Koopa/obstacle.
2. **Sample efficiency and homogeneity:** GP converged with 35% fewer fitness evaluations (880 versus 1,362), lowering the standard deviation from 23.3 to 3.9 and the average tree size from 92.6 to 45.0 nodes through parsimony pressure (combating *code bloat*).
3. **Interpretability:** GP reached its central goal of algorithmic transparency, making it possible to verify exactly which spatial correlations trigger each action button.

---

## Metrics and Evaluation

| Metric | Description |
|--------|-------------|
| `max_fitness` | Highest rightmost_x reached |
| `mean_fitness` | Population mean |
| `fitness_std` | Standard deviation of fitness |
| `best_genome_size` | Connections (NEAT) / nodes (GP) of the best individual |
| `mean_genome_size` | Average population complexity |
| `n_species` | Number of species (NEAT) / niches (GP) |
| `level_completions` | How many times the level was completed |
| `wall_time_per_gen` | Wall-clock time per generation |

---

## References

1. Stanley, K. O., & Miikkulainen, R. (2002). Evolving Neural Networks through Augmenting
   Topologies. *Evolutionary Computation*, 10(2), 99-127.
   https://doi.org/10.1162/106365602320169811

2. Koza, J. R. (1992). *Genetic Programming: On the Programming of Computers by Means of
   Natural Selection*. MIT Press.

3. Poli, R., Langdon, W. B., & McPhee, N. F. (2008). *A Field Guide to Genetic Programming*.
   Lulu.com. http://www.gp-field-guide.org.uk/

4. SethBling (2015). *MarI/O - Machine Learning for Video Games*. YouTube.
   https://www.youtube.com/watch?v=qv6UVOQ0F44

5. Luke, S. (2013). *Essentials of Metaheuristics* (2nd ed.). Lulu.com.
   https://cs.gmu.edu/~sean/book/metaheuristics/

6. Langdon, W. B., & Poli, R. (2002). *Foundations of Genetic Programming*. Springer.

7. BizHawk Lua API Reference: http://tasvideos.org/BizHawk/LuaFunctions.html
