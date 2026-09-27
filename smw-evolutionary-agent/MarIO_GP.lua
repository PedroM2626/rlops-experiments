-- =============================================================================
-- MarI/O-GP  —  Genetic Programming Agent for Super Mario World
-- =============================================================================
-- Author: based on SethBling's MarI/O (NEAT); the GP extension comes from this project.
-- Paradigm: Genetic Programming with expression trees (Koza, 1992).
--
-- Each individual in the population is a set of PROGRAM TREES — one per
-- output button (A, B, X, Y, Up, Down, Left, Right).
-- Each tree maps the perceptual state (13x13 tile grid + sprites)
-- onto a boolean decision (whether or not to press the button).
--
-- Genetic operators implemented:
--   (1) Subtree crossover    (Koza 1992, §6.2)
--   (2) Point mutation       — replaces a node with another of the same arity
--   (3) Hoist mutation       — promotes a subtree upwards (counters bloat)
--   (4) Expansion mutation   — terminal -> new random subtree
--   (5) Collapse mutation    — subtree -> random terminal
--
-- Parsimony pressure: penalty proportional to the total tree size,
-- used to control excessive growth (bloat).
--
-- References:
--   Koza, J.R. (1992). Genetic Programming. MIT Press.
--   Poli, R., Langdon, W.B., McPhee, N.F. (2008). A Field Guide to GP. Lulu.
--   Stanley, K.O., Miikkulainen, R. (2002). Evolving Neural Networks through
--     Augmenting Topologies. Evolutionary Computation 10(2), 99-127.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 0. GAME DETECTION AND BUTTON CONFIGURATION
-- ---------------------------------------------------------------------------
if gameinfo.getromname() == "Super Mario World (USA)" then
    Filename = "DP1.state"
    ButtonNames = { "A", "B", "X", "Y", "Up", "Down", "Left", "Right" }
elseif gameinfo.getromname() == "Super Mario Bros." then
    Filename = "SMB1-1.state"
    ButtonNames = { "A", "B", "Up", "Down", "Left", "Right" }
end

-- ---------------------------------------------------------------------------
-- 1. GP HYPERPARAMETERS
-- ---------------------------------------------------------------------------

-- Perception: NxN tile grid around Mario (same as NEAT)
BoxRadius        = 6
InputSize        = (BoxRadius*2+1)*(BoxRadius*2+1)

-- Population parameters
GP_Population        = 40        -- population size (tuned for convergence under turbo)
GP_TournamentSize    = 5         -- tournament size for selection
GP_CrossoverChance   = 0.80      -- probability of crossover vs. asexual reproduction
GP_MaxDepthInit      = 5         -- maximum depth during initialization (Ramped Half-and-Half)
GP_MaxDepth          = 10        -- hard depth limit (avoids extreme bloat)
GP_MaxNodes          = 200       -- maximum number of nodes per individual

-- Mutation rates (applied sequentially to the child after crossover/copy)
GP_MutPoint          = 0.10      -- P(point mutation)
GP_MutHoist          = 0.05      -- P(hoist mutation)
GP_MutExpansion      = 0.05      -- P(expansion mutation)
GP_MutCollapse       = 0.05      -- P(collapse mutation)
GP_MutSubtree        = 0.10      -- P(replacing a subtree at random)

-- Parsimony: penalty per node above the threshold
GP_ParsePenaltyStart = 50        -- start penalizing beyond this size
GP_ParsePenaltyRate  = 0.5       -- fitness reduced by X per extra node

-- Execution control
TimeoutConstant  = 20
StaleSpecies     = 15            -- generations without improvement before a species is removed
GP_MaxGeneration = 10000         -- generation limit (0 = unlimited)

-- ---------------------------------------------------------------------------
-- 2. GP LANGUAGE PRIMITIVES
-- ---------------------------------------------------------------------------

-- Binary functions (arity 2): take two values and return a value
-- Unary functions (arity 1)
-- Terminals (arity 0): leaves of the tree

-- Formal definition of the function set F:
-- { IF(cond,vtrue,vfalse), AND, OR, NOT, GT, LT, ADD, MUL, MAX, MIN, TANH }
-- Notes:
--   - IF: arity 3 — returns vtrue when cond > 0, otherwise vfalse
--   - All functions operate on real values in [-1, 1]
--   - Boolean results are encoded as 1.0 (true) and -1.0 (false)
--   - The output decision threshold is 0: >0 presses the button

FUNCTIONS = {
    -- name, arity, evaluation function
    { name="IF",   arity=3, eval=function(a,b,c) if a > 0 then return b else return c end end },
    { name="AND",  arity=2, eval=function(a,b)   return (a > 0 and b > 0) and 1 or -1 end },
    { name="OR",   arity=2, eval=function(a,b)   return (a > 0 or  b > 0) and 1 or -1 end },
    { name="NOT",  arity=1, eval=function(a)      return a > 0 and -1 or 1 end },
    { name="GT",   arity=2, eval=function(a,b)   return a > b and 1 or -1 end },
    { name="LT",   arity=2, eval=function(a,b)   return a < b and 1 or -1 end },
    { name="ADD",  arity=2, eval=function(a,b)   return math.max(-1, math.min(1, a+b)) end },
    { name="MUL",  arity=2, eval=function(a,b)   return math.max(-1, math.min(1, a*b)) end },
    { name="MAX",  arity=2, eval=function(a,b)   return math.max(a,b) end },
    { name="MIN",  arity=2, eval=function(a,b)   return math.min(a,b) end },
    { name="TANH", arity=1, eval=function(a)
        local e2 = math.exp(2*a)
        return (e2-1)/(e2+1)
    end },
}

-- Functions grouped by arity for efficient selection
FUNCTIONS_BY_ARITY = {}
for _, f in ipairs(FUNCTIONS) do
    if FUNCTIONS_BY_ARITY[f.arity] == nil then
        FUNCTIONS_BY_ARITY[f.arity] = {}
    end
    table.insert(FUNCTIONS_BY_ARITY[f.arity], f)
end

-- Terminals: each terminal is a function of index -> value in [-1, 1]
-- Generated dynamically from the perception grid
TERMINALS = {}

-- Tile terminals: tile(dx, dy) ∈ {-1, 0, 1}
for dy = -BoxRadius, BoxRadius do
    for dx = -BoxRadius, BoxRadius do
        local ldx, ldy = dx, dy  -- local capture for the closure
        table.insert(TERMINALS, {
            name  = string.format("tile(%d,%d)", ldx*16, ldy*16),
            eval  = function(percept) return percept.tiles[ldy+BoxRadius+1][ldx+BoxRadius+1] end
        })
    end
end

-- Sprite terminals (enemy presence in the cell)
for dy = -BoxRadius, BoxRadius do
    for dx = -BoxRadius, BoxRadius do
        local ldx, ldy = dx, dy
        table.insert(TERMINALS, {
            name  = string.format("spr(%d,%d)", ldx*16, ldy*16),
            eval  = function(percept) return percept.sprites[ldy+BoxRadius+1][ldx+BoxRadius+1] end
        })
    end
end

-- Constant terminals
table.insert(TERMINALS, { name="C_1",  eval=function(p) return  1.0 end })
table.insert(TERMINALS, { name="C_0",  eval=function(p) return  0.0 end })
table.insert(TERMINALS, { name="C_N1", eval=function(p) return -1.0 end })

-- ---------------------------------------------------------------------------
-- 3. GP TREE STRUCTURE
-- ---------------------------------------------------------------------------

-- GP tree node:
--   node.type     = "function" | "terminal"
--   node.func     = reference to the entry in FUNCTIONS (if function)
--   node.terminal = reference to the entry in TERMINALS (if terminal)
--   node.children = { node1, node2, ... }  (length = func.arity)

function newNode_Function(func)
    return { type="function", func=func, children={} }
end

function newNode_Terminal(term)
    return { type="terminal", terminal=term, children={} }
end

-- Deep copy of a node and its subtree
function copyNode(node)
    local n2 = { type=node.type, func=node.func, terminal=node.terminal, children={} }
    for i=1,#node.children do
        n2.children[i] = copyNode(node.children[i])
    end
    return n2
end

-- Counts the number of nodes in a tree
function countNodes(node)
    local c = 1
    for i=1,#node.children do
        c = c + countNodes(node.children[i])
    end
    return c
end

-- Computes the depth of a tree
function treeDepth(node)
    if #node.children == 0 then return 0 end
    local maxd = 0
    for i=1,#node.children do
        local d = treeDepth(node.children[i])
        if d > maxd then maxd = d end
    end
    return maxd + 1
end

-- Renders the tree as a string (S-expression)
function treeToString(node, depth)
    depth = depth or 0
    if node.type == "terminal" then
        return node.terminal.name
    else
        if #node.children == 0 then
            return node.func.name .. "()"
        end
        local parts = {}
        for i=1,#node.children do
            parts[i] = treeToString(node.children[i], depth+1)
        end
        return node.func.name .. "(" .. table.concat(parts, ", ") .. ")"
    end
end

-- Evaluates a tree given a percept (input map)
function evalTree(node, percept)
    if node.type == "terminal" then
        return node.terminal.eval(percept)
    else
        local args = {}
        for i=1,#node.children do
            args[i] = evalTree(node.children[i], percept)
        end
        return node.func.eval(table.unpack(args))
    end
end

-- ---------------------------------------------------------------------------
-- 4. RANDOM TREE GENERATION
-- ---------------------------------------------------------------------------

-- "Ramped Half-and-Half" method (Koza 1992):
--   - Half of the trees are built with the FULL method (up to maximum depth)
--   - Half with the GROW method (up to maximum depth, but it may stop earlier)
-- This guarantees structural diversity in the initial population.

function randomFunction()
    return FUNCTIONS[math.random(#FUNCTIONS)]
end

function randomTerminal()
    return TERMINALS[math.random(#TERMINALS)]
end

function randomFunctionByArity(arity)
    local list = FUNCTIONS_BY_ARITY[arity]
    if list and #list > 0 then
        return list[math.random(#list)]
    end
    return nil
end

-- Generates a tree with the FULL method: all nodes up to 'depth' are functions;
-- nodes at 'depth' are terminals.
function generateFull(depth)
    if depth <= 0 then
        return newNode_Terminal(randomTerminal())
    else
        local f = randomFunction()
        local node = newNode_Function(f)
        for i=1,f.arity do
            node.children[i] = generateFull(depth-1)
        end
        return node
    end
end

-- Generates a tree with the GROW method: it may stop at any point with a
-- probability proportional to the number of terminals vs. total primitives.
function generateGrow(depth)
    if depth <= 0 then
        return newNode_Terminal(randomTerminal())
    end
    local totalPrimitives = #FUNCTIONS + #TERMINALS
    if math.random() < #TERMINALS / totalPrimitives then
        -- Choose a terminal (early leaf)
        return newNode_Terminal(randomTerminal())
    else
        local f = randomFunction()
        local node = newNode_Function(f)
        for i=1,f.arity do
            node.children[i] = generateGrow(depth-1)
        end
        return node
    end
end

-- Generates a random tree using Ramped Half-and-Half
function generateRandomTree(maxDepth)
    if math.random(2) == 1 then
        return generateFull(math.random(1, maxDepth))
    else
        return generateGrow(math.random(1, maxDepth))
    end
end

-- ---------------------------------------------------------------------------
-- 5. GP INDIVIDUAL (SET OF TREES — ONE PER BUTTON)
-- ---------------------------------------------------------------------------

function newIndividual()
    local ind = {}
    ind.trees    = {}    -- one tree per output button
    ind.fitness  = 0
    ind.size     = 0     -- total nodes across all trees
    ind.evalFitness = 0  -- raw fitness before parsimony
    ind.generation  = 0  -- generation in which it was created (for tracking)
    return ind
end

function basicIndividual()
    local ind = newIndividual()
    for b=1,#ButtonNames do
        ind.trees[b] = generateRandomTree(GP_MaxDepthInit)
    end
    ind.size = computeIndividualSize(ind)
    return ind
end

function copyIndividual(ind)
    local ind2 = newIndividual()
    for b=1,#ButtonNames do
        ind2.trees[b] = copyNode(ind.trees[b])
    end
    ind2.fitness     = ind.fitness
    ind2.evalFitness = ind.evalFitness
    ind2.size        = ind.size
    ind2.generation  = ind.generation
    return ind2
end

function computeIndividualSize(ind)
    local total = 0
    for b=1,#ButtonNames do
        total = total + countNodes(ind.trees[b])
    end
    return total
end

-- Evaluates every tree of an individual given a percept; returns the
-- map of buttons to press.
function evaluateIndividual(ind, percept)
    local outputs = {}
    for b=1,#ButtonNames do
        local val = evalTree(ind.trees[b], percept)
        outputs["P1 " .. ButtonNames[b]] = (val > 0)
    end
    return outputs
end

-- ---------------------------------------------------------------------------
-- 6. GENETIC OPERATORS
-- ---------------------------------------------------------------------------

-- 6.1  Listing of every node of a tree (for crossover point selection)
function listNodes(node, lst)
    lst = lst or {}
    table.insert(lst, node)
    for i=1,#node.children do
        listNodes(node.children[i], lst)
    end
    return lst
end

-- 6.2  SUBTREE CROSSOVER (Koza 1992)
-- Swaps two subtrees picked at random between parent1 and parent2.
-- Returns two children.
function crossover(ind1, ind2)
    local child1 = copyIndividual(ind1)
    local child2 = copyIndividual(ind2)

    -- Pick a button at random on which to perform the crossover
    local b = math.random(#ButtonNames)

    local nodes1 = listNodes(child1.trees[b])
    local nodes2 = listNodes(child2.trees[b])

    if #nodes1 == 0 or #nodes2 == 0 then
        return child1, child2
    end

    -- Select random crossover points
    local n1 = nodes1[math.random(#nodes1)]
    local n2 = nodes2[math.random(#nodes2)]

    -- Swap the subtrees (preserving the maximum depth)
    local saved1_children  = n1.children
    local saved1_type      = n1.type
    local saved1_func      = n1.func
    local saved1_terminal  = n1.terminal

    n1.children = n2.children
    n1.type     = n2.type
    n1.func     = n2.func
    n1.terminal = n2.terminal

    n2.children = saved1_children
    n2.type     = saved1_type
    n2.func     = saved1_func
    n2.terminal = saved1_terminal

    -- Check whether the maximum depth is exceeded; if so, revert
    if treeDepth(child1.trees[b]) > GP_MaxDepth or countNodes(child1.trees[b]) > GP_MaxNodes then
        -- Revert child1
        n1.children = saved1_children
        n1.type     = saved1_type
        n1.func     = saved1_func
        n1.terminal = saved1_terminal
    end
    if treeDepth(child2.trees[b]) > GP_MaxDepth or countNodes(child2.trees[b]) > GP_MaxNodes then
        -- Revert child2
        n2.children = saved1_children
        n2.type     = saved1_type
        n2.func     = saved1_func
        n2.terminal = saved1_terminal
    end

    child1.size = computeIndividualSize(child1)
    child2.size = computeIndividualSize(child2)
    return child1, child2
end

-- 6.3  POINT MUTATION
-- Replaces a node with another of the same arity (function->function, terminal->terminal).
function mutatePoint(ind)
    local b = math.random(#ButtonNames)
    local nodes = listNodes(ind.trees[b])
    if #nodes == 0 then return end
    local n = nodes[math.random(#nodes)]
    if n.type == "function" then
        local newF = randomFunctionByArity(n.func.arity)
        if newF then n.func = newF end
    else
        n.terminal = randomTerminal()
    end
end

-- 6.4  HOIST MUTATION (Poli et al. 2008 §4.2.3)
-- Replaces a node with one of its descendants (reduces the tree size).
function mutateHoist(ind)
    local b = math.random(#ButtonNames)
    local nodes = listNodes(ind.trees[b])
    -- Keep only the nodes that have children (non-terminals)
    local candidates = {}
    for _, n in ipairs(nodes) do
        if #n.children > 0 then
            table.insert(candidates, n)
        end
    end
    if #candidates == 0 then return end
    local n = candidates[math.random(#candidates)]
    -- Pick one child and a subtree of that child
    local child = n.children[math.random(#n.children)]
    local childNodes = listNodes(child)
    local replacement = childNodes[math.random(#childNodes)]
    -- Replace n with the replacement (in-place)
    n.type     = replacement.type
    n.func     = replacement.func
    n.terminal = replacement.terminal
    n.children = replacement.children
    ind.size = computeIndividualSize(ind)
end

-- 6.5  EXPANSION MUTATION
-- Replaces a terminal with a new small subtree.
function mutateExpansion(ind)
    local b = math.random(#ButtonNames)
    local nodes = listNodes(ind.trees[b])
    -- Keep the terminals
    local candidates = {}
    for _, n in ipairs(nodes) do
        if n.type == "terminal" then
            table.insert(candidates, n)
        end
    end
    if #candidates == 0 then return end
    local n = candidates[math.random(#candidates)]
    if treeDepth(ind.trees[b]) >= GP_MaxDepth then return end  -- do not expand if already at the limit
    local newSubtree = generateGrow(math.random(1, 2))  -- small subtree
    n.type     = newSubtree.type
    n.func     = newSubtree.func
    n.terminal = newSubtree.terminal
    n.children = newSubtree.children
    ind.size = computeIndividualSize(ind)
end

-- 6.6  COLLAPSE MUTATION
-- Replaces an internal subtree with a random terminal.
function mutateCollapse(ind)
    local b = math.random(#ButtonNames)
    local nodes = listNodes(ind.trees[b])
    -- Keep internal nodes (not the root, so the root is preserved)
    local candidates = {}
    for i=2, #nodes do  -- i=2: skips the root
        if nodes[i].type == "function" then
            table.insert(candidates, nodes[i])
        end
    end
    if #candidates == 0 then return end
    local n = candidates[math.random(#candidates)]
    local term = randomTerminal()
    n.type     = "terminal"
    n.func     = nil
    n.terminal = term
    n.children = {}
    ind.size = computeIndividualSize(ind)
end

-- 6.7  SUBTREE MUTATION
-- Replaces an entire subtree with a new subtree generated at random.
function mutateSubtree(ind)
    local b = math.random(#ButtonNames)
    if math.random(2) == 1 then
        -- Replaces the whole root
        ind.trees[b] = generateRandomTree(GP_MaxDepthInit)
    else
        local nodes = listNodes(ind.trees[b])
        if #nodes <= 1 then
            ind.trees[b] = generateRandomTree(GP_MaxDepthInit)
            return
        end
        local n = nodes[math.random(2, #nodes)]  -- skips the root
        local newSub = generateRandomTree(math.random(1, 3))
        n.type     = newSub.type
        n.func     = newSub.func
        n.terminal = newSub.terminal
        n.children = newSub.children
    end
    ind.size = computeIndividualSize(ind)
end

-- Applies every mutation operator to an individual
function mutateIndividual(ind)
    if math.random() < GP_MutPoint then     mutatePoint(ind)     end
    if math.random() < GP_MutHoist then     mutateHoist(ind)     end
    if math.random() < GP_MutExpansion then mutateExpansion(ind) end
    if math.random() < GP_MutCollapse then  mutateCollapse(ind)  end
    if math.random() < GP_MutSubtree then   mutateSubtree(ind)   end
    ind.size = computeIndividualSize(ind)
end

-- ---------------------------------------------------------------------------
-- 7. TOURNAMENT SELECTION
-- ---------------------------------------------------------------------------
-- Deterministic tournament: selects K individuals at random and returns
-- the best one. Selection pressure is controlled by GP_TournamentSize.

function tournamentSelect(population)
    local best = nil
    for i=1, GP_TournamentSize do
        local candidate = population[math.random(#population)]
        if best == nil or candidate.fitness > best.fitness then
            best = candidate
        end
    end
    return best
end

-- ---------------------------------------------------------------------------
-- 8. APPLYING PARSIMONY
-- ---------------------------------------------------------------------------
-- Penalizes individuals with many nodes to control bloat growth.
-- penalized_fitness = raw_fitness - parsimony_penalty * max(0, size - threshold)
-- Reference: Poli et al. (2008), §4.3 — Lexicographic Parsimony Pressure

function applyParsimony(ind)
    local excess = math.max(0, ind.size - GP_ParsePenaltyStart)
    ind.fitness = ind.evalFitness - GP_ParsePenaltyRate * excess
end

-- ---------------------------------------------------------------------------
-- 9. ENVIRONMENT PERCEPTION (compatible with MarIO.lua)
-- ---------------------------------------------------------------------------

function getPositions()
    if gameinfo.getromname() == "Super Mario World (USA)" then
        marioX = memory.read_s16_le(0x94)
        marioY = memory.read_s16_le(0x96)
        local layer1x = memory.read_s16_le(0x1A)
        local layer1y = memory.read_s16_le(0x1C)
        screenX = marioX - layer1x
        screenY = marioY - layer1y
    elseif gameinfo.getromname() == "Super Mario Bros." then
        marioX = memory.readbyte(0x6D) * 0x100 + memory.readbyte(0x86)
        marioY = memory.readbyte(0x03B8) + 16
        screenX = memory.readbyte(0x03AD)
        screenY = memory.readbyte(0x03B8)
    end
end

function getTileValue(dx, dy)
    if gameinfo.getromname() == "Super Mario World (USA)" then
        local x = math.floor((marioX+dx+8)/16)
        local y = math.floor((marioY+dy)/16)
        local raw = memory.readbyte(0x1C800 + math.floor(x/0x10)*0x1B0 + y*0x10 + x%0x10)
        return raw ~= 0 and 1 or 0
    elseif gameinfo.getromname() == "Super Mario Bros." then
        local x = marioX + dx + 8
        local y = marioY + dy - 16
        local page = math.floor(x/256)%2
        local subx = math.floor((x%256)/16)
        local suby = math.floor((y - 32)/16)
        local addr = 0x500 + page*13*16+suby*16+subx
        if suby >= 13 or suby < 0 then return 0 end
        return memory.readbyte(addr) ~= 0 and 1 or 0
    end
    return 0
end

function getSpritePositions()
    if gameinfo.getromname() == "Super Mario World (USA)" then
        local sprites = {}
        for slot=0,11 do
            local status = memory.readbyte(0x14C8+slot)
            if status ~= 0 then
                local sx = memory.readbyte(0xE4+slot) + memory.readbyte(0x14E0+slot)*256
                local sy = memory.readbyte(0xD8+slot) + memory.readbyte(0x14D4+slot)*256
                sprites[#sprites+1] = {x=sx, y=sy}
            end
        end
        -- Extended sprites
        for slot=0,11 do
            local number = memory.readbyte(0x170B+slot)
            if number ~= 0 then
                local sx = memory.readbyte(0x171F+slot) + memory.readbyte(0x1733+slot)*256
                local sy = memory.readbyte(0x1715+slot) + memory.readbyte(0x1729+slot)*256
                sprites[#sprites+1] = {x=sx, y=sy}
            end
        end
        return sprites
    elseif gameinfo.getromname() == "Super Mario Bros." then
        local sprites = {}
        for slot=0,4 do
            local enemy = memory.readbyte(0xF+slot)
            if enemy ~= 0 then
                local ex = memory.readbyte(0x6E+slot)*0x100 + memory.readbyte(0x87+slot)
                local ey = memory.readbyte(0xCF+slot) + 24
                sprites[#sprites+1] = {x=ex, y=ey}
            end
        end
        return sprites
    end
    return {}
end

-- Builds the percept: a table of indexed tiles and sprites
function buildPercept()
    getPositions()
    local sprites = getSpritePositions()

    -- Initialize the grids
    local tileGrid   = {}
    local spriteGrid = {}
    for row=1, BoxRadius*2+1 do
        tileGrid[row]   = {}
        spriteGrid[row] = {}
        for col=1, BoxRadius*2+1 do
            tileGrid[row][col]   = 0
            spriteGrid[row][col] = -1  -- -1 = no sprite
        end
    end

    -- Fill in the tiles
    for dy=-BoxRadius,BoxRadius do
        for dx=-BoxRadius,BoxRadius do
            local row = dy + BoxRadius + 1
            local col = dx + BoxRadius + 1
            tileGrid[row][col] = getTileValue(dx*16, dy*16)
            if tileGrid[row][col] == 1 and marioY + dy*16 >= 0x1B0 then
                tileGrid[row][col] = 0
            end
        end
    end

    -- Fill in the sprites
    for _, sp in ipairs(sprites) do
        for dy=-BoxRadius,BoxRadius do
            for dx=-BoxRadius,BoxRadius do
                local distx = math.abs(sp.x - (marioX + dx*16))
                local disty = math.abs(sp.y - (marioY + dy*16))
                if distx <= 8 and disty <= 8 then
                    local row = dy + BoxRadius + 1
                    local col = dx + BoxRadius + 1
                    spriteGrid[row][col] = 1
                end
            end
        end
    end

    return { tiles=tileGrid, sprites=spriteGrid }
end

-- ---------------------------------------------------------------------------
-- 10. GP POOL AND GENERATION MANAGEMENT
-- ---------------------------------------------------------------------------

function newGPPool()
    local p = {}
    p.population  = {}
    p.generation  = 0
    p.maxFitness  = 0
    p.currentInd  = 1
    p.currentFrame = 0
    p.logEntries  = {}   -- per-generation metrics history
    return p
end

-- Statistics of the current population
function computePopStats()
    local n = #gpPool.population
    if n == 0 then return 0, 0, 0, 0 end
    local sumF, maxF, sumS = 0, -math.huge, 0
    for _, ind in ipairs(gpPool.population) do
        sumF = sumF + ind.fitness
        if ind.fitness > maxF then maxF = ind.fitness end
        sumS = sumS + ind.size
    end
    local meanF = sumF / n
    local meanS = sumS / n
    -- Standard deviation
    local varF = 0
    for _, ind in ipairs(gpPool.population) do
        varF = varF + (ind.fitness - meanF)^2
    end
    local stdF = math.sqrt(varF / n)
    return meanF, stdF, maxF, meanS
end

-- Evolves a new generation using elitism + tournament + crossover + mutation
function newGPGeneration()
    -- Apply parsimony to every individual
    for _, ind in ipairs(gpPool.population) do
        applyParsimony(ind)
    end

    -- Sort by descending fitness
    table.sort(gpPool.population, function(a,b) return a.fitness > b.fitness end)

    -- Update the maximum fitness
    if gpPool.population[1] and gpPool.population[1].fitness > gpPool.maxFitness then
        gpPool.maxFitness = gpPool.population[1].fitness
        -- Save the best individual
        writeGPFile("best_gp_gen" .. gpPool.generation .. ".gppool")
    end

    -- Metrics log
    local meanF, stdF, maxF, meanS = computePopStats()
    local logEntry = string.format(
        "Gen %d | MaxF=%.1f MeanF=%.1f StdF=%.1f MeanSize=%.1f BestSize=%d",
        gpPool.generation, maxF, meanF, stdF, meanS,
        gpPool.population[1] and gpPool.population[1].size or 0
    )
    table.insert(gpPool.logEntries, logEntry)
    console.writeline(logEntry)

    -- Explicit write to the training log file for external access
    local resFile = io.open("gp_training_results.txt", "a")
    if resFile then
        resFile:write(logEntry .. "\n")
        local bestInd = gpPool.population[1]
        if bestInd then
            resFile:write("--- BEST SYMBOLIC POLICY (GEN " .. gpPool.generation .. " | Fitness " .. math.floor(bestInd.fitness) .. ") ---\n")
            for b=1, #ButtonNames do
                resFile:write("  " .. ButtonNames[b] .. ": " .. treeToString(bestInd.trees[b]) .. "\n")
            end
            resFile:write("------------------------------------------------------------------------\n\n")
        end
        resFile:flush()
        resFile:close()
    end

    -- Elitism: keep the best 2 unchanged
    local eliteCount = 2
    local newPop = {}
    for i=1, math.min(eliteCount, #gpPool.population) do
        table.insert(newPop, copyIndividual(gpPool.population[i]))
    end

    -- Fill the rest with tournament + crossover + mutation
    while #newPop < GP_Population do
        local p1 = tournamentSelect(gpPool.population)
        local child
        if math.random() < GP_CrossoverChance and #gpPool.population > 1 then
            local p2 = tournamentSelect(gpPool.population)
            local c1, c2 = crossover(p1, p2)
            child = c1
            if #newPop < GP_Population then
                mutateIndividual(child)
                table.insert(newPop, child)
                if #newPop < GP_Population then
                    mutateIndividual(c2)
                    table.insert(newPop, c2)
                end
            end
        else
            child = copyIndividual(p1)
            mutateIndividual(child)
            table.insert(newPop, child)
        end
    end

    -- Truncate if needed
    while #newPop > GP_Population do
        table.remove(newPop)
    end

    gpPool.population = newPop
    gpPool.generation = gpPool.generation + 1
    gpPool.currentInd = 1

    writeGPFile(forms.gettext(gpSaveLoadFile))
end

function initializeGPPool()
    gpPool = newGPPool()
    for i=1, GP_Population do
        table.insert(gpPool.population, basicIndividual())
    end
    initializeGPRun()
end

-- ---------------------------------------------------------------------------
-- 11. EXECUTION CONTROL
-- ---------------------------------------------------------------------------

function clearJoypad()
    local ctrl = {}
    for b=1, #ButtonNames do
        ctrl["P1 " .. ButtonNames[b]] = false
    end
    joypad.set(ctrl)
end

function initializeGPRun()
    savestate.load(Filename)
    gpRightmost  = 0
    gpPool.currentFrame = 0
    gpTimeout    = TimeoutConstant
    clearJoypad()
    evaluateGPCurrent()
end

function evaluateGPCurrent()
    local ind = gpPool.population[gpPool.currentInd]
    if ind == nil then return end

    local percept = buildPercept()
    gpController = evaluateIndividual(ind, percept)

    -- L/R and U/D conflicts
    if gpController["P1 Left"] and gpController["P1 Right"] then
        gpController["P1 Left"]  = false
        gpController["P1 Right"] = false
    end
    if gpController["P1 Up"] and gpController["P1 Down"] then
        gpController["P1 Up"]   = false
        gpController["P1 Down"] = false
    end
    joypad.set(gpController)
end

-- ---------------------------------------------------------------------------
-- 12. PERSISTENCE (SAVE / LOAD GP POOL)
-- ---------------------------------------------------------------------------

function nodeToString(node)
    if node.type == "terminal" then
        return "T:" .. node.terminal.name
    else
        local s = "F:" .. node.func.name .. ":" .. #node.children
        for i=1,#node.children do
            s = s .. "|" .. nodeToString(node.children[i])
        end
        return s
    end
end

function writeGPFile(filename)
    local file = io.open(filename, "w")
    if not file then
        console.writeline("ERROR: Could not open the file for writing: " .. filename)
        return
    end
    file:write(gpPool.generation .. "\n")
    file:write(gpPool.maxFitness .. "\n")
    file:write(#gpPool.population .. "\n")
    for _, ind in ipairs(gpPool.population) do
        file:write(ind.fitness .. "\n")
        file:write(ind.evalFitness .. "\n")
        file:write(ind.size .. "\n")
        file:write(#ButtonNames .. "\n")
        for b=1,#ButtonNames do
            file:write(nodeToString(ind.trees[b]) .. "\n")
        end
    end
    -- Metrics log
    file:write("LOG_START\n")
    for _, entry in ipairs(gpPool.logEntries) do
        file:write(entry .. "\n")
    end
    file:write("LOG_END\n")
    file:close()
end

-- Finds a terminal by name
function findTerminalByName(name)
    for _, t in ipairs(TERMINALS) do
        if t.name == name then return t end
    end
    -- Unknown terminal: return zero
    return { name=name, eval=function(p) return 0 end }
end

-- Finds a function by name
function findFunctionByName(fname)
    for _, f in ipairs(FUNCTIONS) do
        if f.name == fname then return f end
    end
    return nil
end

function parseNode(s, pos)
    pos = pos or 1
    local typeChar = s:sub(pos, pos)  -- 'T' or 'F'
    pos = pos + 2  -- skip 'T:' or 'F:'

    if typeChar == "T" then
        -- Read the terminal name up to '|' or '\0'
        local endPos = s:find("|", pos, true)
        local name
        if endPos then
            name = s:sub(pos, endPos-1)
            pos = endPos + 1
        else
            name = s:sub(pos)
            pos = #s + 1
        end
        local term = findTerminalByName(name)
        return newNode_Terminal(term), pos
    elseif typeChar == "F" then
        -- Read the function name
        local colonPos = s:find(":", pos, true)
        local fname = s:sub(pos, colonPos-1)
        pos = colonPos + 1
        -- Read the arity
        local pipeOrEnd = s:find("|", pos, true)
        local arityStr
        if pipeOrEnd then
            arityStr = s:sub(pos, pipeOrEnd-1)
            pos = pipeOrEnd + 1
        else
            arityStr = s:sub(pos)
            pos = #s + 1
        end
        local arity = tonumber(arityStr) or 0
        local func = findFunctionByName(fname)
        if not func then
            -- Fallback: constant terminal
            return newNode_Terminal(findTerminalByName("C_0")), pos
        end
        local node = newNode_Function(func)
        for i=1,arity do
            local child
            child, pos = parseNode(s, pos)
            node.children[i] = child
        end
        return node, pos
    end
    -- Fallback
    return newNode_Terminal(findTerminalByName("C_0")), pos
end

function loadGPFile(filename)
    local file = io.open(filename, "r")
    if not file then
        console.writeline("ERROR: File not found: " .. filename)
        return
    end
    gpPool = newGPPool()
    gpPool.generation = file:read("*number")
    gpPool.maxFitness = file:read("*number")
    local numInds = file:read("*number")
    for i=1, numInds do
        local ind = newIndividual()
        ind.fitness     = file:read("*number")
        ind.evalFitness = file:read("*number")
        ind.size        = file:read("*number")
        local numTrees  = file:read("*number")
        for b=1, numTrees do
            local line = file:read("*line")
            if line then
                ind.trees[b] = parseNode(line)
            else
                ind.trees[b] = generateRandomTree(GP_MaxDepthInit)
            end
        end
        -- Ensure every button has a tree
        for b=numTrees+1, #ButtonNames do
            ind.trees[b] = generateRandomTree(GP_MaxDepthInit)
        end
        ind.size = computeIndividualSize(ind)
        table.insert(gpPool.population, ind)
    end
    file:close()
    forms.settext(gpMaxFitnessLabel, "Max Fitness: " .. math.floor(gpPool.maxFitness))
    initializeGPRun()
end

function saveGPPool()
    writeGPFile(forms.gettext(gpSaveLoadFile))
end

function loadGPPool()
    loadGPFile(forms.gettext(gpSaveLoadFile))
end

-- ---------------------------------------------------------------------------
-- 13. VISUAL DISPLAY (GUI)
-- ---------------------------------------------------------------------------

-- Displays the best individual of the current generation with its symbolic expressions
function displayGP(ind)
    if ind == nil then return end

    -- Background panel
    gui.drawBox(0, 0, 300, 200, 0xC0000000, 0xC0111111)

    -- Title
    gui.drawText(4, 2, "MarI/O-GP  Gen:" .. gpPool.generation, 0xFFFFFF00, 10)
    gui.drawText(4, 14, "MaxFit:" .. math.floor(gpPool.maxFitness) ..
                        "  Size:" .. ind.size, 0xFFFFFF00, 10)

    -- Displays the tile grid (same as NEAT)
    local percept = buildPercept()
    local gridX, gridY = 4, 30
    local cellSize = 4
    gui.drawBox(gridX-1, gridY-1,
                gridX + (BoxRadius*2+1)*cellSize,
                gridY + (BoxRadius*2+1)*cellSize,
                0xFFFFFFFF, 0xFF333333)
    for row=1, BoxRadius*2+1 do
        for col=1, BoxRadius*2+1 do
            local tv = percept.tiles[row][col]
            local sv = percept.sprites[row][col]
            local color
            if sv == 1 then
                color = 0xFFFF0000  -- red = enemy
            elseif tv == 1 then
                color = 0xFFAAAAAA  -- grey = solid tile
            else
                color = 0xFF222222  -- empty
            end
            local px = gridX + (col-1)*cellSize
            local py = gridY + (row-1)*cellSize
            gui.drawBox(px, py, px+cellSize-1, py+cellSize-1, color, color)
        end
    end

    -- Displays the state of the active buttons
    local btnX = gridX + (BoxRadius*2+1)*cellSize + 8
    local btnY = 30
    for b=1, #ButtonNames do
        local pressed = gpController["P1 " .. ButtonNames[b]]
        local color = pressed and 0xFF00FF00 or 0xFF444444
        gui.drawText(btnX, btnY + (b-1)*12, ButtonNames[b], color, 10)
    end

    -- Displays the symbolic expression of the main button (Right — the most informative)
    -- Truncated so that it fits on screen
    local exprY = 140
    gui.drawText(4, exprY, "Right:", 0xFFAAAAFF, 9)
    local expr = treeToString(ind.trees[#ButtonNames])  -- last one = Right
    if #expr > 55 then expr = expr:sub(1, 52) .. "..." end
    gui.drawText(4, exprY+10, expr, 0xFFFFFFFF, 8)

    -- Generation progress bar
    local total = GP_Population
    local measured = gpPool.currentInd - 1
    local pct = total > 0 and math.floor(measured/total*100) or 0
    gui.drawText(4, exprY+22,
        "Evaluating " .. gpPool.currentInd .. "/" .. total ..
        " (" .. pct .. "%)", 0xFFCCCCCC, 8)
end

-- ---------------------------------------------------------------------------
-- 14. PLAY TOP  — runs the best individual found
-- ---------------------------------------------------------------------------

function gpPlayTop()
    table.sort(gpPool.population, function(a,b) return a.fitness > b.fitness end)
    gpPool.currentInd = 1
    gpPool.maxFitness = gpPool.population[1] and gpPool.population[1].fitness or 0
    forms.settext(gpMaxFitnessLabel, "Max Fitness: " .. math.floor(gpPool.maxFitness))
    initializeGPRun()
end

-- Prints the expression of the best individual to the console
function gpPrintBest()
    table.sort(gpPool.population, function(a,b) return a.fitness > b.fitness end)
    local best = gpPool.population[1]
    if best == nil then return end
    console.writeline("=== BEST GP INDIVIDUAL (Gen " .. gpPool.generation .. ") ===")
    console.writeline("Fitness: " .. best.fitness .. "  Size: " .. best.size)
    for b=1, #ButtonNames do
        console.writeline("Button " .. ButtonNames[b] .. ":")
        console.writeline("  " .. treeToString(best.trees[b]))
    end
    console.writeline("=== END ===")
end

-- ---------------------------------------------------------------------------
-- 15. CONTROL FORM
-- ---------------------------------------------------------------------------

function onExitGP()
    forms.destroy(gpForm)
end

if gpPool == nil then
    initializeGPPool()
end

writeGPFile("temp_gp.gppool")
event.onexit(onExitGP)

gpForm            = forms.newform(220, 290, "MarI/O-GP Fitness")
gpMaxFitnessLabel = forms.label(gpForm, "Max Fitness: " .. math.floor(gpPool.maxFitness), 5, 8)
gpShowNetwork     = forms.checkbox(gpForm, "Show Grid", 5, 30)
gpRestartButton   = forms.button(gpForm, "Restart",  initializeGPPool,  5, 55)
gpSaveButton      = forms.button(gpForm, "Save",     saveGPPool,         5, 80)
gpLoadButton      = forms.button(gpForm, "Load",     loadGPPool,        110, 80)
gpSaveLoadFile    = forms.textbox(gpForm, Filename .. ".gppool", 200, 25, nil, 5, 108)
gpSaveLabel       = forms.label(gpForm, "Save/Load:", 5, 134)
gpPlayTopButton   = forms.button(gpForm, "Play Top", gpPlayTop,          5, 158)
gpPrintButton     = forms.button(gpForm, "Print Best", gpPrintBest,     110, 158)
gpHideBanner      = forms.checkbox(gpForm, "Hide Banner", 5, 185)

-- ---------------------------------------------------------------------------
-- 16. MAIN LOOP
-- ---------------------------------------------------------------------------

gpController = {}
gpRightmost  = 0
gpTimeout    = TimeoutConstant

-- Speed up emulation for turbo training
pcall(function() client.speedmode(600) end)

while true do
    -- Background banner
    if not forms.ischecked(gpHideBanner) then
        gui.drawBox(0, 0, 300, 26, 0xD0FFFFFF, 0xD0FFFFFF)
    end

    local ind = gpPool.population[gpPool.currentInd]

    -- Visual display
    if forms.ischecked(gpShowNetwork) then
        displayGP(ind)
    end

    -- Network evaluation (every 5 frames for performance)
    if gpPool.currentFrame % 5 == 0 then
        evaluateGPCurrent()
    end

    joypad.set(gpController)

    -- Timeout and fitness logic
    getPositions()
    if marioX > gpRightmost then
        gpRightmost = marioX
        gpTimeout   = TimeoutConstant
    end

    gpTimeout = gpTimeout - 1

    local timeoutBonus = gpPool.currentFrame / 4
    if gpTimeout + timeoutBonus <= 0 then
        -- Compute the fitness of the current individual
        local fitness = gpRightmost - gpPool.currentFrame / 2
        if gameinfo.getromname() == "Super Mario World (USA)" and gpRightmost > 4816 then
            fitness = fitness + 1000
        end
        if gameinfo.getromname() == "Super Mario Bros." and gpRightmost > 3186 then
            fitness = fitness + 1000
        end
        if fitness == 0 then fitness = -1 end

        ind.evalFitness = fitness
        applyParsimony(ind)

        if ind.fitness > gpPool.maxFitness then
            gpPool.maxFitness = ind.fitness
            forms.settext(gpMaxFitnessLabel, "Max Fitness: " .. math.floor(gpPool.maxFitness))
            writeGPFile("best." .. gpPool.generation .. "." .. forms.gettext(gpSaveLoadFile))
        end

        console.writeline(
            "GP Gen " .. gpPool.generation ..
            " Ind "   .. gpPool.currentInd ..
            "/"       .. #gpPool.population ..
            " Fit:"   .. math.floor(ind.fitness) ..
            " Size:"  .. ind.size
        )

        -- Advance to the next individual
        gpPool.currentInd = gpPool.currentInd + 1
        if gpPool.currentInd > #gpPool.population then
            newGPGeneration()
        end
        initializeGPRun()
    end

    -- Information HUD
    if not forms.ischecked(gpHideBanner) then
        local pct = #gpPool.population > 0
            and math.floor((gpPool.currentInd-1) / #gpPool.population * 100)
            or 0
        gui.drawText(0, 0,
            "GP Gen " .. gpPool.generation ..
            " Ind "   .. gpPool.currentInd ..
            "/" .. #gpPool.population ..
            " (" .. pct .. "%)",
            0xFF000000, 11)
        gui.drawText(0, 12,
            "Fit:" .. math.floor(gpRightmost - gpPool.currentFrame/2 - (gpTimeout+timeoutBonus)*2/3) ..
            "  Max:" .. math.floor(gpPool.maxFitness),
            0xFF000000, 11)
    end

    gpPool.currentFrame = gpPool.currentFrame + 1
    emu.frameadvance()
end
