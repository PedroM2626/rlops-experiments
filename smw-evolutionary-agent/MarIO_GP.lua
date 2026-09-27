-- =============================================================================
-- MarI/O-GP  —  Genetic Programming Agent for Super Mario World
-- =============================================================================
-- Autor: Baseado em MarI/O de SethBling (NEAT); extensão GP por este projeto.
-- Paradigma: Programação Genética com árvores de expressão (Koza, 1992).
--
-- Cada indivíduo na população é um conjunto de ÁRVORES DE PROGRAMA — uma por
-- botão de saída (A, B, X, Y, Up, Down, Left, Right).
-- Cada árvore mapeia o estado perceptual (grade de tiles 13x13 + sprites)
-- para uma decisão booleana (pressionar ou não o botão).
--
-- Operadores genéticos implementados:
--   (1) Crossover de subárvore (Koza 1992, §6.2)
--   (2) Mutação de ponto        — substitui um nó pelo mesmo aridade
--   (3) Mutação de hoist        — sobe uma subárvore (combate bloat)
--   (4) Mutação de expansão     — terminal -> nova subárvore aleatória
--   (5) Mutação de colapso      — subárvore -> terminal aleatório
--
-- Pressão de parsimônia: penalidade proporcional ao tamanho total das árvores
-- para controlar crescimento excessivo (bloat).
--
-- Referências:
--   Koza, J.R. (1992). Genetic Programming. MIT Press.
--   Poli, R., Langdon, W.B., McPhee, N.F. (2008). A Field Guide to GP. Lulu.
--   Stanley, K.O., Miikkulainen, R. (2002). Evolving Neural Networks through
--     Augmenting Topologies. Evolutionary Computation 10(2), 99-127.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 0. DETECÇÃO DE JOGO E CONFIGURAÇÃO DE BOTÕES
-- ---------------------------------------------------------------------------
if gameinfo.getromname() == "Super Mario World (USA)" then
    Filename = "DP1.state"
    ButtonNames = { "A", "B", "X", "Y", "Up", "Down", "Left", "Right" }
elseif gameinfo.getromname() == "Super Mario Bros." then
    Filename = "SMB1-1.state"
    ButtonNames = { "A", "B", "Up", "Down", "Left", "Right" }
end

-- ---------------------------------------------------------------------------
-- 1. HIPERPARÂMETROS GP
-- ---------------------------------------------------------------------------

-- Percepção: grade de tiles NxN ao redor de Mario (mesmo que NEAT)
BoxRadius        = 6
InputSize        = (BoxRadius*2+1)*(BoxRadius*2+1)

-- Parâmetros populacionais
GP_Population        = 40        -- tamanho da população (otimizado para convergência em turbo)
GP_TournamentSize    = 5         -- tamanho do torneio para seleção
GP_CrossoverChance   = 0.80      -- probabilidade de crossover vs. reprodução assexuada
GP_MaxDepthInit      = 5         -- profundidade máxima na inicialização (Ramped Half-and-Half)
GP_MaxDepth          = 10        -- limite máximo de profundidade (evita bloat extremo)
GP_MaxNodes          = 200       -- limite máximo de nós por indivíduo

-- Taxas de mutação (aplicadas sequencialmente ao filho após crossover/cópia)
GP_MutPoint          = 0.10      -- P(mutação de ponto)
GP_MutHoist          = 0.05      -- P(mutação de hoist)
GP_MutExpansion      = 0.05      -- P(mutação de expansão)
GP_MutCollapse       = 0.05      -- P(mutação de colapso)
GP_MutSubtree        = 0.10      -- P(substituir subárvore aleatória)

-- Parsimônia: penalidade por nó acima do limiar
GP_ParsePenaltyStart = 50        -- a partir deste tamanho, penalizar
GP_ParsePenaltyRate  = 0.5       -- fitness reduzido em X por nó extra

-- Controle de execução
TimeoutConstant  = 20
StaleSpecies     = 15            -- gerações sem melhora para eliminar espécie
GP_MaxGeneration = 10000         -- limite de gerações (0 = ilimitado)

-- ---------------------------------------------------------------------------
-- 2. PRIMITIVAS DA LINGUAGEM GP
-- ---------------------------------------------------------------------------

-- Funções binárias (aridade 2): recebem dois valores e retornam um valor
-- Funções unárias (aridade 1)
-- Terminais (aridade 0): folhas da árvore

-- Definição formal do conjunto de funções F:
-- { IF(cond,vtrue,vfalse), AND, OR, NOT, GT, LT, ADD, MUL, MAX, MIN, TANH }
-- Notas:
--   - IF: aridade 3 — se cond > 0 retorna vtrue, caso contrário vfalse
--   - Todas as funções operam sobre valores reais em [-1, 1]
--   - Resultados booleanos são codificados como 1.0 (true) e -1.0 (false)
--   - O threshold de decisão de saída é 0: >0 pressiona o botão

FUNCTIONS = {
    -- nome, aridade, função de avaliação
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

-- Funções agrupadas por aridade para seleção eficiente
FUNCTIONS_BY_ARITY = {}
for _, f in ipairs(FUNCTIONS) do
    if FUNCTIONS_BY_ARITY[f.arity] == nil then
        FUNCTIONS_BY_ARITY[f.arity] = {}
    end
    table.insert(FUNCTIONS_BY_ARITY[f.arity], f)
end

-- Terminais: cada terminal é uma função de índice -> valor em [-1, 1]
-- Gerados dinamicamente a partir da grade de percepção
TERMINALS = {}

-- Terminais de tile: tile(dx, dy) ∈ {-1, 0, 1}
for dy = -BoxRadius, BoxRadius do
    for dx = -BoxRadius, BoxRadius do
        local ldx, ldy = dx, dy  -- captura local para closure
        table.insert(TERMINALS, {
            name  = string.format("tile(%d,%d)", ldx*16, ldy*16),
            eval  = function(percept) return percept.tiles[ldy+BoxRadius+1][ldx+BoxRadius+1] end
        })
    end
end

-- Terminais de sprite (presença de inimigo na célula)
for dy = -BoxRadius, BoxRadius do
    for dx = -BoxRadius, BoxRadius do
        local ldx, ldy = dx, dy
        table.insert(TERMINALS, {
            name  = string.format("spr(%d,%d)", ldx*16, ldy*16),
            eval  = function(percept) return percept.sprites[ldy+BoxRadius+1][ldx+BoxRadius+1] end
        })
    end
end

-- Terminais constantes
table.insert(TERMINALS, { name="C_1",  eval=function(p) return  1.0 end })
table.insert(TERMINALS, { name="C_0",  eval=function(p) return  0.0 end })
table.insert(TERMINALS, { name="C_N1", eval=function(p) return -1.0 end })

-- ---------------------------------------------------------------------------
-- 3. ESTRUTURA DA ÁRVORE GP
-- ---------------------------------------------------------------------------

-- Nó da árvore GP:
--   node.type     = "function" | "terminal"
--   node.func     = referência à entrada em FUNCTIONS (se function)
--   node.terminal = referência à entrada em TERMINALS (se terminal)
--   node.children = { nó1, nó2, ... }  (tamanho = func.arity)

function newNode_Function(func)
    return { type="function", func=func, children={} }
end

function newNode_Terminal(term)
    return { type="terminal", terminal=term, children={} }
end

-- Cópia profunda de um nó e sua subárvore
function copyNode(node)
    local n2 = { type=node.type, func=node.func, terminal=node.terminal, children={} }
    for i=1,#node.children do
        n2.children[i] = copyNode(node.children[i])
    end
    return n2
end

-- Conta o número de nós em uma árvore
function countNodes(node)
    local c = 1
    for i=1,#node.children do
        c = c + countNodes(node.children[i])
    end
    return c
end

-- Calcula a profundidade de uma árvore
function treeDepth(node)
    if #node.children == 0 then return 0 end
    local maxd = 0
    for i=1,#node.children do
        local d = treeDepth(node.children[i])
        if d > maxd then maxd = d end
    end
    return maxd + 1
end

-- Representa a árvore como string (S-expression)
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

-- Avalia uma árvore dado um percept (mapa de entradas)
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
-- 4. GERAÇÃO DE ÁRVORES ALEATÓRIAS
-- ---------------------------------------------------------------------------

-- Método "Ramped Half-and-Half" (Koza 1992):
--   - Metade das árvores geradas pelo método FULL (até depth máxima)
--   - Metade pelo método GROW (até depth máxima, mas pode parar antes)
-- Garante diversidade estrutural na população inicial.

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

-- Gera uma árvore pelo método FULL: todos os nós até 'depth' são funções;
-- nós em 'depth' são terminais.
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

-- Gera uma árvore pelo método GROW: pode parar em qualquer ponto com
-- probabilidade proporcional ao número de terminais vs. primitivas totais.
function generateGrow(depth)
    if depth <= 0 then
        return newNode_Terminal(randomTerminal())
    end
    local totalPrimitives = #FUNCTIONS + #TERMINALS
    if math.random() < #TERMINALS / totalPrimitives then
        -- Escolhe terminal (folha antecipada)
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

-- Gera uma árvore aleatória usando Ramped Half-and-Half
function generateRandomTree(maxDepth)
    if math.random(2) == 1 then
        return generateFull(math.random(1, maxDepth))
    else
        return generateGrow(math.random(1, maxDepth))
    end
end

-- ---------------------------------------------------------------------------
-- 5. INDIVÍDUO GP (CONJUNTO DE ÁRVORES — UMA POR BOTÃO)
-- ---------------------------------------------------------------------------

function newIndividual()
    local ind = {}
    ind.trees    = {}    -- uma árvore por botão de saída
    ind.fitness  = 0
    ind.size     = 0     -- total de nós em todas as árvores
    ind.evalFitness = 0  -- fitness bruto antes da parsimônia
    ind.generation  = 0  -- geração em que foi criado (para tracking)
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

-- Avalia todas as árvores de um indivíduo dado um percept; retorna o
-- mapa de botões a pressionar.
function evaluateIndividual(ind, percept)
    local outputs = {}
    for b=1,#ButtonNames do
        local val = evalTree(ind.trees[b], percept)
        outputs["P1 " .. ButtonNames[b]] = (val > 0)
    end
    return outputs
end

-- ---------------------------------------------------------------------------
-- 6. OPERADORES GENÉTICOS
-- ---------------------------------------------------------------------------

-- 6.1  Listagem de todos os nós de uma árvore (para seleção de ponto de corte)
function listNodes(node, lst)
    lst = lst or {}
    table.insert(lst, node)
    for i=1,#node.children do
        listNodes(node.children[i], lst)
    end
    return lst
end

-- 6.2  CROSSOVER DE SUBÁRVORE (Koza 1992)
-- Troca dois subárvores selecionados aleatoriamente entre pai1 e pai2.
-- Retorna dois filhos.
function crossover(ind1, ind2)
    local child1 = copyIndividual(ind1)
    local child2 = copyIndividual(ind2)

    -- Seleciona aleatoriamente um botão para realizar o crossover
    local b = math.random(#ButtonNames)

    local nodes1 = listNodes(child1.trees[b])
    local nodes2 = listNodes(child2.trees[b])

    if #nodes1 == 0 or #nodes2 == 0 then
        return child1, child2
    end

    -- Seleciona pontos de corte aleatórios
    local n1 = nodes1[math.random(#nodes1)]
    local n2 = nodes2[math.random(#nodes2)]

    -- Troca as subárvores (preservando profundidade máxima)
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

    -- Verificar se excede profundidade máxima; se sim, reverter
    if treeDepth(child1.trees[b]) > GP_MaxDepth or countNodes(child1.trees[b]) > GP_MaxNodes then
        -- Reverter child1
        n1.children = saved1_children
        n1.type     = saved1_type
        n1.func     = saved1_func
        n1.terminal = saved1_terminal
    end
    if treeDepth(child2.trees[b]) > GP_MaxDepth or countNodes(child2.trees[b]) > GP_MaxNodes then
        -- Reverter child2
        n2.children = saved1_children
        n2.type     = saved1_type
        n2.func     = saved1_func
        n2.terminal = saved1_terminal
    end

    child1.size = computeIndividualSize(child1)
    child2.size = computeIndividualSize(child2)
    return child1, child2
end

-- 6.3  MUTAÇÃO DE PONTO
-- Substitui um nó por outro da mesma aridade (função→função, terminal→terminal).
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

-- 6.4  MUTAÇÃO DE HOIST (Poli et al. 2008 §4.2.3)
-- Substitui um nó por um de seus descendentes (reduz tamanho da árvore).
function mutateHoist(ind)
    local b = math.random(#ButtonNames)
    local nodes = listNodes(ind.trees[b])
    -- Filtra apenas nós que têm filhos (não-terminais)
    local candidates = {}
    for _, n in ipairs(nodes) do
        if #n.children > 0 then
            table.insert(candidates, n)
        end
    end
    if #candidates == 0 then return end
    local n = candidates[math.random(#candidates)]
    -- Seleciona um filho e uma subárvore do filho
    local child = n.children[math.random(#n.children)]
    local childNodes = listNodes(child)
    local replacement = childNodes[math.random(#childNodes)]
    -- Substitui n pelo replacement (in-place)
    n.type     = replacement.type
    n.func     = replacement.func
    n.terminal = replacement.terminal
    n.children = replacement.children
    ind.size = computeIndividualSize(ind)
end

-- 6.5  MUTAÇÃO DE EXPANSÃO
-- Substitui um terminal por uma nova subárvore pequena.
function mutateExpansion(ind)
    local b = math.random(#ButtonNames)
    local nodes = listNodes(ind.trees[b])
    -- Filtra terminais
    local candidates = {}
    for _, n in ipairs(nodes) do
        if n.type == "terminal" then
            table.insert(candidates, n)
        end
    end
    if #candidates == 0 then return end
    local n = candidates[math.random(#candidates)]
    if treeDepth(ind.trees[b]) >= GP_MaxDepth then return end  -- não expande se já no limite
    local newSubtree = generateGrow(math.random(1, 2))  -- subárvore pequena
    n.type     = newSubtree.type
    n.func     = newSubtree.func
    n.terminal = newSubtree.terminal
    n.children = newSubtree.children
    ind.size = computeIndividualSize(ind)
end

-- 6.6  MUTAÇÃO DE COLAPSO
-- Substitui uma subárvore interna por um terminal aleatório.
function mutateCollapse(ind)
    local b = math.random(#ButtonNames)
    local nodes = listNodes(ind.trees[b])
    -- Filtra nós internos (não raiz, para preservar a raiz)
    local candidates = {}
    for i=2, #nodes do  -- i=2: pula raiz
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

-- 6.7  MUTAÇÃO DE SUBÁRVORE
-- Substitui uma subárvore inteira por uma nova subárvore gerada aleatoriamente.
function mutateSubtree(ind)
    local b = math.random(#ButtonNames)
    if math.random(2) == 1 then
        -- Substitui a raiz inteira
        ind.trees[b] = generateRandomTree(GP_MaxDepthInit)
    else
        local nodes = listNodes(ind.trees[b])
        if #nodes <= 1 then
            ind.trees[b] = generateRandomTree(GP_MaxDepthInit)
            return
        end
        local n = nodes[math.random(2, #nodes)]  -- pula raiz
        local newSub = generateRandomTree(math.random(1, 3))
        n.type     = newSub.type
        n.func     = newSub.func
        n.terminal = newSub.terminal
        n.children = newSub.children
    end
    ind.size = computeIndividualSize(ind)
end

-- Aplica todos os operadores de mutação a um indivíduo
function mutateIndividual(ind)
    if math.random() < GP_MutPoint then     mutatePoint(ind)     end
    if math.random() < GP_MutHoist then     mutateHoist(ind)     end
    if math.random() < GP_MutExpansion then mutateExpansion(ind) end
    if math.random() < GP_MutCollapse then  mutateCollapse(ind)  end
    if math.random() < GP_MutSubtree then   mutateSubtree(ind)   end
    ind.size = computeIndividualSize(ind)
end

-- ---------------------------------------------------------------------------
-- 7. SELEÇÃO POR TORNEIO
-- ---------------------------------------------------------------------------
-- Torneio determinístico: seleciona K indivíduos aleatoriamente e retorna
-- o melhor. Pressão seletiva controlada por GP_TournamentSize.

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
-- 8. APLICAÇÃO DE PARSIMÔNIA
-- ---------------------------------------------------------------------------
-- Penaliza indivíduos com muitos nós para controlar o crescimento de bloat.
-- fitness_penalizado = fitness_bruto - parsimony_penalty * max(0, size - threshold)
-- Referência: Poli et al. (2008), §4.3 — Lexicographic Parsimony Pressure

function applyParsimony(ind)
    local excess = math.max(0, ind.size - GP_ParsePenaltyStart)
    ind.fitness = ind.evalFitness - GP_ParsePenaltyRate * excess
end

-- ---------------------------------------------------------------------------
-- 9. PERCEPÇÃO DO AMBIENTE (compatível com MarIO.lua)
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

-- Constrói o percept: uma tabela com tiles e sprites indexados
function buildPercept()
    getPositions()
    local sprites = getSpritePositions()

    -- Inicializa grids
    local tileGrid   = {}
    local spriteGrid = {}
    for row=1, BoxRadius*2+1 do
        tileGrid[row]   = {}
        spriteGrid[row] = {}
        for col=1, BoxRadius*2+1 do
            tileGrid[row][col]   = 0
            spriteGrid[row][col] = -1  -- -1 = sem sprite
        end
    end

    -- Preenche tiles
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

    -- Preenche sprites
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
-- 10. POOL GP E GERENCIAMENTO DE GERAÇÃO
-- ---------------------------------------------------------------------------

function newGPPool()
    local p = {}
    p.population  = {}
    p.generation  = 0
    p.maxFitness  = 0
    p.currentInd  = 1
    p.currentFrame = 0
    p.logEntries  = {}   -- histórico de métricas por geração
    return p
end

-- Estatísticas da população atual
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
    -- Desvio padrão
    local varF = 0
    for _, ind in ipairs(gpPool.population) do
        varF = varF + (ind.fitness - meanF)^2
    end
    local stdF = math.sqrt(varF / n)
    return meanF, stdF, maxF, meanS
end

-- Evolui uma nova geração usando elitismo + torneio + crossover + mutação
function newGPGeneration()
    -- Aplica parsimônia a todos os indivíduos
    for _, ind in ipairs(gpPool.population) do
        applyParsimony(ind)
    end

    -- Ordena por fitness decrescente
    table.sort(gpPool.population, function(a,b) return a.fitness > b.fitness end)

    -- Atualiza fitness máximo
    if gpPool.population[1] and gpPool.population[1].fitness > gpPool.maxFitness then
        gpPool.maxFitness = gpPool.population[1].fitness
        -- Salva o melhor indivíduo
        writeGPFile("best_gp_gen" .. gpPool.generation .. ".gppool")
    end

    -- Log de métricas
    local meanF, stdF, maxF, meanS = computePopStats()
    local logEntry = string.format(
        "Gen %d | MaxF=%.1f MeanF=%.1f StdF=%.1f MeanSize=%.1f BestSize=%d",
        gpPool.generation, maxF, meanF, stdF, meanS,
        gpPool.population[1] and gpPool.population[1].size or 0
    )
    table.insert(gpPool.logEntries, logEntry)
    console.writeline(logEntry)

    -- Gravação explícita no arquivo de log do treinamento para acesso externo
    local resFile = io.open("gp_training_results.txt", "a")
    if resFile then
        resFile:write(logEntry .. "\n")
        local bestInd = gpPool.population[1]
        if bestInd then
            resFile:write("--- MELHOR POLITICA SIMBOLICA (GEN " .. gpPool.generation .. " | Fitness " .. math.floor(bestInd.fitness) .. ") ---\n")
            for b=1, #ButtonNames do
                resFile:write("  " .. ButtonNames[b] .. ": " .. treeToString(bestInd.trees[b]) .. "\n")
            end
            resFile:write("------------------------------------------------------------------------\n\n")
        end
        resFile:flush()
        resFile:close()
    end

    -- Elitismo: preserva os 2 melhores sem modificação
    local eliteCount = 2
    local newPop = {}
    for i=1, math.min(eliteCount, #gpPool.population) do
        table.insert(newPop, copyIndividual(gpPool.population[i]))
    end

    -- Preenche o restante com torneio + crossover + mutação
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

    -- Trunca se necessário
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
-- 11. CONTROLE DE EXECUÇÃO
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

    -- Conflitos L/R e U/D
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
-- 12. PERSISTÊNCIA (SALVAR / CARREGAR POOL GP)
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
        console.writeline("ERRO: Não foi possível abrir arquivo para escrita: " .. filename)
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
    -- Log de métricas
    file:write("LOG_START\n")
    for _, entry in ipairs(gpPool.logEntries) do
        file:write(entry .. "\n")
    end
    file:write("LOG_END\n")
    file:close()
end

-- Encontra um terminal pelo nome
function findTerminalByName(name)
    for _, t in ipairs(TERMINALS) do
        if t.name == name then return t end
    end
    -- Terminal desconhecido: retorna zero
    return { name=name, eval=function(p) return 0 end }
end

-- Encontra uma função pelo nome
function findFunctionByName(fname)
    for _, f in ipairs(FUNCTIONS) do
        if f.name == fname then return f end
    end
    return nil
end

function parseNode(s, pos)
    pos = pos or 1
    local typeChar = s:sub(pos, pos)  -- 'T' ou 'F'
    pos = pos + 2  -- pula 'T:' ou 'F:'

    if typeChar == "T" then
        -- Lê nome do terminal até '|' ou '\0'
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
        -- Lê nome da função
        local colonPos = s:find(":", pos, true)
        local fname = s:sub(pos, colonPos-1)
        pos = colonPos + 1
        -- Lê aridade
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
            -- Fallback: terminal constante
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
        console.writeline("ERRO: Arquivo não encontrado: " .. filename)
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
        -- Garante que todos os botões têm árvores
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
-- 13. EXIBIÇÃO VISUAL (GUI)
-- ---------------------------------------------------------------------------

-- Exibe o melhor indivíduo da geração atual com suas expressões simbólicas
function displayGP(ind)
    if ind == nil then return end

    -- Painel de fundo
    gui.drawBox(0, 0, 300, 200, 0xC0000000, 0xC0111111)

    -- Título
    gui.drawText(4, 2, "MarI/O-GP  Gen:" .. gpPool.generation, 0xFFFFFF00, 10)
    gui.drawText(4, 14, "MaxFit:" .. math.floor(gpPool.maxFitness) ..
                        "  Size:" .. ind.size, 0xFFFFFF00, 10)

    -- Exibe a grade de tiles (igual ao NEAT)
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
                color = 0xFFFF0000  -- vermelho = inimigo
            elseif tv == 1 then
                color = 0xFFAAAAAA  -- cinza = tile sólido
            else
                color = 0xFF222222  -- vazio
            end
            local px = gridX + (col-1)*cellSize
            local py = gridY + (row-1)*cellSize
            gui.drawBox(px, py, px+cellSize-1, py+cellSize-1, color, color)
        end
    end

    -- Exibe o estado dos botões ativos
    local btnX = gridX + (BoxRadius*2+1)*cellSize + 8
    local btnY = 30
    for b=1, #ButtonNames do
        local pressed = gpController["P1 " .. ButtonNames[b]]
        local color = pressed and 0xFF00FF00 or 0xFF444444
        gui.drawText(btnX, btnY + (b-1)*12, ButtonNames[b], color, 10)
    end

    -- Exibe a expressão simbólica do primeiro botão (Right — mais informativo)
    -- Truncada para caber na tela
    local exprY = 140
    gui.drawText(4, exprY, "Right:", 0xFFAAAAFF, 9)
    local expr = treeToString(ind.trees[#ButtonNames])  -- último = Right
    if #expr > 55 then expr = expr:sub(1, 52) .. "..." end
    gui.drawText(4, exprY+10, expr, 0xFFFFFFFF, 8)

    -- Barra de progresso da geração
    local total = GP_Population
    local measured = gpPool.currentInd - 1
    local pct = total > 0 and math.floor(measured/total*100) or 0
    gui.drawText(4, exprY+22,
        "Avaliando " .. gpPool.currentInd .. "/" .. total ..
        " (" .. pct .. "%)", 0xFFCCCCCC, 8)
end

-- ---------------------------------------------------------------------------
-- 14. PLAY TOP  — executa o melhor indivíduo encontrado
-- ---------------------------------------------------------------------------

function gpPlayTop()
    table.sort(gpPool.population, function(a,b) return a.fitness > b.fitness end)
    gpPool.currentInd = 1
    gpPool.maxFitness = gpPool.population[1] and gpPool.population[1].fitness or 0
    forms.settext(gpMaxFitnessLabel, "Max Fitness: " .. math.floor(gpPool.maxFitness))
    initializeGPRun()
end

-- Imprime a expressão do melhor indivíduo no console
function gpPrintBest()
    table.sort(gpPool.population, function(a,b) return a.fitness > b.fitness end)
    local best = gpPool.population[1]
    if best == nil then return end
    console.writeline("=== MELHOR INDIVIDUO GP (Gen " .. gpPool.generation .. ") ===")
    console.writeline("Fitness: " .. best.fitness .. "  Tamanho: " .. best.size)
    for b=1, #ButtonNames do
        console.writeline("Botao " .. ButtonNames[b] .. ":")
        console.writeline("  " .. treeToString(best.trees[b]))
    end
    console.writeline("=== FIM ===")
end

-- ---------------------------------------------------------------------------
-- 15. FORMULÁRIO DE CONTROLE
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
-- 16. LOOP PRINCIPAL
-- ---------------------------------------------------------------------------

gpController = {}
gpRightmost  = 0
gpTimeout    = TimeoutConstant

-- Acelera emulação para treinamento turbo
pcall(function() client.speedmode(600) end)

while true do
    -- Background banner
    if not forms.ischecked(gpHideBanner) then
        gui.drawBox(0, 0, 300, 26, 0xD0FFFFFF, 0xD0FFFFFF)
    end

    local ind = gpPool.population[gpPool.currentInd]

    -- Exibição visual
    if forms.ischecked(gpShowNetwork) then
        displayGP(ind)
    end

    -- Avaliação da rede (a cada 5 frames para performance)
    if gpPool.currentFrame % 5 == 0 then
        evaluateGPCurrent()
    end

    joypad.set(gpController)

    -- Lógica de timeout e fitness
    getPositions()
    if marioX > gpRightmost then
        gpRightmost = marioX
        gpTimeout   = TimeoutConstant
    end

    gpTimeout = gpTimeout - 1

    local timeoutBonus = gpPool.currentFrame / 4
    if gpTimeout + timeoutBonus <= 0 then
        -- Calcula fitness do indivíduo atual
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

        -- Avança para o próximo indivíduo
        gpPool.currentInd = gpPool.currentInd + 1
        if gpPool.currentInd > #gpPool.population then
            newGPGeneration()
        end
        initializeGPRun()
    end

    -- HUD de informações
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
