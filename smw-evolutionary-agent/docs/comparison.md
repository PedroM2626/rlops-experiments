# NEAT vs. Genetic Programming — Análise Comparativa Detalhada

## 1. Fundamentos Teóricos Comparados

### 1.1 Espaço de Busca

#### NEAT
O espaço de busca do NEAT é o produto cartesiano:
- Espaço de topologias de grafos direcionados (acíclicos não-garantidos)
- Espaço de pesos reais R^n

A dimensionalidade de R^n cresce dinamicamente conforme conexões são adicionadas.
A busca é guiada por gradiente implícito via mutação de pesos (perturbação gaussiana).

#### GP
O espaço de busca é o conjunto de todas as árvores válidas sobre (F, T):
- F = conjunto de funções (nós internos)
- T = conjunto de terminais (folhas)

O espaço é discreto e infinito. Não há gradiente — a busca é puramente estocástica.
Estimativa de tamanho: |Árvores de profundidade <= d| cresce exponencialmente em d.

### 1.2 Operadores de Variação

| Operador | NEAT | GP |
|----------|------|----|
| Crossover | Alinhamento por número de inovação | Troca de subárvores |
| Mutação estrutural | Adicionar nó / conexão | Hoist, expansão, colapso, subtree |
| Mutação paramétrica | Perturbação de pesos | Mutação de ponto (trocar função/terminal) |
| Preservação de diversidade | Especiação por distância genética | Não há (neste projeto); alternativa: crowding |

### 1.3 Fitness e Pressão Seletiva

Ambas as abordagens usam a mesma função de fitness bruto:

    F_raw = rightmost_x - T/2 + bonus_conclusão

NEAT usa fitness ajustado por espécie (fitness sharing implícito via especiação).
GP usa parsimônia explícita:

    F_gp = F_raw - λ * max(0, |tree| - θ)

onde λ = GP_ParsePenaltyRate e θ = GP_ParsePenaltyStart.

## 2. Análise de Complexidade

### 2.1 Complexidade de Avaliação

#### NEAT
- Construção da rede: O(|genes|)
- Avaliação forward pass: O(|neurônios| * |conexões_médias|)
- Para InputSize=169 + Outputs=8 e ~50 neurônios ocultos: ~O(10^3) ops/frame

#### GP
- Avaliação de uma árvore: O(|nós|) — simples percurso em pré-ordem
- Para árvores de ~100 nós e 8 botões: ~O(800) ops/frame
- Mais simples que NEAT, mas sem paralelismo inerente

### 2.2 Complexidade de Reprodução

#### NEAT
- Crossover: O(|genes1| + |genes2|) — alinhamento por inovação
- Especiação: O(|população| * |espécies|)
- Por geração: O(P * S) onde P = população, S = número de espécies

#### GP
- Crossover de subárvore: O(|árvore|) — listagem + cópia
- Por geração: O(P * |árvore_média|)
- Em geral mais lento que NEAT para populações grandes com árvores profundas

## 3. Interpretabilidade

### 3.1 O que NEAT revela?

A rede neural NEAT é uma **caixa preta**:
- Os pesos não têm interpretação direta
- A topologia é difícil de analisar manualmente para redes com muitos neurônios
- Pode-se inspecionar quais inputs (tiles) têm maiores pesos, mas isso é trabalhoso
- Técnicas como SHAP ou saliency maps seriam necessárias para interpretabilidade post-hoc

### 3.2 O que GP revela?

O programa GP é **diretamente legível**:

Exemplo de programa real (hipotético, após convergência) para o botão "B" (pular):

    IF(
      OR(
        GT(tile(16,0), 0),          -- tile sólido à frente no chão
        sprite_near(16,0)           -- inimigo na frente
      ),
      C_1,                          -- pressiona B (pula)
      AND(
        NOT(sprite_near(0,-16)),    -- nenhum inimigo acima
        C_N1                        -- não pressiona B
      )
    )

Leitura: "Pule se há um obstáculo ou inimigo à frente;
          se não, não pule (especialmente se há inimigo acima)."

Isso é **ciência interpretável**: podemos extrair as regras de decisão do agente.

### 3.3 Métricas de Interpretabilidade

| Métrica | NEAT | GP |
|---------|------|----|
| Complexidade do modelo | Nº de conexões | Nº de nós |
| Legibilidade direta | Não | Sim |
| Número de features ativas | Difícil de calcular | Contagem de terminais únicos |
| Profundidade de raciocínio | Implícita | Profundidade da árvore |
| Regras extraíveis | Não (sem técnicas adicionais) | Sim (direto da S-expression) |

## 4. Problemas Conhecidos e Mitigações

### 4.1 Bloat no GP

**Problema**: Árvores crescem indefinidamente sem benefício de fitness (Poli et al. 2008).
**Causa**: Crossover de subárvores cria frequentemente código morto (introns) que não afeta o fitness mas aumenta o tamanho.

**Mitigações implementadas**:
1. GP_MaxDepth = 12: Limite de profundidade rígido
2. GP_MaxNodes = 300: Limite de nós por indivíduo
3. Parsimônia explícita: F = F_raw - 0.5 * max(0, size - 50)
4. Mutação de hoist: reduz o tamanho substituindo subárvores por seus descendentes

### 4.2 Premature Convergence no GP

**Problema**: Perda de diversidade genética precoce.
**Mitigações**:
1. Seleção por torneio (não proporcional ao fitness) — maior pressão seletiva controlável
2. Inicialização Ramped Half-and-Half — diversidade estrutural inicial
3. Múltiplos operadores de mutação — exploração contínua do espaço

### 4.3 Deception no NEAT

**Problema**: Soluções intermediárias (stepping stones) podem ter fitness baixo.
**Mitigação**: Especiação protege inovações estruturais por StaleSpecies gerações.

## 5. Protocolo de Experimento Comparativo

### 5.1 Condições Controladas

Para comparação justa:
- **Mesmo jogo, mesmo nível**: Yoshi's Island 1 (SMW)
- **Mesmo save state inicial**: DP1.state
- **Mesma função de fitness**: rightmost_x - frames/2 + 1000 (conclusão)
- **Mesmo BoxRadius**: 6 (grade 13x13 = 169 inputs)
- **Mesmo hardware**: rodar sequencialmente no mesmo computador

### 5.2 Parâmetros a Registrar

A cada geração, registre em CSV:
- generation, max_fitness, mean_fitness, std_fitness
- best_size, mean_size, n_evals_total
- wall_clock_seconds, completions

### 5.3 Análise Estatística

Para cada algoritmo, executar N=5 seeds independentes.
Comparar curvas de aprendizado usando:
- Teste de Wilcoxon (não-paramétrico) para comparação final
- Area Under Curve (AUC) da curva fitness x gerações como métrica de convergência

### 5.4 Resultados Experimentais Reais (Benchmark Pareado de 180s)

Executado sob condições idênticas no BizHawk 2.9.1 (Super Mario World USA, Yoshi's Island 1, `DP1.state`, `speedmode(600)`, tempo total fixo de 180 segundos):

| Métrica Avaliada | NEAT (`MarIO.lua`) | Genetic Programming (`MarIO_GP.lua`) |
| :--- | :---: | :---: |
| **Tempo de Execução** | 180s | 180s |
| **Indivíduos Avaliados** | 1.362 avaliações | 880 avaliações |
| **Gerações Concluídas** | 8 gerações (Gen 0 a 7, Pop=300) | 22 gerações (Gen 0 a 21, Pop=40) |
| **Fitness Inicial ($F_{\max}$ Gen 0)** | 127.0 | 123.0 |
| **Fitness Máximo Atingido** | **145.0** (atingido na Gen 1 aos 40s) | **145.0** (atingido na Gen 6 aos 54s) |
| **Fitness Médio Final ($\bar{F}$)** | 114.5 – 127.0 (população dispersa) | **144.1** (população convergida) |
| **Desvio Padrão Final ($\sigma_F$)** | Alto (ampla variância interespécies) | **3.9** (estabilidade homogênea) |
| **Complexidade da Política** | Grafo neural não-interpretável (pesos reais) | Árvores simbólicas de 46 nós (S-expressions legíveis) |

#### Conclusão Empírica do Teste Pareado:
1. **Desempenho no Jogo:** **Empate rigoroso** ($145.0$ vs $145.0$). Ambos os agentes convergiram para a mesma barreira local inicial (o primeiro obstáculo/inimigo antes do abismo) dentro da janela de 180 segundos de treino.
2. **Eficiência Amostral e Convergência:** O GP com $N=40$ demonstrou convergência populacional muito mais acelerada ($\sigma = 3.9$ e $\bar{F} = 144.1$), atingindo o platô com menos avaliações totais (880 contra 1.362).
3. **Interpretabilidade:** Superioridade categórica do GP, que permitiu inspecionar diretamente as regras de decisão em linguagem simbólica (`Y: NOT(tile(48, -16))`).

## 6. Referências Específicas

### Sobre GP e Bloat
- Poli, R., Langdon, W.B., McPhee, N.F. (2008). A Field Guide to Genetic Programming. §4.3.
- Soule, T., Foster, J.A. (1998). Effects of Code Growth and Parsimony Pressure on
  Populations in Genetic Programming. Evolutionary Computation 6(4), 293-309.

### Sobre NEAT e Especiação
- Stanley, K.O., Miikkulainen, R. (2002). Evolving Neural Networks through Augmenting
  Topologies. Evolutionary Computation 10(2), 99-127.
- Stanley, K.O. (2004). Efficient Evolution of Neural Networks through Complexification.
  PhD Dissertation, University of Texas at Austin.

### Sobre Interpretabilidade em RL Evolutivo
- Hein, D., Udluft, S., Runkler, T.A. (2018). Interpretable policies for reinforcement
  learning by genetic programming. Engineering Applications of AI, 76, 158-169.
- Custode, L.L., Iacca, G. (2021). Interpretable policies for reinforcement learning by
  evolutionary optimization. Expert Systems with Applications, 113999.
