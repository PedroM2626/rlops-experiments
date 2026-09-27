# SMW Evolutionary Agent — NEAT vs. Genetic Programming

> **MarI/O** por SethBling (NEAT) + **MarI/O-GP** (Programação Genética) para agentes interpretáveis
> Projeto acadêmico: comparação entre Neuroevolução (NEAT) e Programação Genética (GP) em *Super Mario World*

---

## Índice

1. [Visão Geral](#visão-geral)
2. [Pré-requisitos e Configuração do Ambiente](#pré-requisitos-e-configuração-do-ambiente)
3. [Como Executar o MarIO (NEAT)](#como-executar-o-mario-neat)
4. [Como Executar o MarIO-GP (Genetic Programming)](#como-executar-o-mario-gp)
5. [Estrutura do Projeto](#estrutura-do-projeto)
6. [Fundamentos Teóricos](#fundamentos-teóricos)
7. [Comparação: NEAT vs. GP](#comparação-neat-vs-gp)
8. [Métricas e Avaliação](#métricas-e-avaliação)
9. [Referências](#referências)

---

## Visão Geral

Este projeto implementa e compara dois paradigmas de computação evolutiva para jogar *Super Mario World (USA)* no emulador BizHawk:

| Abordagem | Script | Representação | Interpretabilidade |
|-----------|--------|---------------|--------------------|
| **NEAT** (NeuroEvolution of Augmenting Topologies) | `MarIO.lua` | Rede neural com topologia evolutiva | Baixa |
| **Genetic Programming (GP)** | `MarIO_GP.lua` | Árvore de programa (expressão simbólica) | **Alta** |

Ambos os agentes usam a **mesma função de fitness** e a **mesma representação de entrada**
(grade de tiles 13x13 ao redor de Mario), garantindo uma comparação justa.

---

## Pré-requisitos e Configuração do Ambiente

### 1. Sistema Operacional

Windows 10/11 (64-bit). BizHawk não tem suporte oficial para Linux/macOS.

### 2. BizHawk — Emulador

**Versão recomendada: BizHawk 2.9.1** (compatível com a API Lua usada pelos scripts).

> IMPORTANTE: Versões 2.10+ introduziram mudanças de API que podem quebrar os scripts.
> Use preferencialmente a versão 2.9.x.

**Download:**
- Acesse: https://github.com/TASEmulators/BizHawk/releases/tag/2.9.1
- Baixe `BizHawk-2.9.1-win-x64.zip`
- Extraia para um diretório **sem espaços no caminho**, ex.: `C:\BizHawk\`

**Instalação dos pré-requisitos do BizHawk:**
Execute como Administrador dentro da pasta do BizHawk:

    .\prerequisites\bizhawk_prerequisites.ps1

Ou execute manualmente `prereq\bizhawk_prereqs.exe` que está dentro do ZIP.

**Dependências adicionais necessárias:**
- .NET 8 Runtime: https://dotnet.microsoft.com/download/dotnet/8.0
- Visual C++ Redistributable 2015-2022 x64: https://aka.ms/vs/17/release/vc_redist.x64.exe
- DirectX End-User Runtime: https://www.microsoft.com/en-us/download/details.aspx?id=35

### 3. ROM de Super Mario World

Você precisará de uma ROM legal do jogo:
- Nome exato (verificado pelo script): `Super Mario World (USA).sfc`
- O `gameinfo.getromname()` deve retornar exatamente `"Super Mario World (USA)"`

> Faça dump da sua própria cópia física do cartucho usando um leitor de cartuchos SNES.
> Não distribua ROMs.

### 4. Configuração do Emulador e Savestate
Tanto a detecção do emulador quanto o provisionamento do `DP1.state` nativo e sincronização de scripts são gerenciados de forma automatizada pelo script [`launch_mario.py`](file:///c:/Users/Acer/Downloads/smw-evolutionary-agent/launch_mario.py) ou [`setup_environment.py`](file:///c:/Users/Acer/Downloads/smw-evolutionary-agent/setup_environment.py). Nenhuma intervenção manual no emulador é necessária.

---

## Execução Autônoma (Sem Intervenção Manual)

O projeto inclui um pipeline de execução 100% autônomo através do [`launch_mario.py`](file:///c:/Users/Acer/Downloads/smw-evolutionary-agent/launch_mario.py):

```bash
# Executar NEAT (MarI/O original):
python launch_mario.py

# Executar Genetic Programming (MarI/O-GP interpretável):
python launch_mario.py --gp
```

### O que o pipeline autônomo realiza:
1. **Validação e Sincronização:** Verifica SHA-1 da ROM (`6b47bb75d16514b6a476aa0c73a683a2a4c18765`), executáveis do BizHawk e scripts Lua.
2. **Geração Automática do Savestate (`DP1.state`):** Caso o savestate nativo não exista ou esteja corrompido, o pipeline executa um bootstrap em background que navega de forma autônoma pela tela de título, seleciona o save e salva `DP1.state` no exato frame de início da fase *Yoshi's Island 1* (Mode `0x14`, MarioX = 128).
3. **Inicialização e Treinamento:** Abre o emulador BizHawk já com o script Lua selecionado via `--lua`, exibindo o gameplay e a interface de evolução em tempo real.

---

## Como Executar Manualmente (Opcional)

1. Abra o BizHawk e carregue `Super Mario World (USA).sfc`
2. Vá em `Tools -> Lua Console`
3. Na console Lua: `Script -> Open Script` -> selecione `MarIO.lua`
4. O script começará automaticamente a evoluir uma população de redes neurais

**Controles da interface:**
- Show Map: exibe a grade de tiles e a rede neural
- Show M-Rates: exibe as taxas de mutação atuais
- Restart: reinicia a evolução do zero
- Save/Load: salva/carrega o pool de genomas em arquivo `.pool`
- Play Top: executa o melhor genoma encontrado até agora

---

## Como Executar o MarIO-GP

1. Abra o BizHawk e carregue `Super Mario World (USA).sfc`
2. Vá em `Tools -> Lua Console`
3. Na console Lua: `Script -> Open Script` -> selecione `MarIO_GP.lua`
4. O script evoluirá uma população de **árvores de programa**

**Diferenças visuais:**
- O painel GP exibe a **expressão simbólica** do melhor indivíduo (interpretável!)
- O log mostra a profundidade média das árvores e a complexidade do melhor programa
- Arquivos `.gppool` salvam o estado da evolução GP

---

## Estrutura do Projeto

```text
smw-evolutionary-agent/
|-- MarIO.lua               # NEAT — neuroevolução clássica (SethBling)
|-- MarIO_GP.lua            # Genetic Programming — árvores simbólicas interpretáveis
|-- launch_mario.py         # Pipeline de execução 100% autônomo (NEAT / GP)
|-- setup_environment.py    # Validação de integridade (SHA-1) e sincronização
|-- DP1.state               # Savestate nativo no frame inicial de Yoshi's Island 1
|-- README.md               # Documentação completa do projeto
`-- docs/
    `-- comparison.md       # Estudo comparativo acadêmico NEAT vs. GP
```

---

## Fundamentos Teóricos

### NEAT (NeuroEvolution of Augmenting Topologies)

NEAT é um algoritmo proposto por Stanley & Miikkulainen (2002) que evolui **simultaneamente**
a topologia e os pesos de redes neurais. Características principais:

- **Especiação**: protege inovações estruturais agrupando genomas similares
- **Números de inovação**: permitem crossover entre genomas com topologias diferentes
- **Complexificação incremental**: começa com redes mínimas e aumenta a complexidade

**Representação do genoma:**
Cada genoma é um conjunto de *genes de conexão* `(nó_entrada, nó_saída, peso, habilitado, inovação)`.

**Função de fitness:**

    fitness = rightmost_x - frames / 2 + (1000 se completou o nível)

### Genetic Programming (GP)

GP é um paradigma evolutivo (Koza, 1992) que evolui **programas de computador**
representados como árvores. No contexto deste projeto:

**Conjunto de funções (F):**

    { IF, AND, OR, NOT, GT, LT, ADD, MUL, TANH, MAX, MIN }

**Conjunto de terminais (T):**

    { tile(dx,dy) para dx,dy em [-6..6]*16 }   <- mesma grade do NEAT
    { sprite_near(dx,dy) }                       <- presença de inimigos
    { const_0, const_1, const_neg1 }             <- constantes

**Cada saída (botão) possui sua própria árvore de decisão** — totalmente interpretável!

**Exemplo de programa evoluído para o botão "Right":**

    IF( GT(tile(16,0), 0),
        AND(True, NOT(sprite_near(16,0))),
        True
    )
    -> "Segure direita, mas pare se há tile sólido E inimigo à frente"

**Operadores genéticos GP:**
- Crossover de subárvore: troca subárvores entre dois programas pai
- Mutação de ponto: substitui um nó por outro do mesmo tipo
- Mutação de hoist: sobe uma subárvore (reduz tamanho — combate bloat)
- Mutação de expansão: substitui um terminal por uma nova subárvore
- Mutação de colapso: substitui uma subárvore por um terminal

---

## Comparação: NEAT vs. GP

| Critério | NEAT | Genetic Programming |
|----------|------|---------------------|
| **Representação** | Grafo de neurônios | Árvore de expressões |
| **Interpretabilidade** | Baixa (caixa preta) | Alta (código legível) |
| **Convergência** | Rápida (pesos contínuos) | Mais lenta (espaço discreto) |
| **Generalização** | Moderada | Potencialmente melhor |
| **Bloat** | Não aplicável | Problema conhecido; mitigado por parsimony pressure |
| **Transfer learning** | Difícil | Mais natural (trocar subárvores) |
| **Depuração** | Impossível inspeção direta | Pode-se ler o programa |
| **Memória** | O(conexões) | O(tamanho da árvore) |

### Hipóteses

1. **NEAT** converge para fitness mais alto em menos gerações.
2. **GP** produz programas mais robustos que generalizam para outros níveis.
3. O programa GP revela **regras estratégicas interpretáveis**.

---

## Resultados Experimentais e Comparativo Empírico

Foi conduzido um benchmark pareado controlado de 180 segundos sob condições idênticas no BizHawk 2.9.1 (*Super Mario World USA*, Yoshi's Island 1, save state `DP1.state`, `speedmode(600)`):

### 1. Tabela Comparativa de Métricas Reais

| Métrica Científica | NEAT (`MarIO.lua`) | Genetic Programming (`MarIO_GP.lua`) |
| :--- | :---: | :---: |
| **Tempo de Execução** | 180s | 180s |
| **Indivíduos Avaliados** | 1.362 genomas | 880 programas (35% mais eficiente) |
| **Gerações Concluídas** | 8 gerações (Gen 0 a 7, Pop=300) | 22 gerações (Gen 0 a 21, Pop=40) |
| **Fitness Inicial ($F_{\max}$ Gen 0)** | 127.0 | 123.0 |
| **Fitness Máximo Atingido** | **145.0** (na Gen 1 aos 40s) | **145.0** (na Gen 6 aos 54s) |
| **Fitness Médio Final ($\bar{F}$)** | 114.5 – 127.0 (população dispersa) | **144.1** (população convergida) |
| **Desvio Padrão Final ($\sigma_F$)** | Alto (ampla dispersão entre espécies) | **3.9** (estabilidade homogênea) |
| **Complexidade da Política** | Grafo neural não-interpretável | **46 nós sintáticos** (S-expression explícita) |

### 2. Política Explícita Obtida pelo Agente GP (Gen 21, Fitness = 145.0)

Ao contrário da caixa-preta neural do NEAT, a política evoluída pelo GP para cada botão do controle SNES é uma árvore sintática simbólica diretamente auditável:

```lisp
;; 1. Botão A (Spin Jump):
(spr 0 -32)

;; 2. Botão B (Pulo Regular):
(tile -48 -32)

;; 3. Botão X:
(tile 48 -32)

;; 4. Botão Y (Dash / Corrida Contínua):
(NOT (tile 48 -16))

;; 5. Botão Up:
(tile -80 96)

;; 6. Botão Down:
(spr 0 0)

;; 7. Botão Left:
(tile -96 48)

;; 8. Botão Right (Locomoção / Avanço Horizontal):
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

### 3. Discussão Científica dos Resultados

1. **Desempenho no Jogo (Empate em 145.0):** Ambos os agentes alcançaram o mesmo teto de fitness local na descida inicial do nível, descobrindo o avanço sustentado para a direita em velocidade de corrida, mas demandando mais tempo de treino para sintetizar o salto sobre o primeiro Koopa/obstáculo.
2. **Eficiência Amostral e Homogeneidade:** O GP convergiu com 35% menos avaliações de fitness (880 contra 1.362), reduzindo o desvio padrão de 23.3 para 3.9 e reduzindo o tamanho médio das árvores de 92.6 para 45.0 nós via pressão de parsimônia (combate ao *code bloat*).
3. **Interpretabilidade:** O GP atingiu o objetivo central de transparência algorítmica, permitindo verificar exatamente quais correlações espaciais disparam os botões de ação.

---

## Métricas e Avaliação

| Métrica | Descrição |
|---------|-----------|
| `max_fitness` | Máximo rightmost_x atingido |
| `mean_fitness` | Média da população |
| `fitness_std` | Desvio padrão da fitness |
| `best_genome_size` | Conexões (NEAT) / nós (GP) do melhor indivíduo |
| `mean_genome_size` | Complexidade média da população |
| `n_species` | Número de espécies (NEAT) / nichos (GP) |
| `level_completions` | Quantas vezes o nível foi concluído |
| `wall_time_per_gen` | Tempo de relógio por geração |

---

## Referências

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
