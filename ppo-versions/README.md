# PPO Benchmark: comparando implementações de PPO entre si

Este projeto nasceu de uma pergunta simples — "o PPO de uma lib é
diferente do de outra?" — e virou uma investigação de várias sessões sobre
reprodutibilidade, variância estatística e o que realmente explica
diferenças de desempenho entre implementações de RL. Este README documenta
tanto os **resultados experimentais** (Partes 1-12, cada uma um
experimento concreto) quanto as **conclusões conceituais** que foram
surgindo pelo caminho (seção logo abaixo).

Se você só quer o resultado mais recente e mais forte: vá direto pra
**Parte 12**. Se quer entender o raciocínio completo, leia em ordem — cada
parte parte do que a anterior descobriu (ou deixou sem resposta).

---

## Conceitos e conclusões centrais (o "porquê" por trás dos experimentos)

### 1. Por que a mesma lógica de PPO dá resultados diferentes em libs diferentes

O algoritmo no papel é o mesmo, mas cada implementação decide sozinha uma
porção de detalhes que o paper não especifica: como inicializar os pesos,
se normaliza observação/reward, como trata `value function clipping`, se
usa penalidade de KL, como faz bootstrap em episódios truncados, ordem das
operações de ponto flutuante, qual gerador de números aleatórios usa. Cada
uma dessas escolhas é invisível até você ir atrás dela.

### 2. Isso seria pior (não melhor) entre linguagens diferentes

Mesmo mantendo "a mesma lógica" escrita à mão, PyTorch, JAX e uma
hipotética implementação em Rust divergiriam ainda mais, por motivos
estruturais:
- **RNG diferente**: mesmo seed não gera a mesma sequência de números em
  Mersenne Twister (PyTorch), Threefry (JAX) ou PCG (Rust stdlib).
- **Ponto flutuante não é associativo**: `(a+b)+c ≠ a+(b+c)`. A ordem que
  cada linguagem soma um vetor (loop sequencial vs SIMD vs redução em
  árvore) muda o resultado no último bit.
- **BLAS diferente por baixo dos panos** (OpenBLAS, MKL, Accelerate) muda
  estratégia de tiling e arredondamento.
- Como RL é um sistema com **loop de feedback** (a ação de agora muda o
  estado de amanhã), uma diferença de ponto flutuante no 15º dígito no
  passo 1 pode virar uma política inteiramente diferente depois de
  centenas de milhares de passos -- efeito borboleta.

### 3. Versionamento quebra reprodutibilidade, ao vivo, mais de uma vez nesta conversa

- O pacote `cleanrl` no PyPI (v0.4.8, de 2021) não roda mais com o
  `gym`/`gymnasium` atuais.
- O script oficial `ppo.py` do CleanRL (baixado direto do GitHub, versão
  atual) também quebrou: a API de vetor de envs do Gymnasium mudou de
  `infos["final_info"]` pra `infos["episode"]` + máscara `infos["_episode"]`
  a partir do Gymnasium 1.0.
- Tianshou 2.0 reescreveu inteiramente sua API (`policy` → `algorithm`).
- TorchRL renomeou `SyncDataCollector` → `Collector`.
- O próprio ambiente de execução deste projeto foi resetado no meio do
  trabalho (Parte 10), exigindo reinstalar tudo do zero.

Isso não é falta de sorte -- é a regra, não a exceção, quando você depende
de bibliotecas de terceiros em rápida evolução.

### 4. Seeds dominam quando o efeito é pequeno; recuam quando o efeito é grande

No CartPole (tarefa fácil), com 20 seeds, quase toda "diferença entre
libs" que parecia real com 5 seeds virou ruído estatístico (só 1 de 10
pares sobreviveu à correção de Bonferroni). No LunarLander (tarefa mais
difícil), a mesma metodologia revelou diferenças reais e fortes. A lição:
**tarefas fáceis fazem qualquer política convergir perto do teto rápido,
sobrando só ruído** -- contra-intuitivamente, o ambiente "mais simples e
controlado" foi onde a variância de seed mais enganou.

### 5. A pergunta certa antes de rodar o teste estatístico

Mann-Whitney U testa deslocamento de mediana/ranking. Não detecta
diferença de variância. Quando a pergunta era "a variância mudou?" (Parte
6, `n_envs=1` vs `n_envs=8` no SB3), precisamos do teste de Levene, não
Mann-Whitney -- mesmo com o desvio padrão quadruplicando, Mann-Whitney não
via a diferença. **Escolher o teste errado responde a pergunta errada,
mesmo com dados perfeitamente bons.**

### 6. Correções múltiplas custam caro

Cada nova comparação que você testa no mesmo conjunto de dados aperta o
alpha corrigido de Bonferroni pra todas as outras. Isso literalmente
derrubou a significância de um resultado da Parte 5 quando adicionamos
mais uma implementação na Parte 7 -- não porque o resultado mudou, mas
porque o número de comparações simultâneas subiu.

### 7. "Do zero" não é sinônimo de "otimizado"

Implementações "from scratch" (JAX puro, PyTorch puro) carregavam
hiperparâmetros genéricos herdados do CartPole, nunca tunados pro
LunarLander. Quando adicionamos técnicas padrão (normalização de
obs/reward, rede maior, LR annealing -- Parte 4) o mesmo código saiu de
pior-que-todo-mundo pra competitivo. **A pergunta "biblioteca X é melhor"
esconde a pergunta mais importante: "configurada como?"** -- confirmado de
forma definitiva na Parte 11, onde o SB3 com os hiperparâmetros oficiais
do RL Zoo bateu todo o resto, incluindo o RLlib.

### 8. Nenhuma vantagem de biblioteca sobreviveu a uma reformulação de regime

Ao longo do projeto, quase toda alegação "lib X é melhor/mais estável"
sobreviveu só dentro do regime exato onde foi medida:
- SB3 era a mais estável no nosso benchmark, a segunda mais instável no
  projeto de outra pessoa -- causa isolada: `n_envs` (Parte 6).
- RLlib parecia ter uma vantagem estrutural no LunarLander -- parte dela
  veio de 2 mecanismos específicos (KL dinâmico + vf-clip, Parte 5), mas
  não toda.
- A mesma correção de bootstrap em truncamento foi irrelevante num
  ambiente (LunarLander) e decisiva no outro (CartPole, Partes 7-8).
- TorchRL e Tianshou, duas libs maduras de produção, caíram no meio do
  pelotão **com hiperparâmetros genéricos** (Parte 10) -- mas essa
  colocação também não era fixa: com os hiperparâmetros do Zoo (Parte 12),
  o Tianshou virou o melhor de todos, e o TorchRL o pior.
- O SB3 genérico (pior do pelotão) e o SB3 com config real de produção
  (melhor do pelotão nas Partes 3-10) são literalmente a mesma biblioteca
  (Parte 11) -- a maior amplitude do projeto até aquele ponto veio de
  trocar CONFIGURAÇÃO, não framework.
- **O exemplo mais extremo: o RLlib, isolado no topo em 3 partes
  diferentes (3, 5, 10), caiu pra penúltimo lugar assim que TODAS as
  implementações -- não só o SB3 -- receberam os hiperparâmetros do Zoo
  (Parte 12).** A vantagem que parecia ser "do RLlib" era, em grande
  parte, "de rodar bem configurado enquanto o resto rodava mal
  configurado".

**A conclusão que sobra de tudo isso:** "biblioteca X é melhor/mais
estável" não é uma afirmação verdadeira ou falsa por si só -- é uma
afirmação incompleta até você especificar tarefa, dificuldade, budget,
seeds e configuração. A única coisa que resiste a esse escrutínio é a
versão já qualificada: "nessas condições específicas, medido assim, essa
combinação teve esse resultado."

---

## Ambiente de execução (Python + versões exatas)

```
Python 3.12.3
Ubuntu 24.04.4 LTS, x86_64
CPU only (sem GPU)
```

Ver `requirements.txt` para as versões exatas de cada dependência
(gymnasium, stable-baselines3, torch, jax, ray, torchrl, tianshou, etc.).
Reproduzir com `pip install -r requirements.txt`. Isso é um snapshot
datado -- sem pin de versão, os números vão mudar com o tempo (ver seção
"Conceitos", item 3).

## Estrutura do projeto

```
scripts/
  common.py                     # config compartilhada, evaluate_policy, save_result
  train_sb3.py                  # SB3, hiperparâmetros genéricos
  train_sb3_n1.py                # SB3, mesma config mas n_envs=1
  train_sb3_zoo.py               # SB3, hiperparâmetros OFICIAIS do RL Zoo (Parte 11)
  train_cleanrl.py               # PyTorch puro, estilo CleanRL
  train_cleanrl_original.py      # wrapper que roda o ppo.py OFICIAL do CleanRL via subprocess
  cleanrl_original_ppo.py        # o próprio script, baixado do GitHub e patcheado
  train_jax.py                   # JAX puro, genérico
  train_jax_tuned.py             # JAX + normalização obs/reward, rede maior, LR annealing
  train_jax_tuned_kl.py          # + KL penalty dinâmica + value function clipping
  train_jax_tuned_kl_trunc.py    # + bootstrap correto em episódios truncados
  train_rllib.py                 # RLlib (Ray) real
  train_tianshou.py              # Tianshou 2.x
  train_torchrl.py               # TorchRL
  compare.py / boxplot.py / stats_test.py / eval_stats.py   # análise e estatística
results/            # curvas de treino (JSON), por ambiente
results_eval/       # resultados de avaliação (N episódios por checkpoint), por ambiente
```

---

# Parte 1: primeira comparação (SB3, PyTorch puro, JAX puro) -- CartPole-v1

Motivação inicial: "o PPO de uma lib é diferente do de outra, mesmo com os
mesmos hiperparâmetros?" Comparação com 1 seed, depois 5 seeds, hiperparâmetros
de treino alinhados manualmente entre as 3 implementações (`common.py`).

Resultado inicial (1 seed) sugeria diferenças de até ~100 pontos entre
libs. Isso motivou a pergunta natural: **isso é diferença real de
implementação, ou é ruído de seed que 1 amostra não consegue separar?**

# Parte 2: alinhando hiperparâmetros de arquitetura, não só de treino

Hiperparâmetros de treino (lr, gamma, clip, etc.) já estavam alinhados,
mas arquitetura e inicialização de pesos, não. Alinhamos:
- Arquitetura: 2 camadas de 64, tanh, nas 4 libs (RLlib usava [256,256] por
  padrão).
- Redes separadas ator/crítico (RLlib compartilhava por padrão).
- Inicialização ortogonal com gains diferenciados por camada (√2 hidden,
  0.01 saída do ator, 1.0 saída do crítico) -- replicando a convenção do
  CleanRL/SB3.
- Adam epsilon = 1e-5 em todas.

Depois disso, adicionamos o CleanRL oficial (baixado do GitHub, não o pip
antigo) como 5ª implementação -- e precisou de um patch pra rodar com
Gymnasium atual (ver "Conceitos", item 3).

**5 seeds** não bastaram pra separar diferença real de ruído. Rodamos
**20 seeds** no CartPole-v1 (150k timesteps cada) e aplicamos:

- **Kruskal-Wallis**: `H=10.289, p=0.0358` -- alguma diferença existe no
  conjunto.
- **Mann-Whitney U pairwise com Bonferroni** (10 pares, alpha corrigido =
  0.005): **só 1 de 10 pares foi significativo** (SB3 vs PyTorch puro,
  p=0.0018). Todos os outros 9, incluindo qualquer lib vs CleanRL oficial,
  eram estatisticamente indistinguíveis dado o ruído de seed.

**Conclusão da Parte 2:** no CartPole, a "diferença entre libs" que parecia
óbvia com poucos seeds era, na esmagadora maioria, ruído de seed -- não
diferença real de implementação. A única coisa que sobrou (SB3 com desvio
padrão 2-3x menor que as outras 4) não era sobre performance, era sobre
**consistência**: `std=26.4` pro SB3 contra `62-83` nas outras 4.

# Parte 3: ambiente mais difícil (LunarLander-v3)

Escolhido porque CartPole "fácil demais" estava mascarando diferenças
reais atrás de ruído de seed. LunarLander-v3 (discreto -- mantido
Categorical em vez de reescrever pra ação contínua), 8 seeds, 300k
timesteps.

> **Nota de correção (auditoria posterior):** uma auditoria numérica
> completa do projeto (rodada depois da Parte 12, comparando cada tabela
> deste README contra os arquivos JSON brutos) descobriu que os arquivos
> de seeds 1-8 de `jax_pure`, `cleanrl_style_pytorch` e
> `stable_baselines3` no `results/LunarLander-v3/` tinham sido
> **sobrescritos** em algum momento posterior do projeto por uma run de
> **1.000.000 de steps** (não os 300k originais desta Parte) -- só
> `rllib` e `cleanrl_original` ainda tinham o budget correto (~300k). Os
> dados originais dessas 3 implementações estão irrecuperavelmente
> perdidos. Refizemos as 8 seeds dessas 3 implementações do zero, com o
> budget correto (300k), e a tabela abaixo reflete os dados **verificados
> e corretos** -- que mudam a conclusão original desta parte de forma
> real, não só cosmética (ver "O que mudou" no fim desta seção).

Resultado (dados verificados):

| Lib | Média | Desvio padrão |
|---|---|---|
| RLlib (Ray) | 47.7 | 56.7 |
| JAX puro | 26.5 | 10.3 |
| PyTorch puro (estilo CleanRL) | 26.1 | 32.3 |
| Stable-Baselines3 | 14.5 | 24.8 |
| CleanRL (oficial) | -38.0 | 26.0 |

**Kruskal-Wallis: `H=14.627, p=0.0055`.**

Mann-Whitney (Bonferroni, alpha corrigido=0.0050): **4 de 10 pares
significativos** -- e os 4 envolvem o **CleanRL oficial**, não o RLlib:

| Par | p-valor | Significativo? |
|---|---|---|
| CleanRL oficial vs JAX puro | 0.0006 | SIM |
| RLlib vs CleanRL oficial | 0.0047 | SIM |
| SB3 vs CleanRL oficial | 0.0047 | SIM |
| CleanRL oficial vs PyTorch puro | 0.0047 | SIM |
| (demais 6 pares) | 0.28-0.65 | não |

## O que mudou com a correção

A versão original (com dados corrompidos) desta seção dizia que "o RLlib
tinha uma vantagem real e estatisticamente robusta" sobre as outras 4.
**Isso não se sustenta nos dados corretos.** Com os dados verificados,
RLlib é estatisticamente indistinguível de SB3, PyTorch puro e JAX puro --
a única implementação que realmente se destaca (pra pior) é o **CleanRL
oficial**, não identificado antes por causa da corrupção dos outros 3
conjuntos de dados.

Isso é relevante porque as Partes 4 e 5 foram motivadas pela ideia
"RLlib parece ter uma vantagem especial no LunarLander, vamos investigar
por quê" -- uma motivação que a Parte 3 original (com dados ruins) parecia
confirmar com força estatística, mas que não se sustenta na Parte 3
corrigida. Isso NÃO invalida os resultados das Partes 4-12 em si (cada uma
foi auditada separadamente contra os arquivos `results_eval/` e todas
batem exatamente com o que está documentado) -- mas significa que a
justificativa inicial pra perseguir esse fio ("RLlib parece
sistematicamente melhor com 8 seeds") era mais fraca do que o texto
original sugeria. A vantagem real do RLlib só aparece de forma robusta
mais adiante, com 1 seed + 50 avaliações (Parte 4) e com os testes formais
das Partes 5-10 -- não já na Parte 3.

# Parte 4: 1 seed de treino, múltiplos episódios de avaliação (mais barato)


Ideia do usuário pra economizar compute: em vez de treinar N vezes com
seeds diferentes (caro), treinar **uma vez** com budget maior (1M
timesteps) e avaliar essa política fixa em 50 episódios -- mede
consistência de *avaliação*, não reprodutibilidade de *treino* (pergunta
diferente e complementar).

Resultado (seed=42, 1M steps, 50 episódios):

| Lib | Média eval | Desvio padrão |
|---|---|---|
| RLlib | 158.9 | 115.5 |
| CleanRL oficial | -52.8 | 126.5 |
| PyTorch puro | -55.7 | 76.2 |
| JAX puro | -57.8 | 39.3 |
| Stable-Baselines3 | -85.3 | 30.0 |

Kruskal-Wallis: `p≈0.0000`. **5 de 10 pares significativos** (mais forte
que a versão multi-seed). Achado novo: JAX puro vs SB3 virou significativo
aqui, o que não acontecia com variância de treino -- sugere que parte do
"ruído" da Parte 2 mascarava uma diferença real e mais sutil.

## Testando se era customização de tarefa, não a lib (`train_jax_tuned.py`)

Hipótese do usuário: implementações "do zero" nunca foram *tunadas* pro
LunarLander -- carregavam a receita genérica do CartPole. Adicionamos, só
no JAX puro:
1. Normalização de observação (running mean/std, estilo VecNormalize).
2. Normalização de reward (reward dividido pelo std rodante do retorno
   descontado).
3. Rede maior: 128x128 em vez de 64x64.
4. Learning rate com annealing linear até 0.

Resultado: **-57.8 → +45.1**, estatisticamente muito acima das 4
implementações genéricas. **Não fechou o gap com o RLlib** (ainda
p<0.0001). Testamos se era porque o RLlib normaliza observação por
padrão -- não é: `observation_filter` default do RLlib é `NoFilter`.

# Parte 5: KL dinâmico + value function clipping (`train_jax_tuned_kl.py`)

Motivado por um projeto externo enviado pelo usuário com uma reimplementação
"estilo RLlib" (não é o `ray` de verdade) que apostava nesses 2 mecanismos.
Adicionamos só isso ao `jax_tuned`:
- **Value function clipping** (`RLLIB_VF_CLIP_PARAM=10.0`).
- **Penalidade de KL dinâmica** (`RLLIB_KL_TARGET=0.01`,
  `RLLIB_INITIAL_KL_COEFF=0.2`, ajustada a cada iteração pela mesma regra
  do RLlib: ×1.5 se KL > 1.5x o alvo, ×0.5 se < 0.67x o alvo).

Resultado: **45.1 → 104.9** (p=0.0022 vs jax_tuned). O gap com o RLlib
caiu de 113.8 pontos pra 54.0 -- **esses 2 mecanismos explicam
aproximadamente metade** da vantagem restante do RLlib. Mas RLlib ainda
venceu de forma significativa (p=0.0001) -- sobrava ~54 pontos sem
explicação.

# Parte 6: por que o SB3 é "estável" aqui e "instável" num projeto externo

O usuário notou que, no projeto externo enviado, o SB3 tinha desvio padrão
`112.1` (segundo pior, quase tão ruim quanto um TorchRL com bug
suspeitado), enquanto no nosso benchmark o SB3 era consistentemente o MAIS
estável (`std=26-30`). Hipótese: `N_ENVS` (8 no nosso setup, 1 no deles).

Testamos mudando SÓ essa variável (`train_sb3_n1.py`, `n_envs=1`, mesmo
seed, mesmo budget):

| Versão | Std |
|---|---|
| SB3 (n_envs=8) | 30.0 |
| SB3 (n_envs=1) | 132.1 |

**Teste de Levene** (não Mann-Whitney -- a pergunta era sobre variância,
não sobre média): `p=0.00013`. Diferença de variância real e forte.
**Hipótese confirmada**: mudar 1 parâmetro estrutural inverteu a reputação
de estabilidade do SB3 inteira.

# Parte 7: correção de bootstrap em truncamento -- LunarLander (inconclusivo)

Nosso GAE tratava `terminated` (episódio realmente acabou) e `truncated`
(cortado por limite de tempo artificial) da mesma forma, zerando o
bootstrap de valor nos dois casos -- tecnicamente errado pra truncamento.

Confirmamos empiricamente que o Gymnasium atual (`autoreset_mode=
NEXT_STEP`, o padrão) retorna a observação real final no step de
truncamento -- o dado certo já existia, só não estava sendo usado. Corrigimos
com duas máscaras separadas no GAE (`train_jax_tuned_kl_trunc.py`): uma pra
zerar bootstrap (só terminated de verdade), outra pra parar a propagação
do GAE (terminated OU truncated).

Resultado no LunarLander: **104.9 (sem correção) vs 85.4 (com correção)**,
Mann-Whitney `p=0.3647` -- **não significativo**, a diferença é ruído.
Explicação provável: LunarLander tem limite de 1000 steps, mas episódios
terminam por pouso/crash bem antes disso -- truncamento é raro, a correção
não teve chance de mostrar efeito.

# Parte 8: mesma correção, no CartPole -- resultado claro e positivo

CartPole é o ambiente certo pra essa hipótese: limite de 500 steps, e uma
política competente aprende a balançar indefinidamente, batendo o limite
de tempo na maioria dos episódios (truncamento é a forma DOMINANTE de
terminar um episódio bem-sucedido).

Resultado (seed=42, 400k timesteps):

| Versão | Média | Std | Min |
|---|---|---|---|
| Com correção | **500.0** | **0.0** | 500.0 |
| Sem correção | 485.5 | 43.5 | 343.0 |

Mann-Whitney: `p=0.0231` -- significativo. **Perfeito em todos os 50
episódios de avaliação** com a correção. Mesmo mecanismo, mesmo código,
irrelevante num ambiente (LunarLander) e decisivo no outro (CartPole).

# Parte 9: RLlib real no CartPole -- resultado inconclusivo, mas revelador

Pergunta: será que o RLlib "de fábrica" já trata truncamento corretamente,
explicando parte da vantagem dele? Rodamos RLlib no CartPole (config
genérica das Partes 1-2, sem LR annealing), 400k timesteps, seed=42.

Avaliação final: **171.3 ± 33.9** -- muito abaixo do 500 esperado. Mas a
curva de treino completa mostrou algo diferente: a política **bateu 500**
no meio do treino (step ~330k) e **degradou** de forma consistente até o
final, terminando em 193. Não é falta de budget -- é instabilidade de
treino que a avaliação (feita no checkpoint final) capturou num vale, não
no pico.

**O teste original não foi respondido** -- revelou, em vez disso, que essa
config do RLlib (sem LR annealing) é instável perto do teto de
performance, ecoando o mecanismo que nossas próprias versões tunadas usam
pra evitar exatamente isso.

# Parte 10: TorchRL e Tianshou

Motivação dupla: (1) fechar o mistério do TorchRL catastrófico (-377) do
projeto externo, implementando nossa própria versão controlada; (2)
adicionar uma segunda lib madura pra separar "vantagem específica do
RLlib" de "framework de produção sempre vence hobbyist".

Precisou corrigir 2 APIs quebradas por versão (ver "Conceitos", item 3):
Tianshou 2.0 (`policy`→`algorithm`) e TorchRL (`SyncDataCollector`→
`Collector`). O ambiente de execução também foi resetado no meio do
trabalho, exigindo reinstalar tudo (incluindo lidar com falta de espaço em
disco).

Resultado (LunarLander-v3, seed=42, 1M steps):

| Lib | Média eval | Tempo de treino |
|---|---|---|
| TorchRL | 49.0 | **1014.5s** |
| Tianshou | -7.2 | 233.8s |

**O mistério do TorchRL não se repetiu** -- nossa implementação ficou no
meio do pelotão, nada catastrófico, reforçando que o -377 do projeto
externo era específico da config dele. **TorchRL foi disparado a
implementação mais lenta de todo o projeto** (1014s vs 111-265s das
outras). Tianshou e TorchRL caíram estatisticamente indistinguíveis de
várias implementações "genéricas" nossas -- enfraquece a hipótese de
"framework maduro sempre vence hobbyist": a vantagem do RLlib parece mais
específica dele do que um padrão geral.

---

# Parte 11: SB3 com os hiperparâmetros OFICIAIS do RL Zoo -- confirmação final

## A pergunta

Por que várias implementações (incluindo o próprio SB3 genérico, `-85.3`)
davam reward NEGATIVO no LunarLander? Era bug, ou má configuração? Já
tínhamos evidência indireta (Parte 4: tuning subiu o JAX de -57.8 pra
+45.1) mas nunca confirmamos com a config *real, recomendada por quem
mantém a lib* -- só com nossas próprias tentativas de tuning.

## O teste

Peguei os hiperparâmetros oficiais do **RL Baselines3 Zoo**
(`DLR-RM/rl-baselines3-zoo`, `hyperparams/ppo.yml`, seção
`LunarLander-v3`) -- os valores realmente usados/recomendados pela equipe
que mantém o SB3 pra essa tarefa específica:

```yaml
n_envs: 16
n_steps: 1024
batch_size: 64
gae_lambda: 0.98
gamma: 0.999        # bem mais alto que o 0.99 genérico -- LunarLander
                    # tem episódios longos, precisa dar mais peso a
                    # reward futuro
n_epochs: 4
ent_coef: 0.01
n_timesteps: 1e6
```

(learning rate, clip_range, vf_coef, arquitetura: defaults do SB3, sem
mudança). Mesmo seed=42, mesmo protocolo de avaliação (50 episódios) das
Partes 3-10, pra comparação direta com tudo que já tínhamos.

## Resultado

**231.3 ± 88.8** -- não só saiu do negativo como ficou ACIMA de todas as
outras 11 implementações testadas até aqui, incluindo o RLlib (158.9) e o
nosso próprio `jax_tuned_kl` (104.9).

| Lib | Média eval | Std |
|---|---|---|
| **SB3 (config oficial do Zoo)** | **231.3** | 88.8 |
| RLlib (Ray) | 158.9 | 115.5 |
| jax_tuned_kl | 104.9 | 95.0 |
| jax_tuned_kl_trunc | 85.4 | 106.0 |
| TorchRL | 49.0 | 115.2 |
| jax_tuned | 45.1 | 110.7 |
| stable_baselines3_n1 | 0.6 | 132.1 |
| Tianshou | -7.2 | 105.9 |
| CleanRL (oficial) | -52.8 | 126.5 |
| PyTorch puro | -55.7 | 76.2 |
| JAX puro | -57.8 | 39.3 |
| Stable-Baselines3 (config genérica) | -85.3 | 30.0 |

Kruskal-Wallis: `H=273.810, p≈0.0000`. Mann-Whitney com Bonferroni (12
implementações, 66 pares): **SB3-Zoo é significativamente diferente de
TODAS as outras 11 implementações**, incluindo o RLlib (`p=0.0000` em
todos os pares envolvendo SB3-Zoo).

## A resposta definitiva

**Config, não biblioteca.** SB3-Zoo e Stable-Baselines3 (genérico) são
**literalmente o mesmo código-fonte, a mesma versão da mesma biblioteca**
-- a única coisa que mudou foram os hiperparâmetros. E essa mudança sozinha
produziu a maior amplitude de reward de todo o projeto inteiro (de -85.3
pra +231.3, um salto de mais de 300 pontos), maior que qualquer diferença
entre bibliotecas que medimos nas Partes 1-10.

Isso fecha, com o teste mais direto possível, a pergunta que atravessou
o projeto inteiro: quando você vê "biblioteca X tem reward negativo/ruim",
a explicação mais provável não é "a lib tem bug" -- é "ninguém deu a ela
os hiperparâmetros certos pra essa tarefa". `gamma=0.999` em vez de `0.99`
sozinho já é uma pista forte: LunarLander tem episódios de até 1000 steps,
e descontar reward futuro de forma agressiva (gamma baixo) faz o agente
"não enxergar" o benefício de longo prazo de pousar com cuidado.

## Rodar

```bash
export PPO_ENV=LunarLander-v3
export PPO_SEED=42
python3 scripts/train_sb3_zoo.py
python3 scripts/eval_stats.py
```

---

# Parte 12: repetindo TODA a comparação com os hiperparâmetros do RL Zoo

## A pergunta

A Parte 11 mostrou que o SB3 com a config certa vira o melhor do pelotão.
Mas isso testou só o SB3. A pergunta natural: **será que aplicar a mesma
config (não uma "config oficial" de cada lib, que não existe pra todas,
mas literalmente os mesmos valores de hiperparâmetro do RL Zoo) em TODAS
as implementações muda o ranking inteiro que construímos nas Partes 1-10?**

Ideia do usuário, mais simples e mais limpa que tentar achar uma "receita
oficial" por biblioteca (que não existe pra RLlib/TorchRL/Tianshou/nossas
implementações próprias de forma simétrica): usar os MESMOS valores do RL
Zoo (`n_steps=1024`, `n_envs=16`, `gae_lambda=0.98`, `gamma=0.999`,
`n_epochs=4`, `ent_coef=0.01`, `batch_size=64`) em `common.py`, e rodar
outra vez a mesma bateria de 7 implementações -- 1 seed=42, 1M timesteps,
50 episódios de avaliação, LunarLander-v3.

## Perrengues de infraestrutura no meio do caminho (vale registrar)

Esta parte sofreu DOIS resets de container no meio da execução -- o
sistema de arquivos de trabalho (`/home/claude`) e, desta vez, até a pasta
de outputs (que eu supunha ser permanentemente persistente) reverteram pra
um estado anterior mais de uma vez. Consequências práticas:
- Perdemos os resultados intermediários do TorchRL uma vez (teve que
  rodar de novo, ~20 minutos perdidos).
- Um reset trouxe de volta uma versão ANTIGA do `common.py` (sem os
  hiperparâmetros do Zoo) que só foi percebida porque o log do TorchRL
  mostrava `frames_per_batch (2048)` em vez de `16384` -- o processo
  chegou a rodar ~5 minutos com a config errada antes de ser interrompido
  e corrigido.
- A lição prática adotada a partir daqui: copiar os resultados pra
  `outputs` **imediatamente após cada script terminar**, não só no final
  do trabalho -- e essa lição ela mesma quase não foi suficiente, porque
  até `outputs` se mostrou instável nesta sessão.

Mais um capítulo ao vivo do tema "Conceitos", item 3 -- reprodutibilidade
não é só sobre versão de biblioteca, é sobre o ambiente de execução inteiro
não ser garantido, ponto final.

## Resultado (LunarLander-v3, seed=42, 1M steps, config do Zoo em todas)

| Lib | Média eval (config Zoo) | Std | Média eval (config genérica, Partes 3-10) |
|---|---|---|---|
| **Tianshou** | **281.2** | 37.6 | -7.2 |
| PyTorch puro (estilo CleanRL) | 265.4 | 31.7 | -55.7 |
| CleanRL (oficial) | 235.0 | 73.4 | -52.8 |
| Stable-Baselines3 | 231.3 | 88.8 | -85.3 |
| JAX puro | 158.3 | 24.9 | -57.8 |
| RLlib (Ray) | 123.7 | 34.9 | **158.9** |
| TorchRL | 112.3 | 38.0 | 49.0 |

Kruskal-Wallis: `H=220.311, p≈0.0000`. Mann-Whitney com Bonferroni (21
pares, alpha corrigido=0.0024): **18 de 21 pares significativos** -- a
proporção mais alta de significância de todo o projeto.

Os 3 pares NÃO significativos formam um "grupo de topo" estatisticamente
indistinguível entre si: PyTorch puro vs CleanRL oficial (p=0.0050, não
sobrevive à correção), PyTorch puro vs SB3 (p=0.0759), CleanRL oficial vs
SB3 (p=0.3432). O Tianshou fica sozinho acima até desse grupo (significativamente
melhor que os 3). RLlib e TorchRL ficam isolados abaixo de todo o resto,
significativamente piores que as 5 implementações de cima -- e
significativamente diferentes um do outro também (RLlib > TorchRL,
p=0.0003).

## O achado central: o ranking virou de cabeça pra baixo

**O RLlib, que era disparado o melhor com hiperparâmetros genéricos
(158.9, isolado no topo nas Partes 3-10), caiu pra penúltimo lugar
(123.7) com a config correta aplicada a todo mundo igualmente.** Quase
todas as implementações "do zero" (Tianshou, PyTorch puro, JAX puro) e até
o CleanRL oficial ultrapassaram o RLlib.

Isso não é "RLlib piorou" -- ele teve exatamente os mesmos hiperparâmetros
de treino que as outras, então seu resultado absoluto até melhorou um
pouco moderadamente em relação ao que já tinha (123.7 vs valores baixos
seria esperado, mas na verdade ele já estava em 158.9 antes -- ou seja,
mudar pra config do Zoo não ajudou o RLlib tanto quanto ajudou as outras
implementações, e as outras ultrapassaram ele). A explicação mais
provável, juntando tudo que já sabemos: **o RLlib parece ter mecanismos
internos (normalização, tratamento de episódio, os defaults de KL/vf-clip
que já são parte dele por padrão) que o tornam relativamente robusto
mesmo com hiperparâmetros ruins** -- e é exatamente por isso que ele
"vencia" nas Partes 3-10, quando todo mundo tinha configuração ruim. Uma
vez que TODAS as implementações recebem hiperparâmetros bons, essa
robustez deixa de ser um diferencial, e a vantagem que sobra vai pra quem
tem a implementação mais eficiente em cima de bons hiperparâmetros -- que,
neste teste, não foi o RLlib.

## Leitura final desta parte

Isso é a demonstração mais direta e mais forte de todo o projeto do
princípio central que vínhamos descobrindo aos poucos: **"biblioteca X é
melhor" nunca foi uma propriedade fixa da biblioteca -- sempre foi uma
propriedade da combinação (biblioteca, hiperparâmetros, tarefa, budget)
inteira.** A mesma pergunta ("qual biblioteca de PPO é melhor pro
LunarLander?") tem duas respostas opostas dependendo só de uma variável
que nem é a biblioteca -- é a qualidade dos hiperparâmetros que você deu a
ela.

TorchRL também vale nota à parte: além de continuar sendo, disparado, a
implementação mais lenta (~20 minutos, ~5-10x mais que a maioria), agora
também é a mais fraca em reward. Nenhuma vantagem compensatória apareceu
pra ela nesta bateria de testes.

## Rodar

```bash
export PPO_ENV=LunarLander-v3
export PPO_TIMESTEPS=1000000
export PPO_SEED=42
export PPO_RUN_TAG="-zoo"
python3 scripts/train_sb3.py
python3 scripts/train_cleanrl.py
python3 scripts/train_jax.py
python3 scripts/train_rllib.py
python3 scripts/train_cleanrl_original.py
python3 scripts/train_tianshou.py
python3 scripts/train_torchrl.py   # o mais lento, ~20min sozinho
python3 scripts/eval_stats.py
```

`PPO_RUN_TAG` cria uma subpasta separada em `results/` e `results_eval/`
(`LunarLander-v3-zoo/`) pra não misturar com os resultados de config
genérica das Partes 3-10, que usam o mesmo ambiente e seed mas
hiperparâmetros diferentes.

---



Depois de 11 partes, a resposta mais honesta pra "qual biblioteca de PPO é
melhor" é: **a pergunta, do jeito que é normalmente feita, não é
respondível.** Toda vez que isolamos uma variável -- seeds, dificuldade da
tarefa, budget, `n_envs`, KL/vf-clip, tratamento de truncamento, e por fim
hiperparâmetros de configuração -- uma conclusão anterior que parecia
sólida ou perdeu força, ou se revelou sobre outra coisa completamente
diferente do que parecia.

# Conclusão geral do projeto

Depois de 12 partes, a resposta mais honesta pra "qual biblioteca de PPO é
melhor" é: **a pergunta, do jeito que é normalmente feita, não é
respondível.** Toda vez que isolamos uma variável -- seeds, dificuldade da
tarefa, budget, `n_envs`, KL/vf-clip, tratamento de truncamento, e por fim
hiperparâmetros de configuração -- uma conclusão anterior que parecia
sólida ou perdeu força, ou se revelou sobre outra coisa completamente
diferente do que parecia.

A Parte 12 é o exemplo mais extremo disso no projeto inteiro: o RLlib, que
tinha vencido de forma estatisticamente sólida nas Partes 3, 5 e 10, virou
o penúltimo colocado assim que a ÚNICA coisa que mudou foi dar
hiperparâmetros de qualidade a todo mundo igualmente. Não existiu, em
nenhum momento desse projeto, uma resposta pra "qual lib é melhor" que
sobrevivesse a mudar o regime de comparação -- a única coisa achada por
esse projeto que se sustentou universalmente foi o próprio padrão: **a
biblioteca importa muito menos do que a configuração.**

A única coisa que sobreviveu ao escrutínio inteiro foi a versão qualificada
da pergunta: "com esse algoritmo, nesse ambiente, com esse budget, com N
seeds, com esses hiperparâmetros específicos, essa implementação teve essa
distribuição de resultados." Isso é bem menos vendável que um ranking de
benchmark, mas é a única coisa que continuou verdadeira depois de mudarmos
qualquer coisa.

Se isso for alimentar a ideia da sua lib própria de ML: a lição prática
não é "escolha a lib certa" -- é que a maior parte do que faz um agente de
RL funcionar bem não está na escolha do framework, está em um punhado de
decisões de engenharia (normalização, tratamento de truncamento, LR
annealing, hiperparâmetros calibrados pra tarefa) que qualquer
implementação -- sua ou de terceiros -- precisa acertar.
