# PPO: SB3 vs CleanRL — implementação ou biblioteca?

Experimento fatorial 2×2 + ponte que pergunta se o gap SB3–CleanRL vem
dos **detalhes do algoritmo** ou do **stack de software**.
Roda em **CartPole-v1** e **LunarLander-v3** com hiperparâmetros idênticos.

## 1. Pergunta e desenho

| braço | código | spec de algoritmo | stack |
|---|---|---|---|
| `sb3_torch` | SB3 `PPO` | SB3 | SB3 + Torch |
| `cleanrl_torch` | `train_cleanrl_torch.py --mode cleanrl` | CleanRL | Torch manual |
| `cleanrl_sb3mode_torch` | `train_cleanrl_torch.py --mode sb3` | **SB3** | Torch manual (ponte) |
| `jax_sb3` | `train_jax_ppo.py --mode sb3` | SB3 | JAX + Optax |
| `jax_cleanrl` | `train_jax_ppo.py --mode cleanrl` | CleanRL | JAX + Optax |
| `jax_abl_*` (×4) | `train_jax_ppo.py --mode cleanrl --abl <delta>` | CleanRL + 1 delta SB3 | JAX + Optax (ablação) |

Braços de ablação (§8): `novclip` (sem clip de valor), `noanneal` (LR
constante), `fullmse` (sem o fator 0.5 no MSE), `tboot` (com bootstrap de
timeout).

Contrastes planejados: Q1 gap total; Q2/Q3 efeito de stack com spec fixa;
Q4 a ponte fecha o gap? Q5 efeito de spec dentro do JAX;
Q6 ponte Torch vs JAX com mesma spec (checagem de sanidade).

## 2. O que é "mesmo" e o que difere (ver `specs.py`)

Idênticos: LR 2.5e-4, γ 0.99, λ 0.95, ε 0.2, N_STEPS 2048, batch 64,
epochs 4, ent 0.01, vf 0.5, grad-clip 0.5, Adam eps 1e-5, MLP [64,64]
Tanh com troncos separados, ortho-init (hidden √2, head 0.01/1.0),
vantagem normalizada por minibatch, n_envs=1, seeds {0..4}, CPU.

Deltas de spec (únicas diferenças algorítmicas permitidas):
SB3 = sem clip de valor, LR constante, MSE cheio, bootstrap de timeout;
CleanRL = clip de valor, LR com annealing linear, 0.5×MSE, sem bootstrap.
Tudo documentado e referenciado em `specs.py`.

## 3. Rigor

5 seeds de treino por (braço, env); avaliação determinística com 100
episódios e seeds fixas; curvas = média ± DP entre seeds; contrastes
pareados por seed com bootstrap 95% (N=5000) + Cliff's delta; cada run
salva `meta.json` com spec, hiperparâmetros e versões.

## 4. Como rodar

Pré-requisito: Python 3.10–3.11 com os pacotes de ambos os requirements
(o setup deste estudo usa UM interpretador com torch+sb3+jax juntos;
se preferir separar, `run_all.py --python-torch X --python-jax Y` aceita
dois interpretadores):

```bash
pip install -r requirements-torch.txt   # SB3 + Torch + Gymnasium
pip install -r requirements-jax.txt     # JAX + Optax
```

Notas deste repo em Windows:
- `sb3_shim.py` é carregado automaticamente e contorna o bug
  `cv2/_ARRAY_API not found` (SB3 + NumPy 2) — sem atalho manual.
- Todo o treino é CPU e single-threaded por processo
  (`torch.set_num_threads(1)`, `OMP_NUM_THREADS=1`); o throughput vem de
  rodar jobs em paralelo, não de muitas threads.

Rodadas:

```bash
# smoke test (1 env, 1 seed, 1% do budget; valida o pipeline)
python run_all.py --env CartPole-v1 --seeds 0 --timesteps-scale 0.01

# experimento completo, sequencial (50 runs; lento: ~1 dia em CPU única)
python run_all.py

# recomendado: runner paralelo resume-safe (pula jobs cujo modelo já existe)
python run_full_background.py --workers 8                 # tudo que falta
python run_full_background.py --workers 8 --env LunarLander-v3
python run_full_background.py --workers 8 --only jax_sb3,jax_cleanrl
python run_full_background.py --workers 8 --env LunarLander-v3 \
    --only jax_abl_novclip,jax_abl_noanneal,jax_abl_fullmse,jax_abl_tboot
python run_full_background.py --workers 8 --seeds 5,6,7,8,9  # seeds extras

# pós-processamento (avaliação + estatística + figuras); já roda no fim
# do run_all.py, mas pode ser re-rodado sozinho:
python evaluate.py      # 100 eps determinísticos por modelo -> tables/
python analyze.py       # bootstrap pareado por seed -> contrasts_*.md
python plot_results.py  # curvas de treino + boxplots -> plots/
```

Custos medidos neste hardware (i9-14900HX, CPU): SB3 ≈ 236s (CartPole) /
572s (LunarLander) por run; Torch-manual ≈ 190–200s / 838s; JAX ≈ 500s /
740–1010s. Saídas: `results/models`, `results/curves`, `results/tables`,
`results/plots`, `results/meta`, `results/logs`.

## 5. Leitura dos resultados

Se Q4/Q6 ≈ 0 e Q2/Q3 ≈ 0 mas Q1/Q5 ≠ 0 → diferença é de implementação.
Se Q2/Q3 grandes mesmo com spec fixa → stack/biblioteca pesa.
A ponte (`cleanrl_sb3mode_torch`) é o teste crucial: mesmo código Torch
do CleanRL, mas vestindo a spec do SB3.

## 6. Arquivos

`config.py` hiperparâmetros; `specs.py` deltas; `variants.py` braços;
`train_*.py` treinos; `jax_run_fast.py` (backend JAX padrão de
`train_jax_ppo.py`: update inteiro jitted por epoch via `lax.scan`; mesma
matemática que `jax_run.py`, que fica no repo como referência legível/legada,
~3× mais lenta); `sb3_shim.py` compat cv2/NumPy2; `evaluate.py`,
`analyze*.py`, `plot_*.py` avaliação/estatística/figuras; `run_all.py`
orquestrador sequencial; `run_full_background.py` runner paralelo
resume-safe (pula jobs cujo modelo já existe).
Pesos (`results/models/*`) são ignorados pelo git e regeneráveis.

## 7. Resultados obtidos (10 seeds × 100 eps determinísticos)

**CartPole-v1**: satura no teto de 500 em 4 braços (inclui ambas as specs
"puras"); apenas a ponte manual fica ~440 ± 62 (Q4/Q6 não-nulos pequenos,
resíduo de stack na saturação; env permanece pouco informativo).

**LunarLander-v3** (discriminador, 10 seeds; ablações com 5):

| braço | retorno (média ± DP) |
|---|---|
| CleanRL-code/SB3-spec (ponte) | 151.4 ± 20.2 |
| JAX/SB3-spec | 142.7 ± 29.0 |
| SB3 (Torch lib) | 135.6 ± 72.4 (1 seed colapsada: 8.8) |
| CleanRL (Torch) | 108.1 ± 46.7 |
| JAX/CleanRL-spec | 92.6 ± 44.8 |

Contrastes principais (bootstrap pareado, IC 95%): Q2/Q3/Q4/Q6 ≈ 0 e NS —
o **stack não produz diferença**; Q5 (spec dentro do JAX) = **+50.3
[25.5, 78.1], p<1e-4, δ=0.72** — a spec SB3 supera a CleanRL. Q1 = +27.1
[−28.1, 79.5], NS (variância inflada pela seed colapsada).

**Ablação dos 4 deltas (JAX, CleanRL + 1 delta SB3):** vs `jax_cleanrl`:

| delta trocado | ganho vs CleanRL | IC 95% | p | vs jax_sb3 |
|---|---|---|---|---|
| −clip de valor | **+41.9** | [20.6, 57.8] | <1e-4 | +4.4 (NS — gap fecha) |
| −LR annealing | +20.4 | [4.3, 41.8] | <1e-4 | −17.0 (NS marginal, p=.076) |
| −fator 0.5 no MSE | +0.3 | [−18, 18] | NS | −37.5 |
| +bootstrap timeout | −8.6 | [−27, 12] | NS | −46.2 |

Resposta: o delta dominante é o **clip da loss de valor** (removeu o clip →
recuperou o nível SB3 sozinho); LR annealing é contribuinte secundário;
escala do MSE e bootstrap de timeout não mudam o quadro neste setup.
Tabelas: `results/tables/contrasts_*.md` e `contrasts_ablation_*.md`.

## 8. Limitações / erratas executadas

- **Bug encontrado e corrigido neste estudo (errata)**: `terminal_observation`
  não existe em env gymnasium single-env, então o bootstrap-de-timeout estava
  *inativo* (dead branch) nos braços manuais; e o fator 0.5 do MSE ficava
  engolido no branch com clip. Ambos foram corrigidos (bootstrap a partir da
  obs de truncamento, `vhalf` aplicado nos dois branches) e todos os braços
  com spec SB3 (manual) + ablações relevantes foram **retreinados do zero**;
  os números acima já refletem a correção. A ablação mostra que o delta tinha
  efeito ≈ 0 neste envs (NS), então as conclusões Q2/Q4/Q6 se mantiveram.
- Poder: 10 seeds nos braços principais e 5 nas ablações; Q1 ainda tem IC
  largo por causa da seed colapsada do SB3. `solved` (≥200 LunarLander)
  requer seeds extras ou tuning por env, fora do escopo desta comparação.
- CartPole permanece saturado; um budget menor (~50–100k steps) o tornaria
  discriminativo de novo.


