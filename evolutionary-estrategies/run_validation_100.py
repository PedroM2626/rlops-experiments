"""
Script de Validação Científica Rigorosa: 100 Episódios de Teste Out-of-Sample.

Executa o protocolo acadêmico completo:
  1. Treinamento com semente 42 (G=30 gerações para F1/F3, G=15 para F2)
  2. Extração do vetor ótimo de parâmetros / programa elite
  3. Avaliação de validação em N=100 episódios de teste independentes (semente 9999)
  4. Cálculo de métricas de dispersão, ruído, confiança e viés do vencedor:
     - Média ± SEM
     - 95% Bootstrap Confidence Interval (B=2000)
     - Mediana e IQR
     - Desvio Padrão
     - Signal-to-Noise Ratio (SNR)
     - Coeficiente de Variação (CV)
     - Lacuna de Otimismo / Winner's Curse (f_train - R_test)
     - Taxa de Sucesso (%)
  5. Geração de gráficos estatísticos (boxplots e barras de erro)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import json
import csv
import time
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.environments import ENV_META
from src.family1_direct import SimpleGA, DE, OpenAIES
from src.family2_programs import LinearGP, CartesianGP
from src.family3_eda import CMAES, PBIL
from src.benchmark import run_param_based_v2, run_program_based
from src.evaluation import make_test_eval_fn, evaluate_program_100, compute_statistical_metrics

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ART_DIR = r"C:\Users\Acer\.gemini\antigravity-ide\brain\ba1c8a11-8fe6-4f79-b29a-66466415603f"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(ART_DIR, exist_ok=True)

TEST_SEED = 99999
N_TEST_EPISODES = 100


def build_param_algorithms(pop: int = 32):
    return [
        dict(algo=SimpleGA(pop_size=pop, sigma_init=0.5, cx_prob=0.8), name="SimpleGA", family=1),
        dict(algo=DE(pop_size=pop, F=0.8, CR=0.9), name="DE", family=1),
        dict(algo=OpenAIES(pop_size=pop, sigma=0.05, lr=0.01), name="OpenAI-ES", family=1),
        dict(algo=CMAES(sigma0=0.5), name="CMA-ES", family=3),
        dict(algo=PBIL(pop_size=pop, lr=0.1, lr_sigma=0.05, top_k_frac=0.2), name="PBIL", family=3),
    ]


def build_gp_algorithms(obs_dim: int, act_dim: int):
    return [
        dict(algo=LinearGP(obs_dim=obs_dim, act_dim=act_dim, pop_size=16, prog_len=48, n_extra_regs=8, mut_rate=0.15),
             name="LinearGP", family=2),
        dict(algo=CartesianGP(obs_dim=obs_dim, act_dim=act_dim, pop_size=8, n_cols=30, mut_rate=0.05),
             name="CartesianGP", family=2),
    ]


def main():
    print("=" * 80)
    print("  PROTOCOLO DE VALIDAÇÃO CIENTÍFICA: 100 EPISÓDIOS DE TESTE OUT-OF-SAMPLE")
    print(f"  Episódios de Teste: {N_TEST_EPISODES} | Semente de Teste: {TEST_SEED}")
    print("  Métricas: Média, SEM, Bootstrap CI 95%, SNR, CV, Optimism Gap, Success Rate")
    print("=" * 80)

    envs = ["CartPole-v1", "Acrobot-v1", "Pendulum-v1", "MountainCarContinuous-v0"]
    validation_records = []

    for env_name in envs:
        meta = ENV_META[env_name]
        obs_dim = meta["obs_dim"]
        act_dim = meta["act_dim"]
        print(f"\n{'-'*80}")
        print(f"  Ambiente: {env_name} (obs={obs_dim}, act={act_dim})")
        print(f"{'-'*80}")

        # 1. Compila avaliador JAX de 100 episódios para o ambiente
        test_eval_jax = make_test_eval_fn(env_name, n_episodes=N_TEST_EPISODES)
        test_rng = jax.random.PRNGKey(TEST_SEED)

        # 2. Treina e valida Famílias 1 e 3
        for alg_cfg in build_param_algorithms(pop=32):
            algo_name = alg_cfg["name"]
            family = alg_cfg["family"]
            print(f"  -> Treinando [F{family}] {algo_name} (30 gerações)...", end="", flush=True)

            t_train_start = time.time()
            res = run_param_based_v2(
                algo=alg_cfg["algo"],
                algo_name=algo_name,
                env_name=env_name,
                family=family,
                n_generations=30,
                n_rollouts=2,
                seed=42,
                log_every=999,
            )
            t_train = time.time() - t_train_start
            f_train_best = float(res.best_fitness[-1])
            print(f" Treino: {t_train:.1f}s (Best Train: {f_train_best:.2f})")

            # Avaliação de Teste com 100 episódios
            best_params_jnp = jnp.array(res.best_params)
            returns_test = np.array(test_eval_jax(best_params_jnp, test_rng))

            # Cálculo formal de métricas
            stats = compute_statistical_metrics(returns_test, f_train_best=f_train_best, env_name=env_name)
            record = {
                "algorithm": algo_name,
                "family": family,
                "environment": env_name,
                **stats,
                "train_time_s": round(t_train, 2),
            }
            validation_records.append(record)
            print(f"     [TESTE 100 eps] Média={stats['test_mean']:>8.2f} ± {stats['test_sem']:<5.2f} | "
                  f"CI95%=[{stats['ci95_low']:>7.2f}, {stats['ci95_high']:>7.2f}] | "
                  f"Mediana={stats['test_median']:>7.2f} | Std={stats['test_std']:>6.2f} | "
                  f"SNR={stats['snr']:>5.2f} | Gap={stats['optimism_gap']:>6.2f} | "
                  f"Sucesso={stats['success_rate_pct']:>5.1f}%")

        # 3. Família 2 (LinearGP e CartesianGP) nos ambientes discretos
        if env_name in ["CartPole-v1", "Acrobot-v1"]:
            for alg_cfg in build_gp_algorithms(obs_dim, act_dim):
                algo_name = alg_cfg["name"]
                family = 2
                print(f"  -> Treinando [F2] {algo_name} (15 gerações)...", end="", flush=True)

                t_train_start = time.time()
                res = run_program_based(
                    algo=alg_cfg["algo"],
                    algo_name=algo_name,
                    env_name=env_name,
                    family=2,
                    n_generations=15,
                    n_rollouts=3,
                    seed=42,
                    log_every=999,
                )
                t_train = time.time() - t_train_start
                f_train_best = float(res.best_fitness[-1])
                print(f" Treino: {t_train:.1f}s (Best Train: {f_train_best:.2f})")

                # Teste 100 episódios
                returns_test = evaluate_program_100(res.policy_fn, env_name, n_episodes=N_TEST_EPISODES, seed=TEST_SEED)
                stats = compute_statistical_metrics(returns_test, f_train_best=f_train_best, env_name=env_name)
                record = {
                    "algorithm": algo_name,
                    "family": family,
                    "environment": env_name,
                    **stats,
                    "train_time_s": round(t_train, 2),
                }
                validation_records.append(record)
                print(f"     [TESTE 100 eps] Média={stats['test_mean']:>8.2f} ± {stats['test_sem']:<5.2f} | "
                      f"CI95%=[{stats['ci95_low']:>7.2f}, {stats['ci95_high']:>7.2f}] | "
                      f"Mediana={stats['test_median']:>7.2f} | Std={stats['test_std']:>6.2f} | "
                      f"SNR={stats['snr']:>5.2f} | Gap={stats['optimism_gap']:>6.2f} | "
                      f"Sucesso={stats['success_rate_pct']:>5.1f}%")

    # Salva JSON completo (incluindo arrays brutos para reamostragem/gráficos)
    json_path = os.path.join(OUT_DIR, "validation_100_results.json")
    with open(json_path, "w") as f:
        json.dump(validation_records, f, indent=2)
    with open(os.path.join(ART_DIR, "validation_100_results.json"), "w") as f:
        json.dump(validation_records, f, indent=2)
    print(f"\n[OK] Salvo JSON: {json_path}")

    # Salva CSV resumido
    csv_path = os.path.join(OUT_DIR, "validation_100_summary.csv")
    fieldnames = [
        "algorithm", "family", "environment", "f_train_best", "test_mean", "test_sem",
        "ci95_low", "ci95_high", "test_median", "iqr", "test_std", "snr", "cv",
        "optimism_gap", "success_rate_pct", "train_time_s"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(validation_records)
    with open(os.path.join(ART_DIR, "validation_100_summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(validation_records)
    print(f"[OK] Salvo CSV: {csv_path}")

    # 4. Geração de Gráficos Estatísticos Rigorosos
    print("\nGerando gráficos estatísticos de validação...")
    _generate_statistical_plots(validation_records)

    # 5. Imprime Tabela Markdown Acadêmica
    print("\n" + "=" * 115)
    print("TABELA CONSOLIDADA DE VALIDAÇÃO ESTÍSTICA (100 EPISÓDIOS OUT-OF-SAMPLE)")
    print("=" * 115)
    print(f"{'Algoritmo':<13} {'Fam':<4} {'Ambiente':<24} {'Train Best':>11} {'Test Mean ± SEM':>18} {'95% Bootstrap CI':>21} {'SNR':>6} {'Gap':>8} {'Sucesso%':>9}")
    print("-" * 115)
    for r in validation_records:
        mean_sem = f"{r['test_mean']:.1f} ± {r['test_sem']:.1f}"
        ci_str = f"[{r['ci95_low']:.1f}, {r['ci95_high']:.1f}]"
        print(f"{r['algorithm']:<13} F{r['family']:<3} {r['environment']:<24} {r['f_train_best']:>11.2f} {mean_sem:>18} {ci_str:>21} {r['snr']:>6.2f} {r['optimism_gap']:>8.2f} {r['success_rate_pct']:>8.1f}%")
    print("=" * 115)


def _generate_statistical_plots(records):
    """Gera visualizações estatísticas (barras com erro SEM, boxplots e métricas de ruído)."""
    envs = list(dict.fromkeys(r["environment"] for r in records))
    
    # Cores por família
    COLORS = {1: "#E63946", 2: "#457B9D", 3: "#52B788"}

    # 1. Gráfico de Barras com Intervalo de Confiança 95%
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), facecolor="#0d1117")
    axes = axes.flatten()

    for idx, env in enumerate(envs):
        ax = axes[idx]
        ax.set_facecolor("#161b22")
        ax.set_title(f"{env} — Retorno Médio em Teste (N=100 eps ± 95% CI)", color="white", fontsize=11, fontweight="bold", pad=8)
        ax.set_ylabel("Retorno Acumulado", color="#8b949e", fontsize=9)
        ax.tick_params(colors="#8b949e", labelsize=8)
        for spine in ax.spines.values(): spine.set_edgecolor("#30363d")
        ax.grid(True, axis="y", color="#21262d", linewidth=0.5, linestyle="--")

        sub_records = [r for r in records if r["environment"] == env]
        x_pos = np.arange(len(sub_records))
        names = [f"{r['algorithm']}\n(F{r['family']})" for r in sub_records]
        means = [r["test_mean"] for r in sub_records]
        yerr_low  = [r["test_mean"] - r["ci95_low"] for r in sub_records]
        yerr_high = [r["ci95_high"] - r["test_mean"] for r in sub_records]
        colors = [COLORS[r["family"]] for r in sub_records]

        bars = ax.bar(x_pos, means, yerr=[yerr_low, yerr_high], capsize=4,
                      color=colors, alpha=0.85, edgecolor="#0d1117", linewidth=0.5,
                      error_kw=dict(ecolor="white", lw=1.2, capthick=1.2))

        # Adiciona rótulo de texto com média e SNR
        for i, (m, r) in enumerate(zip(means, sub_records)):
            offset = abs(m) * 0.05 if m != 0 else 5.0
            va = "bottom" if m >= 0 else "top"
            ax.text(x_pos[i], m + (offset if m >= 0 else -offset),
                    f"{m:.1f}\nSNR:{r['snr']:.1f}",
                    ha="center", va=va, color="white", fontsize=7.5, fontweight="bold")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, color="#8b949e", fontsize=8.5)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "validation_100_ci95_bars.png")
    art_path = os.path.join(ART_DIR, "validation_100_ci95_bars.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.savefig(art_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[plot] Salvo: {out_path}")
    print(f"[plot] Salvo: {art_path}")

    # 2. Gráfico do Viés de Otimismo (Winner's Curse: Treino vs Teste)
    fig2, ax2 = plt.subplots(figsize=(13, 6), facecolor="#0d1117")
    ax2.set_facecolor("#161b22")
    ax2.set_title("Lacuna de Otimismo (Winner's Curse): Fitness Treino (2 rollouts) vs Teste Real (100 rollouts)",
                  color="white", fontsize=12, fontweight="bold", pad=10)
    ax2.set_ylabel("Lacuna de Otimismo: Δ = f_train - R_test", color="#8b949e", fontsize=10)
    ax2.tick_params(colors="#8b949e")
    for spine in ax2.spines.values(): spine.set_edgecolor("#30363d")
    ax2.grid(True, axis="y", color="#21262d", linewidth=0.5, linestyle="--")

    labels = [f"{r['algorithm']} (F{r['family']})\n[{r['environment'].split('-')[0]}]" for r in records]
    gaps   = [r["optimism_gap"] for r in records]
    colors = ["#FF6B6B" if g > 15 else "#52B788" if g <= 5 else "#FFD166" for g in gaps]

    x_gaps = np.arange(len(records))
    ax2.bar(x_gaps, gaps, color=colors, alpha=0.85, edgecolor="#0d1117", width=0.7)
    for i, g in enumerate(gaps):
        ax2.text(x_gaps[i], g + (1 if g >= 0 else -3), f"{g:.1f}", ha="center",
                 va="bottom" if g >= 0 else "top", color="white", fontsize=7.5, fontweight="bold")

    ax2.set_xticks(x_gaps)
    ax2.set_xticklabels(labels, color="#8b949e", fontsize=7.5, rotation=45, ha="right")
    ax2.axhline(0, color="#8b949e", linestyle="--", linewidth=0.8)

    plt.tight_layout()
    out_gap = os.path.join(OUT_DIR, "optimism_gap_analysis.png")
    art_gap = os.path.join(ART_DIR, "optimism_gap_analysis.png")
    plt.savefig(out_gap, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
    plt.savefig(art_gap, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
    plt.close()
    print(f"[plot] Salvo: {out_gap}")
    print(f"[plot] Salvo: {art_gap}")


if __name__ == "__main__":
    main()
