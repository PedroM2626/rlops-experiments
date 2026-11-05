"""
Regenerates the data tables of README.md from the result files, so that the documented numbers
cannot drift from the stored results. Tables sit between marker comments:
    <!-- BEGIN:name --> ... <!-- END:name -->
Usage: python generate_readme_tables.py
"""
import os, csv, json

ROOT = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(ROOT, "results")
README = os.path.join(ROOT, "README.md")

ENVS = ["CartPole-v1", "Acrobot-v1", "Pendulum-v1", "MountainCarContinuous-v0"]
SHORT = {"CartPole-v1": "CartPole-v1", "Acrobot-v1": "Acrobot-v1", "Pendulum-v1": "Pendulum-v1",
         "MountainCarContinuous-v0": "MountainCarCont"}
ORDER = ["SimpleGA", "DE", "OpenAI-ES", "LinearGP", "CartesianGP", "CMA-ES", "PBIL"]
FAM = {"SimpleGA": 1, "DE": 1, "OpenAI-ES": 1, "LinearGP": 2, "CartesianGP": 2, "CMA-ES": 3, "PBIL": 3}
FAMNAME = {1: "F1: Direct Solutions", 2: "F2: Programs", 3: "F3: Models (EDA)"}
PARADIGM = {"DE": "rand/1/bin trial vectors", "SimpleGA": "Tournament + SBX crossover", "OpenAI-ES": "NES gradient ascent",
            "CartesianGP": "Positional 2D DAG (1+7)ES", "LinearGP": "Register sequence",
            "CMA-ES": r"Covariance $\mathcal{N}(\mu, \sigma^2 \Sigma)$", "PBIL": "Univariate marginal model"}
DAG = "†"          # LinearGP on Pendulum: see README


def f(x, nd=2):
    x = float(x)
    s = f"{abs(x):.{nd}f}"
    return ("−" if x < 0 and float(s) != 0 else "") + s if x != 0 or True else s


def name(alg, env):
    return f"**{alg}**" + (DAG if (alg == "LinearGP" and env == "Pendulum-v1") else "")


def load_summary():
    rows = {}
    for r in csv.DictReader(open(os.path.join(R, "results_summary.csv"), encoding="utf-8")):
        rows[(r["algorithm"], r["environment"])] = r          # later rows win (stale duplicates are dropped)
    return rows


def table_4_1():
    S = load_summary()
    out = ["| Family | Algorithm | Paradigm | CartPole-v1 (Best Train) | Acrobot-v1 (Best Train) | CartPole Time (s) |",
           "|:---|:---|:---|:---:|:---:|:---:|"]
    for a in ORDER:
        c, b = S[(a, "CartPole-v1")], S[(a, "Acrobot-v1")]
        out.append(f"| **{FAMNAME[FAM[a]]}** | **{a}** | {PARADIGM[a]} | {f(c['final_best_fitness'])} | "
                   f"{f(b['final_best_fitness'])} | {float(c['total_time_s']):.1f}s |")
    return "\n".join(out)


def table_4_2():
    S = load_summary()
    out = ["| Family | Algorithm | Pendulum-v1 (Best) | Pendulum-v1 (Mean Final) | MountainCarCont (Best) | MountainCarCont (Mean Final) | Pendulum Time (s) |",
           "|:---|:---|:---:|:---:|:---:|:---:|:---:|"]
    for a in ORDER:
        p, m = S[(a, "Pendulum-v1")], S[(a, "MountainCarContinuous-v0")]
        out.append(f"| **{FAMNAME[FAM[a]]}** | {name(a, 'Pendulum-v1')} | {f(p['final_best_fitness'])} | {f(p['final_mean_fitness'])} | "
                   f"{f(m['final_best_fitness'])} | {f(m['final_mean_fitness'])} | {float(p['total_time_s']):.1f}s |")
    return "\n".join(out)


def table_5_2():
    V = {(r["algorithm"], r["environment"]): r for r in csv.DictReader(open(os.path.join(R, "validation_100_summary.csv"), encoding="utf-8"))}
    out = ["| Family | Algorithm | Environment | Train Best ($N \\le 3$) | Test Mean $\\pm$ SEM | 95% Bootstrap CI | Median | IQR | Std Dev | SNR | Optimism Gap ($\\Delta$) | Success Rate (%) |",
           "|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"]
    sep = "|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    for i, env in enumerate(ENVS):
        if i:
            out.append(sep)
        for a in ORDER:
            # keep the original ordering convention F1, F2, F3 within an environment
            pass
        for a in ["SimpleGA", "DE", "OpenAI-ES", "LinearGP", "CartesianGP", "CMA-ES", "PBIL"]:
            r = V[(a, env)]
            snr = float(r["snr"])
            snr_s = "$\\infty$" if snr > 1e6 else f"{snr:.2f}"
            gap = float(r["optimism_gap"])
            out.append(f"| **F{FAM[a]}** | {name(a, env)} | {SHORT[env]} | {f(r['f_train_best'])} | {f(r['test_mean'])} $\\pm$ {float(r['test_sem']):.2f} | "
                       f"[{f(r['ci95_low'])}, {f(r['ci95_high'])}] | {f(r['test_median'])} | {float(r['iqr']):.2f} | {float(r['test_std']):.2f} | {snr_s} | "
                       f"{'+' if gap > 0 else ''}{f(gap)} | {float(r['success_rate_pct']):.1f}% |")
    return "\n".join(out)


def table_6_6_2():
    D = json.load(open(os.path.join(R, "multiseed_study.json"), encoding="utf-8"))
    out = ["| Family | Algorithm | Environment | Mean across Seeds $\\pm$ SEM | Stratified 95% CI | IQM | $\\sigma_{\\text{train}}$ | $\\sigma_{\\text{eval}}$ | Variance Ratio $\\rho$ | Success Rate (%) | Mean Train Best | Mean $\\Delta$ |",
           "|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"]
    sep = "|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    idx = {(r["algorithm"], r["environment"]): r for r in D}
    for i, env in enumerate(ENVS):
        if i:
            out.append(sep)
        for a in ORDER:
            r = idx[(a, env)]
            rho = "—" if r["sigma_eval"] < 1e-6 else f"{r['variance_ratio_rho']:.2f}"
            gap = r["mean_optimism_gap"]
            out.append(f"| **F{FAM[a]}** | {name(a, env)} | {SHORT[env]} | {f(r['mean_across_seeds'])} $\\pm$ {r['sem_across_seeds']:.2f} | "
                       f"[{f(r['stratified_ci95_low'])}, {f(r['stratified_ci95_high'])}] | {f(r['iqm'])} | {r['sigma_train']:.2f} | {r['sigma_eval']:.2f} | "
                       f"{rho} | {r['success_rate_pct']:.1f}% | {f(r['mean_train_best'])} | {'+' if gap > 0 else ''}{f(gap)} |")
    return "\n".join(out)


TABLES = {"table_4_1": table_4_1, "table_4_2": table_4_2, "table_5_2": table_5_2, "table_6_6_2": table_6_6_2}


def main():
    s = open(README, encoding="utf-8", newline="").read()
    crlf = "\r\n" in s
    s = s.replace("\r\n", "\n")
    for name_, fn in TABLES.items():
        b, e = f"<!-- BEGIN:{name_} -->", f"<!-- END:{name_} -->"
        if b not in s:
            raise SystemExit(f"missing marker {b}")
        i, j = s.index(b) + len(b), s.index(e)
        s = s[:i] + "\n" + fn() + "\n" + s[j:]
    open(README, "w", encoding="utf-8", newline="").write(s.replace("\n", "\r\n") if crlf else s)
    print("README tables regenerated:", ", ".join(TABLES))


if __name__ == "__main__":
    main()
