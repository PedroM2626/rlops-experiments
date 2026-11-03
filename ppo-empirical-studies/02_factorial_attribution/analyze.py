"""Seed-aggregated stats + planned contrasts with bootstrap CIs.

Reads results/tables/eval_rewards_<env>.csv (cols: <variant>__seed<k>).
Aggregates per-variant means across seeds and tests the 5 planned
contrasts (Q1..Q5, see README) with paired-by-seed differences +
bootstrap 95% CIs and two-sided p-values, plus rank-based effect sizes.

Outputs per env:
  results/tables/stats_<env>.csv      (per-variant seed-aggregated rows)
  results/tables/contrasts_<env>.csv  (one row per planned contrast)
  results/tables/contrasts_<env>.md   (human-readable summary)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import config
import variants

from variants import VARIANTS

CONTRASTS: list[tuple[str, str, str]] = [
    ("Q1_total_gap", "sb3_torch", "cleanrl_torch"),
    ("Q2_stack_given_SB3spec", "sb3_torch", "jax_sb3"),
    ("Q3_stack_given_CleanRLspec", "cleanrl_torch", "jax_cleanrl"),
    ("Q4_specBridge_torch", "sb3_torch", "cleanrl_sb3mode_torch"),
    ("Q5_spec_within_JAX", "jax_sb3", "jax_cleanrl"),
    ("Q6_bridge_vs_JAXSB3", "cleanrl_sb3mode_torch", "jax_sb3"),
]

# Per-delta ablations (README §8): is each CleanRL->SB3 flip, alone,
# enough to close the gap? A = ablated arm, B = cleanrl baseline.
ABLATION_CONTRASTS: list[tuple[str, str, str]] = [
    ("ABL_novclip_vs_cleanrl", "jax_abl_novclip", "jax_cleanrl"),
    ("ABL_noanneal_vs_cleanrl", "jax_abl_noanneal", "jax_cleanrl"),
    ("ABL_fullmse_vs_cleanrl", "jax_abl_fullmse", "jax_cleanrl"),
    ("ABL_tboot_vs_cleanrl", "jax_abl_tboot", "jax_cleanrl"),
    ("ABL_novclip_vs_sb3", "jax_abl_novclip", "jax_sb3"),
    ("ABL_noanneal_vs_sb3", "jax_abl_noanneal", "jax_sb3"),
    ("ABL_fullmse_vs_sb3", "jax_abl_fullmse", "jax_sb3"),
    ("ABL_tboot_vs_sb3", "jax_abl_tboot", "jax_sb3"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=None, choices=config.ENVS)
    args = ap.parse_args()
    envs = [args.env] if args.env else list(config.ENVS)
    from analyze_core import load_rewards, per_variant_seed_means
    from analyze_stats import bootstrap_paired_ci, cliff_delta

    for env_id in envs:
        data = load_rewards(env_id)
        seed_means = per_variant_seed_means(data)
        rng = np.random.default_rng(config.BOOTSTRAP_SEED)
        tdir = Path(variants.TABLES_DIR)
        with open(tdir / f"stats_{env_id}.csv", "w",
                  newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["variant", "n_seeds", "mean", "std", "median",
                        "min", "max"])
            for vid in sorted(seed_means):
                m = seed_means[vid]
                w.writerow([vid, len(m), f"{m.mean():.4f}",
                            f"{m.std(ddof=1) if len(m) > 1 else 0:.4f}",
                            f"{np.median(m):.4f}", f"{m.min():.4f}",
                            f"{m.max():.4f}"])
        def _run_contrasts(pairs):
            out_rows, out_lines = [], []
            for qid, a, b in pairs:
                seeds_a = sorted(int(c.rsplit('__seed', 1)[1])
                                 for c in data if c.startswith(a + "__seed"))
                seeds_b = sorted(int(c.rsplit('__seed', 1)[1])
                                 for c in data if c.startswith(b + "__seed"))
                common = sorted(set(seeds_a) & set(seeds_b))
                if not common:
                    continue
                va = np.asarray([data[f"{a}__seed{s}"].mean()
                                 for s in common])
                vb = np.asarray([data[f"{b}__seed{s}"].mean()
                                 for s in common])
                diffs = va - vb
                bm, lo, hi, p = bootstrap_paired_ci(
                    diffs, config.N_BOOTSTRAP, rng)
                cd = cliff_delta(va, vb)
                out_rows.append([qid, a, b, f"{bm:.4f}",
                                 f"[{lo:.4f},{hi:.4f}]", f"{p:.4f}",
                                 f"{cd:.4f}", len(common)])
                out_lines.append(f"| {qid} | {a} $-$ {b} | {bm:.2f} | "
                                 f"[{lo:.2f}, {hi:.2f}] | {p:.4f} | "
                                 f"{cd:.3f} | {len(common)} |")
            return out_rows, out_lines

        rows, lines = [], [f"# {env_id} planned contrasts",
                           "", "| contrast | A - B | boot_mean | "
                           "95% CI | p | cliff_d | n |",
                           "|---|---|---|---|---|---|---|"]
        rows, ls = _run_contrasts(CONTRASTS)
        lines += ls
        with open(tdir / f"contrasts_{env_id}.csv", "w",
                  newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["contrast", "A", "B", "boot_mean_diff", "ci95",
                        "p_two_sided", "cliff_delta", "n_seeds"])
            w.writerows(rows)
        (tdir / f"contrasts_{env_id}.md").write_text("\n".join(lines),
                                                     encoding="utf-8")
        print(f"[{env_id}] wrote stats + {len(rows)} contrasts")

        arows, als = _run_contrasts(ABLATION_CONTRASTS)
        if arows:
            alines = [f"# {env_id} ablation contrasts (per-delta, JAX)",
                      "", "| contrast | A - B | boot_mean | "
                      "95% CI | p | cliff_d | n |",
                      "|---|---|---|---|---|---|---|"] + als
            with open(tdir / f"contrasts_ablation_{env_id}.csv", "w",
                      newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["contrast", "A", "B", "boot_mean_diff", "ci95",
                            "p_two_sided", "cliff_delta", "n_seeds"])
                w.writerows(arows)
            (tdir / f"contrasts_ablation_{env_id}.md").write_text(
                "\n".join(alines), encoding="utf-8")
            print(f"[{env_id}] wrote {len(arows)} ablation contrasts")


if __name__ == "__main__":
    main()
