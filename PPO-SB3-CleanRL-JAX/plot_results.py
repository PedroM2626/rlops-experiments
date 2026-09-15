"""Seed-mean training curves + per-seed boxplots (matplotlib only)."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=None)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import csv as _csv
    import config as _c
    import variants as _v
    from plot_style import COLORS, LABELS
    from plot_curves import load_curves, interp_mean

    envs = [args.env] if args.env else list(_c.ENVS)
    _v.ensure_dirs()
    for env_id in envs:
        per = load_curves(env_id)
        fig, ax = plt.subplots(figsize=(11, 6))
        for vid, curves in per.items():
            grid, mean, std = interp_mean(curves)
            c = COLORS.get(vid, "gray")
            ax.plot(grid, mean, color=c, lw=2.2,
                    label=f"{LABELS.get(vid, vid)} (n={len(curves)})")
            ax.fill_between(grid, mean - std, mean + std, color=c, alpha=0.18)
        ax.axhline(_c.SOLVED_THRESHOLD[env_id], color="gray", ls="--",
                   lw=1, alpha=0.7)
        ax.text(0.01, 0.97, f"solved={_c.SOLVED_THRESHOLD[env_id]}",
                transform=ax.transAxes, fontsize=9, color="gray", va="top")
        ax.set_xlabel("Environment steps")
        ax.set_ylabel(f"Rolling-{_c.ROLLING_WINDOW} train return (seed-mean)")
        ax.set_title(f"PPO training curves — {env_id} (mean ± SD over seeds)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        out = Path(_v.PLOTS_DIR) / f"training_curves_{env_id}.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        print(f"[{env_id}] training curves -> {out}")

        rp = Path(_v.TABLES_DIR) / f"eval_rewards_{env_id}.csv"
        if rp.exists():
            with open(rp, encoding="utf-8") as f:
                rdr = _csv.DictReader(f)
                cols = rdr.fieldnames or []
                rows = list(rdr)
            vids = sorted({c.rsplit("__seed", 1)[0] for c in cols})
            fig2, ax2 = plt.subplots(figsize=(13, 6.5))
            box_data = [[float(r[c]) for r in rows for c in cols
                  if c.startswith(v + "__seed")] for v in vids]
            box_labels = [LABELS.get(v, v) for v in vids]
            try:
                bp = ax2.boxplot(
                    box_data,
                    tick_labels=box_labels,
                    patch_artist=True, showmeans=True)
            except TypeError:
                bp = ax2.boxplot(
                    box_data,
                    labels=box_labels,
                    patch_artist=True, showmeans=True)
            for patch, v in zip(bp["boxes"], vids):
                patch.set_facecolor(COLORS.get(v, "gray"))
                patch.set_alpha(0.65)
            ax2.axhline(_c.SOLVED_THRESHOLD[env_id], color="gray",
                        ls="--", lw=1, alpha=0.7)
            ax2.set_ylabel(f"Eval return ({_c.N_EVAL_EPISODES} eps/seed)")
            ax2.set_title(f"Deterministic eval returns — {env_id}")
            ax2.grid(True, axis="y", alpha=0.3)
            for lbl in ax2.get_xticklabels():
                lbl.set_rotation(28)
                lbl.set_ha("right")
            fig2.tight_layout()
            out2 = Path(_v.PLOTS_DIR) / f"eval_boxplot_{env_id}.png"
            fig2.savefig(out2, dpi=160)
            plt.close(fig2)
            print(f"[{env_id}] eval boxplot -> {out2}")


if __name__ == "__main__":
    main()
