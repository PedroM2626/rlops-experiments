"""Statistics for the compute_gae off-by-one probe (reads probe_gae/data/).

Arm A = the indexing this study committed (dones[t + 1]); arm B = the SB3/CleanRL
indexing (dones[t]). Same interpreter, libraries, seeds and budget per pair, so
the only difference is which value gates the bootstrap.

needs scipy (as the study's own statistics do).
"""
import csv
import glob
import os
import statistics as st

from scipy import stats

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def report(name, av, bv, seeds):
    diffs = [b - a for a, b in zip(av, bv)]
    se = st.stdev(diffs) / len(diffs) ** 0.5
    tcrit = stats.t.ppf(0.975, len(diffs) - 1)
    t, p_t = stats.ttest_rel(bv, av)
    w, p_w = stats.wilcoxon(bv, av)
    lev, p_lev = stats.levene(av, bv, center="median")
    u, p_u = stats.mannwhitneyu(av, bv, alternative="two-sided")
    print(f"\n=== {name}   n={len(seeds)} paired seeds ===")
    print(f"  A (dones[t+1]): mean {st.mean(av):7.1f}  sd {st.pstdev(av):6.1f}")
    print(f"  B (dones[t]) : mean {st.mean(bv):7.1f}  sd {st.pstdev(bv):6.1f}")
    print(f"  paired B-A   : mean {st.mean(diffs):+.1f}  SE {se:.1f}  "
          f"95% CI [{st.mean(diffs) - tcrit * se:+.1f}, "
          f"{st.mean(diffs) + tcrit * se:+.1f}]")
    print(f"  t={t:+.2f} p={p_t:.4f} | Wilcoxon p={p_w:.4f} | "
          f"Levene p={p_lev:.4f} | MWU p={p_u:.4f}")
    print(f"  effect: {st.mean(diffs) / st.pstdev(av + bv):+.2f} pooled sd")
    if p_t > 0.05 and abs(st.mean(diffs)) > 1e-9:
        need = (2.8 * st.stdev(diffs) / abs(st.mean(diffs))) ** 2
        print(f"  seeds needed for 80% power at this observed effect: {need:.0f} "
              f"(have {len(diffs)})")


def ll_eval(arm):
    """Union every eval matrix present for this arm (the base 15 seeds plus any
    ll_eval_rewards_<arm>_*.csv extension)."""
    out = {}
    for path in sorted(glob.glob(os.path.join(DATA, f"ll_eval_rewards_{arm}*.csv"))):
        rows = list(csv.reader(open(path, encoding="utf-8")))
        for i, header in enumerate(rows[0]):
            out[int(header.rsplit("__seed", 1)[1])] = st.mean(
                [float(r[i]) for r in rows[1:]])
    return out


def cartpole(scale):
    out = {"A": {}, "B": {}}
    with open(os.path.join(DATA, "cartpole_finals.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if float(r["budget_scale"]) == scale:
                out[r["arm"]][int(r["seed"])] = float(r["final_curve_mean20"])
    return out["A"], out["B"]


a, b = ll_eval("A"), ll_eval("B")
seeds = sorted(set(a) & set(b))
report("LunarLander-v3, eval mean over 200 episodes", [a[s] for s in seeds],
       [b[s] for s in seeds], seeds)

for scale in (0.05, 0.1):
    ca, cb = cartpole(scale)
    s2 = sorted(set(ca) & set(cb))
    if s2:
        report(f"CartPole-v1, final training curve, budget scale {scale}",
               [ca[s] for s in s2], [cb[s] for s in s2], s2)
