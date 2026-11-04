"""
Per-initial-angle breakdown of Pendulum-v1 test returns (SAC, TD3, CMA-ES G=250). The 100 test
episodes use the same reset keys for every method (PRNGKey(99999) split in 100, then split again),
so theta_0 of each episode is recovered by resetting the environment with the same keys.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, jax, gymnax

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
env, ep = gymnax.make("Pendulum-v1")
keys = jax.random.split(jax.random.PRNGKey(99999), 100)
th0 = np.array([float(env.reset(jax.random.split(k)[0], ep)[1].theta) for k in keys])
ang = np.abs(((th0 + np.pi) % (2 * np.pi)) - np.pi)          # |theta_0| in [0, pi]

def per_ep(path, fn):
    return fn(json.load(open(os.path.join(R, path))))

sac = np.mean([v["evals"]["30000"]["raw_returns"] for v in json.load(open(os.path.join(R, "sac_pendulum.json"))).values()], 0)
td3 = np.mean([v["evals"]["30000"]["raw_returns"] for v in json.load(open(os.path.join(R, "td3_pendulum.json"))).values()], 0)
cma = [json.load(open(os.path.join(R, "pendulum_stress_test_250.json")))["evaluation_metrics"]["raw_returns"]]
for f in ("pendulum_stress_multiseed.json", "pendulum_cmaes_G250_extra_seeds.json"):
    cma += [v["evaluation_metrics"]["raw_returns"] for v in json.load(open(os.path.join(R, f))).values()]
cma = np.mean(cma, 0)

bins = [(0, np.pi / 3), (np.pi / 3, 2 * np.pi / 3), (2 * np.pi / 3, 5 * np.pi / 6), (5 * np.pi / 6, np.pi + 1e-9)]
out = {"theta0_abs": ang.tolist(), "bins": []}
print(f"{'|theta0| bin':<22}{'n':>4}  {'SAC mean':>9}{'SAC succ':>9}{'TD3 mean':>9}{'TD3 succ':>9}{'CMA mean':>9}{'CMA succ':>9}")
for lo, hi in bins:
    m = (ang >= lo) & (ang < hi)
    row = {"lo": lo, "hi": hi, "n": int(m.sum())}
    cells = []
    for name, r in (("SAC", sac), ("TD3", td3), ("CMA", cma)):
        row[name + "_mean"] = float(r[m].mean()); row[name + "_succ"] = float((r[m] >= -200).mean() * 100)
        cells += [f"{r[m].mean():9.1f}", f"{(r[m] >= -200).mean()*100:8.0f}%"]
    out["bins"].append(row)
    print(f"[{lo/np.pi:.2f}pi,{hi/np.pi:.2f}pi)".ljust(22) + f"{m.sum():>4}  " + "".join(cells))
print("corr(|theta0|, return): SAC %.2f TD3 %.2f CMA %.2f" % tuple(np.corrcoef(ang, r)[0, 1] for r in (sac, td3, cma)))
out["corr"] = {n: float(np.corrcoef(ang, r)[0, 1]) for n, r in (("SAC", sac), ("TD3", td3), ("CMA", cma))}
json.dump(out, open(os.path.join(R, "pendulum_angle_analysis.json"), "w"), indent=2)
