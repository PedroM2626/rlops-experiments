"""
CLI for the TD baselines (rl_baselines.py).
  python run_rl_baselines.py --algo sac --env Pendulum-v1 --steps 30000 --out results/sac_pendulum.json
  python run_rl_baselines.py --algo dqn --env CartPole-v1 --steps 50000 --hp lr=0.001 batch=64 target_every=250
Results are stored per seed ({seed: {evals: {step: metrics}}}) and the run resumes from an existing file.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_baselines import run, HP_DEFAULT


def parse_hp(items):
    hp = {}
    for it in items or []:
        k, v = it.split("=")
        hp[k] = v if k == "act" else (None if v == "None" else (int(v) if v.lstrip("-").isdigit() else float(v)))
    return hp


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=["sac", "td3", "dqn"])
    ap.add_argument("--env", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 202, 303, 404])
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--eval-at", type=int, nargs="+", default=None)
    ap.add_argument("--hp", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ev = set(a.eval_at) if a.eval_at else {a.steps // 6, a.steps // 3, 2 * a.steps // 3, a.steps}
    recs = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for sd in a.seeds:
        if str(sd) in recs:
            continue
        recs[str(sd)] = run(a.algo, a.env, sd, a.steps, parse_hp(a.hp), ev)
        json.dump(recs, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
