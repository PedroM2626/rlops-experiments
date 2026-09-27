"""Learning-curve loading for plot_results.py: collects the per-seed curve CSVs
per variant and interpolates them onto a shared step grid for mean +/- sd bands.
"""


def load_curves(env_id):
    import csv as _csv
    import numpy as _np
    from pathlib import Path as _P
    import config as _c
    import variants as _v
    per: dict[str, list[tuple[_np.ndarray, _np.ndarray]]] = {}
    for p in sorted(_P(_v.CURVES_DIR).glob(f"*_{env_id}_seed*.csv")):
        vid = p.name.split(f"_{env_id}_seed")[0]
        steps, vals = [], []
        with open(p, encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                steps.append(int(row["step"]))
                vals.append(float(row["mean_reward"]))
        if steps:
            per.setdefault(vid, []).append(
                (_np.asarray(steps), _np.asarray(vals)))
    return per


def interp_mean(curves):
    import numpy as _np
    grid = _np.unique(_np.concatenate([s for s, _ in curves]))
    mat = []
    for s, v in curves:
        mat.append(_np.interp(grid, s, v))
    mat = _np.asarray(mat)
    if len(mat) > 1:
        sd = mat.std(0, ddof=1)
    else:
        sd = _np.zeros_like(mat[0])
    return grid, mat.mean(0), sd
