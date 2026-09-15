def bootstrap_paired_ci(diffs, n_boot, rng):
    import numpy as _np
    diffs = _np.asarray(diffs, float)
    boots = rng.choice(diffs, size=(n_boot, len(diffs)),
                       replace=True).mean(axis=1)
    lo, hi = _np.percentile(boots, [2.5, 97.5])
    p = float(2 * min((boots > 0).mean(), (boots < 0).mean()))
    return float(boots.mean()), float(lo), float(hi), float(p)


def cliff_delta(a, b):
    import numpy as _np
    a, b = _np.asarray(a, float), _np.asarray(b, float)
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / max(len(a) * len(b), 1)
