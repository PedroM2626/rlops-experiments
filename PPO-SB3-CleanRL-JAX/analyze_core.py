def load_rewards(env_id):
    import csv as _csv
    import numpy as _np
    from pathlib import Path as _P
    import variants as _v
    p = _P(_v.TABLES_DIR) / f"eval_rewards_{env_id}.csv"
    with open(p, encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    cols = rows and list(rows[0].keys()) or []
    data = {c: _np.asarray([float(r[c]) for r in rows]) for c in cols}
    return data


def per_variant_seed_means(data):
    import numpy as _np
    out: dict[str, list[float]] = {}
    for col, arr in data.items():
        vid, _, seed = col.rpartition("__seed")
        out.setdefault(vid, []).append(float(arr.mean()))
    return {k: _np.asarray(sorted(v)) for k, v in out.items()}
