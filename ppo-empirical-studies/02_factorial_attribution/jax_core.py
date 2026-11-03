"""Parameter initialisation for the JAX PPO agent: an orthogonal init helper
and the actor/critic MLP weights, built to match PyTorch's init semantics.
"""


def orthogonal(key, shape, gain):
    """Orthogonal init with explicit gain (matches torch semantics)."""
    import jax
    init = jax.nn.initializers.orthogonal(gain)
    return init(key, shape).astype("float32")


def init_params(key, obs_dim, act_dim, spec):
    """Fresh actor/critic weights from 8 subkeys of ``key``: orthogonal matrices laid out
    (out, in) like ``nn.Linear``, with the spec's per-head gains and zero biases, all float32."""
    import numpy as _np
    import jax
    import config as _cc
    ks = jax.random.split(key, 8)
    gh, ga, gc = (spec["ortho_hidden_gain"], spec["ortho_actor_gain"],
                  spec["ortho_critic_gain"])
    h1, h2 = _cc.NET_ARCH
    p = {
        "a1w": orthogonal(ks[0], (h1, obs_dim), gh),
        "a1b": _np.zeros(h1, _np.float32),
        "a2w": orthogonal(ks[1], (h2, h1), gh),
        "a2b": _np.zeros(h2, _np.float32),
        "a3w": orthogonal(ks[2], (act_dim, h2), ga),
        "a3b": _np.zeros(act_dim, _np.float32),
        "c1w": orthogonal(ks[3], (h1, obs_dim), gh),
        "c1b": _np.zeros(h1, _np.float32),
        "c2w": orthogonal(ks[4], (h2, h1), gh),
        "c2b": _np.zeros(h2, _np.float32),
        "c3w": orthogonal(ks[5], (1, h2), gc),
        "c3b": _np.zeros(1, _np.float32),
    }
    return {k: _np.asarray(v, dtype=_np.float32) for k, v in p.items()}
