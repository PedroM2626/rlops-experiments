def forward(params, obs):
    """MLP forward -> (logits, values). obs: (B, obs_dim)."""
    import jax.numpy as jnp
    h = jnp.tanh(obs @ params["a1w"].T + params["a1b"])
    h = jnp.tanh(h @ params["a2w"].T + params["a2b"])
    logits = h @ params["a3w"].T + params["a3b"]
    v = jnp.tanh(obs @ params["c1w"].T + params["c1b"])
    v = jnp.tanh(v @ params["c2w"].T + params["c2b"])
    value = (v @ params["c3w"].T + params["c3b"]).squeeze(-1)
    return logits, value


def cat_sample(key, logits):
    import jax
    import jax.numpy as jnp
    return jax.random.categorical(key, logits)


def cat_logprob(logits, actions):
    import jax
    import jax.numpy as jnp
    lse = jax.scipy.special.logsumexp(logits, axis=-1)
    return jnp.take_along_axis(logits, actions[..., None],
                               axis=-1).squeeze(-1) - lse


def cat_entropy(logits):
    import jax
    import jax.numpy as jnp
    lse = jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    logp = logits - lse
    return -jnp.sum(jnp.exp(logp) * logp, axis=-1)
