"""Actor/critic forward pass and categorical-distribution ops (action sampling,
log-probs, entropy) for the JAX PPO agent.
"""


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
    """One Categorical sample per row of unnormalised ``logits`` (Gumbel-max, as torch's
    ``Categorical(logits=...).sample()``)."""
    import jax
    import jax.numpy as jnp
    return jax.random.categorical(key, logits)


def cat_logprob(logits, actions):
    """Log-prob of ``actions`` under the Categorical defined by ``logits``, via logsumexp."""
    import jax
    import jax.numpy as jnp
    lse = jax.scipy.special.logsumexp(logits, axis=-1)
    return jnp.take_along_axis(logits, actions[..., None],
                               axis=-1).squeeze(-1) - lse


def cat_entropy(logits):
    """Shannon entropy -sum(p*log p) of the Categorical defined by ``logits``, one value per row."""
    import jax
    import jax.numpy as jnp
    lse = jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    logp = logits - lse
    return -jnp.sum(jnp.exp(logp) * logp, axis=-1)
