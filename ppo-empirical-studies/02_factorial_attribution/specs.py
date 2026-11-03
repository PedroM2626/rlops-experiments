"""SB3 vs CleanRL algorithm-spec deltas (the ONLY allowed algo differences).

Verified against stable-baselines3 2.x (ppo/ppo.py, common/policies.py,
common/buffers.py, common/on_policy_algorithm.py) and the CleanRL ppo.py
reference implementation.

Both envs here (CartPole-v1, LunarLander-v3) are DISCRETE, which makes the
specs much closer than in the continuous case. Remaining deltas:

1. VALUE-LOSS CLIPPING (clip_range_vf / clip_vloss)
   - SB3 default:            clip_range_vf=None  -> NO value clipping.
   - CleanRL default:        clip_vloss=True     -> clipped value loss with
     the same epsilon as the policy (CLIP_RANGE).
   - Unified in this study:  an explicit boolean SPEC["clip_value_loss"].

2. LEARNING-RATE ANNEALING (anneal_lr)
   - SB3 default:            constant LR (float lr -> ConstantSchedule).
   - CleanRL default:        linear annealing lr * (1 - progress).
   - Unified: SPEC["anneal_lr"] bool. Progress = 1 - global_step/total.

3. ADVANTAGE NORMALISATION SCOPE
   - Both normalise per-MINIBATCH with eps 1e-8 and skip batches of size 1
     (SB3 ppo.py train(); CleanRL ppo.py update loop).
   - No delta: every arm in this study normalises per-minibatch. Logged as
     NORMALIZE_ADVANTAGE="minibatch" in meta.json.

4. ENTROPY TERM
   - CleanRL discrete:  mean Categorical entropy, loss -= ent_coef * mean.
   - SB3 discrete:      entropy_loss = -mean(entropy), added as
     loss = pg + ent_coef*entropy_loss + vf_coef*vf  (identical math).
   - No delta.

5. VALUE LOSS FORMULATION
   - SB3:     F.mse_loss(returns, values_pred) i.e. mean((r-v)^2), no 0.5.
   - CleanRL: 0.5 * mean((v-r)^2)  (also for the clipped branch).
   - Unified: SPEC["value_loss_half"] False (SB3) / True (CleanRL).
     NOTE: with a SHARED vf_coef this is a real 2x scale delta on the value
     gradient. This is intentional: it is part of the published specs, and
     Q4/Q5 quantify exactly how much it matters.

6. NETWORK INITIALISATION (discrete heads)
   - SB3 ActorCriticPolicy ortho_init=True: hidden gain sqrt(2),
     action head gain 0.01, value head gain 1.0, biases 0.
   - CleanRL Agent.layer_init: hidden gain sqrt(2), actor head std 0.01,
     critic head std 1.0, biases 0.
   - No delta for discrete envs (identical). All arms use these gains.
   - (Continuous log-std handling would differ; out of scope: both envs
     here are discrete.)

7. OPTIMISER
   - Both Adam(lr, eps=1e-5). SB3 injects eps=1e-5 for Adam by default;
     CleanRL passes eps=1e-5 explicitly. All arms use Adam + ADAM_EPS.
   - CleanRL defaults to max_grad_norm=0.5; SB3 default 0.5 too. Shared.

8. GAE / TIMEOUT BOOTSTRAP
   - SB3 handles TimeLimit truncation by bootstrapping: on done-with-
     truncation it adds gamma * V(terminal_obs) to the reward and keeps
     episode_starts=True semantics for GAE (see collect_rollouts #633).
   - CleanRL reference stores done = terminated | truncated, so a timeout
     cuts bootstrapping (no terminal-value correction).
   - Unified: SPEC["timeout_bootstrap"] True (SB3) / False (CleanRL).
     All manual arms (torch + JAX) implement the SB3 branch explicitly when
     the SB3 spec is selected. Rationale: without this, "SB3 spec" would
     silently differ by env horizon handling, contaminating Q2/Q4.

9. ROLLOUT / MINIBATCH ORDER
   - SB3: np.random.permutation over the whole rollout per epoch, sliced
     into batch_size chunks (last chunk may be smaller).
   - CleanRL: np.random.shuffle of arange(batch_size), sliced into
     minibatch_size chunks (equal here since n_envs=1 -> batch==rollout).
   - With N_ENVS=1 and N_STEPS % BATCH_SIZE == 0 both reduce to a uniform
     random partition each epoch. All arms implement: permutation per
     epoch via the backend RNG (torch.randperm / jax.random.permutation
     with an explicitly folded key), so the statistical behaviour matches.
"""

from __future__ import annotations

SPEC_SB3: dict = {
    "name": "sb3",
    "clip_value_loss": False,
    "anneal_lr": False,
    "advantage_normalisation": "minibatch",
    "value_loss_half": False,
    "timeout_bootstrap": True,
    "ortho_hidden_gain": 2 ** 0.5,
    "ortho_actor_gain": 0.01,
    "ortho_critic_gain": 1.0,
}

SPEC_CLEANRL: dict = {
    "name": "cleanrl",
    "clip_value_loss": True,
    "anneal_lr": True,
    "advantage_normalisation": "minibatch",
    "value_loss_half": True,
    "timeout_bootstrap": False,
    "ortho_hidden_gain": 2 ** 0.5,
    "ortho_actor_gain": 0.01,
    "ortho_critic_gain": 1.0,
}

SPECS: dict[str, dict] = {"sb3": SPEC_SB3, "cleanrl": SPEC_CLEANRL}


def get_spec(name: str) -> dict:
    """Return the algorithm spec dict for 'sb3' or 'cleanrl'."""
    if name not in SPECS:
        raise ValueError(f"unknown spec {name!r}; expected one of {sorted(SPECS)}")
    return dict(SPECS[name])


# --- Ablation arms (§8 of README): start from the CleanRL spec and flip
# exactly ONE delta to the SB3 value. Run on the JAX stack only (identical
# code, so any observed delta is attributable to the single spec change).
ABLATIONS: dict[str, dict] = {
    "novclip":  {"clip_value_loss": False},
    "noanneal": {"anneal_lr": False},
    "fullmse":  {"value_loss_half": False},
    "tboot":    {"timeout_bootstrap": True},
}


def get_ablated_spec(abl: str) -> dict:
    """CleanRL spec with one SB3 delta applied (see ABLATIONS)."""
    if abl not in ABLATIONS:
        raise ValueError(f"unknown ablation {abl!r}; expected one of "
                         f"{sorted(ABLATIONS)}")
    spec = dict(SPECS["cleanrl"])
    spec.update(ABLATIONS[abl])
    spec["name"] = f"cleanrl+ablation[{abl}]"
    return spec

