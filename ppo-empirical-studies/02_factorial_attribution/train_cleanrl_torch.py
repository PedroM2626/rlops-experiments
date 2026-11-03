"""Arms 2-3/5: manual PyTorch PPO, switchable spec via --mode.

--mode cleanrl: clip_vloss=True, anneal_lr=True, 0.5 value factor,
                no timeout bootstrap.  --mode sb3: no value clip,
                constant LR, full MSE value loss, SB3 timeout bootstrap.
Discrete actions only (both envs are Discrete).
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import config
import variants
from common import (RollingMean, compute_gae, make_env, obs_act_dims,
                    set_all_seeds, write_curve_csv, write_meta_json,
                    package_versions)


def layer_init(layer, std: float):
    """Orthogonal weight init plus zero bias. The arg is named ``std`` (CleanRL's naming) but is
    passed as the orthogonal gain; the gains come from ``spec`` and are identical in both arms."""
    import torch.nn as nn
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


def build_agent(nn, obs_dim: int, act_dim: int, spec: dict):
    """Builds a fresh, untrained PPO actor/critic: two tanh MLPs over ``config.NET_ARCH`` whose
    output layers use the spec's per-head orthogonal gains. Reused by eval_torch.py to reload."""
    class Agent(nn.Module):
        def __init__(self):
            super().__init__()
            gh, ga, gc = (spec["ortho_hidden_gain"],
                          spec["ortho_actor_gain"],
                          spec["ortho_critic_gain"])
            self.actor = nn.Sequential(
                layer_init(nn.Linear(obs_dim, config.NET_ARCH[0]), gh),
                nn.Tanh(),
                layer_init(nn.Linear(config.NET_ARCH[0],
                                     config.NET_ARCH[1]), gh),
                nn.Tanh(),
                layer_init(nn.Linear(config.NET_ARCH[1], act_dim), ga),
            )
            self.critic = nn.Sequential(
                layer_init(nn.Linear(obs_dim, config.NET_ARCH[0]), gh),
                nn.Tanh(),
                layer_init(nn.Linear(config.NET_ARCH[0],
                                     config.NET_ARCH[1]), gh),
                nn.Tanh(),
                layer_init(nn.Linear(config.NET_ARCH[1], 1), gc),
            )

        def forward(self, x):
            return self.actor(x), self.critic(x).squeeze(-1)

        def value(self, x):
            return self.critic(x).squeeze(-1)

        def act(self, x, action=None):
            from torch.distributions import Categorical
            logits = self.actor(x)
            dist = Categorical(logits=logits)
            if action is None:
                action = dist.sample()
            return (action, dist.log_prob(action),
                    dist.entropy(), self.value(x))

    return Agent()


def collect_rollout(env, agent, torch, spec, obs, T, obs_dim):
    """Buffers T single-env steps as (obs, action, logp, reward, done, V(obs)) using a forward
    pass jitted inside this function, so the SB3 spec can timeout-bootstrap a TimeLimit truncation
    with +GAMMA*V(that observation) while clearing its done flag; the CleanRL spec stores done=1."""
    import numpy as _np
    finished: list[float] = []
    b_obs = _np.zeros((T, obs_dim), _np.float32)
    b_act = _np.zeros(T, _np.int64)
    b_logp = _np.zeros(T, _np.float32)
    b_rew = _np.zeros(T, _np.float32)
    b_done = _np.zeros(T, _np.float32)
    b_val = _np.zeros(T, _np.float32)
    last_truly_done = False

    @torch.jit.script
    def net(a1w, a1b, a2w, a2b, a3w, a3b, c1w, c1b, c2w, c2b, c3w, c3b, x):
        ha = torch.tanh(torch.nn.functional.linear(x, a1w, a1b))
        ha = torch.tanh(torch.nn.functional.linear(ha, a2w, a2b))
        logits = torch.nn.functional.linear(ha, a3w, a3b)
        hc = torch.tanh(torch.nn.functional.linear(x, c1w, c1b))
        hc = torch.tanh(torch.nn.functional.linear(hc, c2w, c2b))
        value = torch.nn.functional.linear(hc, c3w, c3b).squeeze(-1)
        return logits, value

    with torch.no_grad():
        for t in range(T):
            ot = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits, v = net(
                agent.actor[0].weight, agent.actor[0].bias,
                agent.actor[2].weight, agent.actor[2].bias,
                agent.actor[4].weight, agent.actor[4].bias,
                agent.critic[0].weight, agent.critic[0].bias,
                agent.critic[2].weight, agent.critic[2].bias,
                agent.critic[4].weight, agent.critic[4].bias,
                ot)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            lp = dist.log_prob(a)
            ai = int(a.item())
            nobs, rew, term, trunc, info = env.step(ai)
            done = bool(term or trunc)
            stored_rew, stored_done = float(rew), float(done)
            if spec["timeout_bootstrap"] and done and trunc:
                # single-env gymnasium has no "terminal_observation"; on
                # TimeLimit truncation `nobs` IS the final observation.
                term_obs = info.get("terminal_observation", nobs)
                tt = torch.as_tensor(
                    _np.asarray(term_obs, dtype=_np.float32)).unsqueeze(0)
                _, tv = net(
                    agent.actor[0].weight, agent.actor[0].bias,
                    agent.actor[2].weight, agent.actor[2].bias,
                    agent.actor[4].weight, agent.actor[4].bias,
                    agent.critic[0].weight, agent.critic[0].bias,
                    agent.critic[2].weight, agent.critic[2].bias,
                    agent.critic[4].weight, agent.critic[4].bias,
                    tt)
                stored_rew = float(rew) + config.GAMMA * float(tv.item())
                stored_done = 0.0
            b_obs[t], b_act[t] = obs, ai
            b_logp[t], b_rew[t] = float(lp.item()), stored_rew
            b_done[t], b_val[t] = stored_done, float(v.item())
            last_truly_done = done
            if "episode" in info:
                finished.append(float(info["episode"]["r"]))
            obs = _np.asarray(env.reset()[0] if done else nobs,
                              dtype=_np.float32)
    return (b_obs, b_act, b_logp, b_rew, b_done, b_val,
            obs, last_truly_done, finished)


def ppo_epochs_fullbatch(agent, torch, nn, opt, spec, batch, n_updates, update):
    """Reference minibatch SGD (CleanRL/SB3): per-minibatch forward+backward+step."""
    o, a_, old_lp, old_v, adv_t, ret_t = batch
    if spec["anneal_lr"]:
        for g in opt.param_groups:
            g["lr"] = config.LEARNING_RATE * (1.0 - (update - 1) / n_updates)
    T = o.shape[0]
    order = np.arange(T)
    for _ in range(config.N_EPOCHS):
        np.random.shuffle(order)
        for mb in order.reshape(-1, config.BATCH_SIZE):
            if len(mb) <= 1:
                continue
            mba = adv_t[mb]
            mba = (mba - mba.mean()) / (mba.std() + 1e-8)
            _, nlp, ent, nv = agent.act(o[mb], a_[mb])
            ratio = torch.exp(nlp - old_lp[mb])
            pg = torch.max(
                -mba * ratio,
                -mba * torch.clamp(ratio, 1 - config.CLIP_RANGE,
                                   1 + config.CLIP_RANGE)).mean()
            vhalf = 0.5 if spec["value_loss_half"] else 1.0
            if spec["clip_value_loss"]:
                v_un = (nv - ret_t[mb]) ** 2
                v_c = torch.clamp(nv - old_v[mb], -config.CLIP_RANGE,
                                  config.CLIP_RANGE) + old_v[mb]
                vf = vhalf * torch.max(v_un, (v_c - ret_t[mb]) ** 2).mean()
            else:
                vf = vhalf * ((nv - ret_t[mb]) ** 2).mean()
            loss = pg - config.ENT_COEF * ent.mean() + config.VF_COEF * vf
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), config.MAX_GRAD_NORM)
            opt.step()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="CartPole-v1", choices=config.ENVS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default="cleanrl", choices=["sb3", "cleanrl"])
    ap.add_argument("--timesteps-scale", type=float, default=1.0)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(1)
    import torch.nn as nn
    import torch.optim as optim
    from specs import get_spec

    spec = get_spec(args.mode)
    variant = ("cleanrl_torch" if args.mode == "cleanrl"
               else "cleanrl_sb3mode_torch")
    variants.ensure_dirs()
    set_all_seeds(args.seed, torch=torch)

    total = variants.timesteps_for(args.env, args.timesteps_scale)
    n_updates = total // config.N_STEPS
    total = n_updates * config.N_STEPS
    log_every = min(config.LOG_INTERVAL, max(total // 50, 1))

    env = make_env(args.env, args.seed)
    obs_dim, act_dim, is_cont = obs_act_dims(env)
    assert not is_cont, "discrete envs only"
    agent = build_agent(nn, obs_dim, act_dim, spec)
    opt = optim.Adam(agent.parameters(), lr=config.LEARNING_RATE,
                     eps=config.ADAM_EPS)
    obs = np.asarray(env.reset(seed=args.seed + 999)[0], dtype=np.float32)
    roll = RollingMean(config.ROLLING_WINDOW)
    curve: list[tuple[int, float]] = []
    gs, last_log = 0, 0
    t0 = time.time()

    for update in range(1, n_updates + 1):
        out = collect_rollout(env, agent, torch, spec, obs,
                              config.N_STEPS, obs_dim)
        b_obs, b_act, b_logp, b_rew, b_done, b_val, obs, last_f, fin = out
        gs += config.N_STEPS
        for r in fin:
            roll.add(r)
        with torch.no_grad():
            lv = float(agent.value(
                torch.as_tensor(obs, dtype=torch.float32)
                .unsqueeze(0)).item())
        last_done = bool(last_f and b_done[-1] > 0.5)
        adv, ret = compute_gae(b_rew, b_val, b_done, lv, last_done,
                               config.GAMMA, config.GAE_LAMBDA)
        batch = (torch.as_tensor(b_obs), torch.as_tensor(b_act),
                 torch.as_tensor(b_logp), torch.as_tensor(b_val),
                 torch.as_tensor(adv), torch.as_tensor(ret))
        ppo_epochs_fullbatch(agent, torch, nn, opt, spec, batch, n_updates,
                             update)
        if gs - last_log >= log_every and len(roll) > 0:
            last_log = gs
            m = roll.mean()
            curve.append((gs, m))
            print(f"[{variant}] {args.env} seed={args.seed} "
                  f"step={gs}/{total} roll20={m:.2f}", flush=True)

    elapsed = time.time() - t0
    from pathlib import Path
    mp = Path(variants.MODELS_DIR) / f"{variant}_{args.env}_seed{args.seed}.pt"
    cp = Path(variants.CURVES_DIR) / f"{variant}_{args.env}_seed{args.seed}.csv"
    mep = Path(variants.META_DIR) / f"{variant}_{args.env}_seed{args.seed}.json"
    torch.save(agent.state_dict(), str(mp))
    write_curve_csv(cp, curve)
    write_meta_json(mep, {
        "variant": variant, "library": "torch-manual", "spec": spec,
        "hyperparams": variants.get_hyperparam_dict(),
        "env": args.env, "seed": args.seed, "total_timesteps": total,
        "elapsed_s": elapsed, "device": "cpu",
        "versions": package_versions(["torch", "gymnasium", "numpy"]),
    })
    print(f"[{variant}] done env={args.env} seed={args.seed} "
          f"steps={total} time={elapsed:.1f}s model={mp}")
    env.close()


if __name__ == "__main__":
    main()



