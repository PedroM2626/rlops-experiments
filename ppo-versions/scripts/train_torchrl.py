"""PPO with TorchRL, following the official tutorial (coding_ppo.html) adapted
for discrete actions -- TorchRL uses a OneHot spec by default for gym.Discrete,
so the right pattern is ProbabilisticActor + OneHotCategorical (not a
plain Categorical), as in the DataCamp/official documentation example."""
import time
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ENV_ID, SEED, TOTAL_TIMESTEPS, PPO_CONFIG, save_result, evaluate_policy, save_eval_result

import numpy as np
import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule

from torchrl.collectors import Collector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import OneHotCategorical, ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE


def layer_init(layer, gain):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


def main():
    cfg = PPO_CONFIG
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    base_env = GymEnv(ENV_ID)
    env = TransformedEnv(base_env, RewardSum())
    env.set_seed(SEED)

    obs_dim = env.observation_spec["observation"].shape[-1]
    act_dim = env.action_spec.shape[-1]  # OneHot -> shape[-1] = n_actions
    h = cfg["hidden_sizes"][0]
    gh, ga, gc = cfg["ortho_gain_hidden"], cfg["ortho_gain_actor_out"], cfg["ortho_gain_critic_out"]

    actor_net = nn.Sequential(
        layer_init(nn.Linear(obs_dim, h), gh), nn.Tanh(),
        layer_init(nn.Linear(h, h), gh), nn.Tanh(),
        layer_init(nn.Linear(h, act_dim), ga),
    )
    actor_module = TensorDictModule(actor_net, in_keys=["observation"], out_keys=["logits"])
    actor = ProbabilisticActor(
        module=actor_module,
        in_keys=["logits"],
        out_keys=["action"],
        spec=env.action_spec,
        distribution_class=OneHotCategorical,
        return_log_prob=True,
    )

    critic_net = nn.Sequential(
        layer_init(nn.Linear(obs_dim, h), gh), nn.Tanh(),
        layer_init(nn.Linear(h, h), gh), nn.Tanh(),
        layer_init(nn.Linear(h, 1), gc),
    )
    critic = ValueOperator(module=critic_net, in_keys=["observation"])

    advantage_module = GAE(
        gamma=cfg["gamma"], lmbda=cfg["gae_lambda"], value_network=critic, average_gae=True
    )
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=cfg["clip_coef"],
        entropy_bonus=True,
        entropy_coeff=cfg["ent_coef"],
        critic_coeff=cfg["vf_coef"],
        normalize_advantage=True,
    )
    optim = torch.optim.Adam(loss_module.parameters(), lr=cfg["lr"], eps=cfg["adam_eps"])

    batch_total = cfg["n_steps"] * cfg["n_envs"]
    minibatch_size = batch_total // cfg["minibatches"]

    collector = Collector(
        env, actor, frames_per_batch=batch_total, total_frames=TOTAL_TIMESTEPS, split_trajs=False,
        auto_register_policy_transforms=True,
    )
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(batch_total), sampler=SamplerWithoutReplacement(),
    )

    reward_history = []
    global_step = 0
    t0 = time.time()

    for tensordict_data in collector:
        global_step += tensordict_data.numel()

        with torch.no_grad():
            advantage_module(tensordict_data)
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view)

        for _ in range(cfg["n_epochs"]):
            for _ in range(batch_total // minibatch_size):
                subdata = replay_buffer.sample(minibatch_size)
                loss_vals = loss_module(subdata)
                loss_value = loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), cfg["max_grad_norm"])
                optim.step()
                optim.zero_grad()

        # episodes that ended in this batch (RewardSum accumulates into "episode_reward")
        done_mask = tensordict_data["next", "done"].squeeze(-1)
        if done_mask.any():
            ep_rewards = tensordict_data["next", "episode_reward"][done_mask].squeeze(-1)
            for r in ep_rewards.tolist():
                reward_history.append((global_step, r))

    elapsed = time.time() - t0
    collector.shutdown()

    smoothed = []
    if reward_history:
        window = []
        for ts, r in reward_history:
            window.append(r)
            if len(window) > 20:
                window.pop(0)
            smoothed.append((ts, float(np.mean(window))))

    save_result("torchrl", elapsed, smoothed)

    def act_fn(obs):
        with torch.no_grad():
            logits = actor_net(torch.as_tensor(np.array(obs), dtype=torch.float32).unsqueeze(0))
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
        return int(action.item())

    eval_returns = evaluate_policy(ENV_ID, act_fn)
    save_eval_result("torchrl", elapsed, eval_returns)


if __name__ == "__main__":
    main()
