"""PPO com Tianshou 2.x. A API mudou bastante na v2 (policy -> algorithm,
classes renomeadas) -- baseado no exemplo oficial test/continuous/test_ppo.py
do repositório, adaptado pra ação discreta (DiscreteActor/DiscreteCritic +
Categorical em vez de Normal)."""
import time
import sys
sys.path.insert(0, "/home/claude/ppo-benchmark/scripts")
from common import ENV_ID, SEED, TOTAL_TIMESTEPS, PPO_CONFIG, save_result, evaluate_policy, save_eval_result

import numpy as np
import torch
import gymnasium as gym

from tianshou.algorithm import PPO
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Collector, CollectStats, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.trainer import OnPolicyTrainerParams
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import DiscreteActor, DiscreteCritic, dist_fn_categorical_from_logits


def main():
    cfg = PPO_CONFIG
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = gym.make(ENV_ID)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    training_envs = DummyVectorEnv([lambda: gym.make(ENV_ID) for _ in range(cfg["n_envs"])])
    training_envs.seed(SEED)

    hidden_sizes = list(cfg["hidden_sizes"])

    net_a = Net(state_shape=obs_dim, hidden_sizes=hidden_sizes)
    actor = DiscreteActor(preprocess_net=net_a, action_shape=act_dim, softmax_output=False)
    net_c = Net(state_shape=obs_dim, hidden_sizes=hidden_sizes)
    critic = DiscreteCritic(preprocess_net=net_c)

    # inicialização ortogonal com os mesmos gains usados nas outras libs
    # (aqui aplicado de forma uniforme por módulo, já que a API não separa
    # facilmente "última camada" do resto sem subclassificar)
    gh = cfg["ortho_gain_hidden"]
    for m in actor.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight, gain=gh)
            torch.nn.init.zeros_(m.bias)
    for m in critic.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight, gain=gh)
            torch.nn.init.zeros_(m.bias)

    optim = AdamOptimizerFactory(lr=cfg["lr"], eps=cfg["adam_eps"])

    policy = ProbabilisticActorPolicy(
        actor=actor,
        dist_fn=dist_fn_categorical_from_logits,
        action_space=env.action_space,
        action_scaling=False,
        action_bound_method=None,
    )

    algorithm = PPO(
        policy=policy,
        critic=critic,
        optim=optim,
        gamma=cfg["gamma"],
        max_grad_norm=cfg["max_grad_norm"],
        eps_clip=cfg["clip_coef"],
        vf_coef=cfg["vf_coef"],
        ent_coef=cfg["ent_coef"],
        gae_lambda=cfg["gae_lambda"],
        value_clip=False,
        advantage_normalization=True,
        return_scaling=False,
    )

    batch_total = cfg["n_steps"] * cfg["n_envs"]
    minibatch_size = batch_total // cfg["minibatches"]

    training_collector = Collector[CollectStats](
        algorithm,
        training_envs,
        VectorReplayBuffer(batch_total * 4, len(training_envs)),
    )

    t0 = time.time()
    result = algorithm.run_training(
        OnPolicyTrainerParams(
            training_collector=training_collector,
            max_epochs=1,
            epoch_num_steps=TOTAL_TIMESTEPS,
            collection_step_num_env_steps=batch_total,
            batch_size=minibatch_size,
            update_step_num_repetitions=cfg["n_epochs"],
            test_in_training=False,
            show_progress=False,
            verbose=False,
        )
    )
    elapsed = time.time() - t0

    # Tianshou não expõe facilmente uma curva de reward no mesmo formato das
    # outras libs sem um test_collector -- registramos só o resultado final
    save_result("tianshou", elapsed, [])

    def act_fn(obs):
        with torch.no_grad():
            obs_t = torch.as_tensor(np.array(obs), dtype=torch.float32).unsqueeze(0)
            logits, _ = actor(obs_t)
            dist = dist_fn_categorical_from_logits(logits)
            action = dist.sample()
        return int(action.item())

    eval_returns = evaluate_policy(ENV_ID, act_fn)
    save_eval_result("tianshou", elapsed, eval_returns)


if __name__ == "__main__":
    main()
