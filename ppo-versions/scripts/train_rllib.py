import time
import sys
sys.path.insert(0, "/home/claude/ppo-benchmark/scripts")
from common import ENV_ID, SEED, TOTAL_TIMESTEPS, PPO_CONFIG, save_result, evaluate_policy, save_eval_result

import ray
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig


def main():
    cfg = PPO_CONFIG
    ray.init(ignore_reinit_error=True, log_to_driver=False, include_dashboard=False)

    train_batch_size = cfg["n_steps"] * cfg["n_envs"]
    minibatch_size = train_batch_size // cfg["minibatches"]

    model_config = DefaultModelConfig(
        fcnet_hiddens=list(cfg["hidden_sizes"]),
        fcnet_activation=cfg["activation"],
        vf_share_layers=False,  # default do RLlib é True; as outras 3 libs usam redes separadas
        fcnet_kernel_initializer="orthogonal_",
        fcnet_kernel_initializer_kwargs={"gain": cfg["ortho_gain_hidden"]},
        # nota: RLlib não expõe gain diferente pro layer de saída via essa API,
        # então a última camada fica com o mesmo gain das hidden (gap documentado no README)
    )

    config = (
        PPOConfig()
        .environment(ENV_ID)
        .env_runners(num_env_runners=0, num_envs_per_env_runner=cfg["n_envs"])
        .rl_module(model_config=model_config)
        .training(
            gamma=cfg["gamma"],
            lr=cfg["lr"],
            train_batch_size=train_batch_size,
            minibatch_size=minibatch_size,
            num_epochs=cfg["n_epochs"],
            lambda_=cfg["gae_lambda"],
            clip_param=cfg["clip_coef"],
            entropy_coeff=cfg["ent_coef"],
            vf_loss_coeff=cfg["vf_coef"],
            grad_clip=cfg["max_grad_norm"],
        )
        .debugging(seed=SEED)
        .framework("torch")
    )

    algo = config.build()

    reward_history = []
    global_step = 0
    t0 = time.time()

    while global_step < TOTAL_TIMESTEPS:
        result = algo.train()
        global_step = result["num_env_steps_sampled_lifetime"]
        mean_r = result.get("env_runners", {}).get("episode_return_mean")
        if mean_r is not None:
            reward_history.append((global_step, mean_r))

    elapsed = time.time() - t0
    save_result("rllib", elapsed, reward_history)

    module = algo.get_module()
    dist_cls = module.get_inference_action_dist_cls()

    def act_fn(obs):
        batch = {"obs": torch.tensor(obs, dtype=torch.float32).unsqueeze(0)}
        out = module.forward_inference(batch)
        dist = dist_cls.from_logits(out["action_dist_inputs"])
        action = dist.sample()
        return int(action.item())

    eval_returns = evaluate_policy(ENV_ID, act_fn)
    save_eval_result("rllib", elapsed, eval_returns)

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
