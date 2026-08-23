"""Train PPO on LunarLander-v3 using TorchRL.
"""

import csv
import time
from collections import deque
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule
from torchrl.collectors import Collector
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs import GymEnv
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator, NormalParamWrapper
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from config import (
    BATCH_SIZE, CLIP_RANGE, CURVES_DIR, ENT_COEF, ENV_ID, GAE_LAMBDA, GAMMA,
    LEARNING_RATE, LOG_INTERVAL, MAX_GRAD_NORM, MLFLOW_EXPERIMENT, MLFLOW_URI,
    MODELS_DIR, N_EPOCHS, N_STEPS, NET_ARCH, SEED, TOTAL_TIMESTEPS, VF_COEF,
    get_hyperparam_dict
)

def set_all_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CURVES_DIR.mkdir(parents=True, exist_ok=True)
    
    set_all_seeds(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    
    with mlflow.start_run(run_name="torchrl"):
        params = get_hyperparam_dict()
        params["implementation"] = "torchrl"
        mlflow.log_params(params)
        
        # 1. Environment
        env = GymEnv(ENV_ID, device=device)
        env.set_seed(SEED)
        
        is_continuous = not hasattr(env.action_spec.space, "n")
        act_dim = env.action_spec.shape[0] if is_continuous else env.action_spec.space.n

        if is_continuous:
            actor_net = MLP(
                in_features=env.observation_spec["observation"].shape[0],
                activation_class=nn.Tanh,
                out_features=2 * act_dim,
                num_cells=NET_ARCH,
            )
            actor_module = NormalParamWrapper(actor_net)
            actor_module = TensorDictModule(actor_module, in_keys=["observation"], out_keys=["loc", "scale"])
            actor = ProbabilisticActor(
                module=actor_module,
                in_keys=["loc", "scale"],
                out_keys=["action"],
                distribution_class=torch.distributions.Normal,
                return_log_prob=True,
            ).to(device)
        else:
            actor_net = MLP(
                in_features=env.observation_spec["observation"].shape[0],
                activation_class=nn.Tanh,
                out_features=act_dim,
                num_cells=NET_ARCH,
            )
            actor_module = TensorDictModule(actor_net, in_keys=["observation"], out_keys=["logits"])
            actor = ProbabilisticActor(
                module=actor_module,
                in_keys=["logits"],
                out_keys=["action"],
                distribution_class=torch.distributions.Categorical,
                return_log_prob=True,
            ).to(device)
        
        value_net = MLP(
            in_features=env.observation_spec["observation"].shape[0],
            activation_class=nn.Tanh,
            out_features=1,
            num_cells=NET_ARCH,
        )
        value_module = ValueOperator(module=value_net, in_keys=["observation"]).to(device)
        
        # 3. Collector
        collector = Collector(
            env,
            actor,
            frames_per_batch=N_STEPS,
            total_frames=TOTAL_TIMESTEPS,
            split_trajs=False,
            device=device,
        )
        
        # 4. Replay Buffer for Minibatching
        replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(N_STEPS),
            sampler=SamplerWithoutReplacement(),
            batch_size=BATCH_SIZE,
        )
        
        # 5. Loss & GAE
        loss_module = ClipPPOLoss(
            actor_network=actor,
            critic_network=value_module,
            clip_epsilon=CLIP_RANGE,
            entropy_bonus=bool(ENT_COEF > 0),
            entropy_coeff=ENT_COEF,
            critic_coeff=VF_COEF,
            loss_critic_type="l2"
        )
        loss_module.make_value_estimator(gamma=GAMMA, lmbda=GAE_LAMBDA)
        
        optimizer = torch.optim.Adam(loss_module.parameters(), lr=LEARNING_RATE, eps=1e-5)
        
        episode_rewards = deque(maxlen=20)
        curve_data = []
        global_step = 0
        last_log_step = 0
        
        print(f"[TorchRL] Starting training for {TOTAL_TIMESTEPS} timesteps")
        
        # 6. Training Loop
        for i, data in enumerate(collector):
            global_step += data.numel()
            data = data.to(device)
            
            # Extract episode rewards
            rewards = data[("next", "reward")]
            dones = data[("next", "done")]
            if dones.any():
                ep_ends = dones.nonzero(as_tuple=True)[0]
                start_idx = 0
                for end_idx in ep_ends:
                    ep_reward = rewards[start_idx:end_idx+1].sum().item()
                    episode_rewards.append(ep_reward)
                    start_idx = end_idx + 1
            
            with torch.no_grad():
                loss_module.value_estimator(
                    data,
                    params=loss_module.critic_network_params,
                    target_params=loss_module.target_critic_network_params,
                )
            
            data = data.reshape(-1)
            replay_buffer.extend(data.cpu())
            
            for _ in range(N_EPOCHS):
                for _ in range(N_STEPS // BATCH_SIZE):
                    subdata = replay_buffer.sample().to(device)
                    loss_vals = loss_module(subdata)
                    
                    loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals.get("loss_entropy", 0.0)
                    )
                    
                    optimizer.zero_grad()
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(loss_module.parameters(), MAX_GRAD_NORM)
                    optimizer.step()
            
            if global_step - last_log_step >= LOG_INTERVAL:
                mean_r = np.mean(episode_rewards) if len(episode_rewards) > 0 else 0.0
                curve_data.append((global_step, mean_r))
                mlflow.log_metric("mean_reward", mean_r, step=global_step)
                print(f"[TorchRL] Step {global_step}/{TOTAL_TIMESTEPS} | Mean reward (20 ep): {mean_r:.2f}")
                last_log_step = global_step
                
        # 7. Save Artifacts
        model_path = MODELS_DIR / "torchrl_ppo.pt"
        actor_state = actor_net.state_dict()
        critic_state = value_net.state_dict()
        unified_state = {}
        for k, v in actor_state.items():
            unified_state[f"actor.{k}"] = v
        for k, v in critic_state.items():
            unified_state[f"critic.{k}"] = v
            
        torch.save({"model_state_dict": unified_state}, str(model_path))
        
        csv_path = CURVES_DIR / "torchrl_curve.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "mean_reward"])
            writer.writerows(curve_data)
            
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(csv_path))
        
        print("[TorchRL] Training complete.")

if __name__ == "__main__":
    main()
