from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

OBS_SIZE = 15
ACTION_SIZE = 8


class PolicyNetwork(nn.Module):
    def __init__(self, obs_size: int = OBS_SIZE, action_size: int = ACTION_SIZE, hidden_size: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_size),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def flatten_parameters(model: nn.Module) -> torch.Tensor:
    vectors = [param.data.view(-1) for param in model.parameters()]
    return torch.cat(vectors).detach().clone()


def set_parameters_from_flat(model: nn.Module, flat_vector: torch.Tensor) -> None:
    pointer = 0
    for param in model.parameters():
        numel = param.numel()
        chunk = flat_vector[pointer : pointer + numel].view_as(param)
        param.data.copy_(chunk)
        pointer += numel


def infer_action(model: nn.Module, observation: np.ndarray) -> np.ndarray:
    obs_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        action = model(obs_tensor).squeeze(0).cpu().numpy()
    return action
