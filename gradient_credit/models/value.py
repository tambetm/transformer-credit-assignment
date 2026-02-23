"""Value network for PPO's critic.

MLP: obs_dim -> 64 -> 64 -> 1 with Tanh activations.
"""

import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)
