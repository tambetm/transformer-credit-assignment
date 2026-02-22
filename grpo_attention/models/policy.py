"""Shared policy network used by all algorithms.

MLP: obs_dim -> 64 -> 64 -> action_dim with Tanh activations.
Outputs a Categorical distribution.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> Categorical:
        logits = self.net(obs)
        return Categorical(logits=logits)

    def get_action(self, obs: torch.Tensor):
        """Sample an action and return (action, log_prob)."""
        dist = self.forward(obs)
        action = dist.sample()
        return action, dist.log_prob(action)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """Return log_prob and entropy for given obs-action pairs."""
        dist = self.forward(obs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy
