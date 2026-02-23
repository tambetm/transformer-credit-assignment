"""Trajectory buffer for collecting and storing episode data."""

from dataclasses import dataclass, field
import torch
import numpy as np


@dataclass
class Episode:
    """A single episode trajectory."""
    observations: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    values: list = field(default_factory=list)  # For PPO
    dones: list = field(default_factory=list)

    def add(self, obs, action, reward, log_prob, value=None, done=False):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        if value is not None:
            self.values.append(value)
        self.dones.append(done)

    @property
    def total_return(self) -> float:
        return sum(self.rewards)

    @property
    def length(self) -> int:
        return len(self.observations)

    def discounted_return(self, gamma: float) -> float:
        G = 0.0
        for r in reversed(self.rewards):
            G = r + gamma * G
        return G

    def to_tensors(self, device="cpu"):
        """Convert episode data to PyTorch tensors."""
        return {
            "observations": torch.tensor(np.array(self.observations), dtype=torch.float32, device=device),
            "actions": torch.tensor(np.array(self.actions), dtype=torch.long, device=device),
            "rewards": torch.tensor(np.array(self.rewards), dtype=torch.float32, device=device),
            "log_probs": torch.tensor(np.array(self.log_probs), dtype=torch.float32, device=device),
            "values": torch.tensor(np.array(self.values), dtype=torch.float32, device=device) if self.values else None,
            "dones": torch.tensor(np.array(self.dones), dtype=torch.float32, device=device),
        }


class TrajectoryBuffer:
    """Collects multiple episodes for batch updates."""

    def __init__(self):
        self.episodes = []

    def add_episode(self, episode: Episode):
        self.episodes.append(episode)

    @property
    def total_timesteps(self) -> int:
        return sum(ep.length for ep in self.episodes)

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    def get_returns(self, gamma: float = 0.99):
        """Get discounted returns for all episodes."""
        return [ep.discounted_return(gamma) for ep in self.episodes]

    def clear(self):
        self.episodes = []

    def to_batch(self, device="cpu"):
        """Flatten all episodes into a single batch of transitions."""
        all_obs, all_act, all_rew, all_logp, all_val, all_done = [], [], [], [], [], []
        ep_starts = [0]

        for ep in self.episodes:
            tensors = ep.to_tensors(device)
            all_obs.append(tensors["observations"])
            all_act.append(tensors["actions"])
            all_rew.append(tensors["rewards"])
            all_logp.append(tensors["log_probs"])
            if tensors["values"] is not None:
                all_val.append(tensors["values"])
            all_done.append(tensors["dones"])
            ep_starts.append(ep_starts[-1] + ep.length)

        batch = {
            "observations": torch.cat(all_obs),
            "actions": torch.cat(all_act),
            "rewards": torch.cat(all_rew),
            "log_probs": torch.cat(all_logp),
            "dones": torch.cat(all_done),
            "ep_starts": ep_starts,
        }
        if all_val:
            batch["values"] = torch.cat(all_val)

        return batch
