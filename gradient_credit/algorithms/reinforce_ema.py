"""REINFORCE with EMA Baseline — uniform credit assignment."""

import torch
import numpy as np

from ..models.policy import PolicyNetwork
from ..utils.ema import EMAStats
from ..utils.buffer import Episode, TrajectoryBuffer
from ..utils.logger import CSVLogger


class ReinforceEMA:
    """REINFORCE with EMA baseline normalization."""

    def __init__(self, env_fn, config: dict, seed: int = 0, log_dir: str = "logs"):
        self.config = config
        self.seed = seed
        self.device = config.get("device", "cpu")

        self.env = env_fn()
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.policy = PolicyNetwork(obs_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config["lr_policy"]
        )
        self.ema = EMAStats(alpha=config["ema_alpha"])
        self.logger = CSVLogger(log_dir)

        self.total_timesteps = 0
        self.total_episodes = 0

    def collect_episodes(self, num_episodes: int) -> TrajectoryBuffer:
        buffer = TrajectoryBuffer()
        for _ in range(num_episodes):
            obs, _ = self.env.reset(seed=self.seed + self.total_episodes)
            episode = Episode()
            done = False
            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    action, log_prob = self.policy.get_action(obs_tensor)
                next_obs, reward, terminated, truncated, info = self.env.step(action.item())
                done = terminated or truncated
                episode.add(obs=obs, action=action.item(), reward=reward,
                            log_prob=log_prob.item(), done=done)
                obs = next_obs
            buffer.add_episode(episode)
            self.total_timesteps += episode.length
            self.total_episodes += 1
        return buffer

    def update(self, buffer: TrajectoryBuffer):
        gamma = self.config["gamma"]
        all_obs, all_actions, all_advantages = [], [], []

        for episode in buffer.episodes:
            G = episode.discounted_return(gamma)
            advantage = self.ema.normalize(G)
            self.ema.update(G)
            tensors = episode.to_tensors(self.device)
            ep_advantages = torch.full(
                (episode.length,), advantage, dtype=torch.float32, device=self.device
            )
            all_obs.append(tensors["observations"])
            all_actions.append(tensors["actions"])
            all_advantages.append(ep_advantages)

        obs_batch = torch.cat(all_obs)
        action_batch = torch.cat(all_actions)
        advantage_batch = torch.cat(all_advantages)

        if len(advantage_batch) > 1:
            advantage_batch = (advantage_batch - advantage_batch.mean()) / (
                advantage_batch.std() + 1e-8
            )

        log_probs, entropy = self.policy.evaluate_actions(obs_batch, action_batch)
        policy_loss = -(log_probs * advantage_batch).mean()
        entropy_loss = -entropy.mean() * 0.01
        loss = policy_loss + entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()

        return {"policy_loss": policy_loss.item(), "entropy": entropy.mean().item()}

    def train(self):
        total_target = self.config["total_timesteps"]
        episodes_per_update = self.config["episodes_per_update"]
        eval_interval = 10
        update_count = 0
        recent_returns = []

        while self.total_timesteps < total_target:
            buffer = self.collect_episodes(episodes_per_update)
            for ep in buffer.episodes:
                recent_returns.append(ep.total_return)
            stats = self.update(buffer)
            update_count += 1

            if update_count % eval_interval == 0:
                mean_return = np.mean(recent_returns[-100:])
                print(
                    f"[REINFORCE] Steps: {self.total_timesteps:>7d} | "
                    f"Episodes: {self.total_episodes:>5d} | "
                    f"Mean Return (100ep): {mean_return:.1f} | "
                    f"Policy Loss: {stats['policy_loss']:.4f}"
                )
                self.logger.log({
                    "timesteps": self.total_timesteps,
                    "episodes": self.total_episodes,
                    "mean_return": mean_return,
                    "policy_loss": stats["policy_loss"],
                    "entropy": stats["entropy"],
                })

        self.logger.close()
        self.env.close()
        return recent_returns
