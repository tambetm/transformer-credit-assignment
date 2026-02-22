"""Proximal Policy Optimization (PPO) with GAE.

Standard PPO implementation with:
- Learned value function (critic) for advantage estimation
- Generalized Advantage Estimation (GAE)
- Clipped surrogate objective
- Same policy network architecture as other algorithms
"""

import torch
import numpy as np
import gymnasium as gym

from ..models.policy import PolicyNetwork
from ..models.value import ValueNetwork
from ..utils.buffer import Episode, TrajectoryBuffer
from ..utils.logger import CSVLogger


class PPO:
    """PPO with GAE and clipped surrogate objective."""

    def __init__(self, env_fn, config: dict, seed: int = 0, log_dir: str = "logs"):
        self.config = config
        self.seed = seed
        self.device = config.get("device", "cpu")

        # Create environment
        self.env = env_fn()
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n

        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create networks
        self.policy = PolicyNetwork(obs_dim, action_dim).to(self.device)
        self.value_net = ValueNetwork(obs_dim).to(self.device)

        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config["lr_policy"]
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=config["lr_value"]
        )

        # Logger
        self.logger = CSVLogger(log_dir)

        # Tracking
        self.total_timesteps = 0
        self.total_episodes = 0

    def collect_episodes(self, min_timesteps: int) -> TrajectoryBuffer:
        """Collect episodes until we have at least min_timesteps transitions."""
        buffer = TrajectoryBuffer()

        while buffer.total_timesteps < min_timesteps:
            obs, _ = self.env.reset(seed=self.seed + self.total_episodes)
            episode = Episode()
            done = False

            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    action, log_prob = self.policy.get_action(obs_tensor)
                    value = self.value_net(obs_tensor)

                next_obs, reward, terminated, truncated, info = self.env.step(
                    action.item()
                )
                done = terminated or truncated

                episode.add(
                    obs=obs,
                    action=action.item(),
                    reward=reward,
                    log_prob=log_prob.item(),
                    value=value.item(),
                    done=done,
                )
                obs = next_obs

            buffer.add_episode(episode)
            self.total_timesteps += episode.length
            self.total_episodes += 1

        return buffer

    def compute_gae(self, buffer: TrajectoryBuffer):
        """Compute GAE advantages and returns-to-go for all episodes."""
        gamma = self.config["gamma"]
        lam = self.config["ppo_gae_lambda"]

        all_advantages = []
        all_returns = []

        for episode in buffer.episodes:
            tensors = episode.to_tensors(self.device)
            rewards = tensors["rewards"]
            values = tensors["values"]
            T = episode.length

            advantages = torch.zeros(T, device=self.device)
            last_gae = 0.0

            for t in reversed(range(T)):
                if t == T - 1:
                    next_value = 0.0  # Terminal
                else:
                    next_value = values[t + 1]

                delta = rewards[t] + gamma * next_value - values[t]
                advantages[t] = last_gae = delta + gamma * lam * last_gae

            returns = advantages + values
            all_advantages.append(advantages)
            all_returns.append(returns)

        return torch.cat(all_advantages), torch.cat(all_returns)

    def update(self, buffer: TrajectoryBuffer):
        """Perform PPO update with multiple epochs."""
        # Get batch data
        batch = buffer.to_batch(self.device)
        advantages, returns = self.compute_gae(buffer)

        obs = batch["observations"]
        actions = batch["actions"]
        old_log_probs = batch["log_probs"]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        clip_eps = self.config["ppo_clip"]
        n_epochs = self.config["ppo_epochs"]
        batch_size = len(obs)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for epoch in range(n_epochs):
            # Shuffle indices for mini-batching
            indices = torch.randperm(batch_size, device=self.device)
            mini_batch_size = min(256, batch_size)

            for start in range(0, batch_size, mini_batch_size):
                end = min(start + mini_batch_size, batch_size)
                idx = indices[start:end]

                # Policy loss
                new_log_probs, entropy = self.policy.evaluate_actions(
                    obs[idx], actions[idx]
                )
                ratio = torch.exp(new_log_probs - old_log_probs[idx])

                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages[idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                new_values = self.value_net(obs[idx])
                value_loss = 0.5 * (new_values - returns[idx]).pow(2).mean()

                # Entropy bonus
                entropy_loss = -entropy.mean() * 0.01

                # Combined loss for policy
                loss = policy_loss + entropy_loss
                self.policy_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.policy_optimizer.step()

                # Separate value update
                self.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
                self.value_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()

        n_updates = n_epochs * ((batch_size + mini_batch_size - 1) // mini_batch_size)
        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
        }

    def train(self):
        """Main training loop."""
        total_target = self.config["total_timesteps"]
        batch_size = self.config["batch_size"]
        eval_interval = 5

        update_count = 0
        recent_returns = []

        while self.total_timesteps < total_target:
            # Collect at least batch_size timesteps
            buffer = self.collect_episodes(batch_size)

            # Track returns
            for ep in buffer.episodes:
                recent_returns.append(ep.total_return)

            # Update
            stats = self.update(buffer)

            update_count += 1

            if update_count % eval_interval == 0:
                mean_return = np.mean(recent_returns[-100:])
                print(
                    f"[PPO]       Steps: {self.total_timesteps:>7d} | "
                    f"Episodes: {self.total_episodes:>5d} | "
                    f"Mean Return (100ep): {mean_return:.1f} | "
                    f"P.Loss: {stats['policy_loss']:.4f} | "
                    f"V.Loss: {stats['value_loss']:.4f}"
                )
                self.logger.log({
                    "timesteps": self.total_timesteps,
                    "episodes": self.total_episodes,
                    "mean_return": mean_return,
                    "policy_loss": stats["policy_loss"],
                    "value_loss": stats["value_loss"],
                    "entropy": stats["entropy"],
                })

        self.logger.close()
        self.env.close()
        return recent_returns
