"""GRPO + Attention-Based Credit Assignment.

Algorithm:
1. Collect episode trajectory
2. Compute A_episode via EMA normalization (same as REINFORCE)
3. Feed state sequence into Credit Assignment Transformer (CAT):
   - Embed states, append [RETURN] token
   - Pass through transformer encoder
   - [RETURN] token predicts episode return (MSE loss)
   - Attention weights from [RETURN] to states = credit assignment weights
4. Per-timestep advantage: A_t = w_t * T * A_episode
5. Policy gradient update using per-timestep advantages
6. Update CAT via return prediction MSE loss
7. Update EMA statistics
"""

import torch
import torch.nn.functional as F
import numpy as np
import gymnasium as gym

from ..models.policy import PolicyNetwork
from ..models.credit_transformer import CreditAssignmentTransformer
from ..utils.ema import EMAStats
from ..utils.buffer import Episode, TrajectoryBuffer
from ..utils.logger import CSVLogger


class GRPOAttention:
    """GRPO with attention-based credit assignment."""

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

        # Create policy
        self.policy = PolicyNetwork(obs_dim, action_dim).to(self.device)
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config["lr_policy"]
        )

        # Create Credit Assignment Transformer
        self.cat = CreditAssignmentTransformer(obs_dim).to(self.device)
        self.cat_optimizer = torch.optim.Adam(
            self.cat.parameters(), lr=config["lr_cat"]
        )

        # EMA baseline
        self.ema = EMAStats(alpha=config["ema_alpha"])

        # Logger
        self.logger = CSVLogger(log_dir)

        # Storage for attention visualization
        self.attention_history = []

        # Tracking
        self.total_timesteps = 0
        self.total_episodes = 0

    def collect_episodes(self, num_episodes: int) -> TrajectoryBuffer:
        """Collect a batch of episodes."""
        buffer = TrajectoryBuffer()

        for _ in range(num_episodes):
            obs, _ = self.env.reset(seed=self.seed + self.total_episodes)
            episode = Episode()
            done = False

            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    action, log_prob = self.policy.get_action(obs_tensor)

                next_obs, reward, terminated, truncated, info = self.env.step(
                    action.item()
                )
                done = terminated or truncated

                episode.add(
                    obs=obs,
                    action=action.item(),
                    reward=reward,
                    log_prob=log_prob.item(),
                    done=done,
                )
                obs = next_obs

            buffer.add_episode(episode)
            self.total_timesteps += episode.length
            self.total_episodes += 1

        return buffer

    def _pad_episodes(self, buffer: TrajectoryBuffer):
        """Pad episode observations to same length for batched CAT forward pass."""
        max_len = max(ep.length for ep in buffer.episodes)
        batch_size = buffer.num_episodes

        obs_dim = buffer.episodes[0].observations[0].shape[0] if hasattr(
            buffer.episodes[0].observations[0], 'shape'
        ) else len(buffer.episodes[0].observations[0])

        padded_obs = torch.zeros(batch_size, max_len, obs_dim, device=self.device)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=self.device)

        for i, ep in enumerate(buffer.episodes):
            T = ep.length
            padded_obs[i, :T] = torch.tensor(
                np.array(ep.observations), dtype=torch.float32, device=self.device
            )
            mask[i, :T] = True

        return padded_obs, mask, max_len

    def update(self, buffer: TrajectoryBuffer):
        """Perform policy and CAT update."""
        gamma = self.config["gamma"]

        # Compute episode returns and EMA advantages
        ep_returns = []
        ep_advantages = []
        for episode in buffer.episodes:
            G = episode.discounted_return(gamma)
            advantage = self.ema.normalize(G)
            self.ema.update(G)
            ep_returns.append(G)
            ep_advantages.append(advantage)

        # Pad episodes for CAT
        padded_obs, mask, max_len = self._pad_episodes(buffer)

        # Forward pass through CAT
        predicted_returns, credit_weights = self.cat(padded_obs, mask)

        # CAT loss: MSE on return prediction
        target_returns = torch.tensor(ep_returns, dtype=torch.float32, device=self.device)
        cat_loss = F.mse_loss(predicted_returns, target_returns)

        # Update CAT
        self.cat_optimizer.zero_grad()
        cat_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.cat.parameters(), 1.0)
        self.cat_optimizer.step()

        # Detach credit weights for policy update (no gradient flow from CAT to policy)
        credit_weights = credit_weights.detach()

        # Build per-timestep advantages using credit weights
        all_obs = []
        all_actions = []
        all_advantages = []

        for i, episode in enumerate(buffer.episodes):
            T = episode.length
            tensors = episode.to_tensors(self.device)

            # Per-timestep advantage: w_t * T * A_episode
            w = credit_weights[i, :T]  # (T,)
            per_step_adv = w * T * ep_advantages[i]

            all_obs.append(tensors["observations"])
            all_actions.append(tensors["actions"])
            all_advantages.append(per_step_adv)

        # Concatenate
        obs_batch = torch.cat(all_obs)
        action_batch = torch.cat(all_actions)
        advantage_batch = torch.cat(all_advantages)

        # Normalize advantages across the batch
        if len(advantage_batch) > 1:
            advantage_batch = (advantage_batch - advantage_batch.mean()) / (
                advantage_batch.std() + 1e-8
            )

        # Policy gradient
        log_probs, entropy = self.policy.evaluate_actions(obs_batch, action_batch)
        policy_loss = -(log_probs * advantage_batch).mean()
        entropy_loss = -entropy.mean() * 0.01

        loss = policy_loss + entropy_loss

        self.policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.policy_optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "cat_loss": cat_loss.item(),
            "entropy": entropy.mean().item(),
            "mean_credit_weight_std": credit_weights[:, :max_len][mask].std().item()
            if mask.any() else 0.0,
        }

    def save_attention_snapshot(self, buffer: TrajectoryBuffer, label: str):
        """Save attention weights for visualization."""
        padded_obs, mask, _ = self._pad_episodes(buffer)
        with torch.no_grad():
            _, credit_weights = self.cat(padded_obs, mask)

        for i, ep in enumerate(buffer.episodes):
            T = ep.length
            self.attention_history.append({
                "label": label,
                "episode": self.total_episodes,
                "timesteps": list(range(T)),
                "attention_weights": credit_weights[i, :T].cpu().numpy().tolist(),
                "rewards": ep.rewards[:T],
                "total_return": ep.total_return,
            })

    def train(self):
        """Main training loop."""
        total_target = self.config["total_timesteps"]
        episodes_per_update = self.config["episodes_per_update"]
        eval_interval = 10

        update_count = 0
        recent_returns = []
        snapshot_points = set()

        # Take attention snapshots at 10%, 50%, 90% of training
        for frac in [0.1, 0.5, 0.9]:
            snapshot_points.add(int(total_target * frac))

        while self.total_timesteps < total_target:
            # Collect episodes
            buffer = self.collect_episodes(episodes_per_update)

            # Track returns
            for ep in buffer.episodes:
                recent_returns.append(ep.total_return)

            # Check if we should save attention snapshot
            for sp in list(snapshot_points):
                if self.total_timesteps >= sp:
                    label = f"step_{sp}"
                    self.save_attention_snapshot(buffer, label)
                    snapshot_points.discard(sp)

            # Update
            stats = self.update(buffer)

            update_count += 1

            if update_count % eval_interval == 0:
                mean_return = np.mean(recent_returns[-100:])
                print(
                    f"[GRPO+Attn] Steps: {self.total_timesteps:>7d} | "
                    f"Episodes: {self.total_episodes:>5d} | "
                    f"Mean Return (100ep): {mean_return:.1f} | "
                    f"P.Loss: {stats['policy_loss']:.4f} | "
                    f"CAT Loss: {stats['cat_loss']:.4f}"
                )
                self.logger.log({
                    "timesteps": self.total_timesteps,
                    "episodes": self.total_episodes,
                    "mean_return": mean_return,
                    "policy_loss": stats["policy_loss"],
                    "cat_loss": stats["cat_loss"],
                    "entropy": stats["entropy"],
                    "credit_weight_std": stats["mean_credit_weight_std"],
                })

        self.logger.close()
        self.env.close()
        return recent_returns
