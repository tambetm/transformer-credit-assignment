"""GRPO + Attention-Based Credit Assignment.

Algorithm:
1. Collect episode trajectory
2. Compute A_episode via EMA normalization (same as REINFORCE)
3. Feed state-action sequence into Credit Assignment Transformer (CAT):
   - Embed state-action pairs, append [RETURN] token
   - Pass through transformer encoder
   - Dedicated cross-attention credit head: [RETURN] queries state-action keys
   - [RETURN] predicts episode return (MSE loss)
4. Blend CAT credit weights with uniform weights (warmup schedule)
5. Per-timestep advantage: A_t = w_blended_t * T * A_episode
6. Policy gradient update
7. Update CAT via return prediction MSE loss
8. Update EMA statistics

Key design choices:
- Actions included in CAT input (credit is about which actions mattered)
- Dedicated credit head with learnable temperature
- Warmup: starts as REINFORCE (uniform weights), gradually shifts to CAT weights
- Normalized return targets for stable CAT training
- Small replay buffer for CAT (prevents catastrophic forgetting of early patterns)
"""

import torch
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import deque

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

        # Create Credit Assignment Transformer (with action embeddings)
        self.cat = CreditAssignmentTransformer(
            obs_dim, action_dim=action_dim, d_model=32, nhead=2,
            num_layers=2, d_ff=64,
        ).to(self.device)
        self.cat_optimizer = torch.optim.Adam(
            self.cat.parameters(), lr=config["lr_cat"]
        )

        # EMA baseline
        self.ema = EMAStats(alpha=config["ema_alpha"])

        # Small replay buffer for CAT training stability
        self.cat_replay = deque(maxlen=100)

        # Logger
        self.logger = CSVLogger(log_dir)

        # Storage for attention visualization
        self.attention_history = []

        # Tracking
        self.total_timesteps = 0
        self.total_episodes = 0

    def _cat_mix_ratio(self) -> float:
        """Compute the mixing ratio for CAT weights vs uniform weights.

        Linear warmup over the first 20% of training:
        - At start: 0.0 (pure uniform = REINFORCE)
        - At 20% training: 0.7 (mostly CAT weights)
        - Stays at 0.7 for rest of training
        """
        total = self.config["total_timesteps"]
        warmup_end = total * 0.2
        max_ratio = 0.7
        if self.total_timesteps < warmup_end:
            return max_ratio * (self.total_timesteps / warmup_end)
        return max_ratio

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

    def _pad_episodes_with_actions(self, episodes):
        """Pad episode observations and actions to same length."""
        max_len = max(ep.length for ep in episodes)
        batch_size = len(episodes)

        obs_dim = episodes[0].observations[0].shape[0] if hasattr(
            episodes[0].observations[0], 'shape'
        ) else len(episodes[0].observations[0])

        padded_obs = torch.zeros(batch_size, max_len, obs_dim, device=self.device)
        padded_act = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=self.device)

        for i, ep in enumerate(episodes):
            T = ep.length
            padded_obs[i, :T] = torch.tensor(
                np.array(ep.observations), dtype=torch.float32, device=self.device
            )
            padded_act[i, :T] = torch.tensor(
                np.array(ep.actions), dtype=torch.long, device=self.device
            )
            mask[i, :T] = True

        return padded_obs, padded_act, mask, max_len

    def _update_cat(self, episodes, returns):
        """Update CAT with return prediction MSE loss."""
        padded_obs, padded_act, mask, max_len = self._pad_episodes_with_actions(episodes)
        target_returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Update running return statistics and normalize targets
        self.cat.update_return_stats(target_returns)
        normalized_targets = self.cat.normalize_returns(target_returns)

        # Forward pass
        predicted_returns, credit_weights = self.cat(padded_obs, mask, padded_act)
        predicted_normalized = self.cat.normalize_returns(predicted_returns)

        # MSE loss on normalized returns — this is the only CAT objective
        # The attention patterns emerge naturally from learning to predict returns
        cat_loss = F.mse_loss(predicted_normalized, normalized_targets)

        self.cat_optimizer.zero_grad()
        cat_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.cat.parameters(), 1.0)
        self.cat_optimizer.step()

        # Compute attention entropy for logging
        eps = 1e-8
        cw = credit_weights[:, :mask.shape[1]].detach()
        log_weights = torch.log(cw + eps)
        entropy_per_pos = -(cw * log_weights) * mask.float()
        attn_entropy = entropy_per_pos.sum(dim=-1).mean().item()

        return {
            "cat_mse": cat_loss.item(),
            "cat_entropy": attn_entropy,
            "cat_temperature": (F.softplus(self.cat.temperature) + 0.1).item(),
        }

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

        # Add to replay buffer
        for ep, ret in zip(buffer.episodes, ep_returns):
            self.cat_replay.append((ep, ret))

        # Train CAT: 1 epoch on current batch + 1 epoch on replay sample
        # Current batch (most relevant data)
        cat_stats = self._update_cat(buffer.episodes, ep_returns)

        # One additional pass on replay for stability
        if len(self.cat_replay) >= 20:
            replay_size = min(len(self.cat_replay), 30)
            indices = np.random.choice(len(self.cat_replay), replay_size, replace=False)
            replay_eps = [self.cat_replay[i][0] for i in indices]
            replay_rets = [self.cat_replay[i][1] for i in indices]
            cat_stats = self._update_cat(replay_eps, replay_rets)

        # Get credit weights for current batch (detached, no grad to policy)
        padded_obs, padded_act, mask, max_len = self._pad_episodes_with_actions(
            buffer.episodes
        )
        with torch.no_grad():
            _, cat_weights = self.cat(padded_obs, mask, padded_act)

        # Blend CAT weights with uniform weights using warmup schedule
        mix = self._cat_mix_ratio()

        # Build per-timestep advantages
        all_obs = []
        all_actions = []
        all_advantages = []

        for i, episode in enumerate(buffer.episodes):
            T = episode.length
            tensors = episode.to_tensors(self.device)

            # Uniform weights: 1/T for each valid step
            uniform_w = torch.ones(T, device=self.device) / T

            # CAT weights for this episode
            w_cat = cat_weights[i, :T]

            # Blended weights: (1-mix)*uniform + mix*CAT
            w = (1.0 - mix) * uniform_w + mix * w_cat

            # Per-timestep advantage: w_t * T * A_episode
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
            "cat_loss": cat_stats.get("cat_mse", 0.0),
            "entropy": entropy.mean().item(),
            "mean_credit_weight_std": cat_weights[:, :max_len][mask].std().item()
            if mask.any() else 0.0,
            "cat_entropy": cat_stats.get("cat_entropy", 0.0),
            "cat_temperature": cat_stats.get("cat_temperature", 1.0),
            "mix_ratio": mix,
        }

    def save_attention_snapshot(self, buffer: TrajectoryBuffer, label: str):
        """Save attention weights for visualization."""
        padded_obs, padded_act, mask, _ = self._pad_episodes_with_actions(
            buffer.episodes
        )
        with torch.no_grad():
            _, credit_weights = self.cat(padded_obs, mask, padded_act)

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
                    f"CAT Loss: {stats['cat_loss']:.4f} | "
                    f"Mix: {stats['mix_ratio']:.2f}"
                )
                self.logger.log({
                    "timesteps": self.total_timesteps,
                    "episodes": self.total_episodes,
                    "mean_return": mean_return,
                    "policy_loss": stats["policy_loss"],
                    "cat_loss": stats["cat_loss"],
                    "entropy": stats["entropy"],
                    "credit_weight_std": stats["mean_credit_weight_std"],
                    "cat_entropy": stats["cat_entropy"],
                    "cat_temperature": stats["cat_temperature"],
                    "mix_ratio": stats["mix_ratio"],
                })

        self.logger.close()
        self.env.close()
        return recent_returns
