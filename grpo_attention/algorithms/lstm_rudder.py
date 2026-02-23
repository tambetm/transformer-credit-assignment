"""LSTM-RUDDER credit assignment (ablation for Transformer-RUDDER).

Same difference-based credit: r_t = g_t - g_{t-1}
But g_t comes from a small LSTM processing the trajectory sequentially.
Same detached advantages, same policy update.

This provides a direct comparison: does the Transformer's parallel attention
help over LSTM's sequential processing for credit assignment?
"""

import torch
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import deque

from ..models.policy import PolicyNetwork
from ..models.lstm_predictor import LSTMReturnPredictor
from ..utils.ema import EMAStats
from ..utils.buffer import Episode, TrajectoryBuffer
from ..utils.logger import CSVLogger


class LSTMRUDDERAlgo:
    """RUDDER with LSTM for credit assignment (ablation)."""

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

        # Create LSTM return predictor (same hidden size as Transformer d_model)
        self.predictor = LSTMReturnPredictor(
            obs_dim, action_dim=action_dim, hidden_size=64, num_layers=1,
        ).to(self.device)
        self.predictor_optimizer = torch.optim.Adam(
            self.predictor.parameters(), lr=config["lr_cat"]
        )

        # EMA baseline
        self.ema = EMAStats(alpha=config["ema_alpha"])

        # Small replay buffer for predictor training stability
        self.predictor_replay = deque(maxlen=100)

        # Logger
        self.logger = CSVLogger(log_dir)

        # Storage for value prediction visualization
        self.value_history = []

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

    def _pad_episodes(self, episodes):
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

    def _update_predictor(self, episodes, returns):
        """Update LSTM predictor with per-timestep MSE loss."""
        padded_obs, padded_act, mask, max_len = self._pad_episodes(episodes)
        target_returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Update running return statistics and normalize targets
        self.predictor.update_return_stats(target_returns)
        normalized_targets = self.predictor.normalize_returns(target_returns)

        # Forward pass
        _, _, pred_normalized = self.predictor(padded_obs, mask, padded_act)

        # MSE loss at every valid timestep
        target_expanded = normalized_targets.unsqueeze(1).expand_as(pred_normalized)

        # Masked MSE: only compute loss on valid (non-padded) positions
        mask_float = mask.float()
        per_step_loss = (pred_normalized - target_expanded) ** 2
        predictor_loss = (per_step_loss * mask_float).sum() / mask_float.sum().clamp(min=1)

        self.predictor_optimizer.zero_grad()
        predictor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
        self.predictor_optimizer.step()

        # Last-step prediction accuracy
        with torch.no_grad():
            last_step_losses = []
            for i, ep in enumerate(episodes):
                T = ep.length
                last_pred = pred_normalized[i, T - 1]
                last_loss = (last_pred - normalized_targets[i]) ** 2
                last_step_losses.append(last_loss.item())

        return {
            "predictor_mse": predictor_loss.item(),
            "predictor_last_step_mse": float(np.mean(last_step_losses)),
        }

    def update(self, buffer: TrajectoryBuffer):
        """Perform policy and predictor update."""
        gamma = self.config["gamma"]

        # Compute episode returns
        ep_returns = []
        for episode in buffer.episodes:
            G = episode.discounted_return(gamma)
            self.ema.update(G)
            ep_returns.append(G)

        # Add to replay buffer
        for ep, ret in zip(buffer.episodes, ep_returns):
            self.predictor_replay.append((ep, ret))

        # Train predictor: 1 pass on current batch + 1 on replay sample
        predictor_stats = self._update_predictor(buffer.episodes, ep_returns)

        if len(self.predictor_replay) >= 20:
            replay_size = min(len(self.predictor_replay), 30)
            indices = np.random.choice(len(self.predictor_replay), replay_size, replace=False)
            replay_eps = [self.predictor_replay[i][0] for i in indices]
            replay_rets = [self.predictor_replay[i][1] for i in indices]
            predictor_stats = self._update_predictor(replay_eps, replay_rets)

        # Get value differences as advantages (detached)
        padded_obs, padded_act, mask, max_len = self._pad_episodes(buffer.episodes)
        with torch.no_grad():
            _, value_diffs, _ = self.predictor(padded_obs, mask, padded_act)

        # Build per-timestep advantages from value differences
        all_obs = []
        all_actions = []
        all_advantages = []

        for i, episode in enumerate(buffer.episodes):
            T = episode.length
            tensors = episode.to_tensors(self.device)

            per_step_adv = value_diffs[i, :T]

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

        # Compute value diff statistics for logging
        valid_diffs = value_diffs[mask]
        value_diff_std = valid_diffs.std().item() if valid_diffs.numel() > 1 else 0.0

        return {
            "policy_loss": policy_loss.item(),
            "predictor_loss": predictor_stats.get("predictor_mse", 0.0),
            "predictor_last_mse": predictor_stats.get("predictor_last_step_mse", 0.0),
            "entropy": entropy.mean().item(),
            "value_diff_std": value_diff_std,
        }

    def save_value_snapshot(self, buffer: TrajectoryBuffer, label: str):
        """Save value predictions for visualization."""
        padded_obs, padded_act, mask, _ = self._pad_episodes(buffer.episodes)
        with torch.no_grad():
            pred_returns, value_diffs, _ = self.predictor(padded_obs, mask, padded_act)

        for i, ep in enumerate(buffer.episodes):
            T = ep.length
            self.value_history.append({
                "label": label,
                "episode": self.total_episodes,
                "timesteps": list(range(T)),
                "value_predictions": pred_returns[i, :T].cpu().numpy().tolist(),
                "value_diffs": value_diffs[i, :T].cpu().numpy().tolist(),
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

        # Take value prediction snapshots at 10%, 50%, 90% of training
        for frac in [0.1, 0.5, 0.9]:
            snapshot_points.add(int(total_target * frac))

        while self.total_timesteps < total_target:
            # Collect episodes
            buffer = self.collect_episodes(episodes_per_update)

            # Track returns
            for ep in buffer.episodes:
                recent_returns.append(ep.total_return)

            # Check if we should save value prediction snapshot
            for sp in list(snapshot_points):
                if self.total_timesteps >= sp:
                    label = f"step_{sp}"
                    self.save_value_snapshot(buffer, label)
                    snapshot_points.discard(sp)

            # Update
            stats = self.update(buffer)

            update_count += 1

            if update_count % eval_interval == 0:
                mean_return = np.mean(recent_returns[-100:])
                print(
                    f"[LSTM-RUDDER] Steps: {self.total_timesteps:>7d} | "
                    f"Episodes: {self.total_episodes:>5d} | "
                    f"Mean Return (100ep): {mean_return:.1f} | "
                    f"P.Loss: {stats['policy_loss']:.4f} | "
                    f"R.Loss: {stats['predictor_loss']:.4f}"
                )
                self.logger.log({
                    "timesteps": self.total_timesteps,
                    "episodes": self.total_episodes,
                    "mean_return": mean_return,
                    "policy_loss": stats["policy_loss"],
                    "predictor_loss": stats["predictor_loss"],
                    "predictor_last_mse": stats["predictor_last_mse"],
                    "entropy": stats["entropy"],
                    "value_diff_std": stats["value_diff_std"],
                })

        self.logger.close()
        self.env.close()
        return recent_returns
