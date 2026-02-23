"""RUDDER-style credit assignment with value differences from causal transformer.

Rerun of experiment 2's approach with the scaled-up predictor.
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
import time

from ..models.policy import PolicyNetwork
from ..models.causal_predictor import CausalReturnPredictor
from ..utils.ema import EMAStats
from ..utils.buffer import Episode, TrajectoryBuffer
from ..utils.logger import CSVLogger
from ..utils.gradient_utils import value_diff_to_advantages


class RUDDERValueDiffAlgo:
    """RUDDER with value differences from causal transformer."""

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
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config["lr_policy"]
        )

        self.predictor = CausalReturnPredictor(
            obs_dim, action_dim=action_dim,
            d_model=config["predictor_d_model"],
            nhead=config["predictor_nhead"],
            num_layers=config["predictor_layers"],
            d_ff=config["predictor_d_ff"],
            dropout=config["predictor_dropout"],
        ).to(self.device)
        self.predictor_optimizer = torch.optim.Adam(
            self.predictor.parameters(), lr=config["lr_predictor"]
        )

        self.ema = EMAStats(alpha=config["ema_alpha"])
        self.replay_buffer = deque(maxlen=config["replay_buffer_size"])
        self.logger = CSVLogger(log_dir)

        self.credit_history = []
        self.total_timesteps = 0
        self.total_episodes = 0
        self.update_times = []

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

    def _pad_episodes(self, episodes):
        max_len = max(ep.length for ep in episodes)
        batch_size = len(episodes)
        obs_dim = len(episodes[0].observations[0]) if not hasattr(
            episodes[0].observations[0], 'shape'
        ) else episodes[0].observations[0].shape[0]

        padded_obs = torch.zeros(batch_size, max_len, obs_dim, device=self.device)
        padded_act = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=self.device)

        for i, ep in enumerate(episodes):
            T = ep.length
            padded_obs[i, :T] = torch.tensor(
                np.array(ep.observations), dtype=torch.float32, device=self.device)
            padded_act[i, :T] = torch.tensor(
                np.array(ep.actions), dtype=torch.long, device=self.device)
            mask[i, :T] = True

        return padded_obs, padded_act, mask

    def _update_predictor(self, episodes, returns):
        padded_obs, padded_act, mask = self._pad_episodes(episodes)
        target_returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        self.predictor.update_return_stats(target_returns)
        normalized_targets = self.predictor.normalize_returns(target_returns)

        _, _, pred_normalized = self.predictor(padded_obs, mask, padded_act)

        target_expanded = normalized_targets.unsqueeze(1).expand_as(pred_normalized)
        mask_float = mask.float()
        per_step_loss = (pred_normalized - target_expanded) ** 2
        loss = (per_step_loss * mask_float).sum() / mask_float.sum().clamp(min=1)

        self.predictor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
        self.predictor_optimizer.step()

        with torch.no_grad():
            last_step_losses = []
            for i, ep in enumerate(episodes):
                T = ep.length
                last_pred = pred_normalized[i, T - 1]
                last_loss = (last_pred - normalized_targets[i]) ** 2
                last_step_losses.append(last_loss.item())

        return {"predictor_mse": loss.item(), "last_step_mse": float(np.mean(last_step_losses))}

    def update(self, buffer: TrajectoryBuffer):
        t_start = time.time()
        gamma = self.config["gamma"]

        ep_returns = []
        for episode in buffer.episodes:
            G = episode.discounted_return(gamma)
            self.ema.update(G)
            ep_returns.append(G)

        for ep, ret in zip(buffer.episodes, ep_returns):
            self.replay_buffer.append((ep, ret))

        # Train predictor
        pred_stats = self._update_predictor(buffer.episodes, ep_returns)
        for _ in range(self.config["predictor_train_steps"] - 1):
            if len(self.replay_buffer) >= 20:
                replay_size = min(len(self.replay_buffer), 30)
                indices = np.random.choice(len(self.replay_buffer), replay_size, replace=False)
                replay_eps = [self.replay_buffer[i][0] for i in indices]
                replay_rets = [self.replay_buffer[i][1] for i in indices]
                pred_stats = self._update_predictor(replay_eps, replay_rets)

        # Get value differences as advantages
        padded_obs, padded_act, mask = self._pad_episodes(buffer.episodes)
        with torch.no_grad():
            _, value_diffs, _ = self.predictor(padded_obs, mask, padded_act)

        all_obs, all_actions, all_advantages = value_diff_to_advantages(
            value_diffs, buffer.episodes, mask, self.device)

        obs_batch = torch.cat(all_obs)
        action_batch = torch.cat(all_actions)
        advantage_batch = torch.cat(all_advantages)

        if len(advantage_batch) > 1:
            advantage_batch = (advantage_batch - advantage_batch.mean()) / (
                advantage_batch.std() + 1e-8)

        log_probs, entropy = self.policy.evaluate_actions(obs_batch, action_batch)
        policy_loss = -(log_probs * advantage_batch).mean()
        entropy_loss = -entropy.mean() * 0.01
        loss = policy_loss + entropy_loss

        self.policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.policy_optimizer.step()

        update_time = time.time() - t_start
        self.update_times.append(update_time)

        valid_diffs = value_diffs[mask]
        return {
            "policy_loss": policy_loss.item(),
            "predictor_mse": pred_stats["predictor_mse"],
            "last_step_mse": pred_stats["last_step_mse"],
            "entropy": entropy.mean().item(),
            "value_diff_std": valid_diffs.std().item() if valid_diffs.numel() > 1 else 0.0,
            "update_time": update_time,
        }

    def save_credit_snapshot(self, buffer, label):
        padded_obs, padded_act, mask = self._pad_episodes(buffer.episodes)
        with torch.no_grad():
            pred_returns, value_diffs, _ = self.predictor(padded_obs, mask, padded_act)
        for i, ep in enumerate(buffer.episodes):
            T = ep.length
            self.credit_history.append({
                "label": label,
                "episode": self.total_episodes,
                "timesteps": list(range(T)),
                "credit_type": "value_diff",
                "credit_values": value_diffs[i, :T].cpu().numpy().tolist(),
                "value_predictions": pred_returns[i, :T].cpu().numpy().tolist(),
                "rewards": ep.rewards[:T],
                "total_return": ep.total_return,
            })

    def train(self):
        total_target = self.config["total_timesteps"]
        episodes_per_update = self.config["episodes_per_update"]
        eval_interval = 10
        update_count = 0
        recent_returns = []

        snapshot_points = set()
        for frac in [0.1, 0.5, 0.9]:
            snapshot_points.add(int(total_target * frac))

        while self.total_timesteps < total_target:
            buffer = self.collect_episodes(episodes_per_update)
            for ep in buffer.episodes:
                recent_returns.append(ep.total_return)

            for sp in list(snapshot_points):
                if self.total_timesteps >= sp:
                    self.save_credit_snapshot(buffer, f"step_{sp}")
                    snapshot_points.discard(sp)

            stats = self.update(buffer)
            update_count += 1

            if update_count % eval_interval == 0:
                mean_return = np.mean(recent_returns[-100:])
                print(
                    f"[RUDDER-VD] Steps: {self.total_timesteps:>7d} | "
                    f"Episodes: {self.total_episodes:>5d} | "
                    f"Mean Return (100ep): {mean_return:.1f} | "
                    f"P.Loss: {stats['policy_loss']:.4f} | "
                    f"Pred.MSE: {stats['predictor_mse']:.4f}"
                )
                self.logger.log({
                    "timesteps": self.total_timesteps,
                    "episodes": self.total_episodes,
                    "mean_return": mean_return,
                    "policy_loss": stats["policy_loss"],
                    "predictor_mse": stats["predictor_mse"],
                    "last_step_mse": stats["last_step_mse"],
                    "entropy": stats["entropy"],
                    "value_diff_std": stats["value_diff_std"],
                    "update_time": stats["update_time"],
                })

        self.logger.close()
        self.env.close()
        return recent_returns
