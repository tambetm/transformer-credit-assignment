"""LSTM-based return predictor for RUDDER-style credit assignment (ablation).

Replaces the causal Transformer with a simple LSTM for return prediction.
Naturally causal: hidden state at time t only depends on inputs up to t.

Architecture:
- State + action embedding: MLP(obs_dim + action_dim -> 64)
- LSTM: input_size=64, hidden_size=64, num_layers=1
- Return prediction head: Linear(64 -> 1) applied at every timestep

Forward pass:
    Input: states (B, T, obs_dim), actions (B, T)
    Output: predicted returns (B, T), value_diffs (B, T), pred_normalized (B, T)
"""

import torch
import torch.nn as nn


class LSTMReturnPredictor(nn.Module):
    """LSTM that predicts episode return at every timestep.

    Naturally causal (hidden state only depends on past).
    Used as an ablation against the Transformer predictor.
    """

    def __init__(self, obs_dim: int, action_dim: int = None,
                 hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim

        # State embedding
        self.state_embed = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
        )

        # Action embedding (learned lookup table for discrete actions)
        if action_dim is not None:
            self.action_embed = nn.Embedding(action_dim, hidden_size)
            self.combine = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Tanh(),
            )
        else:
            self.action_embed = None
            self.combine = None

        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Per-timestep return prediction head (shared across positions)
        self.return_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        # Running return statistics for normalization
        self.register_buffer("return_mean", torch.tensor(0.0))
        self.register_buffer("return_std", torch.tensor(1.0))
        self.register_buffer("return_count", torch.tensor(0.0))

    def update_return_stats(self, returns: torch.Tensor):
        """Update running mean/std of returns for normalized MSE target."""
        batch_mean = returns.mean()
        batch_std = returns.std().clamp(min=1e-6)
        n = returns.numel()

        if self.return_count == 0:
            self.return_mean.copy_(batch_mean)
            self.return_std.copy_(batch_std)
        else:
            alpha = min(0.1, n / (self.return_count + n))
            self.return_mean.mul_(1 - alpha).add_(batch_mean * alpha)
            self.return_std.mul_(1 - alpha).add_(batch_std * alpha)
        self.return_count.add_(n)

    def normalize_returns(self, returns: torch.Tensor) -> torch.Tensor:
        """Normalize return targets."""
        return (returns - self.return_mean) / self.return_std.clamp(min=1e-6)

    def denormalize_returns(self, normalized: torch.Tensor) -> torch.Tensor:
        """Denormalize predicted returns."""
        return normalized * self.return_std.clamp(min=1e-6) + self.return_mean

    def forward(self, states: torch.Tensor, mask: torch.Tensor = None,
                actions: torch.Tensor = None):
        """
        Args:
            states: (batch, seq_len, obs_dim) - padded state sequences
            mask: (batch, seq_len) - True for valid timesteps, False for padding
            actions: (batch, seq_len) - padded action indices

        Returns:
            predicted_returns: (batch, seq_len) - denormalized per-step predictions
            value_diffs: (batch, seq_len) - V_norm(t) - V_norm(t-1), the advantages
            pred_normalized: (batch, seq_len) - normalized predictions (for MSE loss)
        """
        batch_size, seq_len, _ = states.shape

        # Embed states
        state_emb = self.state_embed(states)

        # Optionally embed and combine actions
        if self.action_embed is not None and actions is not None:
            action_emb = self.action_embed(actions)
            embedded = self.combine(torch.cat([state_emb, action_emb], dim=-1))
        else:
            embedded = state_emb

        # Pack padded sequences for efficient LSTM processing
        if mask is not None:
            lengths = mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True, total_length=seq_len
            )
        else:
            lstm_out, _ = self.lstm(embedded)

        # Per-timestep return predictions (in normalized space)
        pred_normalized = self.return_head(lstm_out).squeeze(-1)  # (batch, seq_len)

        # Denormalize for logging/interpretation
        predicted_returns = self.denormalize_returns(pred_normalized)

        # Value differences: A_t = V_norm(t) - V_norm(t-1)
        # V_norm(-1) = 0 because normalize(E[G]) = (E[G] - E[G]) / sigma = 0
        zero_init = torch.zeros(batch_size, 1, device=states.device)
        v_sequence = torch.cat([zero_init, pred_normalized], dim=1)
        value_diffs = v_sequence[:, 1:] - v_sequence[:, :-1]  # (batch, seq_len)

        # Zero out padding positions
        if mask is not None:
            value_diffs = value_diffs * mask.float()

        return predicted_returns, value_diffs, pred_normalized
