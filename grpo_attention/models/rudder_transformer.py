"""RUDDER-style Transformer for per-timestep value prediction.

Replaces RUDDER's LSTM with a causal (autoregressive) transformer.
Key idea: predict total episode return at each timestep using only
information from past timesteps. Per-timestep advantages come from
value differences: Â_t = V̂(t) - V̂(t-1).

These advantages telescope: Σ_t Â_t = V̂(T-1) - V̂(-1) ≈ G - E[G].

Architecture:
- State + action embedding (MLP + learned lookup)
- Sinusoidal positional encoding
- Causal self-attention (each position attends only to positions ≤ t)
- Per-timestep return prediction head (shared across positions)
- No [RETURN] token, no cross-attention credit head

Why causal masking is essential:
- With full attention, every position sees the entire sequence, so V̂(t) ≈ V̂(t')
  for all t, t' → value differences ≈ 0 → no credit assignment signal.
- Causal masking creates an information gradient: position t only knows τ_{0:t}.
- The value difference V̂(t) - V̂(t-1) then measures what timestep t adds
  to the return prediction — a principled credit assignment signal.
- This exactly matches RUDDER's LSTM, which naturally processes left-to-right.
"""

import torch
import torch.nn as nn

from .credit_transformer import SinusoidalPositionalEncoding


class RUDDERTransformer(nn.Module):
    """Causal transformer for RUDDER-style return prediction and credit assignment."""

    def __init__(self, obs_dim: int, action_dim: int = None, d_model: int = 32,
                 nhead: int = 2, num_layers: int = 2, d_ff: int = 64):
        super().__init__()
        self.d_model = d_model
        self.action_dim = action_dim

        # State embedding
        self.state_embed = nn.Sequential(
            nn.Linear(obs_dim, d_model),
            nn.Tanh(),
        )

        # Action embedding (learned lookup table for discrete actions)
        if action_dim is not None:
            self.action_embed = nn.Embedding(action_dim, d_model)
            self.combine = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Tanh(),
            )
        else:
            self.action_embed = None
            self.combine = None

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model)

        # Causal transformer encoder layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "attn": nn.MultiheadAttention(d_model, nhead, batch_first=True),
                "norm1": nn.LayerNorm(d_model),
                "ff": nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.ReLU(),
                    nn.Linear(d_ff, d_model),
                ),
                "norm2": nn.LayerNorm(d_model),
            }))

        # Per-timestep return prediction head (shared across positions)
        self.return_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
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
            states: (batch, seq_len, obs_dim) — padded state sequences
            mask: (batch, seq_len) — True for valid timesteps, False for padding
            actions: (batch, seq_len) — padded action indices

        Returns:
            predicted_returns: (batch, seq_len) — denormalized per-step predictions
            value_diffs: (batch, seq_len) — V̂_norm(t) - V̂_norm(t-1), the advantages
            pred_normalized: (batch, seq_len) — normalized predictions (for MSE loss)
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

        # Add positional encoding
        x = self.pos_enc(embedded)

        # Causal mask: True = blocked (position i cannot attend to j > i)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
            diagonal=1,
        )

        # Key padding mask: True means IGNORE this position
        key_padding_mask = ~mask if mask is not None else None

        # Pass through causal transformer layers
        for layer in self.layers:
            attn_out, _ = layer["attn"](
                x, x, x,
                attn_mask=causal_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            x = layer["norm1"](x + attn_out)
            ff_out = layer["ff"](x)
            x = layer["norm2"](x + ff_out)

        # Per-timestep return predictions (in normalized space)
        pred_normalized = self.return_head(x).squeeze(-1)  # (batch, seq_len)

        # Denormalize for logging/interpretation
        predicted_returns = self.denormalize_returns(pred_normalized)

        # Value differences: Â_t = V̂_norm(t) - V̂_norm(t-1)
        # V̂_norm(-1) = 0 because normalize(E[G]) = (E[G] - E[G]) / σ = 0
        # So the sum telescopes: Σ Â_t = V̂_norm(T-1) ≈ (G - E[G]) / σ
        zero_init = torch.zeros(batch_size, 1, device=x.device)
        v_sequence = torch.cat([zero_init, pred_normalized], dim=1)
        value_diffs = v_sequence[:, 1:] - v_sequence[:, :-1]  # (batch, seq_len)

        # Zero out padding positions
        if mask is not None:
            value_diffs = value_diffs * mask.float()

        return predicted_returns, value_diffs, pred_normalized
