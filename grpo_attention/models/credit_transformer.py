"""Credit Assignment Transformer (CAT).

A lightweight transformer that:
1. Embeds state observations via a small MLP
2. Appends a learnable [RETURN] token
3. Passes through transformer encoder layers
4. Predicts episode return from [RETURN] token's representation
5. Extracts attention weights from [RETURN] token to each state for credit assignment

Architecture:
- State embedding MLP: obs_dim -> 32
- Learnable [RETURN] token: 32-dim
- Transformer encoder: 1 layer, 2 heads, d_model=32, d_ff=64
- Return prediction head: 32 -> 1
- Sinusoidal positional encoding
"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, : x.size(1)]


class CreditAssignmentTransformer(nn.Module):
    """Credit Assignment Transformer (CAT) for redistributing episode advantages."""

    def __init__(self, obs_dim: int, d_model: int = 32, nhead: int = 2,
                 num_layers: int = 1, d_ff: int = 64):
        super().__init__()
        self.d_model = d_model

        # State embedding
        self.state_embed = nn.Sequential(
            nn.Linear(obs_dim, d_model),
            nn.Tanh(),
        )

        # Learnable [RETURN] token
        self.return_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model)

        # Custom transformer layer with accessible attention weights
        # We implement our own to easily extract attention weights
        self.nhead = nhead
        self.num_layers = num_layers

        # Multi-head attention + feedforward for each layer
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

        # Return prediction head
        self.return_head = nn.Linear(d_model, 1)

    def forward(self, states: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            states: (batch, seq_len, obs_dim) — padded state sequences
            mask: (batch, seq_len) — True for valid timesteps, False for padding

        Returns:
            predicted_return: (batch,)
            attention_weights: (batch, seq_len) — credit assignment weights
                (attention from [RETURN] token to each state, averaged over heads)
        """
        batch_size, seq_len, _ = states.shape

        # Embed states
        embedded = self.state_embed(states)  # (batch, seq_len, d_model)

        # Append [RETURN] token
        return_tok = self.return_token.expand(batch_size, -1, -1)
        x = torch.cat([embedded, return_tok], dim=1)  # (batch, seq_len+1, d_model)

        # Add positional encoding
        x = self.pos_enc(x)

        # Build key_padding_mask: True means IGNORE this position
        # mask input: True = valid. We need to invert and add one False for [RETURN] token
        if mask is not None:
            # Add False for [RETURN] token (always valid)
            return_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=mask.device)
            key_padding_mask = torch.cat([~mask, return_mask], dim=1)  # True = pad
        else:
            key_padding_mask = None

        # Pass through transformer layers, capturing attention from last layer
        attn_weights_last = None
        for layer in self.layers:
            # Self-attention with residual
            attn_out, attn_weights_last = layer["attn"](
                x, x, x,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=False,  # Get per-head weights
            )
            x = layer["norm1"](x + attn_out)

            # Feedforward with residual
            ff_out = layer["ff"](x)
            x = layer["norm2"](x + ff_out)

        # Extract [RETURN] token representation (last position)
        return_repr = x[:, -1, :]  # (batch, d_model)

        # Predict return
        predicted_return = self.return_head(return_repr).squeeze(-1)  # (batch,)

        # Extract attention weights from [RETURN] token to states
        # attn_weights_last: (batch, nhead, seq_len+1, seq_len+1)
        # We want: [RETURN] attending to states = last query position, first seq_len keys
        return_to_states = attn_weights_last[:, :, -1, :seq_len]  # (batch, nhead, seq_len)

        # Average over heads
        credit_weights = return_to_states.mean(dim=1)  # (batch, seq_len)

        # Mask out padding positions
        if mask is not None:
            credit_weights = credit_weights * mask.float()

        # Re-normalize so weights sum to 1 over valid positions
        weight_sums = credit_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        credit_weights = credit_weights / weight_sums

        return predicted_return, credit_weights
