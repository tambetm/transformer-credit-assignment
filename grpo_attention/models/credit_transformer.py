"""Credit Assignment Transformer (CAT).

A lightweight transformer that:
1. Embeds state-action pairs via small MLPs
2. Appends a learnable [RETURN] token
3. Passes through transformer encoder layers for contextual representation
4. Uses a dedicated cross-attention credit head: [RETURN] queries state-action keys
5. Predicts episode return from [RETURN] token's representation
6. Credit weights come from the dedicated credit head (not self-attention)

Key improvements over naive approach:
- Actions included: credit assignment is about which *actions* mattered, not just states
- Dedicated credit head: separate from self-attention, trained end-to-end via return prediction
- Learnable temperature: allows sharpening attention toward critical timesteps
- 2 transformer layers for richer contextual representations
- Normalized return targets for more stable MSE training

Architecture:
- State embedding MLP: obs_dim -> d_model
- Action embedding: action_dim -> d_model (learned)
- Combined: state_embed + action_embed projected to d_model
- Learnable [RETURN] token: d_model-dim
- Transformer encoder: 2 layers, 2 heads, d_model=32, d_ff=64
- Dedicated credit cross-attention head: [RETURN] -> state-action sequence
- Return prediction head: d_model -> 1
- Sinusoidal positional encoding
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
            # Projection to combine state + action
            self.combine = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Tanh(),
            )
        else:
            self.action_embed = None
            self.combine = None

        # Learnable [RETURN] token
        self.return_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model)

        # Transformer encoder layers
        self.nhead = nhead
        self.num_layers = num_layers
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

        # Dedicated credit assignment cross-attention head
        # [RETURN] token queries the encoded state-action sequence
        self.credit_query = nn.Linear(d_model, d_model)
        self.credit_key = nn.Linear(d_model, d_model)

        # Learnable temperature for sharpening credit attention
        # Initialize to 1.0; will learn to sharpen (>1) or soften (<1)
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # Return prediction head
        self.return_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )

        # Running statistics for return normalization (not learned)
        self.register_buffer("return_mean", torch.tensor(0.0))
        self.register_buffer("return_std", torch.tensor(1.0))
        self.register_buffer("return_count", torch.tensor(0.0))

    def update_return_stats(self, returns: torch.Tensor):
        """Update running mean/std of returns for normalized MSE target."""
        batch_mean = returns.mean()
        batch_std = returns.std().clamp(min=1e-6)
        n = returns.numel()

        # Welford-like running update
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
            actions: (batch, seq_len) — padded action indices (optional)

        Returns:
            predicted_return: (batch,) — in original (denormalized) scale
            credit_weights: (batch, seq_len) — credit assignment weights (sum to 1)
        """
        batch_size, seq_len, _ = states.shape

        # Embed states
        state_emb = self.state_embed(states)  # (batch, seq_len, d_model)

        # Optionally embed and combine actions
        if self.action_embed is not None and actions is not None:
            action_emb = self.action_embed(actions)  # (batch, seq_len, d_model)
            embedded = self.combine(torch.cat([state_emb, action_emb], dim=-1))
        else:
            embedded = state_emb

        # Append [RETURN] token
        return_tok = self.return_token.expand(batch_size, -1, -1)
        x = torch.cat([embedded, return_tok], dim=1)  # (batch, seq_len+1, d_model)

        # Add positional encoding
        x = self.pos_enc(x)

        # Build key_padding_mask: True means IGNORE this position
        if mask is not None:
            return_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=mask.device)
            key_padding_mask = torch.cat([~mask, return_mask], dim=1)
        else:
            key_padding_mask = None

        # Pass through transformer encoder layers
        for layer in self.layers:
            attn_out, _ = layer["attn"](
                x, x, x,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            x = layer["norm1"](x + attn_out)
            ff_out = layer["ff"](x)
            x = layer["norm2"](x + ff_out)

        # Extract [RETURN] token and state-action representations
        return_repr = x[:, -1:, :]     # (batch, 1, d_model)
        sa_repr = x[:, :seq_len, :]    # (batch, seq_len, d_model)

        # Predict return (in normalized space)
        predicted_normalized = self.return_head(return_repr[:, 0, :]).squeeze(-1)
        predicted_return = self.denormalize_returns(predicted_normalized)

        # Dedicated credit assignment via cross-attention
        # Query from [RETURN], keys from state-action sequence
        q = self.credit_query(return_repr)   # (batch, 1, d_model)
        k = self.credit_key(sa_repr)         # (batch, seq_len, d_model)

        # Scaled dot-product attention with learnable temperature
        # temperature > 1 sharpens, < 1 softens
        temp = F.softplus(self.temperature).clamp(max=3.0) + 0.1  # Positive, capped at 3.1
        scale = math.sqrt(self.d_model)
        attn_logits = torch.bmm(q, k.transpose(1, 2)) / scale  # (batch, 1, seq_len)
        attn_logits = attn_logits * temp  # Apply temperature

        # Mask padding
        if mask is not None:
            attn_logits = attn_logits.masked_fill(~mask.unsqueeze(1), float("-inf"))

        credit_weights = F.softmax(attn_logits, dim=-1).squeeze(1)  # (batch, seq_len)

        # Handle edge case where all positions are masked
        if mask is not None:
            all_masked = ~mask.any(dim=-1, keepdim=True)
            credit_weights = credit_weights.masked_fill(all_masked, 0.0)

        return predicted_return, credit_weights
