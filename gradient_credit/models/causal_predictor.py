"""Causal Return Predictor with gradient-based credit assignment.

Causal transformer for return prediction (d_model=32, 2 heads — same size as
experiments 1 & 2 for comparable results).
Supports gradient-based credit methods:
  - grad_norm: ||dg/d_embedded_t||_2 (unsigned saliency)
  - grad_x_input: (dg/d_embedded_t . embedded_t) (signed contribution)
  - integrated_grads: averaged gradients along interpolation path (signed, principled)

Plus attention-based credit:
  - attention_rollout: multi-layer attention rollout for information flow tracing

Also retains value-difference credit from RUDDER for hybrid and comparison.
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
        return x + self.pe[:, :x.size(1)]


class CausalReturnPredictor(nn.Module):
    """Causal transformer for return prediction with gradient credit methods.

    Same size as experiments 1 & 2 (d_model=32, nhead=2, d_ff=64).
    """

    def __init__(self, obs_dim: int, action_dim: int = None, d_model: int = 32,
                 nhead: int = 2, num_layers: int = 2, d_ff: int = 64,
                 dropout: float = 0.1):
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
                "attn": nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout),
                "norm1": nn.LayerNorm(d_model),
                "ff": nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model),
                ),
                "norm2": nn.LayerNorm(d_model),
                "drop1": nn.Dropout(dropout),
                "drop2": nn.Dropout(dropout),
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
        return (returns - self.return_mean) / self.return_std.clamp(min=1e-6)

    def denormalize_returns(self, normalized: torch.Tensor) -> torch.Tensor:
        return normalized * self.return_std.clamp(min=1e-6) + self.return_mean

    def embed(self, states: torch.Tensor, actions: torch.Tensor = None):
        """Compute embeddings WITHOUT positional encoding."""
        state_emb = self.state_embed(states)
        if self.action_embed is not None and actions is not None:
            action_emb = self.action_embed(actions)
            return self.combine(torch.cat([state_emb, action_emb], dim=-1))
        return state_emb

    def forward_from_embedding(self, embedded: torch.Tensor, mask: torch.Tensor = None):
        """Forward pass from embeddings (adds pos enc + transformer + head).

        Returns normalized per-timestep predictions (batch, seq_len).
        """
        seq_len = embedded.size(1)
        x = self.pos_enc(embedded)

        # Causal mask: True = blocked
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        key_padding_mask = ~mask if mask is not None else None

        for layer in self.layers:
            attn_out, _ = layer["attn"](
                x, x, x,
                attn_mask=causal_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            x = layer["norm1"](x + layer["drop1"](attn_out))
            ff_out = layer["ff"](x)
            x = layer["norm2"](x + layer["drop2"](ff_out))

        pred_normalized = self.return_head(x).squeeze(-1)  # (batch, seq_len)
        return pred_normalized

    def forward(self, states: torch.Tensor, mask: torch.Tensor = None,
                actions: torch.Tensor = None):
        """Full forward pass.

        Returns:
            predicted_returns: (batch, seq_len) denormalized
            value_diffs: (batch, seq_len) normalized value differences
            pred_normalized: (batch, seq_len) normalized predictions
        """
        embedded = self.embed(states, actions)
        pred_normalized = self.forward_from_embedding(embedded, mask)

        predicted_returns = self.denormalize_returns(pred_normalized)

        # Value differences: V_norm(t) - V_norm(t-1), with V_norm(-1) = 0
        batch_size = states.size(0)
        zero_init = torch.zeros(batch_size, 1, device=states.device)
        v_sequence = torch.cat([zero_init, pred_normalized], dim=1)
        value_diffs = v_sequence[:, 1:] - v_sequence[:, :-1]

        if mask is not None:
            value_diffs = value_diffs * mask.float()

        return predicted_returns, value_diffs, pred_normalized

    def forward_with_attention(self, states: torch.Tensor, mask: torch.Tensor = None,
                               actions: torch.Tensor = None):
        """Forward pass that also returns per-layer attention weights.

        Returns:
            predicted_returns: (batch, seq_len) denormalized
            value_diffs: (batch, seq_len) normalized value differences
            pred_normalized: (batch, seq_len) normalized predictions
            attention_weights: list of (batch, seq_len, seq_len) per layer
        """
        embedded = self.embed(states, actions)
        seq_len = embedded.size(1)
        x = self.pos_enc(embedded)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        key_padding_mask = ~mask if mask is not None else None

        attention_weights = []
        for layer in self.layers:
            attn_out, attn_w = layer["attn"](
                x, x, x,
                attn_mask=causal_mask,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=True,
            )
            attention_weights.append(attn_w)  # (B, T, T) averaged over heads
            x = layer["norm1"](x + layer["drop1"](attn_out))
            ff_out = layer["ff"](x)
            x = layer["norm2"](x + layer["drop2"](ff_out))

        pred_normalized = self.return_head(x).squeeze(-1)
        predicted_returns = self.denormalize_returns(pred_normalized)

        batch_size = states.size(0)
        zero_init = torch.zeros(batch_size, 1, device=states.device)
        v_sequence = torch.cat([zero_init, pred_normalized], dim=1)
        value_diffs = v_sequence[:, 1:] - v_sequence[:, :-1]

        if mask is not None:
            value_diffs = value_diffs * mask.float()

        return predicted_returns, value_diffs, pred_normalized, attention_weights

    def get_attention_rollout_credit(self, states, actions, mask):
        """Compute attention rollout credit weights.

        Traces information flow from each input position to the final position
        by multiplying attention matrices across layers (accounting for residual
        connections via the 0.5*A + 0.5*I formulation from Abnar & Zuidema, 2020).

        Returns:
            rollout_credit: (B, T) per-timestep credit (detached)
            value_diffs: (B, T) value differences (detached)
            pred_normalized: (B, T) normalized predictions (detached)
        """
        with torch.no_grad():
            _, value_diffs, pred_normalized, attention_weights = (
                self.forward_with_attention(states, mask, actions))

        B, T = states.size(0), states.size(1)

        # Attention rollout: product of (0.5 * A^l + 0.5 * I) across layers
        # This traces total information flow accounting for residual connections.
        eye = torch.eye(T, device=states.device).unsqueeze(0).expand(B, -1, -1)
        rollout = eye.clone()

        for attn_w in attention_weights:
            # Mix attention with identity to account for residual connection
            residual_attn = 0.5 * attn_w + 0.5 * eye
            rollout = torch.bmm(residual_attn, rollout)

        # Extract flow from each input position to the final valid position
        last_indices = mask.sum(dim=1).long() - 1  # (B,)
        rollout_credit = torch.zeros(B, T, device=states.device)
        for b in range(B):
            rollout_credit[b] = rollout[b, last_indices[b]]

        # Zero out padding
        rollout_credit = rollout_credit * mask.float()

        return rollout_credit.detach(), value_diffs.detach(), pred_normalized.detach()

    def get_gradient_credit(self, states, actions, mask, method='grad_norm',
                            n_ig_steps=10):
        """Compute gradient-based credit for each timestep.

        Args:
            states: (B, T, obs_dim)
            actions: (B, T) action indices
            mask: (B, T) bool mask
            method: 'grad_norm', 'grad_x_input', or 'integrated_grads'
            n_ig_steps: number of interpolation steps for integrated gradients

        Returns:
            credit: (B, T) credit values (detached)
            pred_normalized: (B, T) normalized predictions (detached)
        """
        if method == 'integrated_grads':
            return self._integrated_gradients(states, actions, mask, n_ig_steps)

        B = states.size(0)

        # Compute embeddings and detach to create a new leaf for gradients
        with torch.no_grad():
            embedded_base = self.embed(states, actions)
        embedded = embedded_base.clone().requires_grad_(True)

        # Forward through transformer
        pred_normalized = self.forward_from_embedding(embedded, mask)

        # Get prediction at last valid position for each episode
        last_indices = mask.sum(dim=1).long() - 1  # (B,)
        last_preds = pred_normalized[torch.arange(B, device=states.device), last_indices]

        # Backward — sum over batch (gradients don't mix across batch dim)
        loss = last_preds.sum()
        loss.backward()

        grad = embedded.grad  # (B, T, d_model)

        if method == 'grad_norm':
            credit = grad.norm(dim=-1)  # (B, T) — unsigned
        elif method == 'grad_x_input':
            credit = (grad * embedded.detach()).sum(dim=-1)  # (B, T) — signed
        else:
            raise ValueError(f"Unknown gradient method: {method}")

        # Zero out padding
        if mask is not None:
            credit = credit * mask.float()

        return credit.detach(), pred_normalized.detach()

    def _integrated_gradients(self, states, actions, mask, n_steps=10):
        """Integrated gradients: average gradient along interpolation path.

        IG = (input - baseline) * E[grad(f(baseline + alpha*(input-baseline)))]
        Satisfies the completeness axiom: sum(IG) = f(input) - f(baseline).
        """
        B = states.size(0)

        with torch.no_grad():
            actual_embedded = self.embed(states, actions)
        baseline_embedded = torch.zeros_like(actual_embedded)
        delta = actual_embedded - baseline_embedded

        accumulated_grad = torch.zeros_like(actual_embedded)

        for k in range(1, n_steps + 1):
            alpha = k / n_steps
            interp = baseline_embedded + alpha * delta
            interp = interp.clone().requires_grad_(True)

            pred = self.forward_from_embedding(interp, mask)
            last_indices = mask.sum(dim=1).long() - 1
            last_preds = pred[torch.arange(B, device=states.device), last_indices]
            last_preds.sum().backward()

            accumulated_grad = accumulated_grad + interp.grad

        # IG = (input - baseline) * avg_gradient
        avg_grad = accumulated_grad / n_steps
        ig = delta * avg_grad

        # Sum over embedding dim to get per-timestep credit (signed)
        credit = ig.sum(dim=-1)  # (B, T)

        if mask is not None:
            credit = credit * mask.float()

        # Also get predictions for logging
        with torch.no_grad():
            pred_normalized = self.forward_from_embedding(actual_embedded, mask)

        return credit.detach(), pred_normalized.detach()
