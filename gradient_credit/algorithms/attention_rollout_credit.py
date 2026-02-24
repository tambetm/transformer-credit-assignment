"""Attention Rollout Credit (ARC) — novel credit assignment method.

Core insight: In a causal transformer trained to predict returns, the attention
weights encode which past timesteps influenced the final prediction. Single-layer
attention is shallow. Attention rollout (Abnar & Zuidema, 2020) traces multi-layer
information flow by composing attention matrices across layers, accounting for
residual connections:

    R = prod_l (0.5 * A^l + 0.5 * I)
    flow_t = R[T-1, t]

Combined multiplicatively with value differences:

    A_t = vd_t * (flow_t / mean(flow))

Key properties:
- Zero extra cost: attention weights are already computed during the forward pass
- Multi-hop information flow: captures indirect influence through intermediate positions
- Signed credit (from value diffs), attention-weighted magnitude
- When flow is uniform: reduces to pure value differences
- No backward pass needed (unlike gradient methods)
- Uses the same causal predictor as RUDDER (no extra model)
"""

import torch

from .base_gradient import BaseGradientCreditAlgo
from ..utils.gradient_utils import attention_rollout_credit_to_advantages, compute_credit_stats


class AttentionRolloutCreditAlgo(BaseGradientCreditAlgo):
    """Attention Rollout Credit for credit assignment."""

    algo_tag = "AttnRoll"

    def _compute_credit(self, padded_obs, padded_act, mask):
        # Compute attention rollout credit (returns rollout, value_diffs, pred)
        rollout_credit, value_diffs, pred_normalized = (
            self.predictor.get_attention_rollout_credit(padded_obs, padded_act, mask))

        # Store value_diffs for use in _credit_to_advantages
        self._last_value_diffs = value_diffs

        return rollout_credit, pred_normalized

    def _credit_to_advantages(self, credit, episode_advantages, episodes, mask):
        # Retrieve stored value diffs (computed during _compute_credit)
        value_diffs = self._last_value_diffs

        return attention_rollout_credit_to_advantages(
            credit, value_diffs, episodes, mask, self.device)
