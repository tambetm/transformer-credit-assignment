"""Gradient-Sharpened Value Differences (GSVD) — novel credit assignment method.

Core insight: Value differences (V(t) - V(t-1)) are the best known credit signal
for episodic RL, but they can be diffuse — spreading credit uniformly across many
timesteps. Gradient saliency (||dg/d_emb_t||_2) from the same causal return
predictor identifies which timesteps the model is most sensitive to.

GSVD multiplies value differences by softmax-sharpened gradient saliency:

    A_t = vd_t * softmax(||dg/d_emb_t||_2 / tau) * T

This "sharpens" value differences by upweighting timesteps where the model is
most responsive, while preserving the sign and base structure from value diffs.

Key properties:
- When tau -> inf: reduces to pure value differences (safe fallback)
- When tau -> 0: concentrates credit on highest-saliency timestep
- Signed credit (from value diffs), gradient-informed magnitude
- Adaptive temperature: starts high (trust value diffs) and decreases as
  the predictor improves (trust gradient saliency more)
- Cost: one backward pass (gradient computation) + one forward pass (value diffs)
"""

import torch

from .base_gradient import BaseGradientCreditAlgo
from ..utils.gradient_utils import gsvd_credit_to_advantages


class GSVDCreditAlgo(BaseGradientCreditAlgo):
    """Gradient-Sharpened Value Differences for credit assignment."""

    algo_tag = "GSVD"

    def _compute_credit(self, padded_obs, padded_act, mask):
        # Get gradient norms as saliency signal
        return self.predictor.get_gradient_credit(
            padded_obs, padded_act, mask, method='grad_norm')

    def _credit_to_advantages(self, credit, episode_advantages, episodes, mask):
        # Get value differences from the same predictor
        padded_obs, padded_act, mask_new = self._pad_episodes(episodes)
        with torch.no_grad():
            _, value_diffs, _ = self.predictor(padded_obs, mask_new, padded_act)

        # Adaptive temperature: high when predictor is bad, low when good
        if self.initial_mse is not None and self.initial_mse > 0:
            pred_quality = 1.0 - min(
                1.0, (self.recent_mse or self.initial_mse) / self.initial_mse)
        else:
            pred_quality = 0.0

        # tau_max=10 (≈ uniform weights) -> tau_min=0.5 (sharp focus)
        temperature = 10.0 * (1.0 - pred_quality) + 0.5 * pred_quality

        return gsvd_credit_to_advantages(
            credit, value_diffs, episodes, mask, temperature, self.device)
