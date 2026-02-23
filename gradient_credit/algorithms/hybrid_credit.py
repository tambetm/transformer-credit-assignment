"""Hybrid Credit: adaptive blend of gradient credit and value differences.

beta = gradient weight, adapts based on predictor quality:
  - When predictor is bad (early training): beta=0.7 (trust gradients more)
  - When predictor is good (late training): beta=0.3 (trust value diffs more)
"""

import torch

from .base_gradient import BaseGradientCreditAlgo
from ..utils.gradient_utils import hybrid_credit_to_advantages


class HybridCreditAlgo(BaseGradientCreditAlgo):
    """Adaptive gradient + value difference blend."""

    algo_tag = "Hybrid"

    def _compute_credit(self, padded_obs, padded_act, mask):
        # Returns grad_x_input credit (signed)
        return self.predictor.get_gradient_credit(
            padded_obs, padded_act, mask, method='grad_x_input')

    def _credit_to_advantages(self, credit, episode_advantages, episodes, mask):
        # Also get value differences
        padded_obs, padded_act, mask_new = self._pad_episodes(episodes)
        with torch.no_grad():
            _, value_diffs, _ = self.predictor(padded_obs, mask_new, padded_act)

        # Adaptive beta based on predictor quality
        if self.initial_mse is not None and self.initial_mse > 0:
            pred_quality = 1.0 - min(1.0, (self.recent_mse or self.initial_mse) / self.initial_mse)
        else:
            pred_quality = 0.0
        beta = 0.7 - 0.4 * pred_quality  # 0.7 when bad -> 0.3 when good

        return hybrid_credit_to_advantages(
            credit, value_diffs, episode_advantages, episodes, mask,
            beta=beta, device=self.device)
