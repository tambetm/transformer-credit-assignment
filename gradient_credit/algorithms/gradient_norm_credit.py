"""Gradient Norm Credit (Variant A): ||dg/d_embedded_t||_2 as credit weight.

Unsigned saliency — measures how sensitive the return prediction is to
each timestep's embedding, regardless of direction.
"""

from .base_gradient import BaseGradientCreditAlgo
from ..utils.gradient_utils import unsigned_credit_to_advantages


class GradientNormCreditAlgo(BaseGradientCreditAlgo):
    """Gradient norm saliency for credit assignment."""

    algo_tag = "GradNorm"

    def _compute_credit(self, padded_obs, padded_act, mask):
        return self.predictor.get_gradient_credit(
            padded_obs, padded_act, mask, method='grad_norm')

    def _credit_to_advantages(self, credit, episode_advantages, episodes, mask):
        return unsigned_credit_to_advantages(
            credit, episode_advantages, episodes, mask, self.device)
