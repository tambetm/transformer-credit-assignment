"""Gradient x Input Credit (Variant C): signed (dg/d_embedded_t . embedded_t).

Preserves sign: positive = this input pushes prediction up, negative = down.
Natural element-wise product with episode advantage gives correct sign handling.
"""

from .base_gradient import BaseGradientCreditAlgo
from ..utils.gradient_utils import signed_credit_to_advantages


class GradientInputCreditAlgo(BaseGradientCreditAlgo):
    """Gradient x input signed credit for credit assignment."""

    algo_tag = "GradInput"

    def _compute_credit(self, padded_obs, padded_act, mask):
        return self.predictor.get_gradient_credit(
            padded_obs, padded_act, mask, method='grad_x_input')

    def _credit_to_advantages(self, credit, episode_advantages, episodes, mask):
        return signed_credit_to_advantages(
            credit, episode_advantages, episodes, mask, self.device)
