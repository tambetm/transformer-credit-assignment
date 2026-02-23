"""Integrated Gradients Credit (Variant B): averaged gradients along interpolation path.

More principled than simple gradient norm — satisfies the completeness axiom:
sum(IG) = f(input) - f(baseline). ~10x more expensive due to interpolation steps.
"""

from .base_gradient import BaseGradientCreditAlgo
from ..utils.gradient_utils import signed_credit_to_advantages


class IntegratedGradientsCreditAlgo(BaseGradientCreditAlgo):
    """Integrated gradients for credit assignment."""

    algo_tag = "IG"

    def _compute_credit(self, padded_obs, padded_act, mask):
        n_steps = self.config.get("integrated_grad_steps", 10)
        return self.predictor.get_gradient_credit(
            padded_obs, padded_act, mask, method='integrated_grads',
            n_ig_steps=n_steps)

    def _credit_to_advantages(self, credit, episode_advantages, episodes, mask):
        return signed_credit_to_advantages(
            credit, episode_advantages, episodes, mask, self.device)
