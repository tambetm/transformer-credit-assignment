"""Exponential Moving Average tracker for return statistics.

Used by REINFORCE+EMA and GRPO+Attention for baseline normalization:
    A_episode = (G - mu_EMA) / sigma_EMA
"""


class EMAStats:
    """Track running mean and variance of episode returns via EMA."""

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        self.mean = 0.0
        self.var = 1.0
        self.initialized = False

    def update(self, value: float):
        if not self.initialized:
            self.mean = value
            self.var = 1.0  # Start with unit variance
            self.initialized = True
            return

        delta = value - self.mean
        self.mean += self.alpha * delta
        self.var = (1 - self.alpha) * (self.var + self.alpha * delta ** 2)

    def normalize(self, value: float) -> float:
        """Return normalized advantage: (value - mean) / std."""
        std = max(self.var ** 0.5, 1e-8)
        return (value - self.mean) / std

    def update_batch(self, values):
        """Update with a batch of values."""
        for v in values:
            self.update(v)
