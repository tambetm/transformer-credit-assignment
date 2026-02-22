"""Sparse reward wrapper for CartPole-v1.

Gives reward ONLY at episode termination equal to the total steps survived.
All intermediate rewards are zero.
"""

import gymnasium as gym


class SparseCartPole(gym.Wrapper):
    """CartPole but reward is only given at episode termination (= total steps survived)."""

    def __init__(self, env):
        super().__init__(env)
        self.step_count = 0

    def reset(self, **kwargs):
        self.step_count = 0
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        if terminated or truncated:
            return obs, float(self.step_count), terminated, truncated, info
        return obs, 0.0, terminated, truncated, info


def make_sparse_cartpole(**kwargs):
    """Create a SparseCartPole environment."""
    env = gym.make("CartPole-v1", **kwargs)
    return SparseCartPole(env)
