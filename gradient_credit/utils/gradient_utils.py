"""Gradient computation helpers for credit assignment.

Shared by all gradient credit algorithm variants.
"""

import torch
import numpy as np


def unsigned_credit_to_advantages(credit_weights, episode_advantages, episodes, mask, device="cpu"):
    """Convert unsigned gradient credit (e.g., grad_norm) to per-timestep advantages.

    credit_weights: (B, T) non-negative
    episode_advantages: list of floats, one per episode
    episodes: list of Episode objects
    mask: (B, T) bool

    Returns: list of tensors, one per episode, concatenatable for policy gradient.
    """
    all_obs = []
    all_actions = []
    all_advantages = []

    B = len(episodes)

    for i, episode in enumerate(episodes):
        T = episode.length
        tensors = episode.to_tensors(device)

        # Normalize credit within this episode
        ep_credit = credit_weights[i, :T]
        credit_sum = ep_credit.sum().clamp(min=1e-8)
        w_t = ep_credit / credit_sum  # sums to 1

        # Per-timestep advantage: w_t * T * A_episode
        per_step_adv = w_t * T * episode_advantages[i]

        all_obs.append(tensors["observations"])
        all_actions.append(tensors["actions"])
        all_advantages.append(per_step_adv)

    return all_obs, all_actions, all_advantages


def signed_credit_to_advantages(credit_weights, episode_advantages, episodes, mask, device="cpu"):
    """Convert signed gradient credit (grad_x_input, IG) to per-timestep advantages.

    Signed credit: positive means this input pushes prediction UP.
    On a good episode (A > 0): positive credit -> positive advantage.
    On a bad episode (A < 0): positive credit -> negative advantage.
    Natural element-wise product gives correct sign handling.

    credit_weights: (B, T) signed
    episode_advantages: list of floats
    """
    all_obs = []
    all_actions = []
    all_advantages = []

    for i, episode in enumerate(episodes):
        T = episode.length
        tensors = episode.to_tensors(device)

        ep_credit = credit_weights[i, :T]

        # Scale credit to have unit norm per episode, preserving sign
        credit_scale = ep_credit.abs().sum().clamp(min=1e-8)
        normalized_credit = ep_credit / credit_scale * T

        # Element-wise: credit direction * episode quality
        per_step_adv = normalized_credit * episode_advantages[i]

        all_obs.append(tensors["observations"])
        all_actions.append(tensors["actions"])
        all_advantages.append(per_step_adv)

    return all_obs, all_actions, all_advantages


def value_diff_to_advantages(value_diffs, episodes, mask, device="cpu"):
    """Convert value differences to per-timestep advantages (RUDDER-style).

    value_diffs: (B, T) — V_norm(t) - V_norm(t-1)
    """
    all_obs = []
    all_actions = []
    all_advantages = []

    for i, episode in enumerate(episodes):
        T = episode.length
        tensors = episode.to_tensors(device)
        per_step_adv = value_diffs[i, :T]

        all_obs.append(tensors["observations"])
        all_actions.append(tensors["actions"])
        all_advantages.append(per_step_adv)

    return all_obs, all_actions, all_advantages


def hybrid_credit_to_advantages(gradient_credit, value_diffs, episode_advantages,
                                episodes, mask, beta, device="cpu"):
    """Adaptive blend of gradient credit and value differences.

    A_t = beta * gradient_advantage_t + (1 - beta) * value_diff_t

    gradient_credit: (B, T) signed credit
    value_diffs: (B, T) value differences
    beta: float, gradient weight (0.7 when predictor bad -> 0.3 when good)
    """
    all_obs = []
    all_actions = []
    all_advantages = []

    for i, episode in enumerate(episodes):
        T = episode.length
        tensors = episode.to_tensors(device)

        # Gradient component (signed)
        ep_credit = gradient_credit[i, :T]
        credit_scale = ep_credit.abs().sum().clamp(min=1e-8)
        normalized_credit = ep_credit / credit_scale * T
        grad_adv = normalized_credit * episode_advantages[i]

        # Value diff component
        vd_adv = value_diffs[i, :T]

        # Blend
        per_step_adv = beta * grad_adv + (1 - beta) * vd_adv

        all_obs.append(tensors["observations"])
        all_actions.append(tensors["actions"])
        all_advantages.append(per_step_adv)

    return all_obs, all_actions, all_advantages


def gsvd_credit_to_advantages(gradient_saliency, value_diffs, episodes, mask,
                               temperature, device="cpu"):
    """Gradient-Sharpened Value Differences (GSVD).

    Uses gradient saliency (||dg/d_emb_t||_2) as softmax weights to sharpen
    value differences. Higher saliency = model is more sensitive to this timestep
    = credit assignment is more reliable here.

    A_t = vd_t * softmax(saliency_t / tau) * T

    When tau -> inf: weights -> uniform -> reduces to pure value diffs.
    When tau -> 0: weights concentrate on highest-saliency timestep.

    gradient_saliency: (B, T) non-negative (from grad_norm)
    value_diffs: (B, T) signed value differences
    temperature: float, controls sharpening strength
    """
    all_obs = []
    all_actions = []
    all_advantages = []

    for i, episode in enumerate(episodes):
        T = episode.length
        tensors = episode.to_tensors(device)

        vd = value_diffs[i, :T]
        sal = gradient_saliency[i, :T]

        # Softmax sharpening: weights sum to T (mean weight = 1)
        weights = torch.softmax(sal / max(temperature, 1e-6), dim=0) * T

        per_step_adv = vd * weights

        all_obs.append(tensors["observations"])
        all_actions.append(tensors["actions"])
        all_advantages.append(per_step_adv)

    return all_obs, all_actions, all_advantages


def attention_rollout_credit_to_advantages(rollout_credit, value_diffs, episodes,
                                           mask, device="cpu"):
    """Attention Rollout Credit: combine rollout weights with value differences.

    Uses attention rollout (multi-layer information flow) to weight value diffs.
    Higher rollout flow = model's final prediction depended more on this timestep
    = value difference here is more meaningful.

    A_t = vd_t * (rollout_t / mean(rollout_t))

    When rollout is uniform: reduces to pure value diffs.

    rollout_credit: (B, T) non-negative attention flow weights
    value_diffs: (B, T) signed value differences
    """
    all_obs = []
    all_actions = []
    all_advantages = []

    for i, episode in enumerate(episodes):
        T = episode.length
        tensors = episode.to_tensors(device)

        vd = value_diffs[i, :T]
        rollout = rollout_credit[i, :T]

        # Normalize to mean 1 so uniform rollout = pure value diffs
        rollout_mean = rollout.mean().clamp(min=1e-8)
        rollout_normalized = rollout / rollout_mean

        per_step_adv = vd * rollout_normalized

        all_obs.append(tensors["observations"])
        all_actions.append(tensors["actions"])
        all_advantages.append(per_step_adv)

    return all_obs, all_actions, all_advantages


def compute_credit_stats(credit, mask):
    """Compute statistics about credit distribution for logging."""
    valid = credit[mask] if mask is not None else credit.reshape(-1)
    if valid.numel() == 0:
        return {"credit_mean": 0.0, "credit_std": 0.0, "credit_sparsity": 0.0}

    abs_credit = valid.abs()
    # Sparsity: fraction of total credit in top 10% of positions
    k = max(1, int(0.1 * valid.numel()))
    topk_vals, _ = abs_credit.topk(k)
    sparsity = topk_vals.sum() / abs_credit.sum().clamp(min=1e-8)

    return {
        "credit_mean": valid.mean().item(),
        "credit_std": valid.std().item() if valid.numel() > 1 else 0.0,
        "credit_sparsity": sparsity.item(),
    }
