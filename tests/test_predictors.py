"""Unit tests for return predictors.

Tests:
1. Causal masking: modifying input at position t+k doesn't affect predictions at t
2. Telescoping: sum of value differences equals final prediction minus initial
3. Gradient isolation: policy loss backward doesn't put gradients on predictor params
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from grpo_attention.models.rudder_transformer import RUDDERTransformer
from grpo_attention.models.lstm_predictor import LSTMReturnPredictor
from grpo_attention.models.policy import PolicyNetwork


def test_causal_masking_transformer():
    """Verify that modifying input at position t+k doesn't change predictions at positions <= t."""
    torch.manual_seed(42)
    obs_dim, action_dim, seq_len = 4, 2, 20
    batch_size = 1

    model = RUDDERTransformer(obs_dim, action_dim=action_dim, d_model=32, nhead=2,
                               num_layers=2, d_ff=64)
    model.eval()

    # Create input
    states = torch.randn(batch_size, seq_len, obs_dim)
    actions = torch.randint(0, action_dim, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    # Get predictions with original input
    with torch.no_grad():
        _, _, pred_orig = model(states, mask, actions)

    # Modify input at position t=15, check positions 0..14 unchanged
    modify_pos = 15
    states_modified = states.clone()
    states_modified[0, modify_pos:] = torch.randn(seq_len - modify_pos, obs_dim) * 10
    actions_modified = actions.clone()
    actions_modified[0, modify_pos:] = (actions[0, modify_pos:] + 1) % action_dim

    with torch.no_grad():
        _, _, pred_modified = model(states_modified, mask, actions_modified)

    # Predictions at positions < modify_pos should be identical
    diff_before = (pred_orig[0, :modify_pos] - pred_modified[0, :modify_pos]).abs().max().item()
    # Predictions at positions >= modify_pos can differ
    diff_after = (pred_orig[0, modify_pos:] - pred_modified[0, modify_pos:]).abs().max().item()

    assert diff_before < 1e-5, f"Causal masking violated! Diff before modify_pos: {diff_before}"
    assert diff_after > 1e-5, f"Predictions after modify_pos should differ: {diff_after}"
    print(f"  PASS: Causal masking correct. Diff before={diff_before:.2e}, after={diff_after:.4f}")


def test_causal_masking_lstm():
    """Verify LSTM causality: modifying input at t+k doesn't change predictions at t."""
    torch.manual_seed(42)
    obs_dim, action_dim, seq_len = 4, 2, 20
    batch_size = 1

    model = LSTMReturnPredictor(obs_dim, action_dim=action_dim, hidden_size=64)
    model.eval()

    states = torch.randn(batch_size, seq_len, obs_dim)
    actions = torch.randint(0, action_dim, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    with torch.no_grad():
        _, _, pred_orig = model(states, mask, actions)

    modify_pos = 15
    states_modified = states.clone()
    states_modified[0, modify_pos:] = torch.randn(seq_len - modify_pos, obs_dim) * 10
    actions_modified = actions.clone()
    actions_modified[0, modify_pos:] = (actions[0, modify_pos:] + 1) % action_dim

    with torch.no_grad():
        _, _, pred_modified = model(states_modified, mask, actions_modified)

    diff_before = (pred_orig[0, :modify_pos] - pred_modified[0, :modify_pos]).abs().max().item()
    diff_after = (pred_orig[0, modify_pos:] - pred_modified[0, modify_pos:]).abs().max().item()

    assert diff_before < 1e-5, f"LSTM causality violated! Diff before: {diff_before}"
    assert diff_after > 1e-5, f"LSTM predictions after modify_pos should differ: {diff_after}"
    print(f"  PASS: LSTM causality correct. Diff before={diff_before:.2e}, after={diff_after:.4f}")


def test_telescoping_transformer():
    """Verify that sum of value differences equals final prediction minus initial (0)."""
    torch.manual_seed(42)
    obs_dim, action_dim, seq_len = 4, 2, 30
    batch_size = 3

    model = RUDDERTransformer(obs_dim, action_dim=action_dim, d_model=32, nhead=2,
                               num_layers=2, d_ff=64)
    model.eval()

    states = torch.randn(batch_size, seq_len, obs_dim)
    actions = torch.randint(0, action_dim, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    # Make episode 2 shorter
    mask[2, 20:] = False

    with torch.no_grad():
        _, value_diffs, pred_normalized = model(states, mask, actions)

    for i in range(batch_size):
        T = mask[i].sum().item()
        diff_sum = value_diffs[i, :T].sum().item()
        final_pred = pred_normalized[i, T - 1].item()
        # g_{-1} = 0, so sum of diffs should equal g_{T-1} - 0 = g_{T-1}
        error = abs(diff_sum - final_pred)
        assert error < 1e-4, (
            f"Telescoping failed for episode {i}! "
            f"Sum of diffs={diff_sum:.6f}, final pred={final_pred:.6f}, error={error:.2e}"
        )

    print(f"  PASS: Telescoping property holds for all {batch_size} episodes (max error < 1e-4)")


def test_telescoping_lstm():
    """Verify telescoping for LSTM predictor."""
    torch.manual_seed(42)
    obs_dim, action_dim, seq_len = 4, 2, 30
    batch_size = 3

    model = LSTMReturnPredictor(obs_dim, action_dim=action_dim, hidden_size=64)
    model.eval()

    states = torch.randn(batch_size, seq_len, obs_dim)
    actions = torch.randint(0, action_dim, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[2, 20:] = False

    with torch.no_grad():
        _, value_diffs, pred_normalized = model(states, mask, actions)

    for i in range(batch_size):
        T = mask[i].sum().item()
        diff_sum = value_diffs[i, :T].sum().item()
        final_pred = pred_normalized[i, T - 1].item()
        error = abs(diff_sum - final_pred)
        assert error < 1e-4, (
            f"LSTM telescoping failed for episode {i}! "
            f"Sum of diffs={diff_sum:.6f}, final pred={final_pred:.6f}, error={error:.2e}"
        )

    print(f"  PASS: LSTM telescoping property holds for all {batch_size} episodes")


def test_gradient_isolation_transformer():
    """Verify that policy_loss.backward() puts zero gradient on predictor parameters."""
    torch.manual_seed(42)
    obs_dim, action_dim, seq_len = 4, 2, 10
    batch_size = 2

    predictor = RUDDERTransformer(obs_dim, action_dim=action_dim, d_model=32, nhead=2,
                                   num_layers=2, d_ff=64)
    policy = PolicyNetwork(obs_dim, action_dim)

    states = torch.randn(batch_size, seq_len, obs_dim)
    actions = torch.randint(0, action_dim, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    # Get value diffs (DETACHED)
    _, value_diffs, _ = predictor(states, mask, actions)
    advantages = value_diffs.detach()  # Critical: detach!

    # Compute policy loss using these advantages
    flat_obs = states.reshape(-1, obs_dim)
    flat_actions = actions.reshape(-1)
    flat_advantages = advantages.reshape(-1)

    log_probs, _ = policy.evaluate_actions(flat_obs, flat_actions)
    policy_loss = -(log_probs * flat_advantages).mean()

    # Zero all gradients
    predictor.zero_grad()
    policy.zero_grad()

    # Backward
    policy_loss.backward()

    # Check: predictor params should have NO gradient (or zero gradient)
    for name, param in predictor.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.abs().max().item()
            assert grad_norm == 0.0, (
                f"Gradient isolation failed! Predictor param '{name}' has grad norm {grad_norm}"
            )

    # Check: policy params SHOULD have gradients
    policy_has_grad = False
    for name, param in policy.named_parameters():
        if param.grad is not None and param.grad.abs().max().item() > 0:
            policy_has_grad = True
            break
    assert policy_has_grad, "Policy should have non-zero gradients!"

    print("  PASS: Gradient isolation correct — predictor grads are zero, policy grads are non-zero")


def test_gradient_isolation_lstm():
    """Verify gradient isolation for LSTM predictor."""
    torch.manual_seed(42)
    obs_dim, action_dim, seq_len = 4, 2, 10
    batch_size = 2

    predictor = LSTMReturnPredictor(obs_dim, action_dim=action_dim, hidden_size=64)
    policy = PolicyNetwork(obs_dim, action_dim)

    states = torch.randn(batch_size, seq_len, obs_dim)
    actions = torch.randint(0, action_dim, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    _, value_diffs, _ = predictor(states, mask, actions)
    advantages = value_diffs.detach()

    flat_obs = states.reshape(-1, obs_dim)
    flat_actions = actions.reshape(-1)
    flat_advantages = advantages.reshape(-1)

    log_probs, _ = policy.evaluate_actions(flat_obs, flat_actions)
    policy_loss = -(log_probs * flat_advantages).mean()

    predictor.zero_grad()
    policy.zero_grad()
    policy_loss.backward()

    for name, param in predictor.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.abs().max().item()
            assert grad_norm == 0.0, (
                f"LSTM gradient isolation failed! Param '{name}' has grad norm {grad_norm}"
            )

    policy_has_grad = any(
        p.grad is not None and p.grad.abs().max().item() > 0
        for p in policy.parameters()
    )
    assert policy_has_grad, "Policy should have non-zero gradients!"

    print("  PASS: LSTM gradient isolation correct")


def test_padding_mask():
    """Verify that padded positions have zero value diffs."""
    torch.manual_seed(42)
    obs_dim, action_dim = 4, 2
    batch_size, seq_len = 2, 20

    for ModelClass, name in [
        (RUDDERTransformer, "Transformer"),
        (LSTMReturnPredictor, "LSTM"),
    ]:
        if name == "Transformer":
            model = ModelClass(obs_dim, action_dim=action_dim, d_model=32, nhead=2,
                               num_layers=2, d_ff=64)
        else:
            model = ModelClass(obs_dim, action_dim=action_dim, hidden_size=64)
        model.eval()

        states = torch.randn(batch_size, seq_len, obs_dim)
        actions = torch.randint(0, action_dim, (batch_size, seq_len))
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        mask[0, 12:] = False  # Episode 0 has length 12
        mask[1, 8:] = False   # Episode 1 has length 8

        with torch.no_grad():
            _, value_diffs, _ = model(states, mask, actions)

        # Padded positions should have zero value diffs
        for i in range(batch_size):
            T = mask[i].sum().item()
            padded_diffs = value_diffs[i, T:].abs().max().item()
            assert padded_diffs < 1e-7, (
                f"{name}: Padded value diffs should be zero, got {padded_diffs}"
            )

    print("  PASS: Padding mask works correctly for both Transformer and LSTM")


if __name__ == "__main__":
    print("=" * 60)
    print("Running unit tests for return predictors")
    print("=" * 60)

    print("\n1. Causal masking tests:")
    test_causal_masking_transformer()
    test_causal_masking_lstm()

    print("\n2. Telescoping property tests:")
    test_telescoping_transformer()
    test_telescoping_lstm()

    print("\n3. Gradient isolation tests:")
    test_gradient_isolation_transformer()
    test_gradient_isolation_lstm()

    print("\n4. Padding mask tests:")
    test_padding_mask()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
