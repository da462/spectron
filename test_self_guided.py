#!/usr/bin/env python3
"""
Unit tests for self-guided linear layer implementation.

Run with: python test_self_guided.py
"""

import torch
import torch.nn as nn
from self_guided_linear import (
    LowRankWithSelfGuided,
    CosineTempDecay,
    replace_linear_with_selfguided
)


def test_alpha_decay():
    """Test alpha scheduler produces correct decay"""
    print("Testing alpha decay scheduler...")
    scheduler = CosineTempDecay(guided_steps=100, warmup_steps=10, device='cpu')

    # Test warmup phase (linear 0→1)
    alpha_0 = scheduler.get_alpha(0).item()
    alpha_5 = scheduler.get_alpha(5).item()
    alpha_10 = scheduler.get_alpha(10).item()

    assert abs(alpha_0 - 0.0) < 1e-6, f"Alpha at step 0 should be 0.0, got {alpha_0}"
    assert abs(alpha_5 - 0.5) < 1e-6, f"Alpha at step 5 should be 0.5, got {alpha_5}"
    assert abs(alpha_10 - 1.0) < 1e-6, f"Alpha at step 10 should be 1.0, got {alpha_10}"

    # Test cosine decay phase (1→0)
    alpha_55 = scheduler.get_alpha(55).item()  # Midpoint of decay
    alpha_100 = scheduler.get_alpha(100).item()
    alpha_150 = scheduler.get_alpha(150).item()

    # At midpoint, cosine should be ~0.5
    assert 0.4 < alpha_55 < 0.6, f"Alpha at midpoint should be ~0.5, got {alpha_55}"
    assert abs(alpha_100 - 0.0) < 1e-6, f"Alpha at step 100 should be 0.0, got {alpha_100}"
    assert abs(alpha_150 - 0.0) < 1e-6, f"Alpha after guided steps should be 0.0, got {alpha_150}"

    print("✓ Alpha decay test passed")


def test_forward_pass():
    """Test forward pass produces correct shapes and valid values"""
    print("Testing forward pass...")
    layer = LowRankWithSelfGuided(
        in_features=128,
        out_features=256,
        rank=32,
        reduce_flop=False
    )

    # Test with batch input
    x = torch.randn(4, 16, 128)  # (batch, seq_len, hidden)
    out = layer(x)

    assert out.shape == (4, 16, 256), f"Expected shape (4, 16, 256), got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN values"
    assert not torch.isinf(out).any(), "Output contains Inf values"

    print("✓ Forward pass test passed")


def test_svd_initialization():
    """Test SVD initialization preserves weight approximation"""
    print("Testing SVD initialization...")

    # Create original linear layer
    linear = nn.Linear(128, 256, bias=False)
    nn.init.xavier_normal_(linear.weight)

    # Create self-guided layer from linear
    sg_layer = LowRankWithSelfGuided.from_linear(
        linear, rank=64, method="svd", reduce_flop=False
    )

    # Test 1: Guide layer should exactly match original
    guide_error = torch.norm(sg_layer.guide_linear.weight - linear.weight)
    assert guide_error < 1e-5, f"Guide layer should match original, error: {guide_error}"

    # Test 2: Low-rank should approximate original
    lowrank_weight = sg_layer.lr2 @ sg_layer.lr1
    reconstruction_error = torch.norm(lowrank_weight - linear.weight) / torch.norm(linear.weight)

    # With rank 64 for 128→256, reconstruction should be reasonable
    assert reconstruction_error < 0.5, f"Reconstruction error too high: {reconstruction_error:.4f}"
    print(f"  Reconstruction error: {reconstruction_error:.4f}")
    print("✓ SVD initialization test passed")


def test_phase_transition():
    """Test phase transition disables guide layer"""
    print("Testing phase transition...")
    layer = LowRankWithSelfGuided(128, 256, 32, reduce_flop=False)

    # Check initial state (Phase 1)
    assert layer.self_guided_enabled.item() == True, "Should start enabled"
    assert layer.current_alpha.item() == 1.0, "Alpha should start at 1.0"

    # Transition to Phase 2
    layer.disable_self_guided()

    # Check final state (Phase 2)
    assert layer.self_guided_enabled.item() == False, "Should be disabled after transition"
    assert layer.current_alpha.item() == 0.0, "Alpha should be 0.0 after transition"

    # Test that forward pass still works in Phase 2
    x = torch.randn(2, 128)
    out = layer(x)
    assert out.shape == (2, 256), f"Phase 2 forward pass failed, shape: {out.shape}"

    print("✓ Phase transition test passed")


def test_alpha_update():
    """Test alpha update mechanism"""
    print("Testing alpha update...")
    layer = LowRankWithSelfGuided(128, 256, 32, reduce_flop=False)

    # Update alpha to different values
    test_values = [0.9, 0.7, 0.5, 0.3, 0.1, 0.0]
    for target_alpha in test_values:
        new_alpha = torch.tensor(target_alpha)
        layer.update_alpha(new_alpha)
        actual_alpha = layer.current_alpha.item()
        assert abs(actual_alpha - target_alpha) < 1e-6, \
            f"Alpha not updated correctly: expected {target_alpha}, got {actual_alpha}"

    print("✓ Alpha update test passed")


def test_reduce_flop_mode():
    """Test stochastic guide computation"""
    print("Testing reduce_flop mode...")

    # Create two identical layers: one with reduce_flop, one without
    layer_stochastic = LowRankWithSelfGuided(128, 256, 32, reduce_flop=True)
    layer_deterministic = LowRankWithSelfGuided(128, 256, 32, reduce_flop=False)

    # Copy weights to make them identical
    layer_stochastic.guide_linear.weight.data = layer_deterministic.guide_linear.weight.data.clone()
    layer_stochastic.lr1.data = layer_deterministic.lr1.data.clone()
    layer_stochastic.lr2.data = layer_deterministic.lr2.data.clone()
    if layer_stochastic.bias is not None:
        layer_stochastic.bias.data = layer_deterministic.bias.data.clone()

    # Set same alpha
    alpha = torch.tensor(0.5)
    layer_stochastic.update_alpha(alpha)
    layer_deterministic.update_alpha(alpha)

    # Test eval mode (both should be deterministic)
    layer_stochastic.eval()
    layer_deterministic.eval()

    x = torch.randn(4, 16, 128)
    out_stochastic = layer_stochastic(x)
    out_deterministic = layer_deterministic(x)

    # In eval mode, outputs should match exactly
    diff = torch.norm(out_stochastic - out_deterministic)
    assert diff < 1e-5, f"Outputs should match in eval mode, diff: {diff}"

    print("✓ reduce_flop mode test passed")


def test_checkpoint_state():
    """Test state_dict saves and restores buffers correctly"""
    print("Testing checkpoint state...")
    layer = LowRankWithSelfGuided(128, 256, 32)

    # Modify state
    layer.update_alpha(torch.tensor(0.3))
    layer.current_step.fill_(100)

    # Save state
    state = layer.state_dict()

    # Create new layer and load state
    new_layer = LowRankWithSelfGuided(128, 256, 32)
    new_layer.load_state_dict(state)

    # Verify state restoration
    assert abs(new_layer.current_alpha.item() - 0.3) < 1e-6, \
        f"Alpha not restored from checkpoint: {new_layer.current_alpha.item()}"
    assert new_layer.current_step.item() == 100, \
        f"Step not restored from checkpoint: {new_layer.current_step.item()}"

    # Test phase state restoration
    layer.disable_self_guided()
    state = layer.state_dict()
    new_layer.load_state_dict(state)

    assert new_layer.self_guided_enabled.item() == False, "Phase state not restored"
    assert new_layer.current_alpha.item() == 0.0, "Alpha not restored after phase transition"

    print("✓ Checkpoint state test passed")


def test_replace_function():
    """Test replace_linear_with_selfguided function"""
    print("Testing replace_linear_with_selfguided...")

    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(128, 256)
            self.fc2 = nn.Linear(256, 512)
            self.fc3 = nn.Linear(512, 128)

        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
            return x

    model = SimpleModel()
    original_fc1_weight = model.fc1.weight.data.clone()

    # Replace linear layers
    model = replace_linear_with_selfguided(
        model,
        rank_ratio=0.25,
        method="svd",
        reduce_flop=True
    )

    # Check that layers were replaced
    assert isinstance(model.fc1, LowRankWithSelfGuided), "fc1 not replaced"
    assert isinstance(model.fc2, LowRankWithSelfGuided), "fc2 not replaced"
    assert isinstance(model.fc3, LowRankWithSelfGuided), "fc3 not replaced"

    # Check ranks
    expected_rank_fc1 = int(0.25 * 128)  # rank = 0.25 * in_features
    assert model.fc1.rank == expected_rank_fc1, \
        f"fc1 rank incorrect: expected {expected_rank_fc1}, got {model.fc1.rank}"

    # Test forward pass
    x = torch.randn(2, 128)
    out = model(x)
    assert out.shape == (2, 128), f"Model forward pass failed, shape: {out.shape}"

    print("✓ replace_linear_with_selfguided test passed")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Running Self-Guided Linear Layer Tests")
    print("=" * 60 + "\n")

    try:
        test_alpha_decay()
        test_forward_pass()
        test_svd_initialization()
        test_phase_transition()
        test_alpha_update()
        test_reduce_flop_mode()
        test_checkpoint_state()
        test_replace_function()

        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
        return True
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
