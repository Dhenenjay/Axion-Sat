"""
Unit Tests for LoRA (Low-Rank Adaptation)

Tests verify:
1. LoRA reduces trainable parameters significantly
2. Base model parameters remain frozen during training
3. Only LoRA parameters are trainable
4. Parameter counts are as expected

Author: Axion-Sat Project
Version: 1.0.0
"""

import sys
from pathlib import Path
import unittest
from copy import deepcopy

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn


# ============================================================================
# Mock Models for Testing
# ============================================================================

class SimpleLinearModel(nn.Module):
    """Simple model with linear layers for testing."""
    
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x


class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer.
    
    Adds trainable low-rank matrices A and B to a frozen linear layer:
        output = W_frozen @ x + (B @ A) @ x
    
    where A: (r, in_features), B: (out_features, r), r << min(in, out)
    """
    
    def __init__(self, linear_layer: nn.Linear, rank: int = 4, alpha: float = 1.0):
        """
        Args:
            linear_layer: Frozen linear layer to adapt
            rank: Rank of adaptation matrices (r)
            alpha: Scaling factor for LoRA output
        """
        super().__init__()
        
        self.linear = linear_layer
        self.rank = rank
        self.alpha = alpha
        
        # Freeze original weights
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
        
        # LoRA matrices: A (down-projection), B (up-projection)
        self.lora_A = nn.Parameter(torch.zeros(rank, linear_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(linear_layer.out_features, rank))
        
        # Initialize A with small random values, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
        
        # Scaling factor
        self.scaling = alpha / rank
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original frozen output
        result = self.linear(x)
        
        # Add LoRA adaptation: x @ A^T @ B^T
        lora_output = (x @ self.lora_A.T) @ self.lora_B.T
        result = result + lora_output * self.scaling
        
        return result


def apply_lora_to_model(model: nn.Module, rank: int = 4, alpha: float = 1.0) -> nn.Module:
    """
    Apply LoRA to all Linear layers in a model.
    
    Args:
        model: Model to adapt
        rank: LoRA rank
        alpha: LoRA alpha scaling
        
    Returns:
        Model with LoRA layers applied
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Replace linear layer with LoRA layer
            setattr(model, name, LoRALayer(module, rank=rank, alpha=alpha))
        else:
            # Recursively apply to submodules
            apply_lora_to_model(module, rank=rank, alpha=alpha)
    
    return model


# ============================================================================
# Unit Tests
# ============================================================================

class TestLoRAParameterCounts(unittest.TestCase):
    """Test LoRA parameter counts and reduction."""
    
    def setUp(self):
        """Set up test models."""
        self.model = SimpleLinearModel(input_dim=128, hidden_dim=256, output_dim=128)
        self.lora_rank = 4
    
    def test_original_model_params(self):
        """Test that we can count parameters in the original model."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # All parameters should be trainable initially
        self.assertEqual(total_params, trainable_params)
        
        # Expected params:
        # fc1: 128*256 + 256 = 33,024
        # fc2: 256*256 + 256 = 65,792
        # fc3: 256*128 + 128 = 32,896
        # Total: 131,712
        expected_params = (128 * 256 + 256) + (256 * 256 + 256) + (256 * 128 + 128)
        self.assertEqual(total_params, expected_params)
        
        print(f"\n✓ Original model has {total_params:,} parameters (all trainable)")
    
    def test_lora_reduces_trainable_params(self):
        """Test that LoRA significantly reduces trainable parameters."""
        # Apply LoRA
        lora_model = apply_lora_to_model(deepcopy(self.model), rank=self.lora_rank)
        
        total_params = sum(p.numel() for p in lora_model.parameters())
        trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        # LoRA trainable params for each layer:
        # fc1: rank * (in_features + out_features) = 4 * (128 + 256) = 1,536
        # fc2: rank * (in_features + out_features) = 4 * (256 + 256) = 2,048
        # fc3: rank * (in_features + out_features) = 4 * (256 + 128) = 1,536
        # Total LoRA params: 5,120
        expected_lora_params = (
            self.lora_rank * (128 + 256) +  # fc1
            self.lora_rank * (256 + 256) +  # fc2
            self.lora_rank * (256 + 128)    # fc3
        )
        
        self.assertEqual(trainable_params, expected_lora_params)
        
        # Calculate reduction
        original_trainable = sum(p.numel() for p in self.model.parameters())
        reduction_ratio = trainable_params / original_trainable
        
        self.assertLess(reduction_ratio, 0.1)  # Should be < 10% of original
        
        print(f"\n✓ LoRA model statistics:")
        print(f"  Total params:     {total_params:,}")
        print(f"  Frozen params:    {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
        print(f"  Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        print(f"  Reduction ratio:  {reduction_ratio:.2%} of original")
    
    def test_parameter_shapes(self):
        """Test that LoRA parameter shapes are correct."""
        lora_model = apply_lora_to_model(deepcopy(self.model), rank=self.lora_rank)
        
        # Check fc1 LoRA layer
        fc1_lora = lora_model.fc1
        self.assertEqual(fc1_lora.lora_A.shape, (self.lora_rank, 128))  # (rank, in_features)
        self.assertEqual(fc1_lora.lora_B.shape, (256, self.lora_rank))  # (out_features, rank)
        
        # Check fc2 LoRA layer
        fc2_lora = lora_model.fc2
        self.assertEqual(fc2_lora.lora_A.shape, (self.lora_rank, 256))
        self.assertEqual(fc2_lora.lora_B.shape, (256, self.lora_rank))
        
        # Check fc3 LoRA layer
        fc3_lora = lora_model.fc3
        self.assertEqual(fc3_lora.lora_A.shape, (self.lora_rank, 256))
        self.assertEqual(fc3_lora.lora_B.shape, (128, self.lora_rank))
        
        print(f"\n✓ All LoRA parameter shapes are correct")


class TestFrozenParameters(unittest.TestCase):
    """Test that frozen parameters stay frozen during training."""
    
    def setUp(self):
        """Set up test models."""
        self.model = SimpleLinearModel(input_dim=128, hidden_dim=256, output_dim=128)
        self.lora_model = apply_lora_to_model(deepcopy(self.model), rank=4)
        self.device = torch.device('cpu')
    
    def test_frozen_params_marked_correctly(self):
        """Test that base model parameters are marked as non-trainable."""
        for name, param in self.lora_model.named_parameters():
            if 'lora_' in name:
                # LoRA parameters should be trainable
                self.assertTrue(param.requires_grad, f"{name} should be trainable")
            else:
                # Base model parameters should be frozen
                self.assertFalse(param.requires_grad, f"{name} should be frozen")
        
        print(f"\n✓ All parameters have correct requires_grad flags")
    
    def test_frozen_params_unchanged_after_training_step(self):
        """Test that frozen parameters don't change after a training step."""
        # Store initial frozen parameter values
        frozen_params_before = {}
        for name, param in self.lora_model.named_parameters():
            if not param.requires_grad:  # Frozen parameters
                frozen_params_before[name] = param.data.clone()
        
        # Create dummy data
        batch_size = 16
        x = torch.randn(batch_size, 128)
        target = torch.randn(batch_size, 128)
        
        # Training step
        optimizer = torch.optim.Adam(
            [p for p in self.lora_model.parameters() if p.requires_grad],
            lr=0.001
        )
        
        # Forward pass
        output = self.lora_model(x)
        loss = nn.functional.mse_loss(output, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check that frozen parameters haven't changed
        for name, param in self.lora_model.named_parameters():
            if not param.requires_grad:  # Frozen parameters
                param_before = frozen_params_before[name]
                param_after = param.data
                
                # Parameters should be exactly the same
                self.assertTrue(
                    torch.equal(param_before, param_after),
                    f"Frozen parameter {name} changed during training!"
                )
        
        print(f"\n✓ Frozen parameters unchanged after training step")
        print(f"  Checked {len(frozen_params_before)} frozen parameters")
    
    def test_lora_params_updated_after_training_step(self):
        """Test that LoRA parameters DO change after a training step."""
        # Store initial LoRA parameter values
        lora_params_before = {}
        for name, param in self.lora_model.named_parameters():
            if param.requires_grad and 'lora_' in name:  # LoRA parameters
                lora_params_before[name] = param.data.clone()
        
        # Create dummy data
        batch_size = 16
        x = torch.randn(batch_size, 128)
        target = torch.randn(batch_size, 128)
        
        # Training step
        optimizer = torch.optim.Adam(
            [p for p in self.lora_model.parameters() if p.requires_grad],
            lr=0.001
        )
        
        # Forward pass
        output = self.lora_model(x)
        loss = nn.functional.mse_loss(output, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check that LoRA parameters HAVE changed
        params_changed = 0
        for name, param in self.lora_model.named_parameters():
            if param.requires_grad and 'lora_' in name:  # LoRA parameters
                param_before = lora_params_before[name]
                param_after = param.data
                
                # Parameters should be different
                if not torch.equal(param_before, param_after):
                    params_changed += 1
        
        # At least some LoRA parameters should have changed
        self.assertGreater(params_changed, 0, "LoRA parameters did not update!")
        
        print(f"\n✓ LoRA parameters updated after training step")
        print(f"  {params_changed}/{len(lora_params_before)} LoRA parameters changed")
    
    def test_gradients_only_for_trainable_params(self):
        """Test that gradients are only computed for trainable parameters."""
        # Create dummy data
        batch_size = 16
        x = torch.randn(batch_size, 128)
        target = torch.randn(batch_size, 128)
        
        # Forward pass
        output = self.lora_model(x)
        loss = nn.functional.mse_loss(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        trainable_with_grad = 0
        trainable_total = 0
        
        for name, param in self.lora_model.named_parameters():
            if param.requires_grad:
                trainable_total += 1
                # Trainable parameters should have gradients
                self.assertIsNotNone(param.grad, f"Trainable param {name} has no gradient")
                # Some gradients might be zero if they don't contribute to loss
                # but they should at least be allocated
                if param.grad is not None and not torch.all(param.grad == 0):
                    trainable_with_grad += 1
            else:
                # Frozen parameters should not have gradients
                self.assertTrue(
                    param.grad is None or torch.all(param.grad == 0),
                    f"Frozen param {name} has non-zero gradient"
                )
        
        # At least some trainable parameters should have non-zero gradients
        self.assertGreater(trainable_with_grad, 0, "No trainable parameters have non-zero gradients")
        
        print(f"\n✓ Gradients computed only for trainable parameters")
        print(f"  {trainable_with_grad}/{trainable_total} trainable params have non-zero gradients")


class TestLoRAForwardPass(unittest.TestCase):
    """Test LoRA forward pass behavior."""
    
    def test_lora_changes_output(self):
        """Test that LoRA adaptation changes model output."""
        model = SimpleLinearModel(input_dim=128, hidden_dim=256, output_dim=128)
        lora_model = apply_lora_to_model(deepcopy(model), rank=4)
        
        # Create dummy input
        x = torch.randn(16, 128)
        
        # Get outputs
        with torch.no_grad():
            output_original = model(x)
            output_lora = lora_model(x)
        
        # Initially, outputs should be the same (LoRA B is initialized to zeros)
        self.assertTrue(
            torch.allclose(output_original, output_lora, atol=1e-6),
            "Initial LoRA output should match original"
        )
        
        # Manually modify LoRA parameters with larger values
        for module in lora_model.modules():
            if isinstance(module, LoRALayer):
                # Use larger standard deviation to ensure visible change
                nn.init.normal_(module.lora_A, mean=0, std=0.5)
                nn.init.normal_(module.lora_B, mean=0, std=0.5)
        
        # Now outputs should differ
        with torch.no_grad():
            output_lora_modified = lora_model(x)
        
        # Check that outputs are different
        diff = torch.abs(output_original - output_lora_modified).mean()
        self.assertGreater(
            diff.item(),
            1e-3,
            f"Modified LoRA output should differ significantly (diff={diff.item():.6f})"
        )
        
        print(f"\n✓ LoRA adaptation changes model output correctly")
        print(f"  Mean absolute difference: {diff.item():.6f}")


# ============================================================================
# Test Runner
# ============================================================================

def run_tests():
    """Run all LoRA tests."""
    print("\n" + "=" * 80)
    print("LoRA Parameter Count and Frozen Parameter Tests")
    print("=" * 80)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add tests
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLoRAParameterCounts))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestFrozenParameters))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLoRAForwardPass))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 80)
    if result.wasSuccessful():
        print("✓ All LoRA tests passed!")
    else:
        print(f"✗ {len(result.failures)} test(s) failed")
        print(f"✗ {len(result.errors)} test(s) had errors")
    print("=" * 80)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
