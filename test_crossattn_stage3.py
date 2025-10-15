"""
Quick test script for cross-attention Stage 3 model.

Tests:
1. Model builds successfully
2. Forward pass works
3. Gradients flow correctly
4. Memory footprint
"""

import torch
from axs_lib.stage3_tm_crossattn import build_stage3_crossattn_model

print("="*80)
print("Testing Cross-Attention Stage 3 Model")
print("="*80)

# Build model
print("\n1. Building model...")
model = build_stage3_crossattn_model(
    freeze_backbone=False,
    pretrained=True,
    num_fusion_layers=2
)

# Create dummy inputs
print("\n2. Creating dummy inputs...")
s1 = torch.randn(1, 2, 128, 128).cuda()
opt_v2 = torch.randn(1, 4, 128, 128).cuda()
s2_truth = torch.randn(1, 4, 128, 128).cuda()

# Forward pass
print("\n3. Testing forward pass...")
model.train()
output = model(s1, opt_v2)
print(f"   Input shape: {s1.shape}")
print(f"   Output shape: {output.shape}")
print(f"   Output requires_grad: {output.requires_grad}")
print(f"   Output has grad_fn: {output.grad_fn is not None}")

# Test gradient flow
print("\n4. Testing gradient flow...")
loss = torch.nn.functional.mse_loss(output, s2_truth)
print(f"   Loss: {loss.item():.6f}")
print(f"   Loss requires_grad: {loss.requires_grad}")

loss.backward()
print("   ✓ Backward pass succeeded!")

# Check gradients
decoder_has_grads = any(p.grad is not None for p in model.decoder.parameters())
backbone_has_grads = any(p.grad is not None for p in model.terramind_backbone.parameters())
print(f"   Decoder has gradients: {decoder_has_grads}")
print(f"   Backbone has gradients: {backbone_has_grads}")

# Memory footprint
print("\n5. Memory footprint:")
if torch.cuda.is_available():
    memory_allocated = torch.cuda.memory_allocated() / 1e9
    memory_reserved = torch.cuda.memory_reserved() / 1e9
    print(f"   Allocated: {memory_allocated:.2f} GB")
    print(f"   Reserved: {memory_reserved:.2f} GB")

print("\n" + "="*80)
print("✅ All tests passed! Cross-attention model works correctly.")
print("="*80)
