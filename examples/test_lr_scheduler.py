"""
Test Cosine LR Scheduler with Warmup

Quick test to verify the cosine annealing scheduler with linear warmup works correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from axs_lib.lr_scheduler import CosineAnnealingWarmupLR, CosineAnnealingWarmupEpochLR


def test_step_scheduler():
    """Test step-based scheduler."""
    print("\n" + "=" * 80)
    print("Testing Step-based Cosine LR Scheduler with Warmup")
    print("=" * 80)
    
    # Create dummy optimizer
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    warmup_steps = 100
    total_steps = 1000
    
    scheduler = CosineAnnealingWarmupLR(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        eta_min=1e-6
    )
    
    print(f"\nConfiguration:")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Total steps: {total_steps}")
    print(f"  Base LR: 1e-3")
    print(f"  Min LR: 1e-6")
    
    print(f"\nLR Schedule:")
    print(f"  Step 0:    {scheduler.get_last_lr()[0]:.2e} (start)")
    
    # Warmup phase
    for _ in range(warmup_steps - 1):
        scheduler.step()
    print(f"  Step {warmup_steps}:   {scheduler.get_last_lr()[0]:.2e} (warmup end)")
    
    # Cosine annealing phase
    for _ in range(total_steps // 4 - warmup_steps):
        scheduler.step()
    print(f"  Step {total_steps // 4}:  {scheduler.get_last_lr()[0]:.2e} (25%)")
    
    for _ in range(total_steps // 4):
        scheduler.step()
    print(f"  Step {total_steps // 2}:  {scheduler.get_last_lr()[0]:.2e} (50%)")
    
    for _ in range(total_steps // 4):
        scheduler.step()
    print(f"  Step {3 * total_steps // 4}: {scheduler.get_last_lr()[0]:.2e} (75%)")
    
    for _ in range(total_steps // 4):
        scheduler.step()
    print(f"  Step {total_steps}: {scheduler.get_last_lr()[0]:.2e} (end)")
    
    print("\n✓ Step-based scheduler test passed!")


def test_epoch_scheduler():
    """Test epoch-based scheduler."""
    print("\n" + "=" * 80)
    print("Testing Epoch-based Cosine LR Scheduler with Warmup")
    print("=" * 80)
    
    # Create dummy optimizer
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    warmup_epochs = 5
    total_epochs = 50
    
    scheduler = CosineAnnealingWarmupEpochLR(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=total_epochs,
        eta_min=1e-6
    )
    
    print(f"\nConfiguration:")
    print(f"  Warmup epochs: {warmup_epochs}")
    print(f"  Total epochs: {total_epochs}")
    print(f"  Base LR: 1e-3")
    print(f"  Min LR: 1e-6")
    
    print(f"\nLR Schedule:")
    print(f"  Epoch 0:  {scheduler.get_last_lr()[0]:.2e} (start)")
    
    # Warmup phase
    for _ in range(warmup_epochs - 1):
        scheduler.step()
    print(f"  Epoch {warmup_epochs}: {scheduler.get_last_lr()[0]:.2e} (warmup end)")
    
    # Cosine annealing phase
    for _ in range(total_epochs // 4 - warmup_epochs):
        scheduler.step()
    print(f"  Epoch {total_epochs // 4}: {scheduler.get_last_lr()[0]:.2e} (25%)")
    
    for _ in range(total_epochs // 4):
        scheduler.step()
    print(f"  Epoch {total_epochs // 2}: {scheduler.get_last_lr()[0]:.2e} (50%)")
    
    for _ in range(total_epochs // 4):
        scheduler.step()
    print(f"  Epoch {3 * total_epochs // 4}: {scheduler.get_last_lr()[0]:.2e} (75%)")
    
    for _ in range(total_epochs // 4):
        scheduler.step()
    print(f"  Epoch {total_epochs}: {scheduler.get_last_lr()[0]:.2e} (end)")
    
    print("\n✓ Epoch-based scheduler test passed!")


def test_warmup_behavior():
    """Test that warmup actually increases LR linearly."""
    print("\n" + "=" * 80)
    print("Testing Linear Warmup Behavior")
    print("=" * 80)
    
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
    
    warmup_steps = 10
    total_steps = 100
    
    scheduler = CosineAnnealingWarmupLR(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        eta_min=0.0
    )
    
    print(f"\nTesting linear warmup from 0 to 1.0 over {warmup_steps} steps:")
    
    lrs = []
    for step in range(warmup_steps + 1):
        lr = scheduler.get_last_lr()[0]
        lrs.append(lr)
        expected = step / warmup_steps
        print(f"  Step {step}: LR = {lr:.3f}, Expected = {expected:.3f}")
        scheduler.step()
    
    # Check linearity
    diffs = [lrs[i+1] - lrs[i] for i in range(len(lrs) - 1)]
    is_linear = all(abs(d - diffs[0]) < 1e-6 for d in diffs)
    
    if is_linear:
        print("\n✓ Warmup is perfectly linear!")
    else:
        print("\n✗ Warmup is not linear!")
        return False
    
    return True


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("LR Scheduler Tests")
    print("=" * 80)
    
    try:
        test_step_scheduler()
        test_epoch_scheduler()
        test_warmup_behavior()
        
        print("\n" + "=" * 80)
        print("✓ All tests passed!")
        print("=" * 80)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
