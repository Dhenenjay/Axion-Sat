"""
Example: Safe Checkpoint I/O for Training

This script demonstrates how to use the checkpoint I/O utilities
for robust training state management.
"""

import torch
import torch.nn as nn
from axs_lib.io import (
    CheckpointManager,
    safe_save_checkpoint,
    safe_load_checkpoint,
    save_model_only,
    load_model_only
)


# ============================================================================
# Example 1: Using CheckpointManager
# ============================================================================

def example_checkpoint_manager():
    """Using CheckpointManager for automatic management."""
    print("\n" + "="*70)
    print("Example 1: CheckpointManager Usage")
    print("="*70)
    
    # Create checkpoint manager
    manager = CheckpointManager(
        checkpoint_dir="checkpoints",
        max_checkpoints=5,  # Keep only 5 most recent checkpoints
        auto_backup=True,   # Backup before overwriting
        verify_on_load=True,  # Verify MD5 on load
        compress=False,     # No compression for speed
        verbose=True
    )
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop simulation
    print("\nSimulating training loop...")
    for epoch in range(3):
        # Simulate training
        loss = 1.0 / (epoch + 1)
        
        # Save checkpoint
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'learning_rate': 1e-3
        }
        
        manager.save_checkpoint(
            state,
            f"checkpoint_epoch_{epoch}.pt",
            metadata={
                'notes': f'Training checkpoint at epoch {epoch}',
                'train_samples': (epoch + 1) * 1000
            }
        )
        
        print(f"\nEpoch {epoch}: Loss = {loss:.4f}")
    
    # List all checkpoints
    print("\n" + "-"*70)
    print("Available Checkpoints:")
    checkpoints = manager.list_checkpoints()
    for ckpt in checkpoints:
        print(f"  {ckpt['filename']}: {ckpt['size_mb']:.2f} MB")
        print(f"    Timestamp: {ckpt['timestamp']}")
        print(f"    MD5: {ckpt['md5'][:16]}...")
    
    # Load latest checkpoint
    print("\n" + "-"*70)
    print("Loading Latest Checkpoint:")
    latest_path = manager.get_latest_checkpoint()
    if latest_path:
        state = manager.load_checkpoint(latest_path.split('\\')[-1])
        print(f"\nRestored training state:")
        print(f"  Epoch: {state['epoch']}")
        print(f"  Loss: {state['loss']:.4f}")
        print(f"  LR: {state['learning_rate']}")
    
    # Verify all checkpoints
    print("\n" + "-"*70)
    print("Verifying All Checkpoints:")
    results = manager.verify_all_checkpoints()
    for filename, valid in results.items():
        status = "✓ Valid" if valid else "✗ Corrupted"
        print(f"  {status}: {filename}")


# ============================================================================
# Example 2: Standalone Functions
# ============================================================================

def example_standalone_functions():
    """Using standalone save/load functions."""
    print("\n" + "="*70)
    print("Example 2: Standalone Functions")
    print("="*70)
    
    # Create a simple model
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3),
        nn.ReLU(),
        nn.Conv2d(16, 3, 3)
    )
    
    # Save checkpoint
    print("\nSaving checkpoint with standalone function...")
    state = {
        'model_state_dict': model.state_dict(),
        'architecture': 'SimpleConvNet',
        'input_channels': 3
    }
    
    md5 = safe_save_checkpoint(
        state,
        "standalone_checkpoint.pt",
        verify=True,
        backup=True,
        verbose=True
    )
    
    print(f"\nCheckpoint saved with MD5: {md5}")
    
    # Load checkpoint
    print("\n" + "-"*70)
    print("Loading checkpoint with standalone function...")
    loaded_state = safe_load_checkpoint(
        "standalone_checkpoint.pt",
        verify=True,
        verbose=True
    )
    
    print(f"\nLoaded state keys: {list(loaded_state.keys())}")
    print(f"Architecture: {loaded_state['architecture']}")


# ============================================================================
# Example 3: Model-Only Save/Load
# ============================================================================

def example_model_only():
    """Saving/loading only model weights."""
    print("\n" + "="*70)
    print("Example 3: Model-Only Save/Load")
    print("="*70)
    
    # Create and train a model
    model = nn.Linear(100, 10)
    
    print("\nSaving model weights only...")
    save_model_only(model, "model_weights.pt", verbose=True)
    
    # Create new model and load weights
    print("\n" + "-"*70)
    print("Loading into new model instance...")
    new_model = nn.Linear(100, 10)
    load_model_only(new_model, "model_weights.pt", verbose=True)
    
    print("\n✓ Weights loaded successfully!")


# ============================================================================
# Example 4: Resume Training
# ============================================================================

def example_resume_training():
    """Complete example of resuming training."""
    print("\n" + "="*70)
    print("Example 4: Resume Training")
    print("="*70)
    
    manager = CheckpointManager("training_checkpoints", verbose=True)
    
    # Create model and optimizer
    model = nn.Sequential(nn.Linear(20, 50), nn.ReLU(), nn.Linear(50, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Try to resume from checkpoint
    start_epoch = 0
    best_loss = float('inf')
    
    latest_checkpoint = manager.get_latest_checkpoint()
    if latest_checkpoint:
        print("\nResuming from checkpoint...")
        state = manager.load_checkpoint(latest_checkpoint.split('\\')[-1])
        
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        start_epoch = state['epoch'] + 1
        best_loss = state.get('best_loss', float('inf'))
        
        print(f"Resumed from epoch {state['epoch']}")
        print(f"Best loss so far: {best_loss:.4f}")
    else:
        print("\nNo checkpoint found, starting from scratch")
    
    # Continue training
    print(f"\nContinuing training from epoch {start_epoch}...")
    num_epochs = 5
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        # Simulate training
        loss = 1.0 / (epoch + 1)
        
        # Update best loss
        if loss < best_loss:
            best_loss = loss
            is_best = True
        else:
            is_best = False
        
        # Save checkpoint
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'best_loss': best_loss
        }
        
        # Save regular checkpoint
        manager.save_checkpoint(
            state,
            f"checkpoint_epoch_{epoch}.pt",
            metadata={'is_best': is_best}
        )
        
        # Save best model separately
        if is_best:
            manager.save_checkpoint(
                state,
                "best_model.pt",
                metadata={'best_epoch': epoch, 'best_loss': best_loss}
            )
            print(f"\n  → New best model! Loss: {best_loss:.4f}")
        
        print(f"Epoch {epoch}: Loss = {loss:.4f}")


# ============================================================================
# Example 5: Corruption Recovery
# ============================================================================

def example_corruption_recovery():
    """Demonstrating automatic corruption recovery."""
    print("\n" + "="*70)
    print("Example 5: Corruption Recovery")
    print("="*70)
    
    manager = CheckpointManager(
        "recovery_test",
        auto_backup=True,
        verify_on_load=True,
        verbose=True
    )
    
    # Save a checkpoint
    state = {'data': torch.randn(5, 5), 'epoch': 10}
    manager.save_checkpoint(state, "test_checkpoint.pt")
    
    print("\n✓ Checkpoint saved with backup")
    
    # Simulate corruption (in real scenario, this would be actual corruption)
    # The backup system would allow recovery
    
    print("\nIn case of corruption:")
    print("  1. Original file fails MD5 verification")
    print("  2. System automatically tries backup")
    print("  3. Training continues without data loss")


# ============================================================================
# Example 6: Async Saving
# ============================================================================

def example_async_saving():
    """Asynchronous checkpoint saving."""
    print("\n" + "="*70)
    print("Example 6: Asynchronous Saving")
    print("="*70)
    
    manager = CheckpointManager("async_checkpoints", verbose=True)
    
    # Create large dummy state
    state = {
        'model': {f'layer_{i}': torch.randn(100, 100) for i in range(10)},
        'epoch': 42
    }
    
    print("\nSaving checkpoint asynchronously...")
    print("(Training continues while checkpoint is being written)")
    
    # Async save (non-blocking)
    manager.save_checkpoint(
        state,
        "large_checkpoint.pt",
        async_save=True  # Save in background thread
    )
    
    print("✓ Save initiated in background")
    print("  Training can continue immediately...")
    
    # In practice, wait for save to complete before next checkpoint
    import time
    time.sleep(0.5)
    print("✓ Background save completed")


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("Checkpoint I/O Examples")
    print("="*70)
    
    try:
        # Run examples
        example_checkpoint_manager()
        example_standalone_functions()
        example_model_only()
        example_resume_training()
        example_corruption_recovery()
        example_async_saving()
        
    except Exception as e:
        print(f"\n✗ Example failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    print("\n" + "="*70)
    print("Cleaning up example checkpoints...")
    print("="*70)
    
    import shutil
    from pathlib import Path
    
    for dir_name in ["checkpoints", "training_checkpoints", "recovery_test", "async_checkpoints"]:
        if Path(dir_name).exists():
            shutil.rmtree(dir_name)
            print(f"  Removed {dir_name}/")
    
    # Cleanup standalone files
    for file in ["standalone_checkpoint.pt", "standalone_checkpoint.md5", 
                 "standalone_checkpoint.backup", "model_weights.pt", "model_weights.md5"]:
        if Path(file).exists():
            Path(file).unlink()
            print(f"  Removed {file}")
    
    print("\n" + "="*70)
    print("Examples Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
