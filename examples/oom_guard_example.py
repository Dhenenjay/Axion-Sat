"""
Example: Using OOM Guard for Safe Training

This script demonstrates how to use the OOM Guard utility to automatically
recover from CUDA out-of-memory errors during training.
"""

import torch
import torch.nn as nn
from tools.oom_guard import OOMGuard, oom_safe_step, OOMProtection


# ============================================================================
# Example 1: Using safe_execute() method
# ============================================================================

def example_basic_usage():
    """Basic usage with safe_execute()."""
    print("\n" + "="*70)
    print("Example 1: Basic Usage with safe_execute()")
    print("="*70)
    
    # Initialize OOM Guard
    guard = OOMGuard(config_path="hardware.lowvr.yaml", verbose=True)
    
    # Define a training step function
    def training_step(batch_size, tile_size, model, **kwargs):
        """Simulated training step."""
        print(f"  Executing with batch_size={batch_size}, tile_size={tile_size}")
        
        # Simulate creating tensors
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # This would be your actual training code
        inputs = torch.randn(batch_size, 4, tile_size, tile_size, device=device)
        targets = torch.randn(batch_size, 4, tile_size, tile_size, device=device)
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute loss
        loss = nn.functional.mse_loss(outputs, targets)
        
        return loss.item()
    
    # Create a simple model
    model = nn.Conv2d(4, 4, kernel_size=3, padding=1)
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Define execution context
    context = {
        'batch_size': 8,
        'tile_size': 256,
        'model': model
    }
    
    try:
        # Execute with OOM protection
        result, final_context = guard.safe_execute(
            training_step,
            context=context
        )
        
        print(f"\n✓ Training step completed successfully!")
        print(f"  Loss: {result:.4f}")
        print(f"  Final batch_size: {final_context['batch_size']}")
        print(f"  Final tile_size: {final_context['tile_size']}")
        
    except RuntimeError as e:
        print(f"\n✗ Training failed: {e}")
    
    # Print statistics
    guard.print_stats()


# ============================================================================
# Example 2: Using decorator
# ============================================================================

def example_decorator_usage():
    """Using OOM-safe decorator."""
    print("\n" + "="*70)
    print("Example 2: Decorator Usage")
    print("="*70)
    
    # Initialize OOM Guard
    guard = OOMGuard(config_path="hardware.lowvr.yaml", verbose=True)
    
    # Define context
    context = {
        'batch_size': 8,
        'tile_size': 256
    }
    
    # Decorate the training step
    @oom_safe_step(guard, context=context)
    def training_step(batch_size, tile_size, **kwargs):
        """OOM-safe training step."""
        print(f"  Executing with batch_size={batch_size}, tile_size={tile_size}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Simulate training
        inputs = torch.randn(batch_size, 4, tile_size, tile_size, device=device)
        loss = inputs.mean()
        
        return loss.item()
    
    try:
        # Call the decorated function
        loss = training_step()
        print(f"\n✓ Training step completed!")
        print(f"  Loss: {loss:.4f}")
        
    except RuntimeError as e:
        print(f"\n✗ Training failed: {e}")
    
    guard.print_stats()


# ============================================================================
# Example 3: Using context manager
# ============================================================================

def example_context_manager():
    """Using OOMProtection context manager."""
    print("\n" + "="*70)
    print("Example 3: Context Manager Usage")
    print("="*70)
    
    # Initialize OOM Guard
    guard = OOMGuard(config_path="hardware.lowvr.yaml", verbose=True)
    
    # Initial context
    initial_context = {
        'batch_size': 8,
        'tile_size': 256
    }
    
    try:
        with OOMProtection(guard, initial_context) as ctx:
            print(f"  Starting with batch_size={ctx['batch_size']}, "
                  f"tile_size={ctx['tile_size']}")
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Your training code here
            batch_size = ctx['batch_size']
            tile_size = ctx['tile_size']
            
            inputs = torch.randn(batch_size, 4, tile_size, tile_size, device=device)
            loss = inputs.mean()
            
            print(f"\n✓ Completed successfully!")
            print(f"  Loss: {loss.item():.4f}")
            
    except RuntimeError as e:
        print(f"\n✗ Failed: {e}")
        print(f"  Try again with updated context: {initial_context}")


# ============================================================================
# Example 4: Full training loop with OOM Guard
# ============================================================================

def example_training_loop():
    """Complete training loop with OOM protection."""
    print("\n" + "="*70)
    print("Example 4: Full Training Loop")
    print("="*70)
    
    # Initialize OOM Guard
    guard = OOMGuard(
        config_path="hardware.lowvr.yaml",
        verbose=True,
        log_file="oom_events.log"
    )
    
    # Create a simple model
    model = nn.Sequential(
        nn.Conv2d(4, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 4, kernel_size=3, padding=1)
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    # Training configuration
    initial_batch_size = 8
    initial_tile_size = 256
    num_epochs = 2
    steps_per_epoch = 5
    
    # Training context
    context = {
        'batch_size': initial_batch_size,
        'tile_size': initial_tile_size
    }
    
    def training_step(batch_size, tile_size, **kwargs):
        """Single training step."""
        optimizer.zero_grad()
        
        # Generate synthetic batch
        inputs = torch.randn(batch_size, 4, tile_size, tile_size, device=device)
        targets = torch.randn(batch_size, 4, tile_size, tile_size, device=device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)
        
        epoch_losses = []
        
        for step in range(steps_per_epoch):
            try:
                # Execute training step with OOM protection
                loss, context = guard.safe_execute(
                    training_step,
                    context=context
                )
                
                epoch_losses.append(loss)
                
                print(f"  Step {step + 1}/{steps_per_epoch}: "
                      f"Loss = {loss:.4f}, "
                      f"BS = {context['batch_size']}, "
                      f"TS = {context['tile_size']}")
                
            except RuntimeError as e:
                print(f"  Step {step + 1} failed: {e}")
                break
        
        if epoch_losses:
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"\nEpoch {epoch + 1} Average Loss: {avg_loss:.4f}")
    
    # Print final statistics
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    guard.print_stats()
    
    print("\nFinal training context:")
    print(f"  Batch size: {context['batch_size']}")
    print(f"  Tile size: {context['tile_size']}")
    
    if 'gradient_checkpointing' in context and context['gradient_checkpointing']:
        print("  Gradient checkpointing: Enabled")
    if 'use_fp16' in context and context['use_fp16']:
        print("  Mixed precision: FP16")


# ============================================================================
# Example 5: Memory estimation and configuration
# ============================================================================

def example_memory_estimation():
    """Estimate memory requirements."""
    print("\n" + "="*70)
    print("Example 5: Memory Estimation")
    print("="*70)
    
    from tools.oom_guard import estimate_memory_usage
    
    configurations = [
        # (batch_size, tile_size, description)
        (16, 512, "Large: 16 tiles @ 512x512"),
        (8, 256, "Medium: 8 tiles @ 256x256"),
        (4, 256, "Small: 4 tiles @ 256x256"),
        (2, 256, "Minimal: 2 tiles @ 256x256"),
        (1, 128, "Ultra-minimal: 1 tile @ 128x128"),
    ]
    
    print("\nEstimated VRAM usage for different configurations:")
    print("-" * 70)
    
    for batch_size, tile_size, desc in configurations:
        mem_gb = estimate_memory_usage(
            batch_size=batch_size,
            tile_size=tile_size,
            num_channels=4,
            model_params=600_000_000  # 600M parameters
        )
        
        print(f"{desc:40s} {mem_gb:6.2f} GB")
    
    print("-" * 70)
    
    # Recommendations
    print("\nRecommendations by GPU:")
    print("  8 GB VRAM:  Use batch_size=2, tile_size=256")
    print("  12 GB VRAM: Use batch_size=4, tile_size=256")
    print("  16 GB VRAM: Use batch_size=8, tile_size=256")
    print("  24 GB VRAM: Use batch_size=16, tile_size=256")


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("OOM Guard Usage Examples")
    print("="*70)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"\n✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\n⚠ CUDA not available - running in CPU mode")
    
    # Run examples
    try:
        # Example 1: Basic usage
        example_basic_usage()
        
        # Example 2: Decorator
        # example_decorator_usage()
        
        # Example 3: Context manager
        # example_context_manager()
        
        # Example 4: Full training loop
        # example_training_loop()
        
        # Example 5: Memory estimation
        example_memory_estimation()
        
    except Exception as e:
        print(f"\n✗ Example failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("Examples Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
