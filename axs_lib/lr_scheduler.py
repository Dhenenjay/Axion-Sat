"""
Learning Rate Schedulers with Warmup

Provides cosine annealing learning rate schedulers with linear warmup
for Stage 1 and Stage 2 training.

Author: Axion-Sat Project
Version: 1.0.0
"""

import math
import warnings
from typing import List

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmupLR(_LRScheduler):
    """
    Cosine annealing learning rate scheduler with linear warmup.
    
    This scheduler combines:
    1. Linear warmup: LR increases linearly from 0 to base_lr over warmup_steps
    2. Cosine annealing: LR decreases following cosine curve from base_lr to eta_min
    
    The learning rate is computed as:
        - Warmup phase (step < warmup_steps):
            lr = base_lr * (step / warmup_steps)
        
        - Annealing phase (step >= warmup_steps):
            lr = eta_min + (base_lr - eta_min) * 0.5 * (1 + cos(π * progress))
            where progress = (step - warmup_steps) / (total_steps - warmup_steps)
    
    Args:
        optimizer: Wrapped optimizer
        warmup_steps: Number of warmup steps (linear increase)
        total_steps: Total number of training steps (warmup + annealing)
        eta_min: Minimum learning rate (default: 0.0)
        last_step: The index of last step (default: -1)
    
    Example:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        >>> scheduler = CosineAnnealingWarmupLR(
        ...     optimizer,
        ...     warmup_steps=1000,
        ...     total_steps=10000,
        ...     eta_min=1e-6
        ... )
        >>> for epoch in range(num_epochs):
        ...     for batch in dataloader:
        ...         # Training step
        ...         optimizer.zero_grad()
        ...         loss.backward()
        ...         optimizer.step()
        ...         scheduler.step()  # Step after each batch
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        eta_min: float = 0.0,
        last_step: int = -1
    ):
        """Initialize cosine annealing scheduler with warmup."""
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        
        if total_steps <= warmup_steps:
            raise ValueError(
                f"total_steps ({total_steps}) must be > warmup_steps ({warmup_steps})"
            )
        
        if eta_min < 0:
            raise ValueError(f"eta_min must be >= 0, got {eta_min}")
        
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        
        super().__init__(optimizer, last_epoch=last_step)
    
    def get_lr(self) -> List[float]:
        """
        Compute learning rate for current step.
        
        Returns:
            List of learning rates for each parameter group
        """
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "use `get_last_lr()`.",
                UserWarning
            )
        
        # Current step (0-indexed)
        step = self.last_epoch
        
        if step < self.warmup_steps:
            # Linear warmup phase
            warmup_factor = step / self.warmup_steps if self.warmup_steps > 0 else 1.0
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)  # Clamp to [0, 1]
            
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            
            return [
                self.eta_min + (base_lr - self.eta_min) * cosine_factor
                for base_lr in self.base_lrs
            ]
    
    def _get_closed_form_lr(self) -> List[float]:
        """
        Called by get_lr() when step is called with None.
        Returns the learning rate using closed form computation.
        """
        return self.get_lr()


class CosineAnnealingWarmupEpochLR(_LRScheduler):
    """
    Epoch-based cosine annealing learning rate scheduler with linear warmup.
    
    Similar to CosineAnnealingWarmupLR but operates on epoch granularity
    instead of step granularity.
    
    Args:
        optimizer: Wrapped optimizer
        warmup_epochs: Number of warmup epochs (linear increase)
        total_epochs: Total number of training epochs (warmup + annealing)
        eta_min: Minimum learning rate (default: 0.0)
        last_epoch: The index of last epoch (default: -1)
    
    Example:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        >>> scheduler = CosineAnnealingWarmupEpochLR(
        ...     optimizer,
        ...     warmup_epochs=5,
        ...     total_epochs=100,
        ...     eta_min=1e-6
        ... )
        >>> for epoch in range(num_epochs):
        ...     # Training
        ...     for batch in dataloader:
        ...         # Training step
        ...         pass
        ...     scheduler.step()  # Step after each epoch
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        eta_min: float = 0.0,
        last_epoch: int = -1
    ):
        """Initialize epoch-based cosine annealing scheduler with warmup."""
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")
        
        if total_epochs <= warmup_epochs:
            raise ValueError(
                f"total_epochs ({total_epochs}) must be > warmup_epochs ({warmup_epochs})"
            )
        
        if eta_min < 0:
            raise ValueError(f"eta_min must be >= 0, got {eta_min}")
        
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        
        super().__init__(optimizer, last_epoch=last_epoch)
    
    def get_lr(self) -> List[float]:
        """
        Compute learning rate for current epoch.
        
        Returns:
            List of learning rates for each parameter group
        """
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "use `get_last_lr()`.",
                UserWarning
            )
        
        # Current epoch (0-indexed)
        epoch = self.last_epoch
        
        if epoch < self.warmup_epochs:
            # Linear warmup phase
            warmup_factor = epoch / self.warmup_epochs if self.warmup_epochs > 0 else 1.0
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            progress = min(progress, 1.0)  # Clamp to [0, 1]
            
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            
            return [
                self.eta_min + (base_lr - self.eta_min) * cosine_factor
                for base_lr in self.base_lrs
            ]
    
    def _get_closed_form_lr(self) -> List[float]:
        """
        Called by get_lr() when step is called with None.
        Returns the learning rate using closed form computation.
        """
        return self.get_lr()


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    eta_min: float = 0.0
) -> CosineAnnealingWarmupLR:
    """
    Create a schedule with a learning rate that decreases following a cosine
    curve after a warmup period during which it increases linearly.
    
    This is a convenience function compatible with Hugging Face transformers API.
    
    Args:
        optimizer: The optimizer for which to schedule the learning rate
        num_warmup_steps: The number of steps for the warmup phase
        num_training_steps: The total number of training steps
        num_cycles: The number of waves in the cosine schedule (default: 0.5)
        last_epoch: The index of the last epoch when resuming training
        eta_min: Minimum learning rate
    
    Returns:
        CosineAnnealingWarmupLR scheduler
    
    Note:
        num_cycles controls the frequency of the cosine curve.
        - num_cycles=0.5: One half of a cosine wave (standard)
        - num_cycles=1.0: One full cosine wave
        - num_cycles=2.0: Two full cosine waves
    """
    if num_cycles != 0.5:
        warnings.warn(
            f"num_cycles={num_cycles} is not fully supported yet. "
            f"Using standard cosine annealing with num_cycles=0.5"
        )
    
    return CosineAnnealingWarmupLR(
        optimizer=optimizer,
        warmup_steps=num_warmup_steps,
        total_steps=num_training_steps,
        eta_min=eta_min,
        last_step=last_epoch
    )


def plot_lr_schedule(
    scheduler: _LRScheduler,
    num_steps: int,
    title: str = "Learning Rate Schedule",
    save_path: str = None
) -> None:
    """
    Plot learning rate schedule for visualization.
    
    Args:
        scheduler: LR scheduler to visualize
        num_steps: Number of steps to plot
        title: Plot title
        save_path: Path to save plot (optional)
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available for plotting")
        return
    
    # Compute LR for each step
    lrs = []
    for step in range(num_steps):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(num_steps), lrs, linewidth=2, color='#2ecc71')
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, num_steps)
    
    # Add warmup line if applicable
    if hasattr(scheduler, 'warmup_steps') and scheduler.warmup_steps > 0:
        ax.axvline(
            scheduler.warmup_steps,
            color='red',
            linestyle='--',
            alpha=0.7,
            label=f'Warmup ends (step {scheduler.warmup_steps})'
        )
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved LR schedule plot: {save_path}")
    else:
        plt.show()
    
    plt.close()


# ============================================================================
# Example usage
# ============================================================================

if __name__ == '__main__':
    """Demonstrate LR scheduler usage and visualization."""
    
    print("=" * 80)
    print("Cosine Annealing with Warmup - Examples")
    print("=" * 80)
    
    # Create dummy optimizer
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Example 1: Step-based scheduler
    print("\n1. Step-based scheduler (for gradient accumulation)")
    print("-" * 80)
    
    total_steps = 10000
    warmup_steps = 1000
    
    scheduler_step = CosineAnnealingWarmupLR(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        eta_min=1e-6
    )
    
    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Base LR: {1e-3}")
    print(f"Min LR: {1e-6}")
    print(f"\nLR at key steps:")
    print(f"  Step 0: {scheduler_step.get_last_lr()[0]:.2e}")
    
    for _ in range(warmup_steps - 1):
        scheduler_step.step()
    print(f"  Step {warmup_steps}: {scheduler_step.get_last_lr()[0]:.2e} (warmup end)")
    
    for _ in range(total_steps // 2 - warmup_steps):
        scheduler_step.step()
    print(f"  Step {total_steps // 2}: {scheduler_step.get_last_lr()[0]:.2e} (midpoint)")
    
    for _ in range(total_steps // 2 - 1):
        scheduler_step.step()
    print(f"  Step {total_steps}: {scheduler_step.get_last_lr()[0]:.2e} (end)")
    
    # Example 2: Epoch-based scheduler
    print("\n2. Epoch-based scheduler (for epoch-level updates)")
    print("-" * 80)
    
    total_epochs = 100
    warmup_epochs = 10
    
    # Reset optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    scheduler_epoch = CosineAnnealingWarmupEpochLR(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=total_epochs,
        eta_min=1e-6
    )
    
    print(f"Total epochs: {total_epochs}")
    print(f"Warmup epochs: {warmup_epochs}")
    print(f"Base LR: {1e-3}")
    print(f"Min LR: {1e-6}")
    print(f"\nLR at key epochs:")
    print(f"  Epoch 0: {scheduler_epoch.get_last_lr()[0]:.2e}")
    
    for _ in range(warmup_epochs - 1):
        scheduler_epoch.step()
    print(f"  Epoch {warmup_epochs}: {scheduler_epoch.get_last_lr()[0]:.2e} (warmup end)")
    
    for _ in range(total_epochs // 2 - warmup_epochs):
        scheduler_epoch.step()
    print(f"  Epoch {total_epochs // 2}: {scheduler_epoch.get_last_lr()[0]:.2e} (midpoint)")
    
    for _ in range(total_epochs // 2 - 1):
        scheduler_epoch.step()
    print(f"  Epoch {total_epochs}: {scheduler_epoch.get_last_lr()[0]:.2e} (end)")
    
    # Visualization
    print("\n3. Visualizing LR schedules")
    print("-" * 80)
    
    # Reset schedulers for visualization
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler_vis = CosineAnnealingWarmupLR(
        optimizer,
        warmup_steps=1000,
        total_steps=10000,
        eta_min=1e-6
    )
    
    plot_lr_schedule(
        scheduler_vis,
        num_steps=10000,
        title="Cosine Annealing with Warmup (1000 warmup steps)",
        save_path="lr_schedule_example.png"
    )
    
    print("\n" + "=" * 80)
    print("Examples complete!")
    print("=" * 80)
