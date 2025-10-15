"""
Visualize Learning Rate Schedule

Creates plots showing the learning rate schedule with warmup for training.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import matplotlib.pyplot as plt
import numpy as np
from axs_lib.lr_scheduler import CosineAnnealingWarmupLR


def visualize_lr_schedule(
    warmup_steps: int,
    total_steps: int,
    base_lr: float = 1e-3,
    min_lr: float = 1e-6,
    output_path: str = "lr_schedule.png"
):
    """
    Visualize learning rate schedule.
    
    Args:
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        base_lr: Base learning rate
        min_lr: Minimum learning rate
        output_path: Path to save visualization
    """
    # Create dummy optimizer
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
    
    # Create scheduler
    scheduler = CosineAnnealingWarmupLR(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        eta_min=min_lr
    )
    
    # Collect learning rates
    steps = []
    lrs = []
    
    for step in range(total_steps + 1):
        steps.append(step)
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot LR schedule
    ax.plot(steps, lrs, linewidth=2.5, color='#2ecc71', label='Learning Rate')
    
    # Mark warmup end
    ax.axvline(
        warmup_steps,
        color='#e74c3c',
        linestyle='--',
        linewidth=2,
        alpha=0.7,
        label=f'Warmup ends (step {warmup_steps})'
    )
    
    # Mark base LR
    ax.axhline(
        base_lr,
        color='#3498db',
        linestyle=':',
        linewidth=1.5,
        alpha=0.5,
        label=f'Base LR ({base_lr:.0e})'
    )
    
    # Mark min LR
    ax.axhline(
        min_lr,
        color='#9b59b6',
        linestyle=':',
        linewidth=1.5,
        alpha=0.5,
        label=f'Min LR ({min_lr:.0e})'
    )
    
    # Fill warmup region
    ax.fill_between(
        [0, warmup_steps],
        [0, 0],
        [max(lrs), max(lrs)],
        alpha=0.1,
        color='#e74c3c',
        label='Warmup phase'
    )
    
    # Labels and formatting
    ax.set_xlabel('Training Step', fontsize=13, fontweight='bold')
    ax.set_ylabel('Learning Rate', fontsize=13, fontweight='bold')
    ax.set_title(
        f'Cosine Annealing LR Schedule with Warmup\n'
        f'({warmup_steps:,} warmup steps / {total_steps:,} total steps)',
        fontsize=15,
        fontweight='bold',
        pad=20
    )
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    # Set limits
    ax.set_xlim(0, total_steps)
    ax.set_ylim(0, base_lr * 1.1)
    
    # Add annotations
    ax.annotate(
        'Linear\nWarmup',
        xy=(warmup_steps / 2, base_lr / 2),
        fontsize=11,
        ha='center',
        va='center',
        color='#c0392b',
        fontweight='bold'
    )
    
    ax.annotate(
        'Cosine Annealing',
        xy=(warmup_steps + (total_steps - warmup_steps) / 2, base_lr * 0.7),
        fontsize=11,
        ha='center',
        va='center',
        color='#27ae60',
        fontweight='bold'
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved LR schedule visualization: {output_path}")
    plt.close()


def create_comparison_plot(output_path: str = "lr_schedule_comparison.png"):
    """Create comparison of different warmup ratios."""
    
    total_steps = 10000
    base_lr = 1e-3
    min_lr = 1e-6
    
    warmup_ratios = [0.05, 0.10, 0.20]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        'Effect of Warmup Duration on LR Schedule',
        fontsize=16,
        fontweight='bold',
        y=1.02
    )
    
    for idx, ratio in enumerate(warmup_ratios):
        warmup_steps = int(ratio * total_steps)
        
        # Create scheduler
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
        scheduler = CosineAnnealingWarmupLR(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            eta_min=min_lr
        )
        
        # Collect LRs
        steps = []
        lrs = []
        for step in range(total_steps + 1):
            steps.append(step)
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()
        
        # Plot
        ax = axes[idx]
        ax.plot(steps, lrs, linewidth=2, color='#2ecc71')
        ax.axvline(warmup_steps, color='#e74c3c', linestyle='--', alpha=0.7)
        ax.fill_between(
            [0, warmup_steps],
            [0, 0],
            [base_lr * 1.1, base_lr * 1.1],
            alpha=0.1,
            color='#e74c3c'
        )
        
        ax.set_xlabel('Training Step', fontsize=11)
        ax.set_ylabel('Learning Rate' if idx == 0 else '', fontsize=11)
        ax.set_title(
            f'{ratio*100:.0f}% Warmup\n({warmup_steps:,} steps)',
            fontsize=12,
            fontweight='bold'
        )
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, total_steps)
        ax.set_ylim(0, base_lr * 1.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved comparison plot: {output_path}")
    plt.close()


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("LR Schedule Visualization")
    print("=" * 80)
    
    # Example 1: Typical training schedule
    print("\n1. Creating typical training schedule (10% warmup)...")
    visualize_lr_schedule(
        warmup_steps=1000,
        total_steps=10000,
        base_lr=1e-3,
        min_lr=1e-6,
        output_path="reports/lr_schedule_10pct_warmup.png"
    )
    
    # Example 2: Longer warmup
    print("\n2. Creating schedule with longer warmup (20%)...")
    visualize_lr_schedule(
        warmup_steps=2000,
        total_steps=10000,
        base_lr=1e-3,
        min_lr=1e-6,
        output_path="reports/lr_schedule_20pct_warmup.png"
    )
    
    # Example 3: Comparison plot
    print("\n3. Creating comparison plot...")
    create_comparison_plot(output_path="reports/lr_schedule_comparison.png")
    
    print("\n" + "=" * 80)
    print("✓ Visualization complete!")
    print("=" * 80)
