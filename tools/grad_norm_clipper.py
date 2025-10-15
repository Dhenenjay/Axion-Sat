"""
Gradient Norm Clipping Utility

This module provides gradient norm clipping with logging and statistics tracking.
Helps stabilize training by preventing exploding gradients.

Features:
- Clips gradient norms to a maximum value (default: 1.0)
- Logs when clipping is activated
- Tracks clipping statistics (frequency, magnitude)
- Supports both single and multiple parameter groups
- Compatible with PyTorch optimizers and AMP

Usage:
    from tools.grad_norm_clipper import GradNormClipper
    
    clipper = GradNormClipper(max_norm=1.0, log_clipping=True)
    
    # During training loop
    loss.backward()
    grad_norm = clipper.clip_grad_norm(model.parameters())
    optimizer.step()
    
    # Get statistics
    stats = clipper.get_statistics()
"""

import torch
import torch.nn as nn
from typing import Union, Iterable, Dict, Optional
import warnings
from collections import defaultdict
import time


class GradNormClipper:
    """
    Gradient norm clipper with logging and statistics tracking.
    
    Clips gradients to prevent exploding gradients during training.
    Logs when clipping occurs and tracks statistics over time.
    
    Args:
        max_norm: Maximum gradient norm (default: 1.0)
        norm_type: Type of norm to compute (default: 2.0 for L2 norm)
        log_clipping: Whether to log when clipping occurs (default: True)
        log_frequency: How often to log (every N clips, default: 1)
        error_if_nonfinite: Raise error if gradients are inf/nan (default: False)
        verbose: Print detailed information (default: True)
        
    Example:
        >>> clipper = GradNormClipper(max_norm=1.0)
        >>> 
        >>> # Training loop
        >>> for batch in dataloader:
        ...     loss.backward()
        ...     grad_norm = clipper.clip_grad_norm(model.parameters())
        ...     optimizer.step()
        >>> 
        >>> # Get statistics
        >>> stats = clipper.get_statistics()
        >>> print(f"Clipping rate: {stats['clip_rate']:.2%}")
    """
    
    def __init__(
        self,
        max_norm: float = 1.0,
        norm_type: float = 2.0,
        log_clipping: bool = True,
        log_frequency: int = 1,
        error_if_nonfinite: bool = False,
        verbose: bool = True
    ):
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.log_clipping = log_clipping
        self.log_frequency = log_frequency
        self.error_if_nonfinite = error_if_nonfinite
        self.verbose = verbose
        
        # Statistics tracking
        self.total_steps = 0
        self.clip_count = 0
        self.grad_norms = []
        self.clipped_norms = []
        self.clip_ratios = []
        
        # Logging state
        self.last_clip_step = -1
        self.consecutive_clips = 0
        
        # History for analysis
        self.history = defaultdict(list)
        
    def clip_grad_norm(
        self,
        parameters: Union[torch.Tensor, Iterable[torch.Tensor]],
        max_norm: Optional[float] = None,
        norm_type: Optional[float] = None,
        error_if_nonfinite: Optional[bool] = None
    ) -> float:
        """
        Clip gradient norm of parameters.
        
        Args:
            parameters: Model parameters or iterable of parameters
            max_norm: Maximum norm (overrides default if provided)
            norm_type: Type of norm (overrides default if provided)
            error_if_nonfinite: Error on inf/nan (overrides default if provided)
            
        Returns:
            Total norm of the gradients (before clipping)
            
        Raises:
            RuntimeError: If gradients contain inf/nan and error_if_nonfinite=True
        """
        # Use provided values or fall back to defaults
        max_norm = max_norm if max_norm is not None else self.max_norm
        norm_type = norm_type if norm_type is not None else self.norm_type
        error_if_nonfinite = error_if_nonfinite if error_if_nonfinite is not None else self.error_if_nonfinite
        
        # Convert to list if single tensor
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        
        # Filter parameters with gradients
        parameters = [p for p in parameters if p.grad is not None]
        
        if len(parameters) == 0:
            warnings.warn("No parameters with gradients found!")
            return 0.0
        
        # Compute total gradient norm
        device = parameters[0].grad.device
        
        if norm_type == float('inf'):
            # Infinity norm: max absolute value
            total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
        else:
            # L-p norm
            total_norm = torch.norm(
                torch.stack([
                    torch.norm(p.grad.detach(), norm_type).to(device)
                    for p in parameters
                ]),
                norm_type
            )
        
        # Check for non-finite gradients
        if error_if_nonfinite and not torch.isfinite(total_norm):
            raise RuntimeError(
                f"The total norm of order {norm_type} for gradients is non-finite. "
                f"Gradient norm: {total_norm}"
            )
        
        # Update statistics
        self.total_steps += 1
        grad_norm_value = total_norm.item()
        self.grad_norms.append(grad_norm_value)
        self.history['step'].append(self.total_steps)
        self.history['grad_norm'].append(grad_norm_value)
        
        # Perform clipping if needed
        clipped = False
        if total_norm > max_norm:
            clip_coef = max_norm / (total_norm + 1e-6)
            
            # Clip gradients
            for p in parameters:
                p.grad.detach().mul_(clip_coef)
            
            # Update clipping statistics
            clipped = True
            self.clip_count += 1
            self.clipped_norms.append(grad_norm_value)
            clip_ratio = max_norm / grad_norm_value
            self.clip_ratios.append(clip_ratio)
            self.history['clipped'].append(1)
            self.history['clip_ratio'].append(clip_ratio)
            
            # Update consecutive clip counter
            if self.last_clip_step == self.total_steps - 1:
                self.consecutive_clips += 1
            else:
                self.consecutive_clips = 1
            self.last_clip_step = self.total_steps
            
            # Log clipping event
            if self.log_clipping and (self.clip_count % self.log_frequency == 0):
                self._log_clipping(grad_norm_value, max_norm)
        else:
            self.history['clipped'].append(0)
            self.history['clip_ratio'].append(1.0)
            self.consecutive_clips = 0
        
        return grad_norm_value
    
    def _log_clipping(self, grad_norm: float, max_norm: float):
        """Log clipping event."""
        reduction = (grad_norm - max_norm) / grad_norm * 100
        
        if self.verbose:
            msg = (
                f"[GradClip] Step {self.total_steps}: "
                f"Gradient norm clipped from {grad_norm:.4f} to {max_norm:.4f} "
                f"(reduced by {reduction:.1f}%)"
            )
            
            # Add warning if consecutive clips
            if self.consecutive_clips > 5:
                msg += f" | WARNING: {self.consecutive_clips} consecutive clips!"
            
            print(msg)
    
    def get_statistics(self) -> Dict:
        """
        Get clipping statistics.
        
        Returns:
            Dictionary with statistics:
                - total_steps: Total optimization steps
                - clip_count: Number of times clipping occurred
                - clip_rate: Fraction of steps where clipping occurred
                - avg_grad_norm: Average gradient norm (all steps)
                - max_grad_norm: Maximum gradient norm observed
                - avg_clipped_norm: Average gradient norm when clipped
                - avg_clip_ratio: Average ratio of max_norm / grad_norm when clipped
                - consecutive_clips: Current consecutive clip count
        """
        if self.total_steps == 0:
            return {
                'total_steps': 0,
                'clip_count': 0,
                'clip_rate': 0.0,
                'avg_grad_norm': 0.0,
                'max_grad_norm': 0.0,
                'avg_clipped_norm': 0.0,
                'avg_clip_ratio': 0.0,
                'consecutive_clips': 0
            }
        
        stats = {
            'total_steps': self.total_steps,
            'clip_count': self.clip_count,
            'clip_rate': self.clip_count / self.total_steps,
            'avg_grad_norm': sum(self.grad_norms) / len(self.grad_norms),
            'max_grad_norm': max(self.grad_norms) if self.grad_norms else 0.0,
            'consecutive_clips': self.consecutive_clips
        }
        
        if self.clip_count > 0:
            stats['avg_clipped_norm'] = sum(self.clipped_norms) / len(self.clipped_norms)
            stats['avg_clip_ratio'] = sum(self.clip_ratios) / len(self.clip_ratios)
        else:
            stats['avg_clipped_norm'] = 0.0
            stats['avg_clip_ratio'] = 1.0
        
        return stats
    
    def print_statistics(self):
        """Print formatted statistics."""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("GRADIENT CLIPPING STATISTICS")
        print("="*60)
        print(f"Total steps:           {stats['total_steps']}")
        print(f"Clipping events:       {stats['clip_count']} ({stats['clip_rate']*100:.2f}%)")
        print(f"Consecutive clips:     {stats['consecutive_clips']}")
        print(f"\nGradient Norms:")
        print(f"  Average:             {stats['avg_grad_norm']:.4f}")
        print(f"  Maximum:             {stats['max_grad_norm']:.4f}")
        
        if stats['clip_count'] > 0:
            print(f"\nWhen Clipped:")
            print(f"  Average norm:        {stats['avg_clipped_norm']:.4f}")
            print(f"  Average reduction:   {(1 - stats['avg_clip_ratio'])*100:.2f}%")
        
        print("="*60)
    
    def reset_statistics(self):
        """Reset all statistics."""
        self.total_steps = 0
        self.clip_count = 0
        self.grad_norms = []
        self.clipped_norms = []
        self.clip_ratios = []
        self.last_clip_step = -1
        self.consecutive_clips = 0
        self.history = defaultdict(list)
    
    def get_history(self) -> Dict:
        """
        Get complete history for plotting/analysis.
        
        Returns:
            Dictionary with lists:
                - step: Step numbers
                - grad_norm: Gradient norms at each step
                - clipped: Binary indicator (1 if clipped, 0 otherwise)
                - clip_ratio: Ratio of max_norm / grad_norm (1.0 if not clipped)
        """
        return dict(self.history)
    
    def should_warn(self) -> bool:
        """
        Check if clipping rate is concerning.
        
        Returns:
            True if clipping rate is high (>50%) or many consecutive clips
        """
        if self.total_steps < 10:
            return False
        
        stats = self.get_statistics()
        
        # High clipping rate
        if stats['clip_rate'] > 0.5:
            return True
        
        # Many consecutive clips
        if stats['consecutive_clips'] > 10:
            return True
        
        return False
    
    def get_warning_message(self) -> Optional[str]:
        """
        Get warning message if clipping behavior is concerning.
        
        Returns:
            Warning message string, or None if no warning
        """
        if not self.should_warn():
            return None
        
        stats = self.get_statistics()
        
        messages = []
        
        if stats['clip_rate'] > 0.5:
            messages.append(
                f"⚠️  High clipping rate: {stats['clip_rate']*100:.1f}% of steps are clipped.\n"
                f"   Consider: reducing learning rate, using gradient accumulation, or increasing max_norm"
            )
        
        if stats['consecutive_clips'] > 10:
            messages.append(
                f"⚠️  {stats['consecutive_clips']} consecutive clipping events detected.\n"
                f"   This may indicate training instability. Check your loss function and data."
            )
        
        return "\n".join(messages) if messages else None


# ============================================================================
# Convenience Functions
# ============================================================================

def clip_grad_norm(
    parameters: Union[torch.Tensor, Iterable[torch.Tensor]],
    max_norm: float = 1.0,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    log_clipping: bool = True
) -> float:
    """
    Simple gradient norm clipping with optional logging.
    
    This is a convenience function that creates a one-time clipper.
    For tracking statistics over training, use GradNormClipper class.
    
    Args:
        parameters: Model parameters or iterable of parameters
        max_norm: Maximum gradient norm (default: 1.0)
        norm_type: Type of norm (default: 2.0 for L2)
        error_if_nonfinite: Raise error on inf/nan gradients (default: False)
        log_clipping: Print message if clipping occurs (default: True)
        
    Returns:
        Total norm of gradients (before clipping)
        
    Example:
        >>> loss.backward()
        >>> grad_norm = clip_grad_norm(model.parameters(), max_norm=1.0)
        >>> optimizer.step()
    """
    clipper = GradNormClipper(
        max_norm=max_norm,
        norm_type=norm_type,
        log_clipping=log_clipping,
        error_if_nonfinite=error_if_nonfinite,
        verbose=log_clipping
    )
    
    return clipper.clip_grad_norm(parameters)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == '__main__':
    import torch.optim as optim
    
    print("="*60)
    print("GRADIENT NORM CLIPPER DEMO")
    print("="*60)
    
    # Create dummy model
    model = nn.Sequential(
        nn.Linear(10, 100),
        nn.ReLU(),
        nn.Linear(100, 1)
    )
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Create clipper
    clipper = GradNormClipper(max_norm=1.0, log_clipping=True, verbose=True)
    
    print("\nSimulating training with varying gradient magnitudes...\n")
    
    # Simulate training steps with varying gradient magnitudes
    for step in range(20):
        # Create dummy input/target
        x = torch.randn(32, 10)
        y = torch.randn(32, 1)
        
        # Forward pass
        pred = model(x)
        loss = ((pred - y) ** 2).mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Simulate gradient explosion every 5 steps
        if step % 5 == 4:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad *= 10.0  # Artificially increase gradient
        
        # Clip gradients
        grad_norm = clipper.clip_grad_norm(model.parameters())
        
        # Optimizer step
        optimizer.step()
    
    # Print statistics
    clipper.print_statistics()
    
    # Check for warnings
    warning = clipper.get_warning_message()
    if warning:
        print(f"\n{warning}")
    
    print("\nDemo complete!")
