#!/usr/bin/env python3
"""
timer_ctx.py - Timing and Memory Profiling Context Manager

This module provides a context manager for measuring wall time and peak GPU
memory usage for code blocks. Useful for profiling training, inference, and
data loading operations.

Usage:
    from tools.timer_ctx import Timer
    
    with Timer("model inference"):
        output = model(input)
    
    # With custom logging
    with Timer("training epoch", logger=my_logger, log_level="INFO"):
        train_one_epoch()
    
    # Manual statistics
    timer = Timer("data loading")
    with timer:
        data = load_data()
    print(f"Took {timer.elapsed:.2f}s, peak VRAM: {timer.peak_vram_gb:.2f} GB")

Features:
    - Wall time measurement
    - Peak GPU memory tracking (CUDA)
    - Automatic logging
    - Nested timing support
    - CPU memory tracking (optional)
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Union

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


# ============================================================================
# Configuration
# ============================================================================

# Default logger
DEFAULT_LOGGER = logging.getLogger(__name__)

# ANSI color codes for terminal output
COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_CYAN = "\033[36m"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TimerStats:
    """Statistics collected by Timer context manager."""
    
    name: str
    elapsed_seconds: float
    peak_vram_gb: float
    peak_vram_mb: float
    start_vram_gb: float
    end_vram_gb: float
    vram_delta_gb: float
    peak_cpu_ram_gb: Optional[float] = None
    start_cpu_ram_gb: Optional[float] = None
    end_cpu_ram_gb: Optional[float] = None
    cpu_ram_delta_gb: Optional[float] = None
    
    def __str__(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Timer: {self.name}",
            f"  Wall time: {self.elapsed_seconds:.3f}s",
            f"  Peak VRAM: {self.peak_vram_gb:.2f} GB ({self.peak_vram_mb:.0f} MB)",
            f"  VRAM delta: {self.vram_delta_gb:+.2f} GB",
        ]
        
        if self.peak_cpu_ram_gb is not None:
            lines.append(f"  Peak CPU RAM: {self.peak_cpu_ram_gb:.2f} GB")
            if self.cpu_ram_delta_gb is not None:
                lines.append(f"  CPU RAM delta: {self.cpu_ram_delta_gb:+.2f} GB")
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "peak_vram_gb": round(self.peak_vram_gb, 2),
            "peak_vram_mb": round(self.peak_vram_mb, 0),
            "start_vram_gb": round(self.start_vram_gb, 2),
            "end_vram_gb": round(self.end_vram_gb, 2),
            "vram_delta_gb": round(self.vram_delta_gb, 2),
            "peak_cpu_ram_gb": round(self.peak_cpu_ram_gb, 2) if self.peak_cpu_ram_gb else None,
            "start_cpu_ram_gb": round(self.start_cpu_ram_gb, 2) if self.start_cpu_ram_gb else None,
            "end_cpu_ram_gb": round(self.end_cpu_ram_gb, 2) if self.end_cpu_ram_gb else None,
            "cpu_ram_delta_gb": round(self.cpu_ram_delta_gb, 2) if self.cpu_ram_delta_gb else None,
        }


# ============================================================================
# Timer Context Manager
# ============================================================================

class Timer:
    """
    Context manager for timing code blocks and tracking memory usage.
    
    Attributes:
        name: Descriptive name for the timed block
        elapsed: Elapsed time in seconds (after context exit)
        peak_vram_gb: Peak GPU memory usage in GB (after context exit)
        peak_vram_mb: Peak GPU memory usage in MB (after context exit)
        stats: Full statistics (after context exit)
        
    Example:
        >>> with Timer("data loading"):
        ...     data = load_large_dataset()
        [Timer] data loading: 3.456s | Peak VRAM: 2.34 GB
        
        >>> timer = Timer("inference", auto_log=False)
        >>> with timer:
        ...     output = model(input)
        >>> print(f"Inference took {timer.elapsed:.2f}s")
        Inference took 0.15s
    """
    
    def __init__(
        self,
        name: str = "code block",
        logger: Optional[logging.Logger] = None,
        log_level: Union[str, int] = logging.INFO,
        auto_log: bool = True,
        track_cpu_ram: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        color: bool = True,
    ):
        """
        Initialize Timer context manager.
        
        Args:
            name: Descriptive name for the timed block
            logger: Logger instance (uses default if None)
            log_level: Logging level (INFO, DEBUG, etc.)
            auto_log: Automatically log results on context exit
            track_cpu_ram: Track CPU RAM usage (requires psutil)
            device: PyTorch device to track (default: cuda:0 if available)
            color: Use ANSI colors in log output
        """
        self.name = name
        self.logger = logger or DEFAULT_LOGGER
        self.log_level = log_level if isinstance(log_level, int) else getattr(logging, log_level.upper())
        self.auto_log = auto_log
        self.track_cpu_ram = track_cpu_ram
        self.color = color
        
        # Resolve device
        if device is None and TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device is not None and TORCH_AVAILABLE:
            self.device = torch.device(device)
        else:
            self.device = None
        
        # State
        self._start_time = None
        self._end_time = None
        self._start_vram = None
        self._end_vram = None
        self._peak_vram = None
        self._start_cpu_ram = None
        self._end_cpu_ram = None
        self._peak_cpu_ram = None
        
        # Results (populated after context exit)
        self.elapsed = None
        self.peak_vram_gb = None
        self.peak_vram_mb = None
        self.stats = None
    
    def __enter__(self):
        """Enter context: start timer and record initial memory."""
        self._start_time = time.perf_counter()
        
        # Record GPU memory
        if self.device is not None and self.device.type == "cuda" and TORCH_AVAILABLE:
            torch.cuda.reset_peak_memory_stats(self.device)
            self._start_vram = torch.cuda.memory_allocated(self.device)
        else:
            self._start_vram = 0
        
        # Record CPU memory
        if self.track_cpu_ram and PSUTIL_AVAILABLE:
            process = psutil.Process()
            self._start_cpu_ram = process.memory_info().rss
            self._peak_cpu_ram = self._start_cpu_ram
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context: stop timer, record final memory, log results."""
        self._end_time = time.perf_counter()
        
        # Record GPU memory
        if self.device is not None and self.device.type == "cuda" and TORCH_AVAILABLE:
            self._end_vram = torch.cuda.memory_allocated(self.device)
            self._peak_vram = torch.cuda.max_memory_allocated(self.device)
        else:
            self._end_vram = 0
            self._peak_vram = 0
        
        # Record CPU memory
        if self.track_cpu_ram and PSUTIL_AVAILABLE:
            process = psutil.Process()
            self._end_cpu_ram = process.memory_info().rss
            # Peak CPU is already tracked
        
        # Calculate results
        self.elapsed = self._end_time - self._start_time
        self.peak_vram_gb = self._peak_vram / (1024 ** 3)
        self.peak_vram_mb = self._peak_vram / (1024 ** 2)
        
        # Build statistics
        self.stats = TimerStats(
            name=self.name,
            elapsed_seconds=self.elapsed,
            peak_vram_gb=self.peak_vram_gb,
            peak_vram_mb=self.peak_vram_mb,
            start_vram_gb=self._start_vram / (1024 ** 3),
            end_vram_gb=self._end_vram / (1024 ** 3),
            vram_delta_gb=(self._end_vram - self._start_vram) / (1024 ** 3),
            peak_cpu_ram_gb=self._peak_cpu_ram / (1024 ** 3) if self._peak_cpu_ram else None,
            start_cpu_ram_gb=self._start_cpu_ram / (1024 ** 3) if self._start_cpu_ram else None,
            end_cpu_ram_gb=self._end_cpu_ram / (1024 ** 3) if self._end_cpu_ram else None,
            cpu_ram_delta_gb=(self._end_cpu_ram - self._start_cpu_ram) / (1024 ** 3) 
                if self._end_cpu_ram and self._start_cpu_ram else None,
        )
        
        # Auto-log if enabled
        if self.auto_log:
            self._log_results()
        
        # Don't suppress exceptions
        return False
    
    def _log_results(self):
        """Log timing and memory results."""
        if self.color:
            color_name = COLOR_CYAN
            color_time = COLOR_GREEN
            color_vram = COLOR_YELLOW
            color_reset = COLOR_RESET
        else:
            color_name = color_time = color_vram = color_reset = ""
        
        # Build log message
        message = (
            f"{color_name}[Timer]{color_reset} {self.name}: "
            f"{color_time}{self.elapsed:.3f}s{color_reset}"
        )
        
        if self._peak_vram > 0:
            message += f" | Peak VRAM: {color_vram}{self.peak_vram_gb:.2f} GB{color_reset}"
        
        if self.track_cpu_ram and self._peak_cpu_ram:
            peak_cpu_gb = self._peak_cpu_ram / (1024 ** 3)
            message += f" | Peak CPU RAM: {peak_cpu_gb:.2f} GB"
        
        self.logger.log(self.log_level, message)
    
    def get_current_vram_gb(self) -> float:
        """Get current GPU memory usage in GB."""
        if self.device is not None and self.device.type == "cuda" and TORCH_AVAILABLE:
            return torch.cuda.memory_allocated(self.device) / (1024 ** 3)
        return 0.0
    
    def reset(self):
        """Reset timer (allows reuse of same Timer instance)."""
        self._start_time = None
        self._end_time = None
        self._start_vram = None
        self._end_vram = None
        self._peak_vram = None
        self._start_cpu_ram = None
        self._end_cpu_ram = None
        self._peak_cpu_ram = None
        self.elapsed = None
        self.peak_vram_gb = None
        self.peak_vram_mb = None
        self.stats = None


# ============================================================================
# Convenience Functions
# ============================================================================

@contextmanager
def time_block(name: str = "code block", **kwargs):
    """
    Convenience function for timing a code block.
    
    Args:
        name: Descriptive name
        **kwargs: Additional arguments passed to Timer
        
    Example:
        >>> with time_block("model training"):
        ...     train_model()
    """
    timer = Timer(name, **kwargs)
    with timer:
        yield timer


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string (e.g., "2.5s", "1m 30s", "1h 5m")
        
    Example:
        >>> format_time(2.5)
        '2.50s'
        >>> format_time(90)
        '1m 30s'
        >>> format_time(3665)
        '1h 1m 5s'
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"


def format_memory(bytes_value: Union[int, float]) -> str:
    """
    Format bytes into human-readable string.
    
    Args:
        bytes_value: Memory size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 GB", "512 MB")
        
    Example:
        >>> format_memory(1024**3)
        '1.00 GB'
        >>> format_memory(512 * 1024**2)
        '512.00 MB'
    """
    if bytes_value >= 1024 ** 3:
        return f"{bytes_value / (1024 ** 3):.2f} GB"
    elif bytes_value >= 1024 ** 2:
        return f"{bytes_value / (1024 ** 2):.2f} MB"
    elif bytes_value >= 1024:
        return f"{bytes_value / 1024:.2f} KB"
    else:
        return f"{bytes_value} bytes"


# ============================================================================
# Examples and Testing
# ============================================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    
    print("=" * 79)
    print("Timer Context Manager - Examples")
    print("=" * 79)
    print()
    
    # Example 1: Basic timing
    print("Example 1: Basic timing")
    with Timer("sleep test"):
        time.sleep(0.5)
    print()
    
    # Example 2: Manual statistics
    print("Example 2: Manual statistics")
    timer = Timer("computation", auto_log=False)
    with timer:
        result = sum(range(10_000_000))
    print(f"Elapsed: {timer.elapsed:.3f}s")
    print(f"Result: {result}")
    print()
    
    # Example 3: GPU memory tracking (if available)
    if TORCH_AVAILABLE and torch.cuda.is_available():
        print("Example 3: GPU memory tracking")
        with Timer("tensor allocation"):
            x = torch.randn(10000, 10000, device="cuda")
            y = x @ x.T
        print()
    
    # Example 4: CPU RAM tracking
    if PSUTIL_AVAILABLE:
        print("Example 4: CPU RAM tracking")
        with Timer("large list allocation", track_cpu_ram=True):
            big_list = [i for i in range(10_000_000)]
        print()
    
    # Example 5: Nested timers
    print("Example 5: Nested timers")
    with Timer("outer operation"):
        time.sleep(0.1)
        with Timer("inner operation 1"):
            time.sleep(0.2)
        with Timer("inner operation 2"):
            time.sleep(0.15)
    print()
    
    # Example 6: Full statistics
    print("Example 6: Full statistics")
    timer = Timer("detailed test", auto_log=False, track_cpu_ram=True)
    with timer:
        time.sleep(0.3)
    print(timer.stats)
    print()
    print(f"Stats dict: {timer.stats.to_dict()}")
    print()
    
    # Example 7: Convenience function
    print("Example 7: Convenience function")
    with time_block("quick test") as t:
        time.sleep(0.1)
        current_vram = t.get_current_vram_gb()
    print()
    
    print("=" * 79)
    print("All examples completed!")
    print("=" * 79)
