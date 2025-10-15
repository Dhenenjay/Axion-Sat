"""
Training logger with JSONL output format.

This module provides structured logging for training runs with:
- JSONL (JSON Lines) format for easy parsing
- Automatic timestamp tracking
- GPU memory monitoring
- Loss and metric tracking
- Step/epoch/batch information
- Custom metadata support

JSONL format: Each line is a valid JSON object, making it easy to:
- Stream logs during training
- Parse with standard tools (jq, pandas, etc.)
- Append to existing logs
- Process incrementally

Example output format:
{"timestamp": "2025-10-13T10:30:45.123456", "step": 100, "epoch": 1, "loss": 0.123, "gpu_mem_gb": 8.5}
{"timestamp": "2025-10-13T10:30:46.234567", "step": 101, "epoch": 1, "loss": 0.119, "gpu_mem_gb": 8.5}
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import warnings


try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch not available, GPU memory logging disabled")


class TrainingLogger:
    """
    Logger for training metrics in JSONL format.
    
    Args:
        log_file: Path to log file (will be created if doesn't exist)
        append: Whether to append to existing file (default: False)
        auto_flush: Flush after each write (default: True for real-time updates)
        include_gpu_memory: Log GPU memory usage (default: True)
        include_timestamp: Include ISO format timestamp (default: True)
        log_system_info: Log system info on initialization (default: True)
        verbose: Print logs to console as well (default: False)
        
    Example:
        >>> logger = TrainingLogger("logs/training.jsonl")
        >>> 
        >>> for step in range(100):
        ...     loss = train_step()
        ...     logger.log_step(
        ...         step=step,
        ...         epoch=0,
        ...         loss=loss,
        ...         lr=0.001
        ...     )
        >>> 
        >>> logger.close()
    """
    
    def __init__(
        self,
        log_file: Union[str, Path],
        append: bool = False,
        auto_flush: bool = True,
        include_gpu_memory: bool = True,
        include_timestamp: bool = True,
        log_system_info: bool = True,
        verbose: bool = False
    ):
        self.log_file = Path(log_file)
        self.auto_flush = auto_flush
        self.include_gpu_memory = include_gpu_memory
        self.include_timestamp = include_timestamp
        self.verbose = verbose
        
        # Create parent directory
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Open file
        mode = 'a' if append else 'w'
        self.file_handle = open(self.log_file, mode)
        
        # Track state
        self.start_time = time.time()
        self.step_count = 0
        
        # Log system info
        if log_system_info:
            self._log_system_info()
    
    def _log_system_info(self):
        """Log system information at initialization."""
        info = {
            'event': 'training_start',
            'timestamp': self._get_timestamp(),
            'start_time_unix': self.start_time,
        }
        
        if HAS_TORCH and torch.cuda.is_available():
            info['cuda_available'] = True
            info['cuda_device_count'] = torch.cuda.device_count()
            info['cuda_device_name'] = torch.cuda.get_device_name(0)
            info['cuda_device_capability'] = torch.cuda.get_device_capability(0)
            info['total_vram_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
        else:
            info['cuda_available'] = False
        
        self._write_log(info)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.now().isoformat()
    
    def _get_gpu_memory(self) -> Optional[Dict[str, float]]:
        """Get GPU memory statistics."""
        if not HAS_TORCH or not torch.cuda.is_available():
            return None
        
        device = torch.cuda.current_device()
        return {
            'gpu_mem_allocated_gb': torch.cuda.memory_allocated(device) / 1e9,
            'gpu_mem_reserved_gb': torch.cuda.memory_reserved(device) / 1e9,
            'gpu_mem_max_allocated_gb': torch.cuda.max_memory_allocated(device) / 1e9,
        }
    
    def _write_log(self, log_dict: Dict[str, Any]):
        """Write a log entry to file."""
        # Write JSONL format (one JSON object per line)
        json_str = json.dumps(log_dict, default=str)
        self.file_handle.write(json_str + '\n')
        
        if self.auto_flush:
            self.file_handle.flush()
        
        if self.verbose:
            print(json_str)
    
    def log_step(
        self,
        step: int,
        epoch: Optional[int] = None,
        batch: Optional[int] = None,
        loss: Optional[float] = None,
        **metrics
    ):
        """
        Log a training step.
        
        Args:
            step: Global training step
            epoch: Current epoch (optional)
            batch: Batch number within epoch (optional)
            loss: Loss value (optional)
            **metrics: Additional metrics to log (lr, accuracy, etc.)
            
        Example:
            >>> logger.log_step(
            ...     step=100,
            ...     epoch=1,
            ...     batch=50,
            ...     loss=0.123,
            ...     lr=0.001,
            ...     accuracy=0.95
            ... )
        """
        log_dict = {
            'event': 'train_step',
            'step': step,
        }
        
        if self.include_timestamp:
            log_dict['timestamp'] = self._get_timestamp()
            log_dict['elapsed_seconds'] = time.time() - self.start_time
        
        if epoch is not None:
            log_dict['epoch'] = epoch
        
        if batch is not None:
            log_dict['batch'] = batch
        
        if loss is not None:
            log_dict['loss'] = float(loss)
        
        # Add GPU memory if enabled
        if self.include_gpu_memory:
            gpu_mem = self._get_gpu_memory()
            if gpu_mem:
                log_dict.update(gpu_mem)
        
        # Add custom metrics
        for key, value in metrics.items():
            # Convert tensors to Python scalars
            if HAS_TORCH and isinstance(value, torch.Tensor):
                value = value.item()
            log_dict[key] = value
        
        self._write_log(log_dict)
        self.step_count += 1
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        **metrics
    ):
        """
        Log epoch summary.
        
        Args:
            epoch: Epoch number
            train_loss: Average training loss for epoch
            val_loss: Validation loss for epoch
            **metrics: Additional epoch metrics
            
        Example:
            >>> logger.log_epoch(
            ...     epoch=1,
            ...     train_loss=0.123,
            ...     val_loss=0.145,
            ...     train_acc=0.95,
            ...     val_acc=0.93
            ... )
        """
        log_dict = {
            'event': 'epoch_end',
            'epoch': epoch,
        }
        
        if self.include_timestamp:
            log_dict['timestamp'] = self._get_timestamp()
            log_dict['elapsed_seconds'] = time.time() - self.start_time
        
        if train_loss is not None:
            log_dict['train_loss'] = float(train_loss)
        
        if val_loss is not None:
            log_dict['val_loss'] = float(val_loss)
        
        # Add custom metrics
        for key, value in metrics.items():
            if HAS_TORCH and isinstance(value, torch.Tensor):
                value = value.item()
            log_dict[key] = value
        
        self._write_log(log_dict)
    
    def log_validation(
        self,
        step: int,
        epoch: Optional[int] = None,
        val_loss: Optional[float] = None,
        **metrics
    ):
        """
        Log validation results.
        
        Args:
            step: Global training step
            epoch: Current epoch
            val_loss: Validation loss
            **metrics: Validation metrics
            
        Example:
            >>> logger.log_validation(
            ...     step=1000,
            ...     epoch=1,
            ...     val_loss=0.145,
            ...     val_acc=0.93,
            ...     val_psnr=28.5
            ... )
        """
        log_dict = {
            'event': 'validation',
            'step': step,
        }
        
        if self.include_timestamp:
            log_dict['timestamp'] = self._get_timestamp()
            log_dict['elapsed_seconds'] = time.time() - self.start_time
        
        if epoch is not None:
            log_dict['epoch'] = epoch
        
        if val_loss is not None:
            log_dict['val_loss'] = float(val_loss)
        
        # Add GPU memory if enabled
        if self.include_gpu_memory:
            gpu_mem = self._get_gpu_memory()
            if gpu_mem:
                log_dict.update(gpu_mem)
        
        # Add custom metrics
        for key, value in metrics.items():
            if HAS_TORCH and isinstance(value, torch.Tensor):
                value = value.item()
            log_dict[key] = value
        
        self._write_log(log_dict)
    
    def log_checkpoint(
        self,
        step: int,
        epoch: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
        is_best: bool = False,
        **metadata
    ):
        """
        Log checkpoint save event.
        
        Args:
            step: Global training step
            epoch: Current epoch
            checkpoint_path: Path where checkpoint was saved
            is_best: Whether this is the best model so far
            **metadata: Additional checkpoint metadata
        """
        log_dict = {
            'event': 'checkpoint_saved',
            'step': step,
            'is_best': is_best,
        }
        
        if self.include_timestamp:
            log_dict['timestamp'] = self._get_timestamp()
        
        if epoch is not None:
            log_dict['epoch'] = epoch
        
        if checkpoint_path is not None:
            log_dict['checkpoint_path'] = str(checkpoint_path)
        
        for key, value in metadata.items():
            log_dict[key] = value
        
        self._write_log(log_dict)
    
    def log_event(
        self,
        event_name: str,
        **data
    ):
        """
        Log a custom event.
        
        Args:
            event_name: Name of the event
            **data: Event data
            
        Example:
            >>> logger.log_event('model_saved', path='model.pt', size_mb=123.4)
            >>> logger.log_event('early_stopping', patience_exceeded=True)
        """
        log_dict = {
            'event': event_name,
        }
        
        if self.include_timestamp:
            log_dict['timestamp'] = self._get_timestamp()
        
        for key, value in data.items():
            if HAS_TORCH and isinstance(value, torch.Tensor):
                value = value.item()
            log_dict[key] = value
        
        self._write_log(log_dict)
    
    def close(self):
        """Close the log file."""
        if hasattr(self, 'file_handle') and self.file_handle:
            # Log training end
            self._write_log({
                'event': 'training_end',
                'timestamp': self._get_timestamp(),
                'total_elapsed_seconds': time.time() - self.start_time,
                'total_steps': self.step_count
            })
            
            self.file_handle.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


# ============================================================================
# Utility Functions
# ============================================================================

def read_jsonl_logs(log_file: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Read JSONL log file into a list of dictionaries.
    
    Args:
        log_file: Path to JSONL log file
        
    Returns:
        List of log entries as dictionaries
        
    Example:
        >>> logs = read_jsonl_logs("logs/training.jsonl")
        >>> print(f"Total steps: {len([l for l in logs if l['event'] == 'train_step'])}")
    """
    logs = []
    
    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError as e:
                    warnings.warn(f"Failed to parse log line: {line[:100]}... Error: {e}")
    
    return logs


def filter_logs(
    logs: List[Dict[str, Any]],
    event: Optional[str] = None,
    min_step: Optional[int] = None,
    max_step: Optional[int] = None,
    epoch: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Filter log entries based on criteria.
    
    Args:
        logs: List of log entries
        event: Filter by event type
        min_step: Minimum step (inclusive)
        max_step: Maximum step (inclusive)
        epoch: Filter by epoch
        
    Returns:
        Filtered list of log entries
        
    Example:
        >>> logs = read_jsonl_logs("logs/training.jsonl")
        >>> train_logs = filter_logs(logs, event='train_step', epoch=1)
    """
    filtered = logs
    
    if event is not None:
        filtered = [l for l in filtered if l.get('event') == event]
    
    if min_step is not None:
        filtered = [l for l in filtered if l.get('step', 0) >= min_step]
    
    if max_step is not None:
        filtered = [l for l in filtered if l.get('step', float('inf')) <= max_step]
    
    if epoch is not None:
        filtered = [l for l in filtered if l.get('epoch') == epoch]
    
    return filtered


def extract_metric(
    logs: List[Dict[str, Any]],
    metric_name: str,
    default: Any = None
) -> List[Any]:
    """
    Extract a specific metric from log entries.
    
    Args:
        logs: List of log entries
        metric_name: Name of metric to extract
        default: Default value if metric not present
        
    Returns:
        List of metric values
        
    Example:
        >>> logs = read_jsonl_logs("logs/training.jsonl")
        >>> train_logs = filter_logs(logs, event='train_step')
        >>> losses = extract_metric(train_logs, 'loss')
        >>> print(f"Min loss: {min(losses)}")
    """
    return [log.get(metric_name, default) for log in logs]


def logs_to_dataframe(logs: List[Dict[str, Any]]):
    """
    Convert logs to pandas DataFrame.
    
    Args:
        logs: List of log entries
        
    Returns:
        pandas DataFrame
        
    Example:
        >>> import pandas as pd
        >>> logs = read_jsonl_logs("logs/training.jsonl")
        >>> df = logs_to_dataframe(logs)
        >>> print(df[df['event'] == 'train_step'][['step', 'loss']].head())
    """
    try:
        import pandas as pd
        return pd.DataFrame(logs)
    except ImportError:
        raise ImportError("pandas is required for logs_to_dataframe()")


# ============================================================================
# Example / Testing
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Training Logger Example")
    print("="*70)
    
    # Create logger
    log_file = "test_training.jsonl"
    
    with TrainingLogger(log_file, verbose=True) as logger:
        print("\nSimulating training loop...")
        
        # Simulate training
        for epoch in range(2):
            print(f"\nEpoch {epoch}")
            
            # Training steps
            for batch in range(5):
                step = epoch * 5 + batch
                loss = 1.0 / (step + 1)  # Simulated decreasing loss
                
                logger.log_step(
                    step=step,
                    epoch=epoch,
                    batch=batch,
                    loss=loss,
                    lr=0.001,
                    grad_norm=0.5
                )
            
            # Epoch summary
            logger.log_epoch(
                epoch=epoch,
                train_loss=0.5 / (epoch + 1),
                val_loss=0.6 / (epoch + 1),
                train_acc=0.8 + epoch * 0.05,
                val_acc=0.75 + epoch * 0.05
            )
            
            # Checkpoint
            logger.log_checkpoint(
                step=step,
                epoch=epoch,
                checkpoint_path=f"checkpoint_epoch_{epoch}.pt",
                is_best=(epoch == 1)
            )
    
    print("\n" + "="*70)
    print("Reading and analyzing logs...")
    print("="*70)
    
    # Read logs
    logs = read_jsonl_logs(log_file)
    print(f"\nTotal log entries: {len(logs)}")
    
    # Filter train steps
    train_logs = filter_logs(logs, event='train_step')
    print(f"Training steps: {len(train_logs)}")
    
    # Extract losses
    losses = extract_metric(train_logs, 'loss')
    print(f"Loss range: {min(losses):.4f} - {max(losses):.4f}")
    
    # Filter epoch summaries
    epoch_logs = filter_logs(logs, event='epoch_end')
    print(f"Epochs completed: {len(epoch_logs)}")
    
    # Cleanup
    import os
    os.remove(log_file)
    print(f"\nCleaned up test file: {log_file}")
    
    print("\n" + "="*70)
    print("Example complete!")
    print("="*70)
