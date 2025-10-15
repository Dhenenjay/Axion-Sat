"""
Disk space utilities for monitoring and managing available disk space.

Functions for checking free disk space and waiting until sufficient space
is available before proceeding with operations.
"""

import os
import sys
import time
from pathlib import Path
from typing import Union


def get_free_gb(path: Union[str, Path]) -> float:
    """
    Get available free disk space in GB for the drive containing the given path.
    
    Args:
        path: Path to check (file or directory). Uses the drive/mount containing this path.
        
    Returns:
        Free space in gigabytes (GB) as a float.
        
    Examples:
        >>> free = get_free_gb('/home/user/data')
        >>> print(f"Free space: {free:.2f} GB")
        Free space: 45.32 GB
        
        >>> free = get_free_gb('C:\\Users\\Documents')
        >>> print(f"Free space: {free:.2f} GB")
        Free space: 123.45 GB
    """
    path = Path(path).resolve()
    
    # Ensure path exists or use parent directory
    if not path.exists():
        path = path.parent
    
    if sys.platform == 'win32':
        # Windows: Use ctypes to call GetDiskFreeSpaceExW
        import ctypes
        
        free_bytes = ctypes.c_ulonglong(0)
        total_bytes = ctypes.c_ulonglong(0)
        total_free_bytes = ctypes.c_ulonglong(0)
        
        # GetDiskFreeSpaceExW returns available space (respecting quotas)
        ret = ctypes.windll.kernel32.GetDiskFreeSpaceExW(
            str(path),
            ctypes.pointer(free_bytes),
            ctypes.pointer(total_bytes),
            ctypes.pointer(total_free_bytes)
        )
        
        if ret == 0:
            raise OSError(f"Failed to get disk space for {path}")
        
        return free_bytes.value / (1024 ** 3)
    
    else:
        # Unix/Linux/Mac: Use statvfs
        stat = os.statvfs(path)
        # f_bavail is available blocks for unprivileged users
        # f_frsize is fragment size (block size)
        free_bytes = stat.f_bavail * stat.f_frsize
        return free_bytes / (1024 ** 3)


def wait_for_space(
    path: Union[str, Path],
    min_free_gb: float,
    poll_s: int = 30,
    timeout_s: int = None
) -> float:
    """
    Wait until sufficient free disk space is available.
    
    Polls the disk space every `poll_s` seconds until the free space
    meets or exceeds `min_free_gb`. Prints status messages to console.
    
    Args:
        path: Path to monitor (file or directory).
        min_free_gb: Minimum required free space in GB.
        poll_s: Polling interval in seconds (default: 30).
        timeout_s: Optional timeout in seconds. If None, waits indefinitely.
                   Raises TimeoutError if timeout is exceeded.
    
    Returns:
        Current free space in GB when condition is met.
        
    Raises:
        TimeoutError: If timeout_s is specified and exceeded.
        KeyboardInterrupt: If user interrupts with Ctrl+C.
        
    Examples:
        >>> # Wait until at least 50 GB is free
        >>> free = wait_for_space('/data/output', min_free_gb=50.0)
        >>> print(f"Ready to proceed with {free:.2f} GB free")
        
        >>> # Wait with 1-minute polling and 30-minute timeout
        >>> try:
        ...     free = wait_for_space('/data', min_free_gb=100, poll_s=60, timeout_s=1800)
        ... except TimeoutError:
        ...     print("Timeout waiting for disk space")
    """
    path = Path(path).resolve()
    start_time = time.time()
    first_check = True
    
    while True:
        free_gb = get_free_gb(path)
        
        if free_gb >= min_free_gb:
            if not first_check:
                # Print success message if we were waiting
                print(f"\n✅ Sufficient space available: {free_gb:.2f} GB free "
                      f"(required: {min_free_gb:.2f} GB)")
            return free_gb
        
        # Calculate elapsed time
        elapsed = time.time() - start_time
        
        # Check timeout
        if timeout_s is not None and elapsed > timeout_s:
            raise TimeoutError(
                f"Timeout after {elapsed:.1f}s waiting for {min_free_gb:.2f} GB. "
                f"Current free space: {free_gb:.2f} GB"
            )
        
        # Print warning message
        deficit = min_free_gb - free_gb
        if first_check:
            print(f"\n⚠️  INSUFFICIENT DISK SPACE")
            print(f"   Path: {path}")
            print(f"   Current: {free_gb:.2f} GB free")
            print(f"   Required: {min_free_gb:.2f} GB")
            print(f"   Need: {deficit:.2f} GB more")
            print(f"\n   Waiting for space to free up...")
            print(f"   Checking every {poll_s}s (Press Ctrl+C to abort)")
            first_check = False
        else:
            # Update status on same line
            elapsed_str = f"{int(elapsed)}s" if elapsed < 60 else f"{int(elapsed/60)}m"
            print(f"   [{elapsed_str}] Still {deficit:.2f} GB short "
                  f"({free_gb:.2f} / {min_free_gb:.2f} GB)... ", 
                  end='\r', flush=True)
        
        # Sleep for polling interval
        try:
            time.sleep(poll_s)
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            raise


def get_drive_info(path: Union[str, Path]) -> dict:
    """
    Get detailed drive/filesystem information.
    
    Args:
        path: Path to check.
        
    Returns:
        Dictionary with keys:
            - 'total_gb': Total space in GB
            - 'used_gb': Used space in GB
            - 'free_gb': Free space in GB
            - 'percent_used': Percentage of space used
            - 'path': Resolved path checked
            
    Examples:
        >>> info = get_drive_info('/data')
        >>> print(f"Drive: {info['percent_used']:.1f}% full")
        Drive: 67.3% full
    """
    path = Path(path).resolve()
    
    if not path.exists():
        path = path.parent
    
    if sys.platform == 'win32':
        import ctypes
        
        free_bytes = ctypes.c_ulonglong(0)
        total_bytes = ctypes.c_ulonglong(0)
        total_free_bytes = ctypes.c_ulonglong(0)
        
        ret = ctypes.windll.kernel32.GetDiskFreeSpaceExW(
            str(path),
            ctypes.pointer(free_bytes),
            ctypes.pointer(total_bytes),
            ctypes.pointer(total_free_bytes)
        )
        
        if ret == 0:
            raise OSError(f"Failed to get disk space for {path}")
        
        free_gb = free_bytes.value / (1024 ** 3)
        total_gb = total_bytes.value / (1024 ** 3)
        used_gb = total_gb - free_gb
        
    else:
        stat = os.statvfs(path)
        total_gb = (stat.f_blocks * stat.f_frsize) / (1024 ** 3)
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
        used_gb = total_gb - free_gb
    
    percent_used = (used_gb / total_gb * 100) if total_gb > 0 else 0
    
    return {
        'total_gb': total_gb,
        'used_gb': used_gb,
        'free_gb': free_gb,
        'percent_used': percent_used,
        'path': str(path)
    }


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes into human-readable string.
    
    Args:
        bytes_value: Number of bytes.
        
    Returns:
        Formatted string (e.g., "1.5 GB", "234.5 MB").
        
    Examples:
        >>> format_bytes(1536)
        '1.5 KB'
        >>> format_bytes(1073741824)
        '1.0 GB'
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


# Convenience function for quick checks
def has_space(path: Union[str, Path], required_gb: float) -> bool:
    """
    Check if path has at least the required amount of free space.
    
    Args:
        path: Path to check.
        required_gb: Required free space in GB.
        
    Returns:
        True if sufficient space available, False otherwise.
        
    Examples:
        >>> if has_space('/output', 50.0):
        ...     print("OK to proceed")
        ... else:
        ...     print("Need more space")
    """
    return get_free_gb(path) >= required_gb


if __name__ == '__main__':
    # Demo usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Check disk space')
    parser.add_argument('path', nargs='?', default='.', help='Path to check')
    parser.add_argument('--wait', type=float, help='Wait for this many GB to be free')
    parser.add_argument('--poll', type=int, default=30, help='Polling interval in seconds')
    
    args = parser.parse_args()
    
    if args.wait:
        print(f"Waiting for {args.wait:.2f} GB free space on {args.path}")
        free = wait_for_space(args.path, args.wait, poll_s=args.poll)
        print(f"✅ Space available: {free:.2f} GB")
    else:
        info = get_drive_info(args.path)
        print(f"\nDisk space for: {info['path']}")
        print(f"  Total:  {info['total_gb']:.2f} GB")
        print(f"  Used:   {info['used_gb']:.2f} GB ({info['percent_used']:.1f}%)")
        print(f"  Free:   {info['free_gb']:.2f} GB")
