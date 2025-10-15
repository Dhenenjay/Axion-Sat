"""
Path Utilities: Windows-Safe File I/O Operations

Provides utilities for cross-platform path handling with special focus on
Windows compatibility. All path operations use pathlib.Path for consistency.

Key Features:
    - Windows-safe path normalization
    - Automatic backslash handling
    - Path validation and sanitization
    - Cross-platform compatibility

Windows Path Issues:
    1. **Backslash separators**: Windows uses '\\' while Unix uses '/'
    2. **Drive letters**: Windows has C:, D:, etc.
    3. **UNC paths**: Network paths like \\\\server\\share
    4. **Case insensitivity**: Windows paths are case-insensitive
    5. **Path length limits**: MAX_PATH = 260 characters (can be extended)
    6. **Reserved names**: CON, PRN, AUX, NUL, COM1-9, LPT1-9
    7. **Invalid characters**: < > : " | ? *

Best Practices:
    - Always use pathlib.Path for path operations
    - Use Path.resolve() for absolute paths
    - Use Path.as_posix() for forward slashes
    - Use raw strings (r"...") or forward slashes in literals
    - Avoid string concatenation for paths

Usage:
    >>> from axs_lib.path_utils import ensure_path, safe_mkdir, list_files
    >>> 
    >>> # Safe path creation
    >>> path = ensure_path("data/tiles/sample.npz")  # Works on Windows & Unix
    >>> 
    >>> # Safe directory creation
    >>> safe_mkdir("outputs/checkpoints")
    >>> 
    >>> # Cross-platform file listing
    >>> files = list_files("data", pattern="*.npz")

Author: Axion-Sat Project
Version: 1.0.0
"""

import os
import sys
import platform
from pathlib import Path, PurePath, PureWindowsPath, PurePosixPath
from typing import Union, List, Optional, Iterator
import warnings


# ============================================================================
# Platform Detection
# ============================================================================

def is_windows() -> bool:
    """Check if running on Windows."""
    return platform.system() == 'Windows' or sys.platform == 'win32'


def is_linux() -> bool:
    """Check if running on Linux."""
    return platform.system() == 'Linux' or sys.platform.startswith('linux')


def is_mac() -> bool:
    """Check if running on macOS."""
    return platform.system() == 'Darwin' or sys.platform == 'darwin'


# ============================================================================
# Path Conversion & Normalization
# ============================================================================

def ensure_path(path: Union[str, Path, PurePath], resolve: bool = False) -> Path:
    """
    Convert path-like object to pathlib.Path with proper normalization.
    
    Handles:
    - String paths with forward or backward slashes
    - Existing Path objects
    - Path-like objects
    - Automatic normalization
    
    Args:
        path: Path as string, Path, or path-like object
        resolve: If True, resolve to absolute path
        
    Returns:
        Normalized Path object
        
    Examples:
        >>> ensure_path("data/tiles/sample.npz")
        WindowsPath('data/tiles/sample.npz')  # On Windows
        
        >>> ensure_path("C:\\Users\\data")  # Backslashes work
        WindowsPath('C:/Users/data')
        
        >>> ensure_path(Path("data") / "tiles")
        WindowsPath('data/tiles')
    """
    if isinstance(path, Path):
        p = path
    else:
        # Convert string to Path (handles both / and \\ automatically)
        p = Path(path)
    
    # Optionally resolve to absolute path
    if resolve:
        p = p.resolve()
    
    return p


def normalize_path(path: Union[str, Path], posix: bool = False) -> str:
    """
    Normalize path to consistent format.
    
    Args:
        path: Path to normalize
        posix: If True, use forward slashes (POSIX style)
        
    Returns:
        Normalized path string
        
    Examples:
        >>> normalize_path("data\\tiles\\file.npz")
        'data/tiles/file.npz'  # On all platforms
        
        >>> normalize_path("data/tiles/file.npz", posix=True)
        'data/tiles/file.npz'
    """
    p = ensure_path(path)
    
    if posix:
        return p.as_posix()
    else:
        return str(p)


def to_posix_path(path: Union[str, Path]) -> str:
    """
    Convert path to POSIX format (forward slashes).
    
    Useful for:
    - Configuration files
    - URLs
    - Cross-platform compatibility
    
    Args:
        path: Path to convert
        
    Returns:
        Path with forward slashes
        
    Examples:
        >>> to_posix_path("C:\\Users\\data\\file.txt")
        'C:/Users/data/file.txt'
    """
    return ensure_path(path).as_posix()


def to_native_path(path: Union[str, Path]) -> str:
    """
    Convert path to native platform format.
    
    Args:
        path: Path to convert
        
    Returns:
        Path in native format (backslashes on Windows, forward slashes on Unix)
        
    Examples:
        >>> to_native_path("data/tiles/file.npz")
        'data\\tiles\\file.npz'  # On Windows
        'data/tiles/file.npz'    # On Unix
    """
    return str(ensure_path(path))


# ============================================================================
# Path Validation
# ============================================================================

def validate_path(path: Union[str, Path], must_exist: bool = False) -> bool:
    """
    Validate if path is valid and optionally exists.
    
    Args:
        path: Path to validate
        must_exist: If True, check if path exists
        
    Returns:
        True if path is valid (and exists if must_exist=True)
        
    Examples:
        >>> validate_path("data/tiles")
        True
        
        >>> validate_path("data/tiles", must_exist=True)
        False  # If directory doesn't exist
    """
    try:
        p = ensure_path(path)
        
        if must_exist:
            return p.exists()
        
        return True
    except (ValueError, OSError):
        return False


def is_safe_filename(filename: str) -> bool:
    """
    Check if filename is safe (no invalid characters or reserved names).
    
    Windows reserved names: CON, PRN, AUX, NUL, COM1-9, LPT1-9
    Invalid characters: < > : " | ? * and control characters
    
    Args:
        filename: Filename to check (without path)
        
    Returns:
        True if filename is safe
        
    Examples:
        >>> is_safe_filename("data.npz")
        True
        
        >>> is_safe_filename("data<test>.npz")
        False  # Contains invalid character
        
        >>> is_safe_filename("CON.txt")
        False  # Reserved name on Windows
    """
    if not filename or filename in ('.', '..'):
        return False
    
    # Check for invalid characters
    invalid_chars = '<>:"|?*'
    if any(char in filename for char in invalid_chars):
        return False
    
    # Check for control characters
    if any(ord(char) < 32 for char in filename):
        return False
    
    # Check for Windows reserved names
    if is_windows():
        reserved_names = {
            'CON', 'PRN', 'AUX', 'NUL',
            'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
            'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
        }
        
        # Get filename without extension
        name = Path(filename).stem.upper()
        if name in reserved_names:
            return False
    
    return True


def sanitize_filename(filename: str, replacement: str = '_') -> str:
    """
    Sanitize filename by replacing invalid characters.
    
    Args:
        filename: Filename to sanitize
        replacement: Character to use for replacements
        
    Returns:
        Sanitized filename
        
    Examples:
        >>> sanitize_filename("data<test>.npz")
        'data_test_.npz'
        
        >>> sanitize_filename("my file:name.txt")
        'my file_name.txt'
    """
    # Replace invalid characters
    invalid_chars = '<>:"|?*'
    for char in invalid_chars:
        filename = filename.replace(char, replacement)
    
    # Remove control characters
    filename = ''.join(char for char in filename if ord(char) >= 32)
    
    # Handle Windows reserved names
    if is_windows():
        name = Path(filename).stem
        ext = Path(filename).suffix
        
        reserved_names = {
            'CON', 'PRN', 'AUX', 'NUL',
            'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
            'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
        }
        
        if name.upper() in reserved_names:
            filename = f"{name}{replacement}{ext}"
    
    return filename


# ============================================================================
# Directory Operations
# ============================================================================

def safe_mkdir(path: Union[str, Path], parents: bool = True, exist_ok: bool = True) -> Path:
    """
    Safely create directory with proper error handling.
    
    Args:
        path: Directory path to create
        parents: Create parent directories if needed
        exist_ok: Don't raise error if directory exists
        
    Returns:
        Path object for created directory
        
    Examples:
        >>> safe_mkdir("outputs/checkpoints/stage1")
        WindowsPath('outputs/checkpoints/stage1')
    """
    p = ensure_path(path)
    p.mkdir(parents=parents, exist_ok=exist_ok)
    return p


def list_files(
    directory: Union[str, Path],
    pattern: str = "*",
    recursive: bool = False,
    absolute: bool = False
) -> List[Path]:
    """
    List files in directory with pattern matching.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern (e.g., "*.npz", "**/*.py")
        recursive: Search recursively
        absolute: Return absolute paths
        
    Returns:
        List of Path objects
        
    Examples:
        >>> files = list_files("data/tiles", pattern="*.npz")
        >>> files = list_files("data", pattern="**/*.npz", recursive=True)
    """
    dir_path = ensure_path(directory, resolve=absolute)
    
    if not dir_path.exists():
        warnings.warn(f"Directory does not exist: {dir_path}")
        return []
    
    if not dir_path.is_dir():
        warnings.warn(f"Path is not a directory: {dir_path}")
        return []
    
    if recursive:
        files = list(dir_path.rglob(pattern))
    else:
        files = list(dir_path.glob(pattern))
    
    # Filter out directories
    files = [f for f in files if f.is_file()]
    
    if absolute:
        files = [f.resolve() for f in files]
    
    return sorted(files)


def iter_files(
    directory: Union[str, Path],
    pattern: str = "*",
    recursive: bool = False
) -> Iterator[Path]:
    """
    Iterate over files in directory (generator for large directories).
    
    Args:
        directory: Directory to search
        pattern: Glob pattern
        recursive: Search recursively
        
    Yields:
        Path objects for each matching file
        
    Examples:
        >>> for file in iter_files("data/tiles", "*.npz"):
        ...     process(file)
    """
    dir_path = ensure_path(directory)
    
    if not dir_path.exists() or not dir_path.is_dir():
        return
    
    if recursive:
        iterator = dir_path.rglob(pattern)
    else:
        iterator = dir_path.glob(pattern)
    
    for item in iterator:
        if item.is_file():
            yield item


# ============================================================================
# Path Information
# ============================================================================

def get_path_info(path: Union[str, Path]) -> dict:
    """
    Get detailed information about a path.
    
    Args:
        path: Path to inspect
        
    Returns:
        Dictionary with path information
        
    Examples:
        >>> info = get_path_info("data/tiles/sample.npz")
        >>> print(info['absolute_path'])
    """
    p = ensure_path(path)
    
    info = {
        'path': str(p),
        'absolute_path': str(p.resolve()) if p.exists() else None,
        'posix_path': p.as_posix(),
        'name': p.name,
        'stem': p.stem,
        'suffix': p.suffix,
        'parent': str(p.parent),
        'exists': p.exists(),
        'is_file': p.is_file() if p.exists() else None,
        'is_dir': p.is_dir() if p.exists() else None,
        'is_absolute': p.is_absolute(),
        'platform': platform.system()
    }
    
    if p.exists():
        stat = p.stat()
        info.update({
            'size_bytes': stat.st_size,
            'size_mb': stat.st_size / (1024 ** 2),
            'modified_time': stat.st_mtime,
        })
    
    return info


def print_path_info(path: Union[str, Path]):
    """
    Print formatted path information.
    
    Args:
        path: Path to inspect
        
    Examples:
        >>> print_path_info("data/tiles/sample.npz")
    """
    info = get_path_info(path)
    
    print("=" * 80)
    print(f"Path Information: {info['name']}")
    print("=" * 80)
    print(f"  Path:          {info['path']}")
    print(f"  Absolute:      {info['absolute_path']}")
    print(f"  POSIX:         {info['posix_path']}")
    print(f"  Name:          {info['name']}")
    print(f"  Stem:          {info['stem']}")
    print(f"  Suffix:        {info['suffix']}")
    print(f"  Parent:        {info['parent']}")
    print(f"  Exists:        {info['exists']}")
    print(f"  Is File:       {info['is_file']}")
    print(f"  Is Directory:  {info['is_dir']}")
    print(f"  Is Absolute:   {info['is_absolute']}")
    print(f"  Platform:      {info['platform']}")
    
    if info['exists']:
        print(f"  Size:          {info['size_mb']:.2f} MB ({info['size_bytes']:,} bytes)")
    
    print("=" * 80)


# ============================================================================
# CLI for Testing
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Path utilities testing and diagnostics"
    )
    
    parser.add_argument(
        'path',
        nargs='?',
        default='.',
        help='Path to inspect (default: current directory)'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List files in directory'
    )
    
    parser.add_argument(
        '--pattern',
        default='*',
        help='File pattern for listing (default: *)'
    )
    
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Search recursively'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("Path Utilities - Testing & Diagnostics")
    print("=" * 80)
    print()
    
    print(f"Platform:   {platform.system()}")
    print(f"Is Windows: {is_windows()}")
    print(f"Is Linux:   {is_linux()}")
    print(f"Is macOS:   {is_mac()}")
    print()
    
    # Test path operations
    test_path = args.path
    print(f"Testing path: {test_path}")
    print()
    
    # Print path info
    print_path_info(test_path)
    print()
    
    # List files if requested
    if args.list:
        path = ensure_path(test_path)
        if path.is_dir():
            print(f"Listing files in {path} (pattern: {args.pattern}, recursive: {args.recursive})")
            print("-" * 80)
            
            files = list_files(path, pattern=args.pattern, recursive=args.recursive)
            
            if files:
                for i, file in enumerate(files[:20], 1):  # Show first 20
                    print(f"  {i:3d}. {file}")
                
                if len(files) > 20:
                    print(f"  ... and {len(files) - 20} more files")
                
                print(f"\nTotal: {len(files)} file(s)")
            else:
                print("  No files found")
        else:
            print("Path is not a directory - cannot list files")
    
    # Test filename sanitization
    print("\nFilename Sanitization Tests:")
    print("-" * 80)
    test_filenames = [
        "normal_file.txt",
        "file<with>special.txt",
        "CON.txt",
        "data:test.npz",
        "file|name?.csv"
    ]
    
    for filename in test_filenames:
        is_safe = is_safe_filename(filename)
        sanitized = sanitize_filename(filename)
        print(f"  {filename:30s} → Safe: {is_safe:5} → {sanitized}")
    
    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)
