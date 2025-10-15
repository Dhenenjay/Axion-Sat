#!/usr/bin/env python3
"""
win_path_check.py - Windows Path Length Validator

This tool checks all file paths in the project to ensure they don't exceed
Windows path length limitations. Windows has historically had a MAX_PATH
limit of 260 characters, though this can be extended with long path support.

For safety, this tool enforces a 240-character limit to leave headroom for:
- File operations that append temporary suffixes
- User home directories that may be deeply nested
- Working directory changes during runtime

Usage:
    python tools/win_path_check.py                    # Check all files
    python tools/win_path_check.py --limit 260        # Custom limit
    python tools/win_path_check.py --fix              # Suggest fixes
    python tools/win_path_check.py --dir ./outputs    # Check specific dir

Exit Codes:
    0 - All paths valid
    1 - Paths exceed limit (warnings)
    2 - Critical error

Why This Matters on Windows:
    - Legacy MAX_PATH = 260 characters
    - Many tools/APIs still enforce this limit
    - Errors like "FileNotFoundError" or "OSError: [Errno 2]"
    - Issues with git, file explorers, installers
    - Prevention is easier than fixing nested directories
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple


# ============================================================================
# Configuration
# ============================================================================

# Default path length limit (conservative for Windows compatibility)
DEFAULT_PATH_LIMIT = 240

# Windows theoretical maximum (with long path support enabled)
WINDOWS_MAX_PATH = 260

# ANSI color codes
COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_RED = "\033[31m"
COLOR_CYAN = "\033[36m"

# Directories to exclude from scanning
EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    ".eggs",
    "*.egg-info",
}

# Files to always exclude
EXCLUDE_FILES = {
    ".DS_Store",
    "Thumbs.db",
    "desktop.ini",
}


# ============================================================================
# Helper Functions
# ============================================================================

def print_header(message: str) -> None:
    """Print formatted header."""
    print(f"\n{COLOR_BOLD}{COLOR_CYAN}{'=' * 79}{COLOR_RESET}")
    print(f"{COLOR_BOLD}{COLOR_GREEN}{message}{COLOR_RESET}")
    print(f"{COLOR_BOLD}{COLOR_CYAN}{'=' * 79}{COLOR_RESET}\n")


def print_warning(message: str) -> None:
    """Print warning message."""
    print(f"{COLOR_YELLOW}⚠ WARNING: {message}{COLOR_RESET}")


def print_error(message: str) -> None:
    """Print error message."""
    print(f"{COLOR_BOLD}{COLOR_RED}✗ ERROR: {message}{COLOR_RESET}")


def print_success(message: str) -> None:
    """Print success message."""
    print(f"{COLOR_GREEN}✓ {message}{COLOR_RESET}")


def print_info(message: str) -> None:
    """Print info message."""
    print(f"{COLOR_CYAN}{message}{COLOR_RESET}")


def should_exclude(path: Path) -> bool:
    """Check if path should be excluded from scanning."""
    # Check if any part of the path matches exclude patterns
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return True
        if part in EXCLUDE_FILES:
            return True
        # Wildcard matching for patterns like *.egg-info
        for exclude_pattern in EXCLUDE_DIRS:
            if "*" in exclude_pattern:
                pattern = exclude_pattern.replace("*", "")
                if pattern in part:
                    return True
    return False


def get_absolute_path_length(path: Path) -> int:
    """
    Get the absolute path length of a file.
    
    Args:
        path: Path to check
        
    Returns:
        Length of absolute path string
    """
    try:
        abs_path = path.resolve()
        return len(str(abs_path))
    except Exception:
        # If resolve fails, use absolute path
        return len(str(path.absolute()))


def scan_directory(root_dir: Path, limit: int) -> Tuple[List[Tuple[Path, int]], int]:
    """
    Recursively scan directory for files and check path lengths.
    
    Args:
        root_dir: Root directory to scan
        limit: Path length limit
        
    Returns:
        Tuple of (violations list, total files scanned)
    """
    violations = []
    total_files = 0
    
    try:
        for path in root_dir.rglob("*"):
            # Skip excluded paths
            if should_exclude(path):
                continue
            
            # Only check files, not directories
            if not path.is_file():
                continue
            
            total_files += 1
            path_length = get_absolute_path_length(path)
            
            if path_length > limit:
                violations.append((path, path_length))
    
    except PermissionError as e:
        print_warning(f"Permission denied: {e}")
    except Exception as e:
        print_error(f"Error scanning directory: {e}")
    
    return violations, total_files


def suggest_fixes(violations: List[Tuple[Path, int]], limit: int) -> None:
    """
    Suggest fixes for path length violations.
    
    Args:
        violations: List of (path, length) tuples
        limit: Path length limit
    """
    print_header("Suggested Fixes")
    
    print("To resolve path length issues on Windows:\n")
    
    print(f"{COLOR_BOLD}1. Enable Long Path Support (Windows 10+):{COLOR_RESET}")
    print("   • Open Registry Editor (regedit)")
    print("   • Navigate to: HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Control\\FileSystem")
    print("   • Set LongPathsEnabled = 1 (DWORD)")
    print("   • Restart computer\n")
    
    print(f"{COLOR_BOLD}2. Shorten Directory Names:{COLOR_RESET}")
    print("   • Move project closer to drive root (e.g., C:\\Projects\\)")
    print("   • Avoid deeply nested subdirectories")
    print("   • Use shorter, descriptive names\n")
    
    print(f"{COLOR_BOLD}3. Use Symbolic Links (mklink):{COLOR_RESET}")
    print("   • Create junction/symlink at shorter path")
    print("   • Example: mklink /J C:\\AS C:\\Users\\YourName\\Deep\\Path\\Axion-Sat\n")
    
    print(f"{COLOR_BOLD}4. Specific File Recommendations:{COLOR_RESET}")
    
    # Group violations by directory
    violations_by_dir: Dict[Path, List[Tuple[Path, int]]] = {}
    for path, length in violations:
        parent = path.parent
        if parent not in violations_by_dir:
            violations_by_dir[parent] = []
        violations_by_dir[parent].append((path, length))
    
    # Show most problematic directories
    sorted_dirs = sorted(
        violations_by_dir.items(),
        key=lambda x: max(length for _, length in x[1]),
        reverse=True
    )
    
    for i, (parent_dir, files) in enumerate(sorted_dirs[:5], 1):
        excess = max(length for _, length in files) - limit
        print(f"\n   {i}. {parent_dir}")
        print(f"      Excess: {excess} characters")
        print(f"      Files affected: {len(files)}")
        
        # Suggest renaming
        parent_name = parent_dir.name
        if len(parent_name) > 20:
            short_name = parent_name[:15] + "..."
            print(f"      Suggestion: Rename '{parent_name}' → '{short_name}'")


def format_path_for_display(path: Path, max_length: int = 70) -> str:
    """
    Format path for display, truncating if necessary.
    
    Args:
        path: Path to format
        max_length: Maximum display length
        
    Returns:
        Formatted path string
    """
    path_str = str(path)
    if len(path_str) <= max_length:
        return path_str
    
    # Truncate middle of path
    prefix_len = max_length // 2 - 2
    suffix_len = max_length // 2 - 2
    return f"{path_str[:prefix_len]}...{path_str[-suffix_len:]}"


# ============================================================================
# Main Logic
# ============================================================================

def check_paths(
    root_dir: Path,
    limit: int = DEFAULT_PATH_LIMIT,
    suggest_fixes_flag: bool = False,
    verbose: bool = False
) -> int:
    """
    Check all paths in directory for length violations.
    
    Args:
        root_dir: Root directory to scan
        limit: Maximum path length
        suggest_fixes_flag: Whether to suggest fixes
        verbose: Print all checked files
        
    Returns:
        Exit code (0 = success, 1 = warnings, 2 = error)
    """
    print_header(f"Windows Path Length Check - Limit: {limit} characters")
    
    # Validate root directory
    if not root_dir.exists():
        print_error(f"Directory not found: {root_dir}")
        return 2
    
    if not root_dir.is_dir():
        print_error(f"Not a directory: {root_dir}")
        return 2
    
    print_info(f"Scanning directory: {root_dir.absolute()}")
    print_info(f"Excluding: {', '.join(sorted(EXCLUDE_DIRS))}\n")
    
    # Scan directory
    violations, total_files = scan_directory(root_dir, limit)
    
    # Print results
    print(f"{COLOR_BOLD}Scan Results:{COLOR_RESET}")
    print(f"  Total files scanned: {total_files}")
    print(f"  Path violations: {len(violations)}")
    
    if not violations:
        print_success(f"All paths are within the {limit}-character limit!")
        return 0
    
    # Sort violations by path length (worst first)
    violations.sort(key=lambda x: x[1], reverse=True)
    
    # Print violations
    print(f"\n{COLOR_BOLD}{COLOR_YELLOW}Path Length Violations:{COLOR_RESET}\n")
    
    for i, (path, length) in enumerate(violations, 1):
        excess = length - limit
        rel_path = path.relative_to(root_dir) if path.is_relative_to(root_dir) else path
        
        print(f"{i}. {COLOR_RED}[{length} chars, +{excess} over limit]{COLOR_RESET}")
        print(f"   {format_path_for_display(rel_path)}")
        
        if verbose:
            print(f"   Full path: {path.absolute()}")
        
        print()
    
    # Show summary
    print(f"{COLOR_BOLD}Summary:{COLOR_RESET}")
    max_length = max(length for _, length in violations)
    avg_length = sum(length for _, length in violations) / len(violations)
    
    print(f"  Longest path: {max_length} characters")
    print(f"  Average violation: {int(avg_length)} characters")
    print(f"  Limit: {limit} characters")
    print(f"  Worst excess: +{max_length - limit} characters\n")
    
    # Warnings about potential issues
    print_warning("Long paths may cause issues with:")
    print("  • File Explorer and some Windows applications")
    print("  • Git operations (clone, checkout, etc.)")
    print("  • Installers and deployment tools")
    print("  • Legacy software without long path support")
    print()
    
    # Suggest fixes if requested
    if suggest_fixes_flag:
        suggest_fixes(violations, limit)
    else:
        print_info("Run with --fix to see suggested fixes\n")
    
    return 1


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check Windows path lengths in project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/win_path_check.py
  python tools/win_path_check.py --limit 260
  python tools/win_path_check.py --fix
  python tools/win_path_check.py --dir ./outputs --verbose
  
Windows Path Length Info:
  • Legacy MAX_PATH: 260 characters
  • Recommended limit: 240 characters (safety margin)
  • Can be extended with LongPathsEnabled registry key
  • Python 3.6+ supports long paths on Windows 10+
        """
    )
    
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path.cwd(),
        help="Directory to scan (default: current directory)"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_PATH_LIMIT,
        help=f"Path length limit (default: {DEFAULT_PATH_LIMIT})"
    )
    
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Suggest fixes for violations"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print full paths for violations"
    )
    
    parser.add_argument(
        "--strict",
        action="store_true",
        help=f"Use strict Windows limit ({WINDOWS_MAX_PATH} chars)"
    )
    
    args = parser.parse_args()
    
    # Override limit if strict mode
    limit = WINDOWS_MAX_PATH if args.strict else args.limit
    
    try:
        return check_paths(
            root_dir=args.dir,
            limit=limit,
            suggest_fixes_flag=args.fix,
            verbose=args.verbose
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 2
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 2


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    sys.exit(main())
