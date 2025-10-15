#!/usr/bin/env python3
"""
03_pin_versions.py - Pin Python Package Versions

This script captures the current Python environment's installed packages
using `pip freeze` and writes them to `pip-freeze.txt` with a UTC timestamp
header for version tracking and reproducibility.

Usage:
    python scripts/03_pin_versions.py

Output:
    - Prints frozen package list to console
    - Writes to pip-freeze.txt in project root

Purpose:
    - Document exact package versions for reproducibility
    - Track dependency changes over time
    - Enable environment cloning for CI/CD or other developers
"""

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


# ============================================================================
# Configuration
# ============================================================================

# Project root directory (parent of scripts/)
PROJECT_ROOT = Path(__file__).parent.parent

# Output file name
OUTPUT_FILE = PROJECT_ROOT / "pip-freeze.txt"

# ANSI color codes for terminal output
COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"
COLOR_GREEN = "\033[32m"
COLOR_BLUE = "\033[34m"
COLOR_YELLOW = "\033[33m"
COLOR_RED = "\033[31m"
COLOR_CYAN = "\033[36m"


# ============================================================================
# Helper Functions
# ============================================================================

def print_header(message: str) -> None:
    """Print a formatted header message."""
    print(f"\n{COLOR_BOLD}{COLOR_CYAN}{'=' * 79}{COLOR_RESET}")
    print(f"{COLOR_BOLD}{COLOR_GREEN}▶ {message}{COLOR_RESET}")
    print(f"{COLOR_BOLD}{COLOR_CYAN}{'=' * 79}{COLOR_RESET}\n")


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"\n{COLOR_BOLD}{COLOR_GREEN}✓ {message}{COLOR_RESET}\n")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"\n{COLOR_BOLD}{COLOR_RED}✗ {message}{COLOR_RESET}\n", file=sys.stderr)


def print_info(message: str) -> None:
    """Print an info message."""
    print(f"{COLOR_CYAN}{message}{COLOR_RESET}")


# ============================================================================
# Main Functions
# ============================================================================

def get_pip_freeze_output() -> str:
    """
    Run `pip freeze` and return the output as a string.
    
    Returns:
        str: Output from `pip freeze` command
        
    Raises:
        subprocess.CalledProcessError: If pip freeze fails
        FileNotFoundError: If pip is not found
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print_error(f"pip freeze failed with exit code {e.returncode}")
        print(f"Error output: {e.stderr}", file=sys.stderr)
        raise
    except FileNotFoundError:
        print_error("Python or pip not found in current environment")
        raise


def generate_header() -> str:
    """
    Generate a header comment with timestamp and metadata.
    
    Returns:
        str: Formatted header with timestamp
    """
    now_utc = datetime.now(timezone.utc)
    timestamp = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
    iso_timestamp = now_utc.isoformat()
    
    python_version = sys.version.split()[0]
    python_impl = sys.implementation.name
    
    header = f"""# ============================================================================
# pip-freeze.txt - Pinned Python Package Versions
# ============================================================================
#
# This file contains the exact versions of all Python packages installed
# in the Axion-Sat environment at the time of generation.
#
# Generated: {timestamp}
# ISO 8601:  {iso_timestamp}
# Python:    {python_version} ({python_impl})
# Platform:  {sys.platform}
#
# Purpose:
#   - Reproducible environment setup
#   - Dependency version tracking
#   - CI/CD environment synchronization
#
# Usage:
#   Install exact versions:
#     pip install -r pip-freeze.txt
#
#   Compare with current environment:
#     pip freeze > current.txt
#     diff pip-freeze.txt current.txt
#
# Regenerate this file:
#   python scripts/03_pin_versions.py
#
# ============================================================================

"""
    return header


def get_package_count(freeze_output: str) -> int:
    """
    Count the number of packages in pip freeze output.
    
    Args:
        freeze_output: Output from pip freeze
        
    Returns:
        int: Number of packages
    """
    # Filter out empty lines and comments
    lines = [line for line in freeze_output.strip().split("\n") if line and not line.startswith("#")]
    return len(lines)


def format_package_list(freeze_output: str) -> str:
    """
    Format the package list with optional categorization.
    
    Args:
        freeze_output: Raw output from pip freeze
        
    Returns:
        str: Formatted package list
    """
    # For now, just return as-is
    # Future enhancement: could categorize packages (ML, geospatial, dev, etc.)
    return freeze_output


def write_freeze_file(output_path: Path, content: str) -> None:
    """
    Write the freeze content to file.
    
    Args:
        output_path: Path to output file
        content: Content to write
        
    Raises:
        IOError: If file write fails
    """
    try:
        output_path.write_text(content, encoding="utf-8")
    except IOError as e:
        print_error(f"Failed to write to {output_path}: {e}")
        raise


def print_freeze_output(content: str, max_lines: int = 50) -> None:
    """
    Print freeze output to console, with optional truncation.
    
    Args:
        content: Content to print
        max_lines: Maximum lines to print before truncating
    """
    lines = content.split("\n")
    
    # Print header
    print(f"{COLOR_BOLD}{COLOR_BLUE}Package List:{COLOR_RESET}\n")
    
    # Print lines
    if len(lines) <= max_lines:
        print(content)
    else:
        # Print first portion
        print("\n".join(lines[:max_lines]))
        print(f"\n{COLOR_YELLOW}... ({len(lines) - max_lines} more lines, see {OUTPUT_FILE.name} for full list) ...{COLOR_RESET}\n")


def main() -> int:
    """
    Main entry point.
    
    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    print_header("Pinning Python Package Versions")
    
    try:
        # Step 1: Get pip freeze output
        print_info("▸ Running pip freeze...")
        freeze_output = get_pip_freeze_output()
        package_count = get_package_count(freeze_output)
        print_success(f"Found {package_count} installed packages")
        
        # Step 2: Generate header
        print_info("▸ Generating file header...")
        header = generate_header()
        
        # Step 3: Format package list
        print_info("▸ Formatting package list...")
        formatted_packages = format_package_list(freeze_output)
        
        # Step 4: Combine header and packages
        full_content = header + formatted_packages
        
        # Step 5: Write to file
        print_info(f"▸ Writing to {OUTPUT_FILE.relative_to(PROJECT_ROOT)}...")
        write_freeze_file(OUTPUT_FILE, full_content)
        
        # Step 6: Print to console
        print_freeze_output(formatted_packages, max_lines=50)
        
        # Success summary
        print(f"{COLOR_BOLD}{COLOR_CYAN}{'─' * 79}{COLOR_RESET}")
        print(f"{COLOR_BOLD}Summary:{COLOR_RESET}")
        print(f"  • Packages:     {COLOR_GREEN}{package_count}{COLOR_RESET}")
        print(f"  • Output file:  {COLOR_BLUE}{OUTPUT_FILE.relative_to(PROJECT_ROOT)}{COLOR_RESET}")
        print(f"  • File size:    {COLOR_YELLOW}{len(full_content):,} bytes{COLOR_RESET}")
        print(f"{COLOR_BOLD}{COLOR_CYAN}{'─' * 79}{COLOR_RESET}")
        
        print_success("Package versions pinned successfully!")
        
        print_info("Next steps:")
        print(f"  • Commit {OUTPUT_FILE.name} to version control")
        print(f"  • Share with team for environment synchronization")
        print(f"  • Use for CI/CD environment setup")
        print()
        
        return 0
        
    except subprocess.CalledProcessError:
        print_error("Failed to execute pip freeze")
        return 1
    except FileNotFoundError:
        print_error("Python or pip not found")
        return 1
    except IOError as e:
        print_error(f"File I/O error: {e}")
        return 1
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    sys.exit(main())
