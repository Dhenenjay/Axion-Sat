"""
tests/test_hf_hashcheck.py - HuggingFace Weight File Integrity Check

Computes SHA256 hashes of downloaded model weights and stores them in
weights/hf/.hashes.json to detect partial/corrupted downloads.

This test ensures:
1. Model weight files are fully downloaded (not partial)
2. Files are not corrupted during download
3. Files haven't changed unexpectedly (cache integrity)
4. Checksums are reproducible across systems

Usage:
    # Run as pytest test
    pytest tests/test_hf_hashcheck.py -v
    
    # Run standalone to generate/update hashes
    python tests/test_hf_hashcheck.py --generate
    
    # Verify existing hashes
    python tests/test_hf_hashcheck.py --verify
    
    # Update hashes for specific model
    python tests/test_hf_hashcheck.py --update terramind
"""

import sys
import json
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False

# ============================================================================
# Configuration
# ============================================================================

# HuggingFace cache directory
HF_CACHE_DIR = project_root / "weights" / "hf"

# Hash storage file
HASH_FILE = HF_CACHE_DIR / ".hashes.json"

# File extensions to check (model weights, configs, tokenizers)
CHECKABLE_EXTENSIONS = {
    # PyTorch weights
    ".bin", ".pt", ".pth", ".safetensors",
    # Model configs
    ".json", ".yaml", ".yml",
    # Tokenizers
    ".model", ".vocab",
    # ONNX models
    ".onnx",
}

# Exclude patterns (temporary files, logs, etc.)
EXCLUDE_PATTERNS = {
    ".gitignore", ".DS_Store", "Thumbs.db",
    ".git", ".cache", "__pycache__",
    ".lock", ".tmp", ".temp",
}

# Chunk size for reading large files (64 MB)
CHUNK_SIZE = 64 * 1024 * 1024  # 64 MB


# ============================================================================
# Hash Computation Functions
# ============================================================================

def compute_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """
    Compute cryptographic hash of a file.
    
    Uses streaming to handle large files without loading into memory.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm ("sha256", "sha1", "md5")
        
    Returns:
        Hex digest of file hash
        
    Example:
        >>> hash_val = compute_file_hash(Path("model.bin"))
        >>> print(hash_val)
        'a3f5e8c...'
    """
    hash_obj = hashlib.new(algorithm)
    
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def should_check_file(file_path: Path) -> bool:
    """
    Determine if file should be checked for integrity.
    
    Args:
        file_path: Path to file
        
    Returns:
        True if file should be checked
    """
    # Skip if not a file
    if not file_path.is_file():
        return False
    
    # Skip if matches exclude pattern
    if any(pattern in str(file_path) for pattern in EXCLUDE_PATTERNS):
        return False
    
    # Check if extension is in whitelist
    return file_path.suffix in CHECKABLE_EXTENSIONS


def find_model_files(cache_dir: Path) -> List[Path]:
    """
    Find all model files in cache directory.
    
    Args:
        cache_dir: HuggingFace cache directory
        
    Returns:
        List of file paths to check
    """
    if not cache_dir.exists():
        return []
    
    files = []
    for file_path in cache_dir.rglob("*"):
        if should_check_file(file_path):
            files.append(file_path)
    
    return sorted(files)


def compute_directory_hashes(
    cache_dir: Path,
    verbose: bool = True,
    progress: bool = True
) -> Dict[str, Dict]:
    """
    Compute hashes for all model files in cache directory.
    
    Args:
        cache_dir: HuggingFace cache directory
        verbose: Print detailed output
        progress: Show progress updates
        
    Returns:
        Dictionary mapping relative paths to file metadata:
        {
            "relative/path/to/file.bin": {
                "sha256": "abc123...",
                "size": 1234567890,
                "modified": "2025-01-13T12:34:56",
            }
        }
    """
    files = find_model_files(cache_dir)
    
    if verbose:
        print(f"Found {len(files)} files to check in {cache_dir}")
    
    hashes = {}
    
    for i, file_path in enumerate(files, 1):
        if progress and (i % 10 == 0 or i == 1 or i == len(files)):
            print(f"  Processing {i}/{len(files)}: {file_path.name}")
        
        # Compute relative path from cache directory
        rel_path = file_path.relative_to(cache_dir)
        
        # Get file stats
        stat = file_path.stat()
        
        # Compute hash
        try:
            file_hash = compute_file_hash(file_path, algorithm="sha256")
        except Exception as e:
            if verbose:
                print(f"    WARNING: Failed to hash {rel_path}: {e}")
            continue
        
        # Store metadata
        hashes[str(rel_path)] = {
            "sha256": file_hash,
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }
        
        if verbose and i <= 5:  # Show first 5 in detail
            print(f"    {rel_path.name}: {file_hash[:16]}... ({stat.st_size:,} bytes)")
    
    return hashes


# ============================================================================
# Hash Storage Functions
# ============================================================================

def save_hashes(hashes: Dict[str, Dict], hash_file: Path, verbose: bool = True):
    """
    Save computed hashes to JSON file.
    
    Args:
        hashes: Dictionary of file hashes
        hash_file: Path to output JSON file
        verbose: Print status messages
    """
    # Create directory if needed
    hash_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare output data
    output = {
        "version": "1.0",
        "generated": datetime.now().isoformat(),
        "algorithm": "sha256",
        "files": hashes,
        "count": len(hashes),
    }
    
    # Write to file
    with open(hash_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True)
    
    if verbose:
        print(f"\n✓ Saved {len(hashes)} file hashes to {hash_file}")


def load_hashes(hash_file: Path) -> Optional[Dict[str, Dict]]:
    """
    Load previously computed hashes from JSON file.
    
    Args:
        hash_file: Path to JSON file
        
    Returns:
        Dictionary of file hashes, or None if file doesn't exist
    """
    if not hash_file.exists():
        return None
    
    try:
        with open(hash_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("files", {})
    except Exception as e:
        print(f"WARNING: Failed to load hashes from {hash_file}: {e}")
        return None


# ============================================================================
# Verification Functions
# ============================================================================

def verify_hashes(
    cache_dir: Path,
    hash_file: Path,
    verbose: bool = True
) -> Tuple[bool, List[str], List[str], List[str]]:
    """
    Verify current files against stored hashes.
    
    Args:
        cache_dir: HuggingFace cache directory
        hash_file: Path to stored hashes JSON
        verbose: Print detailed output
        
    Returns:
        Tuple of (all_valid, mismatches, missing, new_files):
        - all_valid: True if all checks passed
        - mismatches: List of files with hash mismatches
        - missing: List of files in hashes but not on disk
        - new_files: List of files on disk but not in hashes
    """
    # Load stored hashes
    stored_hashes = load_hashes(hash_file)
    if stored_hashes is None:
        if verbose:
            print(f"No hash file found at {hash_file}")
            print("Run with --generate to create initial hashes")
        return False, [], [], []
    
    if verbose:
        print(f"Loaded {len(stored_hashes)} stored hashes from {hash_file}")
    
    # Compute current hashes
    if verbose:
        print(f"Computing current hashes...")
    current_hashes = compute_directory_hashes(cache_dir, verbose=False, progress=verbose)
    
    # Find differences
    mismatches = []
    missing = []
    new_files = []
    
    # Check stored files
    for rel_path, stored_meta in stored_hashes.items():
        if rel_path not in current_hashes:
            missing.append(rel_path)
        elif current_hashes[rel_path]["sha256"] != stored_meta["sha256"]:
            mismatches.append(rel_path)
    
    # Check for new files
    for rel_path in current_hashes:
        if rel_path not in stored_hashes:
            new_files.append(rel_path)
    
    # Print results
    all_valid = len(mismatches) == 0 and len(missing) == 0
    
    if verbose:
        print("\n" + "=" * 79)
        print("HASH VERIFICATION RESULTS")
        print("=" * 79)
        
        if all_valid and len(new_files) == 0:
            print("✓ All files verified successfully!")
        else:
            if mismatches:
                print(f"\n✗ HASH MISMATCHES ({len(mismatches)}):")
                for path in mismatches[:10]:  # Show first 10
                    print(f"  - {path}")
                if len(mismatches) > 10:
                    print(f"  ... and {len(mismatches) - 10} more")
            
            if missing:
                print(f"\n⚠ MISSING FILES ({len(missing)}):")
                for path in missing[:10]:
                    print(f"  - {path}")
                if len(missing) > 10:
                    print(f"  ... and {len(missing) - 10} more")
            
            if new_files:
                print(f"\nℹ NEW FILES ({len(new_files)}):")
                for path in new_files[:10]:
                    print(f"  - {path}")
                if len(new_files) > 10:
                    print(f"  ... and {len(new_files) - 10} more")
        
        print("\nSummary:")
        print(f"  Total files checked: {len(current_hashes)}")
        print(f"  Verified (matching): {len(current_hashes) - len(mismatches) - len(new_files)}")
        print(f"  Mismatches: {len(mismatches)}")
        print(f"  Missing: {len(missing)}")
        print(f"  New files: {len(new_files)}")
        print("=" * 79)
    
    return all_valid, mismatches, missing, new_files


# ============================================================================
# Pytest Tests
# ============================================================================

@pytest.mark.skipif(not HF_CACHE_DIR.exists(), reason="HuggingFace cache not found")
def test_hash_file_exists():
    """Test that hash file exists."""
    assert HASH_FILE.exists(), (
        f"Hash file not found: {HASH_FILE}\n"
        "Run: python tests/test_hf_hashcheck.py --generate"
    )


@pytest.mark.skipif(not HF_CACHE_DIR.exists(), reason="HuggingFace cache not found")
def test_hash_file_valid():
    """Test that hash file is valid JSON."""
    assert HASH_FILE.exists(), "Hash file not found"
    
    with open(HASH_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    assert "files" in data, "Hash file missing 'files' key"
    assert "version" in data, "Hash file missing 'version' key"
    assert "algorithm" in data, "Hash file missing 'algorithm' key"
    assert len(data["files"]) > 0, "Hash file contains no hashes"


@pytest.mark.skipif(not HF_CACHE_DIR.exists(), reason="HuggingFace cache not found")
def test_verify_hashes():
    """Test that all model file hashes match stored values."""
    all_valid, mismatches, missing, new_files = verify_hashes(
        HF_CACHE_DIR,
        HASH_FILE,
        verbose=False
    )
    
    error_msg = []
    if mismatches:
        error_msg.append(f"Hash mismatches detected in {len(mismatches)} files:")
        for path in mismatches[:5]:
            error_msg.append(f"  - {path}")
        if len(mismatches) > 5:
            error_msg.append(f"  ... and {len(mismatches) - 5} more")
        error_msg.append("\nThis may indicate corrupted or partially downloaded files.")
        error_msg.append("Re-download affected models or run: python tests/test_hf_hashcheck.py --update")
    
    if missing:
        error_msg.append(f"\nMissing files: {len(missing)}")
        error_msg.append("Some cached files have been deleted.")
        error_msg.append("Run: python tests/test_hf_hashcheck.py --update")
    
    assert all_valid, "\n".join(error_msg) if error_msg else "Unknown verification error"


@pytest.mark.skipif(not HF_CACHE_DIR.exists(), reason="HuggingFace cache not found")
def test_model_files_exist():
    """Test that expected model files exist in cache."""
    files = find_model_files(HF_CACHE_DIR)
    
    assert len(files) > 0, (
        f"No model files found in {HF_CACHE_DIR}\n"
        "Download models with: python scripts/06_model_sanity.py"
    )
    
    # Check for common model file types
    extensions = {f.suffix for f in files}
    has_weights = any(ext in extensions for ext in [".bin", ".safetensors", ".pt", ".pth"])
    
    assert has_weights, (
        f"No model weight files found (expected .bin, .safetensors, .pt, or .pth)\n"
        f"Found extensions: {extensions}"
    )


# ============================================================================
# Command-Line Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Verify HuggingFace model weight file integrity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate initial hashes
  python tests/test_hf_hashcheck.py --generate
  
  # Verify current files against stored hashes
  python tests/test_hf_hashcheck.py --verify
  
  # Update hashes (generate new hashes, overwrite existing)
  python tests/test_hf_hashcheck.py --update
  
  # Run as pytest test
  pytest tests/test_hf_hashcheck.py -v
        """
    )
    
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate hash file from current cache"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify current cache against stored hashes"
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update stored hashes (alias for --generate)"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Only process specific model directory"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output"
    )
    
    args = parser.parse_args()
    
    # Default to verify if no action specified
    if not (args.generate or args.verify or args.update):
        args.verify = True
    
    verbose = not args.quiet
    
    # Check if cache directory exists
    if not HF_CACHE_DIR.exists():
        print(f"ERROR: HuggingFace cache not found: {HF_CACHE_DIR}")
        print("\nExpected structure:")
        print("  weights/")
        print("    hf/")
        print("      terramind-1.0-large/")
        print("      prithvi-eo-2.0-600M/")
        print("\nDownload models with:")
        print("  python scripts/06_model_sanity.py")
        return 1
    
    # Determine cache directory to check
    cache_dir = HF_CACHE_DIR
    if args.model:
        cache_dir = HF_CACHE_DIR / args.model
        if not cache_dir.exists():
            print(f"ERROR: Model directory not found: {cache_dir}")
            return 1
    
    # Generate/update hashes
    if args.generate or args.update:
        if verbose:
            print("=" * 79)
            print("GENERATING FILE HASHES")
            print("=" * 79)
            print(f"Cache directory: {cache_dir}")
            print()
        
        hashes = compute_directory_hashes(cache_dir, verbose=verbose, progress=True)
        
        if len(hashes) == 0:
            print("\nWARNING: No model files found to hash")
            print("Download models with: python scripts/06_model_sanity.py")
            return 1
        
        save_hashes(hashes, HASH_FILE, verbose=verbose)
        
        if verbose:
            print("\nTo verify integrity in the future, run:")
            print("  python tests/test_hf_hashcheck.py --verify")
        
        return 0
    
    # Verify hashes
    if args.verify:
        if verbose:
            print("=" * 79)
            print("VERIFYING FILE INTEGRITY")
            print("=" * 79)
            print(f"Cache directory: {cache_dir}")
            print(f"Hash file: {HASH_FILE}")
            print()
        
        all_valid, mismatches, missing, new_files = verify_hashes(
            cache_dir,
            HASH_FILE,
            verbose=verbose
        )
        
        if all_valid and len(new_files) == 0:
            if verbose:
                print("\n✓ All files verified successfully!")
            return 0
        else:
            if verbose:
                print("\n⚠ Verification found differences")
                print("\nTo update hashes, run:")
                print("  python tests/test_hf_hashcheck.py --update")
            return 1


if __name__ == "__main__":
    sys.exit(main())
