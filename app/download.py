"""
app/download.py - Satellite Data Download Manager

Provides robust downloading of satellite imagery with resume support, progress
tracking, checksum verification, and manifest generation.

Features:
    - Resume interrupted downloads
    - Progress bars with speed/ETA
    - MD5/SHA256 checksum verification
    - Automatic retry on failure
    - Manifest CSV tracking
    - Parallel downloads (optional)
    - Cloud-optimized GeoTIFF (COG) support

Usage:
    >>> from app.download import download_asset, download_search_results
    >>> 
    >>> # Download single asset
    >>> result = download_asset(
    ...     url="https://sentinel-cogs.s3.amazonaws.com/.../B04.tif",
    ...     output_dir="data/raw/sentinel-2-l2a/S2A_...",
    ...     filename="B04.tif"
    ... )
    >>> 
    >>> # Download all assets from search results
    >>> manifest = download_search_results(results, max_workers=4)
"""

import os
import sys
import csv
import hashlib
import requests
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from urllib.parse import urlparse
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Progress bar support
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("WARNING: tqdm not available. Install with: pip install tqdm")

# Parallel download support
try:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False

# ============================================================================
# Configuration
# ============================================================================

# Data directories
DATA_DIR = project_root / "data"
RAW_DIR = DATA_DIR / "raw"
MANIFEST_DIR = DATA_DIR / "manifests"

# Download settings
CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB chunks
CONNECT_TIMEOUT = 30  # seconds
READ_TIMEOUT = 300  # seconds (5 minutes for large files)
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# User agent
USER_AGENT = "axion-sat/1.0 (satellite-data-downloader)"


# ============================================================================
# Checksum Calculation
# ============================================================================

def calculate_checksum(
    file_path: Path,
    algorithm: str = "md5",
    chunk_size: int = CHUNK_SIZE
) -> str:
    """
    Calculate file checksum.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm ("md5", "sha256", "sha1")
        chunk_size: Read chunk size
        
    Returns:
        Hex digest of checksum
    """
    hash_obj = hashlib.new(algorithm)
    
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def verify_checksum(
    file_path: Path,
    expected_checksum: str,
    algorithm: str = "md5"
) -> bool:
    """
    Verify file checksum matches expected value.
    
    Args:
        file_path: Path to file
        expected_checksum: Expected checksum hex string
        algorithm: Hash algorithm
        
    Returns:
        True if checksums match
    """
    actual = calculate_checksum(file_path, algorithm)
    return actual.lower() == expected_checksum.lower()


# ============================================================================
# Download Functions
# ============================================================================

def download_asset(
    url: str,
    output_dir: Path,
    filename: Optional[str] = None,
    checksum: Optional[str] = None,
    checksum_algorithm: str = "md5",
    resume: bool = True,
    max_retries: int = MAX_RETRIES,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Download a single asset with resume support and verification.
    
    Args:
        url: Asset URL to download
        output_dir: Output directory
        filename: Output filename (None = infer from URL)
        checksum: Expected checksum for verification
        checksum_algorithm: Hash algorithm ("md5", "sha256")
        resume: Enable resume for partial downloads
        max_retries: Maximum retry attempts
        verbose: Show progress bar
        
    Returns:
        Dictionary with download metadata:
        {
            "url": str,
            "file_path": Path,
            "size_bytes": int,
            "checksum": str,
            "status": "success" | "failed" | "skipped",
            "error": Optional[str],
            "download_time_seconds": float,
        }
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine filename
    if filename is None:
        filename = Path(urlparse(url).path).name
    
    output_path = output_dir / filename
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    
    # Check if file already exists and is complete
    if output_path.exists() and not temp_path.exists():
        if checksum:
            if verify_checksum(output_path, checksum, checksum_algorithm):
                if verbose:
                    print(f"✓ Skipped (already exists): {filename}")
                return {
                    "url": url,
                    "file_path": output_path,
                    "size_bytes": output_path.stat().st_size,
                    "checksum": checksum,
                    "status": "skipped",
                    "error": None,
                    "download_time_seconds": 0.0,
                }
        else:
            # No checksum provided, assume complete
            if verbose:
                print(f"✓ Skipped (already exists): {filename}")
            return {
                "url": url,
                "file_path": output_path,
                "size_bytes": output_path.stat().st_size,
                "checksum": None,
                "status": "skipped",
                "error": None,
                "download_time_seconds": 0.0,
            }
    
    # Download with retries
    start_time = datetime.now()
    last_error = None
    
    for attempt in range(max_retries):
        try:
            # Determine starting position (resume support)
            initial_pos = 0
            mode = "wb"
            
            if resume and temp_path.exists():
                initial_pos = temp_path.stat().st_size
                mode = "ab"
                if verbose and attempt == 0:
                    print(f"Resuming download: {filename} (from {initial_pos:,} bytes)")
            
            # Setup headers
            headers = {"User-Agent": USER_AGENT}
            if initial_pos > 0:
                headers["Range"] = f"bytes={initial_pos}-"
            
            # Make request
            response = requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)
            )
            response.raise_for_status()
            
            # Get total size
            total_size = int(response.headers.get("content-length", 0)) + initial_pos
            
            # Download with progress bar
            if TQDM_AVAILABLE and verbose:
                progress = tqdm(
                    total=total_size,
                    initial=initial_pos,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=filename,
                )
            else:
                progress = None
            
            # Write data
            with open(temp_path, mode) as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        if progress:
                            progress.update(len(chunk))
            
            if progress:
                progress.close()
            
            # Move from temp to final location
            if output_path.exists():
                output_path.unlink()
            temp_path.rename(output_path)
            
            # Verify checksum if provided
            if checksum:
                if verbose:
                    print(f"  Verifying checksum...")
                
                if not verify_checksum(output_path, checksum, checksum_algorithm):
                    raise ValueError(f"Checksum mismatch for {filename}")
            
            # Calculate checksum if not provided
            actual_checksum = checksum
            if not actual_checksum:
                actual_checksum = calculate_checksum(output_path, checksum_algorithm)
            
            # Success
            download_time = (datetime.now() - start_time).total_seconds()
            
            if verbose:
                print(f"✓ Downloaded: {filename} ({output_path.stat().st_size:,} bytes)")
            
            return {
                "url": url,
                "file_path": output_path,
                "size_bytes": output_path.stat().st_size,
                "checksum": actual_checksum,
                "status": "success",
                "error": None,
                "download_time_seconds": download_time,
            }
        
        except Exception as e:
            last_error = str(e)
            
            if verbose:
                print(f"  Attempt {attempt + 1}/{max_retries} failed: {e}")
            
            if attempt < max_retries - 1:
                import time
                time.sleep(RETRY_DELAY)
    
    # All retries failed
    download_time = (datetime.now() - start_time).total_seconds()
    
    if verbose:
        print(f"✗ Failed: {filename} - {last_error}")
    
    # Cleanup temp file
    if temp_path.exists():
        temp_path.unlink()
    
    return {
        "url": url,
        "file_path": output_path,
        "size_bytes": 0,
        "checksum": None,
        "status": "failed",
        "error": last_error,
        "download_time_seconds": download_time,
    }


# ============================================================================
# Batch Download Functions
# ============================================================================

def download_item_assets(
    item_id: str,
    collection_id: str,
    assets: Dict[str, str],
    output_base: Optional[Path] = None,
    asset_filter: Optional[List[str]] = None,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Download all assets for a STAC item.
    
    Args:
        item_id: STAC item ID
        collection_id: Collection ID (for directory structure)
        assets: Dictionary mapping asset keys to URLs
        output_base: Base output directory (default: data/raw)
        asset_filter: List of asset keys to download (None = all)
        verbose: Show progress
        
    Returns:
        List of download results
    """
    if output_base is None:
        output_base = RAW_DIR
    
    # Create item directory: data/raw/{collection}/{item_id}/
    item_dir = output_base / collection_id / item_id
    
    if verbose:
        print(f"\nDownloading {collection_id}/{item_id}")
        print(f"  Output: {item_dir}")
    
    # Filter assets
    if asset_filter:
        assets = {k: v for k, v in assets.items() if k in asset_filter}
    
    if verbose:
        print(f"  Assets: {list(assets.keys())}")
    
    # Download each asset
    results = []
    for asset_key, asset_url in assets.items():
        if verbose:
            print(f"\n  Asset: {asset_key}")
        
        result = download_asset(
            url=asset_url,
            output_dir=item_dir,
            filename=f"{asset_key}.tif",  # Assume GeoTIFF
            verbose=verbose
        )
        
        result["asset_key"] = asset_key
        result["item_id"] = item_id
        result["collection_id"] = collection_id
        
        results.append(result)
    
    return results


def download_search_results(
    search_results: List[Dict[str, Any]],
    output_base: Optional[Path] = None,
    asset_filter: Optional[List[str]] = None,
    max_items: Optional[int] = None,
    max_workers: int = 1,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Download assets from multiple search results.
    
    Args:
        search_results: List of results from search_* functions
        output_base: Base output directory
        asset_filter: Asset keys to download (e.g., ["B02", "B03", "B04"])
        max_items: Maximum number of items to download
        max_workers: Number of parallel download threads (1 = sequential)
        verbose: Show progress
        
    Returns:
        Dictionary with download summary and manifest
    """
    if output_base is None:
        output_base = RAW_DIR
    
    # Limit items if requested
    if max_items:
        search_results = search_results[:max_items]
    
    if verbose:
        print("=" * 79)
        print(f"DOWNLOADING {len(search_results)} ITEMS")
        print("=" * 79)
    
    # Collect all download tasks
    all_results = []
    
    if max_workers > 1 and PARALLEL_AVAILABLE:
        # Parallel downloads
        if verbose:
            print(f"Using {max_workers} parallel workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for search_result in search_results:
                metadata = search_result["metadata"]
                assets = search_result["assets"]
                
                future = executor.submit(
                    download_item_assets,
                    item_id=metadata["id"],
                    collection_id=metadata["collection"],
                    assets=assets,
                    output_base=output_base,
                    asset_filter=asset_filter,
                    verbose=False  # Disable per-item verbosity in parallel mode
                )
                futures.append(future)
            
            for future in as_completed(futures):
                results = future.result()
                all_results.extend(results)
                
                if verbose:
                    success_count = sum(1 for r in results if r["status"] == "success")
                    item_id = results[0]["item_id"] if results else "unknown"
                    print(f"✓ Completed: {item_id} ({success_count}/{len(results)} assets)")
    else:
        # Sequential downloads
        for search_result in search_results:
            metadata = search_result["metadata"]
            assets = search_result["assets"]
            
            results = download_item_assets(
                item_id=metadata["id"],
                collection_id=metadata["collection"],
                assets=assets,
                output_base=output_base,
                asset_filter=asset_filter,
                verbose=verbose
            )
            
            all_results.extend(results)
    
    # Generate summary
    success_count = sum(1 for r in all_results if r["status"] == "success")
    skipped_count = sum(1 for r in all_results if r["status"] == "skipped")
    failed_count = sum(1 for r in all_results if r["status"] == "failed")
    total_bytes = sum(r["size_bytes"] for r in all_results if r["status"] in ["success", "skipped"])
    total_time = sum(r["download_time_seconds"] for r in all_results)
    
    summary = {
        "total_assets": len(all_results),
        "success": success_count,
        "skipped": skipped_count,
        "failed": failed_count,
        "total_bytes": total_bytes,
        "total_time_seconds": total_time,
        "download_results": all_results,
    }
    
    if verbose:
        print("\n" + "=" * 79)
        print("DOWNLOAD SUMMARY")
        print("=" * 79)
        print(f"Total assets: {len(all_results)}")
        print(f"  Successful: {success_count}")
        print(f"  Skipped (already exist): {skipped_count}")
        print(f"  Failed: {failed_count}")
        print(f"Total size: {total_bytes / 1e9:.2f} GB")
        print(f"Total time: {total_time:.1f} seconds")
        if total_time > 0:
            print(f"Average speed: {total_bytes / total_time / 1e6:.2f} MB/s")
        print("=" * 79)
    
    return summary


# ============================================================================
# Manifest Generation
# ============================================================================

def save_manifest(
    download_results: List[Dict[str, Any]],
    manifest_file: Optional[Path] = None,
    verbose: bool = True
) -> Path:
    """
    Save download manifest to CSV.
    
    Args:
        download_results: List of download result dictionaries
        manifest_file: Output CSV path (None = auto-generate)
        verbose: Print status
        
    Returns:
        Path to manifest file
    """
    # Auto-generate manifest filename
    if manifest_file is None:
        MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        manifest_file = MANIFEST_DIR / f"raw_{timestamp}.csv"
    
    manifest_file = Path(manifest_file)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Write CSV
    with open(manifest_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "collection_id",
            "item_id",
            "asset_key",
            "url",
            "file_path",
            "size_bytes",
            "checksum",
            "status",
            "error",
            "download_time_seconds",
        ])
        
        writer.writeheader()
        
        for result in download_results:
            writer.writerow({
                "collection_id": result.get("collection_id", ""),
                "item_id": result.get("item_id", ""),
                "asset_key": result.get("asset_key", ""),
                "url": result["url"],
                "file_path": str(result["file_path"]),
                "size_bytes": result["size_bytes"],
                "checksum": result.get("checksum", ""),
                "status": result["status"],
                "error": result.get("error", ""),
                "download_time_seconds": result["download_time_seconds"],
            })
    
    if verbose:
        print(f"\n✓ Manifest saved: {manifest_file}")
        print(f"  Total records: {len(download_results)}")
    
    return manifest_file


def load_manifest(manifest_file: Path) -> List[Dict[str, Any]]:
    """
    Load download manifest from CSV.
    
    Args:
        manifest_file: Path to manifest CSV
        
    Returns:
        List of download records
    """
    records = []
    
    with open(manifest_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            row["size_bytes"] = int(row["size_bytes"]) if row["size_bytes"] else 0
            row["download_time_seconds"] = float(row["download_time_seconds"]) if row["download_time_seconds"] else 0.0
            records.append(row)
    
    return records


# ============================================================================
# Testing & Examples
# ============================================================================

if __name__ == "__main__":
    print("=" * 79)
    print("DOWNLOAD MODULE TEST")
    print("=" * 79)
    print()
    
    # Test single file download (small test file)
    print("TEST 1: Single File Download")
    print("-" * 79)
    
    test_url = "https://httpbin.org/bytes/1048576"  # 1 MB test file
    test_dir = DATA_DIR / "test" / "download"
    
    result = download_asset(
        url=test_url,
        output_dir=test_dir,
        filename="test_1mb.bin",
        verbose=True
    )
    
    print(f"Status: {result['status']}")
    print(f"Size: {result['size_bytes']:,} bytes")
    print(f"Time: {result['download_time_seconds']:.2f} seconds")
    print()
    
    # Test resume (download again, should skip)
    print("TEST 2: Resume (Should Skip)")
    print("-" * 79)
    
    result2 = download_asset(
        url=test_url,
        output_dir=test_dir,
        filename="test_1mb.bin",
        verbose=True
    )
    
    print(f"Status: {result2['status']}")
    print()
    
    # Test manifest generation
    print("TEST 3: Manifest Generation")
    print("-" * 79)
    
    mock_results = [result, result2]
    manifest_path = save_manifest(mock_results, verbose=True)
    
    print()
    print("Manifest contents:")
    loaded = load_manifest(manifest_path)
    for record in loaded:
        print(f"  {record['status']}: {record['file_path']}")
    
    print()
    print("=" * 79)
    print("TESTS COMPLETE")
    print("=" * 79)
    print()
    print("Usage example:")
    print("  from app.download import download_search_results")
    print("  from app.stac_fetch import search_s2_l2a_cloudfree")
    print("  ")
    print("  # Search for data")
    print("  results = search_s2_l2a_cloudfree(bbox, date)")
    print("  ")
    print("  # Download top 5 results")
    print("  summary = download_search_results(")
    print("      results,")
    print("      asset_filter=['B02', 'B03', 'B04', 'B08'],")
    print("      max_items=5")
    print("  )")
    print("  ")
    print("  # Save manifest")
    print("  save_manifest(summary['download_results'])")
