"""
app/stac_fetch.py - STAC Data Fetching

Provides high-level functions for searching satellite imagery using STAC APIs.
Supports Sentinel-1 GRD, Sentinel-2 L2A, and HLS (Harmonized Landsat Sentinel).

Features:
    - Sentinel-1 GRD search with orbit/polarization filters
    - Cloud-free Sentinel-2 search
    - HLS temporal neighbors search
    - Automatic sorting by acquisition proximity
    - Asset href extraction
    - Metadata extraction

Usage:
    >>> from app.stac_fetch import search_s1_grd, search_s2_l2a_cloudfree
    >>> 
    >>> # Search for Sentinel-1
    >>> bbox = [31.5, -1.5, 34.0, 0.5]  # Lake Victoria
    >>> items = search_s1_grd(bbox, "2024-06-15", days=5)
    >>> 
    >>> # Search for cloud-free Sentinel-2
    >>> items = search_s2_l2a_cloudfree(bbox, "2024-06-15", max_cloud=10)
"""

import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timedelta
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("WARNING: PyYAML not available. Install with: pip install pyyaml")

try:
    from pystac_client import Client
    from pystac import Item
    PYSTAC_AVAILABLE = True
except ImportError:
    PYSTAC_AVAILABLE = False
    print("WARNING: pystac-client not available. Install with: pip install pystac-client")

# ============================================================================
# Configuration Loading
# ============================================================================

# Load STAC endpoints configuration
CONFIG_FILE = project_root / "configs" / "stac.endpoints.yaml"

def load_stac_config() -> Dict:
    """
    Load STAC endpoints configuration from YAML file.
    
    Returns:
        Configuration dictionary
    """
    if not YAML_AVAILABLE:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")
    
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"STAC config not found: {CONFIG_FILE}")
    
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Load configuration at module import
try:
    STAC_CONFIG = load_stac_config()
    EARTHSEARCH_URL = STAC_CONFIG['endpoints']['earthsearch']['url']
    MPC_URL = STAC_CONFIG['endpoints']['mpc']['url']
    LPCLOUD_URL = STAC_CONFIG['endpoints']['lpcloud']['url']
except Exception as e:
    warnings.warn(f"Failed to load STAC config: {e}")
    STAC_CONFIG = {}
    EARTHSEARCH_URL = "https://earth-search.aws.element84.com/v1"
    MPC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
    LPCLOUD_URL = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD"


# ============================================================================
# Utility Functions
# ============================================================================

def parse_date_range(
    date_iso: str,
    days: int = 7
) -> Tuple[datetime, datetime]:
    """
    Parse date and create date range for STAC search.
    
    Args:
        date_iso: ISO date string (YYYY-MM-DD)
        days: Number of days before and after date (total window = 2*days)
        
    Returns:
        Tuple of (start_date, end_date) as datetime objects
        
    Example:
        >>> start, end = parse_date_range("2024-06-15", days=5)
        >>> print(start, end)
        2024-06-10 2024-06-20
    """
    center_date = datetime.fromisoformat(date_iso)
    start_date = center_date - timedelta(days=days)
    end_date = center_date + timedelta(days=days)
    
    return start_date, end_date


def format_datetime_range(start: datetime, end: datetime) -> str:
    """
    Format datetime range for STAC query.
    
    Args:
        start: Start datetime
        end: End datetime
        
    Returns:
        ISO 8601 interval string
        
    Example:
        >>> format_datetime_range(datetime(2024, 6, 10), datetime(2024, 6, 20))
        '2024-06-10T00:00:00Z/2024-06-20T23:59:59Z'
    """
    start_str = start.strftime("%Y-%m-%dT00:00:00Z")
    end_str = end.strftime("%Y-%m-%dT23:59:59Z")
    return f"{start_str}/{end_str}"


def calculate_temporal_distance(item: Item, target_date: datetime) -> float:
    """
    Calculate temporal distance (in days) from target date.
    
    Args:
        item: STAC Item
        target_date: Target date
        
    Returns:
        Absolute difference in days
    """
    item_datetime = item.datetime
    if item_datetime is None:
        # Fallback to properties
        item_datetime = datetime.fromisoformat(
            item.properties.get("datetime", "").replace("Z", "+00:00")
        )
    
    delta = abs((item_datetime - target_date).total_seconds() / 86400)
    return delta


def extract_asset_hrefs(item: Item, asset_keys: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Extract asset HREFs from STAC Item.
    
    Args:
        item: STAC Item
        asset_keys: List of asset keys to extract (None = all assets)
        
    Returns:
        Dictionary mapping asset keys to HREFs
        
    Example:
        >>> hrefs = extract_asset_hrefs(item, ["B02", "B03", "B04"])
        >>> print(hrefs)
        {'B02': 'https://...', 'B03': 'https://...', 'B04': 'https://...'}
    """
    assets = {}
    
    for key, asset in item.assets.items():
        if asset_keys is None or key in asset_keys:
            assets[key] = asset.href
    
    return assets


def extract_item_metadata(item: Item) -> Dict[str, Any]:
    """
    Extract essential metadata from STAC Item.
    
    Args:
        item: STAC Item
        
    Returns:
        Dictionary with metadata fields
    """
    properties = item.properties
    
    metadata = {
        "id": item.id,
        "collection": item.collection_id,
        "datetime": item.datetime.isoformat() if item.datetime else None,
        "bbox": item.bbox,
        "geometry": item.geometry,
        "platform": properties.get("platform"),
        "instruments": properties.get("instruments"),
        "constellation": properties.get("constellation"),
    }
    
    # Add collection-specific metadata
    if "sentinel-1" in item.collection_id.lower():
        metadata.update({
            "orbit_direction": properties.get("sat:orbit_state"),
            "relative_orbit": properties.get("sat:relative_orbit"),
            "polarizations": properties.get("sar:polarizations"),
        })
    elif "sentinel-2" in item.collection_id.lower():
        metadata.update({
            "cloud_cover": properties.get("eo:cloud_cover"),
            "grid_code": properties.get("grid:code"),
        })
    elif "hls" in item.collection_id.lower():
        metadata.update({
            "cloud_cover": properties.get("eo:cloud_cover"),
            "tile_id": properties.get("hls:tile_id"),
        })
    
    return metadata


# ============================================================================
# Sentinel-1 GRD Search
# ============================================================================

def search_s1_grd(
    bbox: List[float],
    date_iso: str,
    days: int = 5,
    orbit: Optional[str] = None,
    polarizations: Tuple[str, ...] = ("VV", "VH"),
    max_results: int = 50,
    endpoint: str = "earthsearch",
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Search for Sentinel-1 GRD imagery, sorted by temporal proximity.
    
    Searches for Sentinel-1 SAR imagery in Ground Range Detected (GRD) format
    within a temporal window around the target date. Results are sorted by
    acquisition date proximity to the target.
    
    Args:
        bbox: Bounding box [west, south, east, north] (EPSG:4326)
        date_iso: Target date as ISO string (YYYY-MM-DD)
        days: Search window (±days from target date)
        orbit: Orbit direction filter ("ASCENDING", "DESCENDING", None=both)
        polarizations: Required polarizations (e.g., ("VV", "VH"))
        max_results: Maximum number of results to return
        endpoint: STAC endpoint to use ("earthsearch" or "mpc")
        verbose: Print search details
        
    Returns:
        List of dictionaries containing:
        - item: STAC Item object
        - metadata: Essential metadata
        - assets: Dictionary of asset HREFs
        - temporal_distance: Days from target date
        
    Example:
        >>> bbox = [31.5, -1.5, 34.0, 0.5]  # Lake Victoria
        >>> items = search_s1_grd(bbox, "2024-06-15", days=5)
        >>> 
        >>> for result in items[:3]:
        ...     print(f"{result['metadata']['id']}: {result['temporal_distance']:.1f} days")
        ...     print(f"  Orbit: {result['metadata']['orbit_direction']}")
        ...     print(f"  Polarizations: {result['metadata']['polarizations']}")
    """
    if not PYSTAC_AVAILABLE:
        raise ImportError(
            "pystac-client is required. Install with: pip install pystac-client"
        )
    
    # Get endpoint URL
    if endpoint == "earthsearch":
        url = EARTHSEARCH_URL
        collection_id = "sentinel-1-grd"
    elif endpoint == "mpc":
        url = MPC_URL
        collection_id = "sentinel-1-grd"
    else:
        raise ValueError(f"Unknown endpoint: {endpoint}")
    
    if verbose:
        print(f"Searching Sentinel-1 GRD on {endpoint}")
        print(f"  Date: {date_iso} ±{days} days")
        print(f"  BBox: {bbox}")
        if orbit:
            print(f"  Orbit: {orbit}")
        print(f"  Polarizations: {polarizations}")
    
    # Parse date range
    start_date, end_date = parse_date_range(date_iso, days)
    datetime_range = format_datetime_range(start_date, end_date)
    target_date = datetime.fromisoformat(date_iso)
    
    # Connect to STAC API
    catalog = Client.open(url)
    
    # Build query
    query_params = {
        "collections": [collection_id],
        "bbox": bbox,
        "datetime": datetime_range,
        "limit": max_results,
    }
    
    # Add property filters if needed
    if orbit or polarizations:
        query_params["query"] = {}
        
        if orbit:
            query_params["query"]["sat:orbit_state"] = {"eq": orbit.upper()}
        
        if polarizations:
            # Filter for items containing all required polarizations
            query_params["query"]["sar:polarizations"] = {
                "in": list(polarizations)
            }
    
    # Execute search
    search = catalog.search(**query_params)
    items = list(search.items())
    
    if verbose:
        print(f"  Found {len(items)} items")
    
    # Process results
    results = []
    for item in items:
        # Calculate temporal distance
        distance = calculate_temporal_distance(item, target_date)
        
        # Extract metadata
        metadata = extract_item_metadata(item)
        
        # Extract asset HREFs (VV, VH bands)
        assets = extract_asset_hrefs(item, list(polarizations))
        
        results.append({
            "item": item,
            "metadata": metadata,
            "assets": assets,
            "temporal_distance": distance,
        })
    
    # Sort by temporal proximity
    results.sort(key=lambda x: x["temporal_distance"])
    
    if verbose:
        print(f"  Closest item: {results[0]['temporal_distance']:.2f} days")
    
    return results


# ============================================================================
# Sentinel-2 L2A Cloud-Free Search
# ============================================================================

def search_s2_l2a_cloudfree(
    bbox: List[float],
    date_iso: str,
    days: int = 7,
    max_cloud: float = 20.0,
    max_results: int = 50,
    endpoint: str = "earthsearch",
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Search for cloud-free Sentinel-2 L2A imagery.
    
    Searches for Sentinel-2 Level-2A surface reflectance imagery with low
    cloud cover. Results are sorted by temporal proximity and cloud cover.
    
    Args:
        bbox: Bounding box [west, south, east, north] (EPSG:4326)
        date_iso: Target date as ISO string (YYYY-MM-DD)
        days: Search window (±days from target date)
        max_cloud: Maximum cloud cover percentage (0-100)
        max_results: Maximum number of results to return
        endpoint: STAC endpoint to use ("earthsearch" or "mpc")
        verbose: Print search details
        
    Returns:
        List of dictionaries containing:
        - item: STAC Item object
        - metadata: Essential metadata (includes cloud_cover)
        - assets: Dictionary of asset HREFs
        - temporal_distance: Days from target date
        - cloud_cover: Cloud cover percentage
        
    Example:
        >>> bbox = [31.5, -1.5, 34.0, 0.5]
        >>> items = search_s2_l2a_cloudfree(bbox, "2024-06-15", max_cloud=10)
        >>> 
        >>> for result in items[:3]:
        ...     print(f"{result['metadata']['id']}")
        ...     print(f"  Cloud: {result['cloud_cover']:.1f}%")
        ...     print(f"  Distance: {result['temporal_distance']:.1f} days")
    """
    if not PYSTAC_AVAILABLE:
        raise ImportError(
            "pystac-client is required. Install with: pip install pystac-client"
        )
    
    # Get endpoint URL
    if endpoint == "earthsearch":
        url = EARTHSEARCH_URL
        collection_id = "sentinel-2-l2a"
    elif endpoint == "mpc":
        url = MPC_URL
        collection_id = "sentinel-2-l2a"
    else:
        raise ValueError(f"Unknown endpoint: {endpoint}")
    
    if verbose:
        print(f"Searching Sentinel-2 L2A on {endpoint}")
        print(f"  Date: {date_iso} ±{days} days")
        print(f"  BBox: {bbox}")
        print(f"  Max cloud: {max_cloud}%")
    
    # Parse date range
    start_date, end_date = parse_date_range(date_iso, days)
    datetime_range = format_datetime_range(start_date, end_date)
    target_date = datetime.fromisoformat(date_iso)
    
    # Connect to STAC API
    catalog = Client.open(url)
    
    # Build query with cloud cover filter
    query_params = {
        "collections": [collection_id],
        "bbox": bbox,
        "datetime": datetime_range,
        "query": {
            "eo:cloud_cover": {"lte": max_cloud}
        },
        "limit": max_results,
    }
    
    # Execute search
    search = catalog.search(**query_params)
    items = list(search.items())
    
    if verbose:
        print(f"  Found {len(items)} items")
    
    # Process results
    results = []
    for item in items:
        # Calculate temporal distance
        distance = calculate_temporal_distance(item, target_date)
        
        # Extract metadata
        metadata = extract_item_metadata(item)
        
        # Get cloud cover
        cloud_cover = item.properties.get("eo:cloud_cover", 100.0)
        
        # Extract asset HREFs (common bands for low-VRAM: B02, B03, B04, B08)
        asset_keys = ["B02", "B03", "B04", "B08", "B11", "B12", "SCL"]
        assets = extract_asset_hrefs(item, asset_keys)
        
        results.append({
            "item": item,
            "metadata": metadata,
            "assets": assets,
            "temporal_distance": distance,
            "cloud_cover": cloud_cover,
        })
    
    # Sort by temporal proximity first, then cloud cover
    results.sort(key=lambda x: (x["temporal_distance"], x["cloud_cover"]))
    
    if verbose and results:
        print(f"  Best item: {results[0]['temporal_distance']:.2f} days, "
              f"{results[0]['cloud_cover']:.1f}% cloud")
    
    return results


# ============================================================================
# HLS Temporal Neighbors Search
# ============================================================================

def search_hls_neighbors(
    bbox: List[float],
    date_iso: str,
    k: int = 2,
    window_days: int = 40,
    max_cloud: float = 50.0,
    product: str = "both",
    endpoint: str = "lpcloud",
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Search for HLS temporal neighbors (k nearest acquisitions).
    
    Searches for Harmonized Landsat Sentinel-2 (HLS) imagery and returns the
    k nearest acquisitions in time. Useful for temporal interpolation or
    finding cloud-free alternatives.
    
    Args:
        bbox: Bounding box [west, south, east, north] (EPSG:4326)
        date_iso: Target date as ISO string (YYYY-MM-DD)
        k: Number of neighbors to return (before and after, total = 2*k)
        window_days: Maximum search window (±days from target)
        max_cloud: Maximum cloud cover percentage (0-100)
        product: HLS product ("L30"=Landsat, "S30"=Sentinel-2, "both")
        endpoint: STAC endpoint (only "lpcloud" supports HLS)
        verbose: Print search details
        
    Returns:
        List of dictionaries containing:
        - item: STAC Item object
        - metadata: Essential metadata
        - assets: Dictionary of asset HREFs
        - temporal_distance: Days from target date
        - cloud_cover: Cloud cover percentage
        
    Example:
        >>> bbox = [31.5, -1.5, 34.0, 0.5]
        >>> items = search_hls_neighbors(bbox, "2024-06-15", k=2, window_days=40)
        >>> 
        >>> print(f"Found {len(items)} neighbors")
        >>> for result in items:
        ...     days = result['temporal_distance']
        ...     direction = "before" if days < 0 else "after"
        ...     print(f"  {abs(days):.1f} days {direction}")
    """
    if not PYSTAC_AVAILABLE:
        raise ImportError(
            "pystac-client is required. Install with: pip install pystac-client"
        )
    
    if endpoint != "lpcloud":
        raise ValueError("HLS is only available on 'lpcloud' endpoint")
    
    # Get collection IDs
    collections = []
    if product in ["L30", "both"]:
        collections.append("HLSL30.v2.0")
    if product in ["S30", "both"]:
        collections.append("HLSS30.v2.0")
    
    if not collections:
        raise ValueError(f"Invalid HLS product: {product}")
    
    if verbose:
        print(f"Searching HLS on {endpoint}")
        print(f"  Date: {date_iso} ±{window_days} days")
        print(f"  BBox: {bbox}")
        print(f"  Collections: {collections}")
        print(f"  Target neighbors: {k} before + {k} after = {2*k} total")
    
    # Parse date range
    start_date, end_date = parse_date_range(date_iso, window_days)
    datetime_range = format_datetime_range(start_date, end_date)
    target_date = datetime.fromisoformat(date_iso)
    
    # Connect to STAC API
    url = LPCLOUD_URL
    catalog = Client.open(url)
    
    # Build query
    query_params = {
        "collections": collections,
        "bbox": bbox,
        "datetime": datetime_range,
        "query": {
            "eo:cloud_cover": {"lte": max_cloud}
        },
        "limit": 100,  # Get more than needed for neighbor selection
    }
    
    # Execute search
    search = catalog.search(**query_params)
    items = list(search.items())
    
    if verbose:
        print(f"  Found {len(items)} items")
    
    # Process results
    results = []
    for item in items:
        # Calculate temporal distance (preserve sign for before/after)
        item_datetime = item.datetime
        if item_datetime is None:
            item_datetime = datetime.fromisoformat(
                item.properties.get("datetime", "").replace("Z", "+00:00")
            )
        
        distance = (item_datetime - target_date).total_seconds() / 86400
        
        # Extract metadata
        metadata = extract_item_metadata(item)
        
        # Get cloud cover
        cloud_cover = item.properties.get("eo:cloud_cover", 100.0)
        
        # Extract asset HREFs (HLS bands)
        asset_keys = ["B02", "B03", "B04", "B05", "B06", "B07", "Fmask"]
        assets = extract_asset_hrefs(item, asset_keys)
        
        results.append({
            "item": item,
            "metadata": metadata,
            "assets": assets,
            "temporal_distance": distance,
            "cloud_cover": cloud_cover,
        })
    
    # Separate before and after
    before = [r for r in results if r["temporal_distance"] < 0]
    after = [r for r in results if r["temporal_distance"] >= 0]
    
    # Sort: before by descending time (closest last), after by ascending
    before.sort(key=lambda x: x["temporal_distance"], reverse=True)
    after.sort(key=lambda x: x["temporal_distance"])
    
    # Select k neighbors from each side
    neighbors = before[:k] + after[:k]
    
    # Sort final results by absolute distance
    neighbors.sort(key=lambda x: abs(x["temporal_distance"]))
    
    if verbose:
        print(f"  Selected {len(neighbors)} neighbors")
        if neighbors:
            print(f"  Closest: {abs(neighbors[0]['temporal_distance']):.2f} days")
    
    return neighbors


# ============================================================================
# Convenience Functions
# ============================================================================

def print_search_results(results: List[Dict[str, Any]], max_items: int = 10):
    """
    Pretty-print search results.
    
    Args:
        results: Search results from search_* functions
        max_items: Maximum number of items to print
    """
    print(f"\nSearch Results ({len(results)} items):")
    print("=" * 79)
    
    for i, result in enumerate(results[:max_items], 1):
        metadata = result['metadata']
        
        print(f"\n{i}. {metadata['id']}")
        print(f"   Collection: {metadata['collection']}")
        print(f"   Date: {metadata['datetime']}")
        print(f"   Temporal distance: {result['temporal_distance']:.2f} days")
        
        if 'cloud_cover' in result:
            print(f"   Cloud cover: {result['cloud_cover']:.1f}%")
        
        if 'orbit_direction' in metadata:
            print(f"   Orbit: {metadata['orbit_direction']}")
        
        print(f"   Assets: {list(result['assets'].keys())}")
    
    if len(results) > max_items:
        print(f"\n... and {len(results) - max_items} more items")
    
    print("=" * 79)


# ============================================================================
# Testing & Examples
# ============================================================================

if __name__ == "__main__":
    print("=" * 79)
    print("STAC DATA FETCHING TEST")
    print("=" * 79)
    print()
    
    # Check dependencies
    if not PYSTAC_AVAILABLE:
        print("ERROR: pystac-client not available")
        print("Install with: pip install pystac-client")
        exit(1)
    
    print("✓ pystac-client available")
    print()
    
    # Test configuration
    bbox = [31.5, -1.5, 34.0, 0.5]  # Lake Victoria
    date = "2024-06-15"
    
    print("Test Area: Lake Victoria")
    print(f"BBox: {bbox}")
    print(f"Target Date: {date}")
    print()
    
    # Test 1: Sentinel-1 GRD
    print("=" * 79)
    print("TEST 1: Sentinel-1 GRD Search")
    print("=" * 79)
    
    try:
        results = search_s1_grd(bbox, date, days=5, verbose=True)
        print_search_results(results, max_items=3)
    except Exception as e:
        print(f"ERROR: {e}")
    
    print()
    
    # Test 2: Sentinel-2 L2A Cloud-Free
    print("=" * 79)
    print("TEST 2: Sentinel-2 L2A Cloud-Free Search")
    print("=" * 79)
    
    try:
        results = search_s2_l2a_cloudfree(bbox, date, days=7, max_cloud=20, verbose=True)
        print_search_results(results, max_items=3)
    except Exception as e:
        print(f"ERROR: {e}")
    
    print()
    
    # Test 3: HLS Neighbors
    print("=" * 79)
    print("TEST 3: HLS Temporal Neighbors Search")
    print("=" * 79)
    
    try:
        results = search_hls_neighbors(bbox, date, k=2, window_days=40, verbose=True)
        print_search_results(results, max_items=4)
    except Exception as e:
        print(f"ERROR: {e}")
    
    print()
    print("=" * 79)
    print("TESTS COMPLETE")
    print("=" * 79)
