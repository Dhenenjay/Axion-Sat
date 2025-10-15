"""
app/geocode.py - Geocoding Service with Caching

Provides geocoding functionality using Geopy/Nominatim with persistent
on-disk caching to minimize API calls and improve performance.

Features:
    - Place name to bounding box conversion
    - Centroid calculation
    - GeoJSON output format
    - Persistent JSON cache
    - Rate limiting (Nominatim ToS compliance)
    - Retry logic for transient failures

Usage:
    >>> from app.geocode import geocode_place
    >>> 
    >>> # Geocode a place
    >>> result = geocode_place("Lake Victoria")
    >>> print(result['display_name'])
    'Lake Victoria, Kenya/Tanzania/Uganda'
    >>> 
    >>> # Access bounding box
    >>> bbox = result['bbox_geojson']
    >>> print(bbox['coordinates'])  # [[lon, lat], [lon, lat], ...]
    >>> 
    >>> # Access centroid
    >>> centroid = result['centroid']
    >>> print(centroid)  # {'lon': 32.5, 'lat': -1.0}
"""

import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
from datetime import datetime
import warnings

# Suppress SSL warnings (common with Nominatim)
warnings.filterwarnings("ignore", message="Unverified HTTPS request")

try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderUnavailable, GeocoderServiceError
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False
    print("WARNING: geopy not available. Install with: pip install geopy")

# ============================================================================
# Configuration
# ============================================================================

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Cache directory and file
CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_FILE = CACHE_DIR / "geocode.json"

# User agent for Nominatim (required by ToS)
USER_AGENT = "axion-sat-water-segmentation/1.0"

# Rate limiting (Nominatim ToS: max 1 request/second)
MIN_REQUEST_INTERVAL = 1.0  # seconds

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2.0  # seconds

# Last request timestamp (for rate limiting)
_last_request_time = 0.0


# ============================================================================
# Cache Management
# ============================================================================

def load_cache() -> Dict[str, Dict]:
    """
    Load geocoding cache from disk.
    
    Returns:
        Dictionary mapping query strings to geocoding results
    """
    if not CACHE_FILE.exists():
        return {}
    
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("queries", {})
    except Exception as e:
        print(f"WARNING: Failed to load geocoding cache: {e}")
        return {}


def save_cache(cache: Dict[str, Dict]):
    """
    Save geocoding cache to disk.
    
    Args:
        cache: Dictionary mapping query strings to geocoding results
    """
    # Create cache directory if needed
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Prepare output data
    output = {
        "version": "1.0",
        "last_updated": datetime.now().isoformat(),
        "count": len(cache),
        "queries": cache,
    }
    
    # Write to file
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, sort_keys=True)
    except Exception as e:
        print(f"WARNING: Failed to save geocoding cache: {e}")


def get_cached_result(query: str) -> Optional[Dict]:
    """
    Get cached geocoding result for a query.
    
    Args:
        query: Query string (place name)
        
    Returns:
        Cached result dictionary, or None if not found
    """
    cache = load_cache()
    
    # Normalize query for cache lookup
    normalized_query = query.strip().lower()
    
    return cache.get(normalized_query)


def cache_result(query: str, result: Dict):
    """
    Cache a geocoding result.
    
    Args:
        query: Query string (place name)
        result: Geocoding result dictionary
    """
    cache = load_cache()
    
    # Normalize query for cache storage
    normalized_query = query.strip().lower()
    
    # Add metadata
    result_with_meta = {
        **result,
        "cached_at": datetime.now().isoformat(),
        "original_query": query,
    }
    
    cache[normalized_query] = result_with_meta
    save_cache(cache)


# ============================================================================
# Rate Limiting
# ============================================================================

def wait_for_rate_limit():
    """
    Enforce rate limit (Nominatim ToS: 1 request/second).
    
    Waits if necessary to ensure minimum interval between requests.
    """
    global _last_request_time
    
    current_time = time.time()
    time_since_last = current_time - _last_request_time
    
    if time_since_last < MIN_REQUEST_INTERVAL:
        sleep_time = MIN_REQUEST_INTERVAL - time_since_last
        time.sleep(sleep_time)
    
    _last_request_time = time.time()


# ============================================================================
# GeoJSON Conversion
# ============================================================================

def bbox_to_geojson(bbox: List[float]) -> Dict:
    """
    Convert bounding box to GeoJSON Polygon.
    
    Args:
        bbox: Bounding box as [south, north, west, east] (lat/lon)
        
    Returns:
        GeoJSON Polygon dictionary
        
    Example:
        >>> bbox = [-1.5, 0.5, 31.5, 34.0]  # Lake Victoria approx
        >>> geojson = bbox_to_geojson(bbox)
        >>> print(geojson['type'])
        'Polygon'
    """
    south, north, west, east = bbox
    
    # Create polygon coordinates (exterior ring, counterclockwise)
    # GeoJSON uses [longitude, latitude] order
    coordinates = [[
        [west, south],   # SW corner
        [east, south],   # SE corner
        [east, north],   # NE corner
        [west, north],   # NW corner
        [west, south],   # Close polygon
    ]]
    
    return {
        "type": "Polygon",
        "coordinates": coordinates,
        "bbox": [west, south, east, north],  # [minX, minY, maxX, maxY]
    }


def calculate_centroid(bbox: List[float]) -> Dict[str, float]:
    """
    Calculate centroid of bounding box.
    
    Args:
        bbox: Bounding box as [south, north, west, east]
        
    Returns:
        Dictionary with 'lon' and 'lat' keys
    """
    south, north, west, east = bbox
    
    return {
        "lon": (west + east) / 2.0,
        "lat": (south + north) / 2.0,
    }


# ============================================================================
# Geocoding Functions
# ============================================================================

def geocode_place(
    query: str,
    use_cache: bool = True,
    timeout: int = 10,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Geocode a place name to bounding box and centroid.
    
    Uses Nominatim geocoding service with persistent caching to minimize
    API calls. Respects Nominatim Terms of Service (1 request/second limit).
    
    Args:
        query: Place name to geocode (e.g., "Lake Victoria", "Cairo, Egypt")
        use_cache: Use cached results if available (default: True)
        timeout: Request timeout in seconds (default: 10)
        verbose: Print status messages (default: False)
        
    Returns:
        Dictionary with keys:
        - bbox_geojson: GeoJSON Polygon of bounding box
        - centroid: Dictionary with 'lon' and 'lat' keys
        - display_name: Human-readable place name
        - osm_id: OpenStreetMap ID
        - osm_type: OpenStreetMap type (node/way/relation)
        - importance: Nominatim importance score (0-1)
        - cached: Boolean indicating if result was from cache
        
    Example:
        >>> result = geocode_place("Lake Victoria")
        >>> print(result['display_name'])
        'Lake Victoria, Kenya/Tanzania/Uganda'
        >>> 
        >>> bbox = result['bbox_geojson']
        >>> print(bbox['coordinates'][0])  # Exterior ring
        [[31.5, -1.5], [34.0, -1.5], [34.0, 0.5], [31.5, 0.5], [31.5, -1.5]]
        >>> 
        >>> centroid = result['centroid']
        >>> print(f"Center: {centroid['lat']:.2f}°, {centroid['lon']:.2f}°")
        Center: -0.50°, 32.75°
        
    Raises:
        ImportError: If geopy is not installed
        ValueError: If query is empty or invalid
        RuntimeError: If geocoding fails after retries
    """
    if not GEOPY_AVAILABLE:
        raise ImportError(
            "geopy is required for geocoding. "
            "Install with: pip install geopy"
        )
    
    # Validate query
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    query = query.strip()
    
    # Check cache first
    if use_cache:
        cached_result = get_cached_result(query)
        if cached_result is not None:
            if verbose:
                print(f"✓ Using cached result for: {query}")
            # Mark as cached
            cached_result["cached"] = True
            return cached_result
    
    if verbose:
        print(f"Geocoding: {query}")
    
    # Initialize geocoder
    geolocator = Nominatim(
        user_agent=USER_AGENT,
        timeout=timeout,
    )
    
    # Attempt geocoding with retries
    location = None
    last_error = None
    
    for attempt in range(MAX_RETRIES):
        try:
            # Enforce rate limit
            wait_for_rate_limit()
            
            # Query Nominatim
            location = geolocator.geocode(
                query,
                exactly_one=True,
                addressdetails=True,
            )
            
            if location is not None:
                break  # Success
            else:
                if verbose:
                    print(f"  No results found for: {query}")
                raise ValueError(f"No results found for: {query}")
        
        except (GeocoderTimedOut, GeocoderUnavailable) as e:
            last_error = e
            if verbose:
                print(f"  Attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
        
        except GeocoderServiceError as e:
            # Don't retry on service errors (usually invalid query)
            raise RuntimeError(f"Geocoding service error: {e}")
    
    # Check if geocoding succeeded
    if location is None:
        raise RuntimeError(
            f"Geocoding failed after {MAX_RETRIES} attempts: {last_error}"
        )
    
    # Extract bounding box
    # Nominatim returns: [south, north, west, east]
    bbox_raw = location.raw.get("boundingbox", [])
    if len(bbox_raw) != 4:
        raise RuntimeError(f"Invalid bounding box from Nominatim: {bbox_raw}")
    
    bbox = [float(coord) for coord in bbox_raw]
    
    # Convert to GeoJSON
    bbox_geojson = bbox_to_geojson(bbox)
    
    # Calculate centroid
    centroid = calculate_centroid(bbox)
    
    # Prepare result
    result = {
        "bbox_geojson": bbox_geojson,
        "centroid": centroid,
        "display_name": location.address,
        "osm_id": location.raw.get("osm_id"),
        "osm_type": location.raw.get("osm_type"),
        "importance": location.raw.get("importance", 0.0),
        "place_rank": location.raw.get("place_rank"),
        "cached": False,
    }
    
    # Cache result
    if use_cache:
        cache_result(query, result)
        if verbose:
            print(f"✓ Cached result for: {query}")
    
    if verbose:
        print(f"✓ Geocoded: {result['display_name']}")
        print(f"  Centroid: {centroid['lat']:.4f}°, {centroid['lon']:.4f}°")
    
    return result


def batch_geocode(
    queries: List[str],
    use_cache: bool = True,
    verbose: bool = False
) -> Dict[str, Dict]:
    """
    Geocode multiple place names.
    
    Args:
        queries: List of place names to geocode
        use_cache: Use cached results if available
        verbose: Print progress messages
        
    Returns:
        Dictionary mapping queries to results
        
    Example:
        >>> queries = ["Lake Victoria", "Cairo, Egypt", "London, UK"]
        >>> results = batch_geocode(queries)
        >>> for query, result in results.items():
        ...     print(f"{query}: {result['centroid']}")
    """
    results = {}
    
    if verbose:
        print(f"Batch geocoding {len(queries)} queries...")
    
    for i, query in enumerate(queries, 1):
        if verbose:
            print(f"[{i}/{len(queries)}] {query}")
        
        try:
            result = geocode_place(query, use_cache=use_cache, verbose=False)
            results[query] = result
        except Exception as e:
            if verbose:
                print(f"  ERROR: {e}")
            results[query] = {"error": str(e)}
    
    if verbose:
        success_count = sum(1 for r in results.values() if "error" not in r)
        print(f"\n✓ Successfully geocoded {success_count}/{len(queries)} queries")
    
    return results


# ============================================================================
# Cache Management Functions
# ============================================================================

def clear_cache(verbose: bool = True):
    """
    Clear the geocoding cache.
    
    Args:
        verbose: Print status message
    """
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()
        if verbose:
            print(f"✓ Cleared geocoding cache: {CACHE_FILE}")
    else:
        if verbose:
            print("Cache file does not exist")


def list_cached_queries(verbose: bool = True) -> List[str]:
    """
    List all cached queries.
    
    Args:
        verbose: Print list to console
        
    Returns:
        List of cached query strings
    """
    cache = load_cache()
    queries = sorted(cache.keys())
    
    if verbose:
        print(f"Cached queries ({len(queries)}):")
        for query in queries:
            cached_at = cache[query].get("cached_at", "unknown")
            display_name = cache[query].get("display_name", "unknown")
            print(f"  - {query}")
            print(f"    → {display_name}")
            print(f"    (cached: {cached_at})")
    
    return queries


def get_cache_stats() -> Dict[str, Any]:
    """
    Get statistics about the geocoding cache.
    
    Returns:
        Dictionary with cache statistics
    """
    cache = load_cache()
    
    return {
        "total_queries": len(cache),
        "cache_file": str(CACHE_FILE),
        "cache_exists": CACHE_FILE.exists(),
        "cache_size_bytes": CACHE_FILE.stat().st_size if CACHE_FILE.exists() else 0,
    }


# ============================================================================
# Testing & Validation
# ============================================================================

if __name__ == "__main__":
    print("=" * 79)
    print("GEOCODING SERVICE TEST")
    print("=" * 79)
    print()
    
    # Check dependencies
    if not GEOPY_AVAILABLE:
        print("ERROR: geopy not available")
        print("Install with: pip install geopy")
        exit(1)
    
    print("✓ geopy available")
    print(f"Cache file: {CACHE_FILE}")
    print()
    
    # Test queries
    test_queries = [
        "Lake Victoria",
        "Nile Delta, Egypt",
        "Amazon River, Brazil",
        "Dead Sea",
        "Great Lakes, USA",
    ]
    
    print("Testing geocoding with sample queries...")
    print()
    
    for query in test_queries:
        print(f"Query: {query}")
        
        try:
            result = geocode_place(query, verbose=False)
            
            print(f"  Display name: {result['display_name']}")
            print(f"  Centroid: {result['centroid']['lat']:.4f}°, {result['centroid']['lon']:.4f}°")
            print(f"  Importance: {result['importance']:.4f}")
            print(f"  Cached: {result['cached']}")
            print()
            
            # Show bounding box
            bbox = result['bbox_geojson']['bbox']
            print(f"  Bounding box:")
            print(f"    West:  {bbox[0]:.4f}°")
            print(f"    South: {bbox[1]:.4f}°")
            print(f"    East:  {bbox[2]:.4f}°")
            print(f"    North: {bbox[3]:.4f}°")
            print()
        
        except Exception as e:
            print(f"  ERROR: {e}")
            print()
    
    # Show cache stats
    stats = get_cache_stats()
    print("=" * 79)
    print("Cache Statistics:")
    print(f"  Total cached queries: {stats['total_queries']}")
    print(f"  Cache size: {stats['cache_size_bytes']:,} bytes")
    print("=" * 79)
    print()
    
    print("To list cached queries:")
    print("  python -c \"from app.geocode import list_cached_queries; list_cached_queries()\"")
    print()
    print("To clear cache:")
    print("  python -c \"from app.geocode import clear_cache; clear_cache()\"")
