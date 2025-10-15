#!/usr/bin/env python3
"""
test_imports.py - Early Import Validation Test

This test module attempts to import every planned module and dependency
to catch path issues, missing DLLs, and import errors early in development.

This is especially important on Windows where DLL dependencies (GDAL, CUDA,
etc.) can cause cryptic import failures.

Usage:
    pytest tests/test_imports.py -v
    python tests/test_imports.py

Purpose:
    - Detect missing dependencies before runtime
    - Catch Windows-specific DLL issues (GDAL, CUDA, etc.)
    - Validate Python environment setup
    - Ensure all required packages are installed
"""

import sys
import warnings
from pathlib import Path
from typing import List, Tuple

import pytest


# ============================================================================
# Test Configuration
# ============================================================================

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Suppress non-critical warnings during import testing
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch")


# ============================================================================
# Core Python Standard Library Imports
# ============================================================================

def test_import_stdlib():
    """Test standard library imports."""
    import json
    import os
    import subprocess
    import sys
    import tempfile
    from datetime import datetime, timezone
    from pathlib import Path

    assert json is not None
    assert os is not None
    assert subprocess is not None
    assert sys is not None
    assert tempfile is not None
    assert datetime is not None
    assert timezone is not None
    assert Path is not None


# ============================================================================
# Core Scientific Python Stack
# ============================================================================

def test_import_numpy():
    """Test NumPy import."""
    import numpy as np

    assert np is not None
    assert hasattr(np, "__version__")
    print(f"  NumPy version: {np.__version__}")


def test_import_scipy():
    """Test SciPy import (optional but recommended)."""
    try:
        import scipy
        assert scipy is not None
        print(f"  SciPy version: {scipy.__version__}")
    except ImportError:
        pytest.skip("SciPy not installed (optional)")


def test_import_pandas():
    """Test pandas import."""
    import pandas as pd

    assert pd is not None
    assert hasattr(pd, "__version__")
    print(f"  pandas version: {pd.__version__}")


def test_import_sklearn():
    """Test scikit-learn import."""
    import sklearn

    assert sklearn is not None
    assert hasattr(sklearn, "__version__")
    print(f"  scikit-learn version: {sklearn.__version__}")


# ============================================================================
# Deep Learning Framework (PyTorch)
# ============================================================================

def test_import_torch():
    """Test PyTorch import - critical for all model operations."""
    import torch

    assert torch is not None
    assert hasattr(torch, "__version__")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  cuDNN version: {torch.backends.cudnn.version()}")


def test_import_torchvision():
    """Test torchvision import."""
    import torchvision

    assert torchvision is not None
    assert hasattr(torchvision, "__version__")
    print(f"  torchvision version: {torchvision.__version__}")


def test_import_timm():
    """Test timm (PyTorch Image Models) import."""
    try:
        import timm
        assert timm is not None
        assert hasattr(timm, "__version__")
        print(f"  timm version: {timm.__version__}")
    except ImportError:
        pytest.skip("timm not installed (required for ViT models)")


def test_import_einops():
    """Test einops import (for tensor operations)."""
    try:
        import einops
        assert einops is not None
        print(f"  einops version: {einops.__version__}")
    except ImportError:
        pytest.skip("einops not installed (required for advanced tensor ops)")


# ============================================================================
# Geospatial Libraries (Critical for Windows DLL issues)
# ============================================================================

def test_import_gdal():
    """Test GDAL/OGR import - common source of Windows DLL errors."""
    try:
        from osgeo import gdal, ogr, osr

        assert gdal is not None
        assert ogr is not None
        assert osr is not None
        print(f"  GDAL version: {gdal.__version__}")
        print(f"  GDAL data path: {gdal.GetConfigOption('GDAL_DATA')}")
    except ImportError as e:
        pytest.fail(
            f"GDAL import failed - common Windows DLL issue.\n"
            f"Error: {e}\n"
            f"Solution: Install GDAL via conda or OSGeo4W"
        )


def test_import_rasterio():
    """Test rasterio import - depends on GDAL."""
    try:
        import rasterio
        import rasterio.features
        import rasterio.transform
        import rasterio.warp

        assert rasterio is not None
        print(f"  rasterio version: {rasterio.__version__}")
        print(f"  rasterio GDAL version: {rasterio.__gdal_version__}")
    except ImportError as e:
        pytest.fail(
            f"rasterio import failed - likely GDAL DLL issue.\n"
            f"Error: {e}\n"
            f"Solution: Ensure GDAL is properly installed"
        )


def test_import_geopandas():
    """Test geopandas import."""
    try:
        import geopandas as gpd

        assert gpd is not None
        print(f"  geopandas version: {gpd.__version__}")
    except ImportError as e:
        pytest.fail(
            f"geopandas import failed.\n"
            f"Error: {e}\n"
            f"Solution: pip install geopandas or conda install geopandas"
        )


def test_import_shapely():
    """Test shapely import - geometry operations."""
    try:
        import shapely
        from shapely.geometry import Point, Polygon

        assert shapely is not None
        assert Point is not None
        assert Polygon is not None
        print(f"  shapely version: {shapely.__version__}")
    except ImportError as e:
        pytest.fail(f"shapely import failed: {e}")


def test_import_pyproj():
    """Test pyproj import - coordinate reference systems."""
    try:
        import pyproj

        assert pyproj is not None
        print(f"  pyproj version: {pyproj.__version__}")
    except ImportError as e:
        pytest.fail(f"pyproj import failed: {e}")


def test_import_fiona():
    """Test fiona import - vector data I/O."""
    try:
        import fiona

        assert fiona is not None
        print(f"  fiona version: {fiona.__version__}")
    except ImportError:
        pytest.skip("fiona not installed (optional)")


# ============================================================================
# STAC and Satellite Data Libraries
# ============================================================================

def test_import_pystac():
    """Test pystac import."""
    try:
        import pystac

        assert pystac is not None
        print(f"  pystac version: {pystac.__version__}")
    except ImportError:
        pytest.skip("pystac not installed")


def test_import_pystac_client():
    """Test pystac-client import."""
    try:
        import pystac_client

        assert pystac_client is not None
        print(f"  pystac-client version: {pystac_client.__version__}")
    except ImportError:
        pytest.skip("pystac-client not installed")


def test_import_planetary_computer():
    """Test planetary-computer import (optional)."""
    try:
        import planetary_computer

        assert planetary_computer is not None
        print(f"  planetary-computer version: {planetary_computer.__version__}")
    except ImportError:
        pytest.skip("planetary-computer not installed (optional)")


# ============================================================================
# Web Framework and API
# ============================================================================

def test_import_fastapi():
    """Test FastAPI import."""
    try:
        import fastapi
        from fastapi import FastAPI

        assert fastapi is not None
        assert FastAPI is not None
        print(f"  FastAPI version: {fastapi.__version__}")
    except ImportError:
        pytest.skip("FastAPI not installed")


def test_import_uvicorn():
    """Test uvicorn import."""
    try:
        import uvicorn

        assert uvicorn is not None
        print(f"  uvicorn version: {uvicorn.__version__}")
    except ImportError:
        pytest.skip("uvicorn not installed")


def test_import_pydantic():
    """Test pydantic import."""
    try:
        import pydantic

        assert pydantic is not None
        print(f"  pydantic version: {pydantic.__version__}")
    except ImportError:
        pytest.skip("pydantic not installed")


# ============================================================================
# Visualization Libraries
# ============================================================================

def test_import_matplotlib():
    """Test matplotlib import."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt

        assert matplotlib is not None
        assert plt is not None
        print(f"  matplotlib version: {matplotlib.__version__}")
    except ImportError:
        pytest.skip("matplotlib not installed (optional)")


def test_import_pillow():
    """Test PIL/Pillow import."""
    try:
        from PIL import Image

        assert Image is not None
        import PIL
        print(f"  Pillow version: {PIL.__version__}")
    except ImportError:
        pytest.skip("Pillow not installed")


def test_import_opencv():
    """Test OpenCV import (optional)."""
    try:
        import cv2

        assert cv2 is not None
        print(f"  OpenCV version: {cv2.__version__}")
    except ImportError:
        pytest.skip("OpenCV not installed (optional)")


# ============================================================================
# Utilities and Helpers
# ============================================================================

def test_import_requests():
    """Test requests import - HTTP client."""
    import requests

    assert requests is not None
    print(f"  requests version: {requests.__version__}")


def test_import_tqdm():
    """Test tqdm import - progress bars."""
    try:
        import tqdm

        assert tqdm is not None
        print(f"  tqdm version: {tqdm.__version__}")
    except ImportError:
        pytest.skip("tqdm not installed (optional)")


def test_import_dotenv():
    """Test python-dotenv import - environment variable loading."""
    try:
        import dotenv

        assert dotenv is not None
        print(f"  python-dotenv version: {dotenv.__version__}")
    except ImportError:
        pytest.skip("python-dotenv not installed")


def test_import_yaml():
    """Test PyYAML import."""
    try:
        import yaml

        assert yaml is not None
        print(f"  PyYAML version: {yaml.__version__}")
    except ImportError:
        pytest.skip("PyYAML not installed (optional)")


# ============================================================================
# Project-Specific Modules (Axion-Sat)
# ============================================================================

def test_import_vc_lib():
    """Test vc_lib package import (if exists)."""
    try:
        import vc_lib

        assert vc_lib is not None
        print("  vc_lib package found")
    except ImportError:
        pytest.skip("vc_lib package not yet implemented")


def test_import_vc_lib_data():
    """Test vc_lib.data module import."""
    try:
        import vc_lib.data

        assert vc_lib.data is not None
        print("  vc_lib.data module found")
    except ImportError:
        pytest.skip("vc_lib.data module not yet implemented")


def test_import_vc_lib_models():
    """Test vc_lib.models module import."""
    try:
        import vc_lib.models

        assert vc_lib.models is not None
        print("  vc_lib.models module found")
    except ImportError:
        pytest.skip("vc_lib.models module not yet implemented")


def test_import_vc_lib_training():
    """Test vc_lib.training module import."""
    try:
        import vc_lib.training

        assert vc_lib.training is not None
        print("  vc_lib.training module found")
    except ImportError:
        pytest.skip("vc_lib.training module not yet implemented")


def test_import_vc_lib_evaluation():
    """Test vc_lib.evaluation module import."""
    try:
        import vc_lib.evaluation

        assert vc_lib.evaluation is not None
        print("  vc_lib.evaluation module found")
    except ImportError:
        pytest.skip("vc_lib.evaluation module not yet implemented")


def test_import_vc_lib_utils():
    """Test vc_lib.utils module import."""
    try:
        import vc_lib.utils

        assert vc_lib.utils is not None
        print("  vc_lib.utils module found")
    except ImportError:
        pytest.skip("vc_lib.utils module not yet implemented")


# ============================================================================
# System Information
# ============================================================================

def test_python_version():
    """Check Python version."""
    version_info = sys.version_info
    print(f"  Python version: {version_info.major}.{version_info.minor}.{version_info.micro}")
    assert version_info.major == 3, "Python 3 required"
    assert version_info.minor >= 11, "Python 3.11+ recommended"


def test_platform_info():
    """Print platform information."""
    import platform

    print(f"  Platform: {platform.system()} {platform.release()}")
    print(f"  Architecture: {platform.machine()}")
    print(f"  Processor: {platform.processor()}")


# ============================================================================
# Summary Report
# ============================================================================

def test_generate_import_report():
    """Generate a summary report of all imports."""
    report = []
    report.append("\n" + "=" * 79)
    report.append("IMPORT TEST SUMMARY")
    report.append("=" * 79)

    # Core dependencies
    critical_imports = [
        ("numpy", "Core numerical computing"),
        ("pandas", "Data manipulation"),
        ("torch", "Deep learning framework"),
        ("rasterio", "Geospatial raster I/O"),
        ("geopandas", "Geospatial data manipulation"),
        ("pystac_client", "STAC catalog access"),
    ]

    report.append("\nCritical Dependencies:")
    for module_name, description in critical_imports:
        try:
            __import__(module_name)
            status = "✓ OK"
        except ImportError:
            status = "✗ MISSING"
        report.append(f"  [{status}] {module_name:20s} - {description}")

    # Optional dependencies
    optional_imports = [
        ("timm", "Vision Transformer models"),
        ("einops", "Advanced tensor operations"),
        ("opencv-python", "Computer vision utilities"),
        ("matplotlib", "Visualization"),
        ("tqdm", "Progress bars"),
    ]

    report.append("\nOptional Dependencies:")
    for module_name, description in optional_imports:
        try:
            # Handle special cases
            if module_name == "opencv-python":
                __import__("cv2")
            else:
                __import__(module_name)
            status = "✓ OK"
        except ImportError:
            status = "- Not installed"
        report.append(f"  [{status}] {module_name:20s} - {description}")

    report.append("\n" + "=" * 79)

    print("\n".join(report))


# ============================================================================
# Main Entry Point (for standalone execution)
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 79)
    print("AXION-SAT IMPORT VALIDATION TEST")
    print("=" * 79)
    print("\nRunning import tests to detect missing dependencies and DLL issues...")
    print("This is especially important on Windows where GDAL/CUDA DLLs can fail.\n")

    # Run tests manually (without pytest)
    import inspect

    test_functions = [
        obj for name, obj in inspect.getmembers(sys.modules[__name__])
        if inspect.isfunction(obj) and name.startswith("test_")
    ]

    passed = 0
    failed = 0
    skipped = 0

    for test_func in test_functions:
        test_name = test_func.__name__
        try:
            print(f"\n[TEST] {test_name}")
            test_func()
            passed += 1
            print(f"  ✓ PASSED")
        except AssertionError as e:
            failed += 1
            print(f"  ✗ FAILED: {e}")
        except Exception as e:
            if "skip" in str(e).lower():
                skipped += 1
                print(f"  - SKIPPED: {e}")
            else:
                failed += 1
                print(f"  ✗ ERROR: {e}")

    # Print summary
    print("\n" + "=" * 79)
    print("TEST SUMMARY")
    print("=" * 79)
    print(f"  Passed:  {passed}")
    print(f"  Failed:  {failed}")
    print(f"  Skipped: {skipped}")
    print(f"  Total:   {len(test_functions)}")
    print("=" * 79 + "\n")

    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)
