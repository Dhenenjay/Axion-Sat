#!/usr/bin/env python3
"""
tools/size_report.py - Dataset Size and Quality Report

Analyzes tile datasets and generates comprehensive reports including:
- Number of tiles per split (train/val/test)
- Mean cloud coverage percentage
- Total disk footprint
- Data quality metrics
- Coverage statistics

Usage:
    python tools/size_report.py data/index/dataset_tiles.csv
    python tools/size_report.py data/index/*.csv --format table
    python tools/size_report.py data/index/demo/*.csv --export report.json

Author: Axion-Sat Project
Version: 1.0.0
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import json

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("WARNING: pandas not available - install with: pip install pandas")


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SplitStats:
    """Statistics for a single split (train/val/test)."""
    split_name: str
    num_tiles: int
    total_size_bytes: int
    mean_tile_size_bytes: float
    mean_cloud_percent: Optional[float]
    mean_valid_percent: Optional[float]
    tile_paths: List[str]
    
    @property
    def total_size_mb(self) -> float:
        """Total size in MB."""
        return self.total_size_bytes / (1024 * 1024)
    
    @property
    def total_size_gb(self) -> float:
        """Total size in GB."""
        return self.total_size_bytes / (1024 * 1024 * 1024)
    
    @property
    def mean_tile_size_mb(self) -> float:
        """Mean tile size in MB."""
        return self.mean_tile_size_bytes / (1024 * 1024)


@dataclass
class DatasetReport:
    """Complete dataset report."""
    dataset_name: str
    index_file: str
    total_tiles: int
    total_size_bytes: int
    splits: Dict[str, SplitStats]
    
    @property
    def total_size_gb(self) -> float:
        """Total dataset size in GB."""
        return self.total_size_bytes / (1024 * 1024 * 1024)


# ============================================================================
# Analysis Functions
# ============================================================================

def get_tile_metadata(tile_path: Path) -> Dict:
    """
    Extract metadata from a tile NPZ file.
    
    Args:
        tile_path: Path to tile NPZ file
        
    Returns:
        Dictionary with tile metadata
    """
    metadata = {
        'path': str(tile_path),
        'exists': tile_path.exists(),
        'size_bytes': 0,
        'has_sar': False,
        'has_optical': False,
        'cloud_percent': None,
        'valid_percent': None
    }
    
    if not tile_path.exists():
        return metadata
    
    # Get file size
    metadata['size_bytes'] = tile_path.stat().st_size
    
    # Load tile to check contents
    try:
        tile = np.load(tile_path)
        bands = list(tile.files)
        
        # Check for SAR and optical
        metadata['has_sar'] = any(b.startswith('s1_') for b in bands)
        metadata['has_optical'] = any(b.startswith('s2_') for b in bands)
        
        # Estimate cloud coverage from valid data percentage
        # (Lower valid % often indicates clouds/shadows)
        if 's2_b4' in bands:
            optical_data = tile['s2_b4']
            total_pixels = optical_data.size
            valid_pixels = np.sum(~np.isnan(optical_data))
            metadata['valid_percent'] = (valid_pixels / total_pixels) * 100.0
            
            # Rough cloud estimate: assume missing data is clouds
            # (This is approximate - real SCL band would be better)
            metadata['cloud_percent'] = 100.0 - metadata['valid_percent']
        
        tile.close()
        
    except Exception as e:
        print(f"Warning: Could not read tile {tile_path.name}: {e}")
    
    return metadata


def analyze_split(
    split_name: str,
    tile_paths: List[str],
    verbose: bool = False
) -> SplitStats:
    """
    Analyze tiles for a single split.
    
    Args:
        split_name: Name of split (train/val/test)
        tile_paths: List of tile file paths
        verbose: Print progress
        
    Returns:
        SplitStats object
    """
    if verbose:
        print(f"  Analyzing {split_name} split ({len(tile_paths)} tiles)...", end=' ', flush=True)
    
    total_size = 0
    tile_sizes = []
    cloud_percentages = []
    valid_percentages = []
    
    for tile_path_str in tile_paths:
        tile_path = Path(tile_path_str)
        metadata = get_tile_metadata(tile_path)
        
        total_size += metadata['size_bytes']
        tile_sizes.append(metadata['size_bytes'])
        
        if metadata['cloud_percent'] is not None:
            cloud_percentages.append(metadata['cloud_percent'])
        
        if metadata['valid_percent'] is not None:
            valid_percentages.append(metadata['valid_percent'])
    
    # Compute statistics
    mean_tile_size = np.mean(tile_sizes) if tile_sizes else 0.0
    mean_cloud = np.mean(cloud_percentages) if cloud_percentages else None
    mean_valid = np.mean(valid_percentages) if valid_percentages else None
    
    if verbose:
        print("✓")
    
    return SplitStats(
        split_name=split_name,
        num_tiles=len(tile_paths),
        total_size_bytes=total_size,
        mean_tile_size_bytes=mean_tile_size,
        mean_cloud_percent=mean_cloud,
        mean_valid_percent=mean_valid,
        tile_paths=tile_paths
    )


def analyze_dataset(
    index_file: Path,
    verbose: bool = False
) -> DatasetReport:
    """
    Analyze entire dataset from index CSV.
    
    Args:
        index_file: Path to tile index CSV
        verbose: Print progress
        
    Returns:
        DatasetReport object
    """
    if not HAS_PANDAS:
        raise ImportError("pandas required. Install with: pip install pandas")
    
    if verbose:
        print(f"\nAnalyzing: {index_file.name}")
    
    # Load index
    try:
        df = pd.read_csv(index_file)
    except Exception as e:
        raise ValueError(f"Could not load index file {index_file}: {e}")
    
    # Validate columns
    required_cols = ['tile_path', 'split']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Index missing required columns: {missing_cols}")
    
    # Analyze by split
    splits = {}
    for split_name in ['train', 'val', 'test']:
        split_df = df[df['split'] == split_name]
        if len(split_df) > 0:
            tile_paths = split_df['tile_path'].tolist()
            splits[split_name] = analyze_split(split_name, tile_paths, verbose=verbose)
    
    # Overall statistics
    total_tiles = len(df)
    total_size = sum(s.total_size_bytes for s in splits.values())
    
    return DatasetReport(
        dataset_name=index_file.stem,
        index_file=str(index_file),
        total_tiles=total_tiles,
        total_size_bytes=total_size,
        splits=splits
    )


# ============================================================================
# Formatting Functions
# ============================================================================

def format_bytes(bytes_val: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


def print_report_text(report: DatasetReport):
    """Print report in text format."""
    print()
    print("=" * 79)
    print(f"DATASET SIZE REPORT: {report.dataset_name}")
    print("=" * 79)
    print()
    
    print(f"Index file: {report.index_file}")
    print(f"Total tiles: {report.total_tiles:,}")
    print(f"Total size: {format_bytes(report.total_size_bytes)} ({report.total_size_gb:.3f} GB)")
    print()
    
    # Split breakdown
    print("Split Breakdown:")
    print("-" * 79)
    print(f"{'Split':<10} {'Tiles':<10} {'Size':<15} {'Mean Size':<15} {'Cloud %':<10} {'Valid %':<10}")
    print("-" * 79)
    
    for split_name in ['train', 'val', 'test']:
        if split_name in report.splits:
            split = report.splits[split_name]
            
            cloud_str = f"{split.mean_cloud_percent:.1f}%" if split.mean_cloud_percent is not None else "N/A"
            valid_str = f"{split.mean_valid_percent:.1f}%" if split.mean_valid_percent is not None else "N/A"
            
            print(
                f"{split_name:<10} "
                f"{split.num_tiles:<10,} "
                f"{format_bytes(split.total_size_bytes):<15} "
                f"{format_bytes(int(split.mean_tile_size_bytes)):<15} "
                f"{cloud_str:<10} "
                f"{valid_str:<10}"
            )
    
    print("-" * 79)
    print()
    
    # Distribution
    if report.splits:
        print("Distribution:")
        total = report.total_tiles
        for split_name in ['train', 'val', 'test']:
            if split_name in report.splits:
                split = report.splits[split_name]
                pct = (split.num_tiles / total) * 100.0 if total > 0 else 0.0
                bar_width = int(pct / 2)
                bar = "█" * bar_width
                print(f"  {split_name:<6} {bar:<50} {pct:5.1f}% ({split.num_tiles:,} tiles)")
        print()


def print_report_table(report: DatasetReport):
    """Print report in compact table format."""
    print(f"\n{report.dataset_name}")
    print(f"{'Total:':<8} {report.total_tiles:>6,} tiles  {report.total_size_gb:>6.2f} GB")
    
    for split_name in ['train', 'val', 'test']:
        if split_name in report.splits:
            split = report.splits[split_name]
            cloud_str = f"{split.mean_cloud_percent:>5.1f}%" if split.mean_cloud_percent is not None else "  N/A"
            print(
                f"{split_name:<8} {split.num_tiles:>6,} tiles  "
                f"{split.total_size_mb:>8.1f} MB  "
                f"cloud: {cloud_str}"
            )


def print_report_csv(report: DatasetReport):
    """Print report in CSV format."""
    print("dataset,split,num_tiles,size_bytes,size_mb,size_gb,mean_cloud_percent,mean_valid_percent")
    
    for split_name in ['train', 'val', 'test']:
        if split_name in report.splits:
            split = report.splits[split_name]
            cloud = split.mean_cloud_percent if split.mean_cloud_percent is not None else ""
            valid = split.mean_valid_percent if split.mean_valid_percent is not None else ""
            print(
                f"{report.dataset_name},{split_name},{split.num_tiles},"
                f"{split.total_size_bytes},{split.total_size_mb:.2f},{split.total_size_gb:.4f},"
                f"{cloud},{valid}"
            )


def export_report_json(report: DatasetReport, output_path: Path):
    """Export report as JSON."""
    data = {
        'dataset_name': report.dataset_name,
        'index_file': report.index_file,
        'total_tiles': report.total_tiles,
        'total_size_bytes': report.total_size_bytes,
        'total_size_gb': report.total_size_gb,
        'splits': {}
    }
    
    for split_name, split_stats in report.splits.items():
        data['splits'][split_name] = {
            'num_tiles': split_stats.num_tiles,
            'total_size_bytes': split_stats.total_size_bytes,
            'total_size_mb': split_stats.total_size_mb,
            'total_size_gb': split_stats.total_size_gb,
            'mean_tile_size_bytes': split_stats.mean_tile_size_bytes,
            'mean_tile_size_mb': split_stats.mean_tile_size_mb,
            'mean_cloud_percent': split_stats.mean_cloud_percent,
            'mean_valid_percent': split_stats.mean_valid_percent
        }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Exported report to {output_path}")


# ============================================================================
# Multi-Dataset Analysis
# ============================================================================

def analyze_multiple_datasets(
    index_files: List[Path],
    format: str = 'text',
    verbose: bool = False
) -> List[DatasetReport]:
    """
    Analyze multiple datasets and generate comparative report.
    
    Args:
        index_files: List of index CSV files
        format: Output format
        verbose: Print progress
        
    Returns:
        List of DatasetReport objects
    """
    reports = []
    
    for index_file in index_files:
        try:
            report = analyze_dataset(index_file, verbose=verbose)
            reports.append(report)
        except Exception as e:
            print(f"Error analyzing {index_file}: {e}")
    
    return reports


def print_summary_table(reports: List[DatasetReport]):
    """Print summary comparison table for multiple datasets."""
    print()
    print("=" * 79)
    print("DATASET COMPARISON SUMMARY")
    print("=" * 79)
    print()
    
    # Header
    print(f"{'Dataset':<30} {'Total Tiles':<15} {'Size (GB)':<15} {'Mean Cloud %':<15}")
    print("-" * 79)
    
    # Rows
    for report in reports:
        # Calculate overall mean cloud
        cloud_vals = [
            s.mean_cloud_percent for s in report.splits.values()
            if s.mean_cloud_percent is not None
        ]
        mean_cloud = np.mean(cloud_vals) if cloud_vals else None
        cloud_str = f"{mean_cloud:.1f}%" if mean_cloud is not None else "N/A"
        
        print(
            f"{report.dataset_name:<30} "
            f"{report.total_tiles:<15,} "
            f"{report.total_size_gb:<15.3f} "
            f"{cloud_str:<15}"
        )
    
    # Totals
    if reports:
        total_tiles = sum(r.total_tiles for r in reports)
        total_size_gb = sum(r.total_size_gb for r in reports)
        print("-" * 79)
        print(
            f"{'TOTAL':<30} "
            f"{total_tiles:<15,} "
            f"{total_size_gb:<15.3f}"
        )
    
    print()


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate size and quality reports for tile datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single dataset
  python tools/size_report.py data/index/dataset_tiles.csv
  
  # Analyze multiple datasets
  python tools/size_report.py data/index/*.csv
  
  # Compact table format
  python tools/size_report.py data/index/*.csv --format table
  
  # CSV format
  python tools/size_report.py data/index/*.csv --format csv > report.csv
  
  # Export to JSON
  python tools/size_report.py data/index/dataset.csv --export report.json
  
  # Verbose output
  python tools/size_report.py data/index/*.csv --verbose
        """
    )
    
    parser.add_argument(
        'index_files',
        nargs='+',
        type=Path,
        help='One or more tile index CSV files'
    )
    
    parser.add_argument(
        '--format', '-f',
        choices=['text', 'table', 'csv'],
        default='text',
        help='Output format (default: text)'
    )
    
    parser.add_argument(
        '--export', '-e',
        type=Path,
        help='Export to JSON file'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Only show summary comparison (for multiple datasets)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    if not HAS_PANDAS:
        print("ERROR: pandas is required")
        print("Install with: pip install pandas")
        return 1
    
    # Expand wildcards and validate files
    index_files = []
    for pattern_path in args.index_files:
        if '*' in str(pattern_path):
            # Expand glob pattern
            matches = list(pattern_path.parent.glob(pattern_path.name))
            index_files.extend(matches)
        else:
            if pattern_path.exists():
                index_files.append(pattern_path)
            else:
                print(f"Warning: File not found: {pattern_path}")
    
    if not index_files:
        print("ERROR: No valid index files found")
        return 1
    
    # Analyze datasets
    reports = analyze_multiple_datasets(
        index_files,
        format=args.format,
        verbose=args.verbose
    )
    
    if not reports:
        print("ERROR: No datasets could be analyzed")
        return 1
    
    # Output results
    if len(reports) == 1:
        # Single dataset
        report = reports[0]
        
        if args.format == 'text':
            print_report_text(report)
        elif args.format == 'table':
            print_report_table(report)
        elif args.format == 'csv':
            print_report_csv(report)
        
        if args.export:
            export_report_json(report, args.export)
    
    else:
        # Multiple datasets
        if args.summary_only:
            print_summary_table(reports)
        else:
            # Show individual reports
            if args.format == 'text':
                for report in reports:
                    print_report_text(report)
            elif args.format == 'table':
                for report in reports:
                    print_report_table(report)
            elif args.format == 'csv':
                # CSV: combine all datasets
                print("dataset,split,num_tiles,size_bytes,size_mb,size_gb,mean_cloud_percent,mean_valid_percent")
                for report in reports:
                    for split_name in ['train', 'val', 'test']:
                        if split_name in report.splits:
                            split = report.splits[split_name]
                            cloud = split.mean_cloud_percent if split.mean_cloud_percent is not None else ""
                            valid = split.mean_valid_percent if split.mean_valid_percent is not None else ""
                            print(
                                f"{report.dataset_name},{split_name},{split.num_tiles},"
                                f"{split.total_size_bytes},{split.total_size_mb:.2f},{split.total_size_gb:.4f},"
                                f"{cloud},{valid}"
                            )
            
            # Show summary
            if args.format != 'csv':
                print_summary_table(reports)
            
            # Export first report if requested
            if args.export and reports:
                export_report_json(reports[0], args.export)
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
