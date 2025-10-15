#!/usr/bin/env python3
"""
tools/manifest_lint.py - Tile Manifest Linter

Validates tile manifests and datasets for common issues:
- Duplicate tile paths or IDs
- Missing or corrupted tile files
- Missing band arrays (SAR or optical)
- Inconsistent shapes across tiles
- Invalid split assignments
- Broken metadata links
- Orphaned files

Usage:
    python tools/manifest_lint.py data/index/tiles.csv
    python tools/manifest_lint.py data/index/*.csv --fix
    python tools/manifest_lint.py data/index/tiles.csv --check-bands --verbose

Author: Axion-Sat Project
Version: 1.0.0
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import json

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("ERROR: pandas is required. Install with: pip install pandas")
    sys.exit(1)


# ============================================================================
# Configuration
# ============================================================================

# Expected band names
EXPECTED_SAR_BANDS = ['s1_vv', 's1_vh']
EXPECTED_OPTICAL_BANDS = ['s2_b2', 's2_b3', 's2_b4', 's2_b5', 's2_b6', 
                          's2_b7', 's2_b8', 's2_b8a', 's2_b11', 's2_b12']
EXPECTED_ALL_BANDS = EXPECTED_SAR_BANDS + EXPECTED_OPTICAL_BANDS

VALID_SPLITS = {'train', 'val', 'test'}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class LintIssue:
    """Represents a linting issue found in the dataset."""
    severity: str  # 'error', 'warning', 'info'
    category: str  # 'duplicate', 'missing', 'invalid', etc.
    message: str
    tile_path: Optional[str] = None
    details: Dict = field(default_factory=dict)
    
    def __str__(self) -> str:
        severity_icon = {
            'error': '✗',
            'warning': '⚠',
            'info': 'ℹ'
        }
        icon = severity_icon.get(self.severity, '•')
        
        msg = f"{icon} [{self.severity.upper()}] {self.category}: {self.message}"
        if self.tile_path:
            msg += f"\n  Tile: {self.tile_path}"
        if self.details:
            for key, value in self.details.items():
                msg += f"\n  {key}: {value}"
        return msg


@dataclass
class LintReport:
    """Summary of linting results."""
    total_tiles: int = 0
    issues: List[LintIssue] = field(default_factory=list)
    checked_bands: bool = False
    checked_files: bool = False
    
    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == 'error')
    
    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == 'warning')
    
    @property
    def info_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == 'info')
    
    def is_clean(self) -> bool:
        """Check if no errors found."""
        return self.error_count == 0
    
    def print_summary(self):
        """Print summary of linting results."""
        print("\n" + "=" * 80)
        print("MANIFEST LINT REPORT")
        print("=" * 80)
        print(f"\nTotal tiles checked: {self.total_tiles}")
        print(f"Errors:   {self.error_count}")
        print(f"Warnings: {self.warning_count}")
        print(f"Info:     {self.info_count}")
        
        if self.is_clean():
            print("\n✓ No errors found! Dataset is clean.")
        else:
            print(f"\n✗ Found {self.error_count} error(s) that must be fixed.")
        
        print("=" * 80)


# ============================================================================
# Linting Functions
# ============================================================================

def check_duplicate_paths(df: pd.DataFrame, report: LintReport) -> None:
    """Check for duplicate tile paths in manifest."""
    if 'tile_path' not in df.columns:
        report.issues.append(LintIssue(
            severity='error',
            category='schema',
            message="Missing 'tile_path' column in manifest"
        ))
        return
    
    duplicates = df[df.duplicated(subset=['tile_path'], keep=False)]
    
    if len(duplicates) > 0:
        dup_paths = duplicates['tile_path'].unique()
        for path in dup_paths:
            count = len(df[df['tile_path'] == path])
            report.issues.append(LintIssue(
                severity='error',
                category='duplicate',
                message=f"Duplicate tile path appears {count} times",
                tile_path=path
            ))


def check_duplicate_ids(df: pd.DataFrame, report: LintReport) -> None:
    """Check for duplicate tile IDs in manifest."""
    if 'tile_id' not in df.columns:
        report.issues.append(LintIssue(
            severity='info',
            category='schema',
            message="No 'tile_id' column found (optional)"
        ))
        return
    
    duplicates = df[df.duplicated(subset=['tile_id'], keep=False)]
    
    if len(duplicates) > 0:
        dup_ids = duplicates['tile_id'].unique()
        for tile_id in dup_ids:
            count = len(df[df['tile_id'] == tile_id])
            paths = df[df['tile_id'] == tile_id]['tile_path'].tolist()
            report.issues.append(LintIssue(
                severity='error',
                category='duplicate',
                message=f"Duplicate tile ID appears {count} times",
                details={'tile_id': tile_id, 'paths': paths[:3]}
            ))


def check_file_existence(df: pd.DataFrame, report: LintReport, base_dir: Optional[Path] = None) -> None:
    """Check if tile files actually exist on disk."""
    missing_files = []
    
    for idx, row in df.iterrows():
        tile_path = Path(row['tile_path'])
        
        # Handle relative paths
        if not tile_path.is_absolute() and base_dir:
            tile_path = base_dir / tile_path
        
        if not tile_path.exists():
            missing_files.append(str(row['tile_path']))
            report.issues.append(LintIssue(
                severity='error',
                category='missing',
                message="Tile file does not exist",
                tile_path=str(row['tile_path'])
            ))
    
    if missing_files:
        report.issues.append(LintIssue(
            severity='error',
            category='missing',
            message=f"Found {len(missing_files)} missing tile files"
        ))


def check_tile_bands(
    df: pd.DataFrame, 
    report: LintReport, 
    base_dir: Optional[Path] = None,
    sample_size: int = 0
) -> None:
    """Check that tiles contain expected band arrays."""
    if sample_size > 0:
        sample_df = df.sample(min(sample_size, len(df)))
    else:
        sample_df = df
    
    tiles_to_check = len(sample_df)
    tiles_checked = 0
    
    for idx, row in sample_df.iterrows():
        tile_path = Path(row['tile_path'])
        
        # Handle relative paths
        if not tile_path.is_absolute() and base_dir:
            tile_path = base_dir / tile_path
        
        if not tile_path.exists():
            continue  # Already reported as missing
        
        try:
            # Load tile
            tile = np.load(tile_path)
            bands = set(tile.files)
            tile.close()
            
            # Check for missing SAR bands
            missing_sar = set(EXPECTED_SAR_BANDS) - bands
            if missing_sar:
                report.issues.append(LintIssue(
                    severity='error',
                    category='missing_bands',
                    message=f"Missing SAR bands: {missing_sar}",
                    tile_path=str(row['tile_path'])
                ))
            
            # Check for missing optical bands
            missing_optical = set(EXPECTED_OPTICAL_BANDS) - bands
            if missing_optical:
                report.issues.append(LintIssue(
                    severity='warning',
                    category='missing_bands',
                    message=f"Missing optical bands: {missing_optical}",
                    tile_path=str(row['tile_path'])
                ))
            
            # Check for unexpected bands
            unexpected = bands - set(EXPECTED_ALL_BANDS)
            if unexpected:
                report.issues.append(LintIssue(
                    severity='info',
                    category='extra_bands',
                    message=f"Unexpected bands found: {unexpected}",
                    tile_path=str(row['tile_path'])
                ))
            
            tiles_checked += 1
            
        except Exception as e:
            report.issues.append(LintIssue(
                severity='error',
                category='corrupted',
                message=f"Failed to load tile: {e}",
                tile_path=str(row['tile_path'])
            ))
    
    if sample_size > 0:
        report.issues.append(LintIssue(
            severity='info',
            category='sampling',
            message=f"Checked bands in {tiles_checked}/{tiles_to_check} sampled tiles"
        ))


def check_band_shapes(
    df: pd.DataFrame, 
    report: LintReport, 
    base_dir: Optional[Path] = None,
    sample_size: int = 10
) -> None:
    """Check that all bands in tiles have consistent shapes."""
    sample_df = df.sample(min(sample_size, len(df)))
    
    expected_shape = None
    
    for idx, row in sample_df.iterrows():
        tile_path = Path(row['tile_path'])
        
        if not tile_path.is_absolute() and base_dir:
            tile_path = base_dir / tile_path
        
        if not tile_path.exists():
            continue
        
        try:
            tile = np.load(tile_path)
            
            # Get shapes of all bands
            shapes = {band: tile[band].shape for band in tile.files}
            tile.close()
            
            # Check all bands have same shape
            unique_shapes = set(shapes.values())
            if len(unique_shapes) > 1:
                report.issues.append(LintIssue(
                    severity='error',
                    category='inconsistent_shape',
                    message="Bands have different shapes within tile",
                    tile_path=str(row['tile_path']),
                    details={'shapes': {k: str(v) for k, v in shapes.items()}}
                ))
            
            # Track expected shape across tiles
            tile_shape = list(unique_shapes)[0] if unique_shapes else None
            if tile_shape:
                if expected_shape is None:
                    expected_shape = tile_shape
                elif tile_shape != expected_shape:
                    report.issues.append(LintIssue(
                        severity='warning',
                        category='inconsistent_shape',
                        message=f"Tile shape {tile_shape} differs from expected {expected_shape}",
                        tile_path=str(row['tile_path'])
                    ))
        
        except Exception as e:
            pass  # Already reported as corrupted


def check_split_validity(df: pd.DataFrame, report: LintReport) -> None:
    """Check that split assignments are valid."""
    if 'split' not in df.columns:
        report.issues.append(LintIssue(
            severity='warning',
            category='schema',
            message="No 'split' column found"
        ))
        return
    
    invalid_splits = df[~df['split'].isin(VALID_SPLITS)]
    
    for idx, row in invalid_splits.iterrows():
        report.issues.append(LintIssue(
            severity='error',
            category='invalid_split',
            message=f"Invalid split value: '{row['split']}' (expected: {VALID_SPLITS})",
            tile_path=str(row['tile_path']) if 'tile_path' in row else None
        ))


def check_split_distribution(df: pd.DataFrame, report: LintReport) -> None:
    """Check that split distribution is reasonable."""
    if 'split' not in df.columns:
        return
    
    split_counts = df['split'].value_counts()
    total = len(df)
    
    for split_name in VALID_SPLITS:
        count = split_counts.get(split_name, 0)
        pct = (count / total * 100) if total > 0 else 0
        
        if count == 0:
            report.issues.append(LintIssue(
                severity='warning',
                category='split_distribution',
                message=f"Split '{split_name}' has no tiles"
            ))
        elif pct < 5:
            report.issues.append(LintIssue(
                severity='warning',
                category='split_distribution',
                message=f"Split '{split_name}' has very few tiles: {count} ({pct:.1f}%)"
            ))


def check_metadata_files(df: pd.DataFrame, report: LintReport, base_dir: Optional[Path] = None) -> None:
    """Check if metadata JSON files exist for tiles."""
    missing_metadata = []
    
    for idx, row in df.iterrows():
        tile_path = Path(row['tile_path'])
        
        if not tile_path.is_absolute() and base_dir:
            tile_path = base_dir / tile_path
        
        json_path = tile_path.with_suffix('.json')
        
        if tile_path.exists() and not json_path.exists():
            missing_metadata.append(str(row['tile_path']))
    
    if missing_metadata:
        report.issues.append(LintIssue(
            severity='info',
            category='missing_metadata',
            message=f"Found {len(missing_metadata)} tiles without metadata JSON files"
        ))


def check_orphaned_files(df: pd.DataFrame, report: LintReport, tile_dir: Path) -> None:
    """Check for tile files not referenced in manifest."""
    if not tile_dir.exists():
        return
    
    # Get all NPZ files in directory
    all_tiles = set(tile_dir.rglob('*.npz'))
    
    # Get tiles referenced in manifest
    manifest_tiles = set()
    for tile_path in df['tile_path']:
        path = Path(tile_path)
        if not path.is_absolute():
            path = tile_dir / path
        manifest_tiles.add(path.resolve())
    
    # Find orphans
    orphaned = []
    for tile in all_tiles:
        if tile.resolve() not in manifest_tiles:
            orphaned.append(str(tile.relative_to(tile_dir)))
    
    if orphaned:
        report.issues.append(LintIssue(
            severity='warning',
            category='orphaned',
            message=f"Found {len(orphaned)} tile files not in manifest",
            details={'examples': orphaned[:5]}
        ))


def check_required_columns(df: pd.DataFrame, report: LintReport) -> None:
    """Check that required columns exist in manifest."""
    required = ['tile_path']
    recommended = ['tile_id', 'split', 'date']
    
    for col in required:
        if col not in df.columns:
            report.issues.append(LintIssue(
                severity='error',
                category='schema',
                message=f"Missing required column: '{col}'"
            ))
    
    for col in recommended:
        if col not in df.columns:
            report.issues.append(LintIssue(
                severity='info',
                category='schema',
                message=f"Missing recommended column: '{col}'"
            ))


# ============================================================================
# Main Linter
# ============================================================================

def lint_manifest(
    manifest_path: Path,
    check_bands: bool = False,
    check_files: bool = True,
    check_orphans: bool = False,
    sample_size: int = 0,
    base_dir: Optional[Path] = None,
    verbose: bool = False
) -> LintReport:
    """
    Lint a tile manifest for issues.
    
    Args:
        manifest_path: Path to manifest CSV
        check_bands: Check band arrays in tile files
        check_files: Check that tile files exist
        check_orphans: Check for orphaned tile files
        sample_size: Number of tiles to sample for band checks (0 = all)
        base_dir: Base directory for resolving relative paths
        verbose: Print progress messages
        
    Returns:
        LintReport with found issues
    """
    report = LintReport()
    
    if verbose:
        print(f"\nLinting manifest: {manifest_path.name}")
    
    # Load manifest
    try:
        df = pd.read_csv(manifest_path)
        report.total_tiles = len(df)
    except Exception as e:
        report.issues.append(LintIssue(
            severity='error',
            category='load_error',
            message=f"Failed to load manifest: {e}"
        ))
        return report
    
    if verbose:
        print(f"  Loaded {len(df)} tiles")
    
    # Determine base directory
    if base_dir is None:
        base_dir = manifest_path.parent.parent  # Assume manifest in data/index/
    
    # Run checks
    if verbose:
        print("  Checking schema...")
    check_required_columns(df, report)
    
    if verbose:
        print("  Checking for duplicates...")
    check_duplicate_paths(df, report)
    check_duplicate_ids(df, report)
    
    if verbose:
        print("  Checking split assignments...")
    check_split_validity(df, report)
    check_split_distribution(df, report)
    
    if check_files:
        if verbose:
            print("  Checking file existence...")
        check_file_existence(df, report, base_dir)
        report.checked_files = True
    
    if check_bands:
        if verbose:
            print(f"  Checking band arrays{' (sampling)' if sample_size > 0 else ''}...")
        check_tile_bands(df, report, base_dir, sample_size)
        check_band_shapes(df, report, base_dir, sample_size=min(10, len(df)))
        report.checked_bands = True
    
    if check_files:
        if verbose:
            print("  Checking metadata files...")
        check_metadata_files(df, report, base_dir)
    
    if check_orphans:
        if verbose:
            print("  Checking for orphaned files...")
        tile_dir = base_dir / 'tiles'
        check_orphaned_files(df, report, tile_dir)
    
    return report


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Lint tile manifests for issues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic linting
  python tools/manifest_lint.py data/index/tiles.csv
  
  # Check band arrays (slower)
  python tools/manifest_lint.py data/index/tiles.csv --check-bands
  
  # Sample 100 tiles for band checking
  python tools/manifest_lint.py data/index/tiles.csv --check-bands --sample 100
  
  # Check multiple manifests
  python tools/manifest_lint.py data/index/*.csv
  
  # Check for orphaned files
  python tools/manifest_lint.py data/index/tiles.csv --check-orphans
  
  # Verbose output
  python tools/manifest_lint.py data/index/tiles.csv --verbose
        """
    )
    
    parser.add_argument(
        'manifests',
        nargs='+',
        type=Path,
        help='Manifest CSV file(s) to lint'
    )
    
    parser.add_argument(
        '--check-bands',
        action='store_true',
        help='Check band arrays in tile files (slower)'
    )
    
    parser.add_argument(
        '--check-orphans',
        action='store_true',
        help='Check for orphaned tile files not in manifest'
    )
    
    parser.add_argument(
        '--no-check-files',
        action='store_true',
        help='Skip checking if tile files exist'
    )
    
    parser.add_argument(
        '--sample',
        type=int,
        default=0,
        help='Sample size for band checks (0 = all tiles)'
    )
    
    parser.add_argument(
        '--base-dir',
        type=Path,
        help='Base directory for resolving relative paths'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--errors-only',
        action='store_true',
        help='Only show errors, hide warnings and info'
    )
    
    parser.add_argument(
        '--json',
        type=Path,
        help='Export results to JSON file'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Expand glob patterns
    manifests = []
    for pattern in args.manifests:
        if '*' in str(pattern):
            matches = list(pattern.parent.glob(pattern.name))
            manifests.extend(matches)
        else:
            if pattern.exists():
                manifests.append(pattern)
            else:
                print(f"Warning: Manifest not found: {pattern}")
    
    if not manifests:
        print("ERROR: No manifest files found")
        return 1
    
    # Lint each manifest
    all_reports = []
    
    for manifest_path in manifests:
        report = lint_manifest(
            manifest_path,
            check_bands=args.check_bands,
            check_files=not args.no_check_files,
            check_orphans=args.check_orphans,
            sample_size=args.sample,
            base_dir=args.base_dir,
            verbose=args.verbose
        )
        all_reports.append((manifest_path, report))
    
    # Print results
    print("\n" + "=" * 80)
    print("MANIFEST LINTING RESULTS")
    print("=" * 80)
    
    total_errors = 0
    total_warnings = 0
    
    for manifest_path, report in all_reports:
        print(f"\n{manifest_path.name}:")
        print(f"  Tiles: {report.total_tiles}")
        print(f"  Errors: {report.error_count}")
        print(f"  Warnings: {report.warning_count}")
        print(f"  Info: {report.info_count}")
        
        total_errors += report.error_count
        total_warnings += report.warning_count
        
        # Print issues
        for issue in report.issues:
            if args.errors_only and issue.severity != 'error':
                continue
            print(f"\n{issue}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Manifests checked: {len(all_reports)}")
    print(f"Total tiles: {sum(r.total_tiles for _, r in all_reports)}")
    print(f"Total errors: {total_errors}")
    print(f"Total warnings: {total_warnings}")
    
    if total_errors == 0:
        print("\n✓ All manifests are clean!")
        exit_code = 0
    else:
        print(f"\n✗ Found {total_errors} error(s) that must be fixed")
        exit_code = 1
    
    # Export to JSON if requested
    if args.json:
        export_data = []
        for manifest_path, report in all_reports:
            export_data.append({
                'manifest': str(manifest_path),
                'total_tiles': report.total_tiles,
                'error_count': report.error_count,
                'warning_count': report.warning_count,
                'info_count': report.info_count,
                'issues': [
                    {
                        'severity': i.severity,
                        'category': i.category,
                        'message': i.message,
                        'tile_path': i.tile_path,
                        'details': i.details
                    }
                    for i in report.issues
                ]
            })
        
        with open(args.json, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\n✓ Exported results to {args.json}")
    
    print("=" * 80)
    
    return exit_code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nLinting interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
