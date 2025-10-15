"""
Visualize Stage 2 Benchmark Results

Creates plots from benchmark CSV data showing:
- Memory usage vs tile size
- Throughput vs tile size
- Throughput vs batch size
- Time breakdown (forward/backward for training mode)

Usage:
    python scripts/visualize_benchmarks.py --input logs/stage2_bench.csv
    python scripts/visualize_benchmarks.py --input logs/stage2_bench.csv --output reports/bench_plots.png
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_benchmark_data(csv_path: Path) -> pd.DataFrame:
    """Load benchmark CSV data."""
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} benchmark results from {csv_path}")
    return df


def plot_benchmarks(df: pd.DataFrame, output_path: Path = None):
    """Create comprehensive benchmark visualizations."""
    
    # Filter successful results
    if 'error' in df.columns:
        df = df[df['error'].isna()]
    # else: keep all rows
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Stage 2 Benchmark Results', fontsize=16, fontweight='bold', y=0.995)
    
    # Plot 1: Memory usage vs tile size
    ax = axes[0, 0]
    for batch_size in sorted(df['batch_size'].unique()):
        data = df[df['batch_size'] == batch_size]
        ax.plot(data['tile_size'], data['memory_peak_mb'], 
                marker='o', linewidth=2, markersize=8, label=f'Batch={batch_size}')
    
    ax.set_xlabel('Tile Size', fontsize=11)
    ax.set_ylabel('Peak Memory (MB)', fontsize=11)
    ax.set_title('GPU Memory Usage', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Throughput vs tile size
    ax = axes[0, 1]
    for batch_size in sorted(df['batch_size'].unique()):
        data = df[df['batch_size'] == batch_size]
        ax.plot(data['tile_size'], data['throughput_samples_per_sec'],
                marker='s', linewidth=2, markersize=8, label=f'Batch={batch_size}')
    
    ax.set_xlabel('Tile Size', fontsize=11)
    ax.set_ylabel('Throughput (samples/sec)', fontsize=11)
    ax.set_title('Inference Throughput', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Average time vs configuration
    ax = axes[1, 0]
    time_col = 'total_time_ms' if 'total_time_ms' in df.columns else 'avg_time_ms'
    
    x_labels = []
    times = []
    colors = []
    
    for _, row in df.iterrows():
        label = f"{row['tile_size']}\nB{row['batch_size']}"
        x_labels.append(label)
        times.append(row[time_col])
        # Color by tile size
        colors.append('#2ecc71' if row['tile_size'] == 256 else '#3498db')
    
    ax.bar(range(len(times)), times, color=colors, alpha=0.7)
    ax.set_xlabel('Configuration (Tile Size / Batch)', fontsize=11)
    ax.set_ylabel('Time (ms)', fontsize=11)
    ax.set_title('Processing Time per Configuration', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Memory vs Throughput tradeoff
    ax = axes[1, 1]
    for tile_size in sorted(df['tile_size'].unique()):
        data = df[df['tile_size'] == tile_size]
        ax.scatter(data['memory_peak_mb'], data['throughput_samples_per_sec'],
                  s=data['batch_size']*100, alpha=0.6, label=f'Tile={tile_size}')
    
    ax.set_xlabel('Peak Memory (MB)', fontsize=11)
    ax.set_ylabel('Throughput (samples/sec)', fontsize=11)
    ax.set_title('Memory vs Throughput Tradeoff', fontsize=12, fontweight='bold')
    ax.legend(title='Tile Size\n(size=batch)', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"✓ Saved plot to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def print_comparison_table(df: pd.DataFrame):
    """Print comparison table for tile sizes."""
    print("\n" + "=" * 80)
    print("Tile Size Comparison (256 vs 384)")
    print("=" * 80)
    
    for batch_size in sorted(df['batch_size'].unique()):
        print(f"\nBatch Size: {batch_size}")
        print("-" * 80)
        
        tile_256 = df[(df['tile_size'] == 256) & (df['batch_size'] == batch_size)]
        tile_384 = df[(df['tile_size'] == 384) & (df['batch_size'] == batch_size)]
        
        if len(tile_256) == 0 or len(tile_384) == 0:
            print("  Missing data for comparison")
            continue
        
        # Get values
        time_256 = tile_256.iloc[0].get('total_time_ms', tile_256.iloc[0].get('avg_time_ms', 0))
        time_384 = tile_384.iloc[0].get('total_time_ms', tile_384.iloc[0].get('avg_time_ms', 0))
        
        mem_256 = tile_256.iloc[0]['memory_peak_mb']
        mem_384 = tile_384.iloc[0]['memory_peak_mb']
        
        thr_256 = tile_256.iloc[0]['throughput_samples_per_sec']
        thr_384 = tile_384.iloc[0]['throughput_samples_per_sec']
        
        # Calculate differences
        time_ratio = time_384 / time_256 if time_256 > 0 else 0
        mem_ratio = mem_384 / mem_256 if mem_256 > 0 else 0
        thr_ratio = thr_384 / thr_256 if thr_256 > 0 else 0
        
        print(f"  Time:       256={time_256:.1f}ms, 384={time_384:.1f}ms (ratio: {time_ratio:.2f}x)")
        print(f"  Memory:     256={mem_256:.0f}MB, 384={mem_384:.0f}MB (ratio: {mem_ratio:.2f}x)")
        print(f"  Throughput: 256={thr_256:.1f}/s, 384={thr_384:.1f}/s (ratio: {thr_ratio:.2f}x)")
        
        # Pixels processed
        pixels_256 = 256 * 256
        pixels_384 = 384 * 384
        pixels_ratio = pixels_384 / pixels_256
        
        print(f"\n  Pixels:     256={pixels_256:,}, 384={pixels_384:,} (ratio: {pixels_ratio:.2f}x)")
        print(f"  Efficiency: 256={thr_256*pixels_256:.0f} px/s, 384={thr_384*pixels_384:.0f} px/s")


def main():
    parser = argparse.ArgumentParser(description='Visualize benchmark results')
    parser.add_argument('--input', type=Path, default=Path('logs/stage2_bench.csv'),
                       help='Input CSV file')
    parser.add_argument('--output', type=Path, default=None,
                       help='Output plot file (default: show plot)')
    
    args = parser.parse_args()
    
    # Load data
    df = load_benchmark_data(args.input)
    
    # Print comparison
    print_comparison_table(df)
    
    # Create plots
    plot_benchmarks(df, args.output)


if __name__ == '__main__':
    main()
