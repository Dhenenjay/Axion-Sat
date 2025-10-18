"""
Production Stage 1 Pangaea Benchmark

Replicates TerraMind paper results using our optimized Stage 1 pipeline:
- Direct TerraMind generator (no TiM, no tokenizers)
- Gamma correction 0.7
- Simple, fast, accurate

Target Datasets (from radar chart):
- Sen1Floods11: 90.9%
- BurnScars: 83.6%
- MADOS: 75.6%
- FiveBillionPixels: 69.2%
- SpaceNet7: 63.0%
- CTM-SS: 55.8%
- PASTIS: 43.1%
- DynamicEarthNet: 39.3%
- AI4Farms: 28.1%
"""

import os
import sys
from pathlib import Path
import argparse
from datetime import datetime

# Setup paths
project_root = Path(__file__).parent
pangaea_dir = project_root / "pangaea-bench"
os.chdir(pangaea_dir)
sys.path.insert(0, str(pangaea_dir))
sys.path.insert(0, str(project_root))

print("="*80)
print("🚀 PRODUCTION STAGE 1 PANGAEA BENCHMARK")
print("="*80)
print(f"Using: Simple TerraMind Generator + γ=0.7")
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
print()

# Dataset configurations matching paper
BENCHMARK_DATASETS = {
    'sen1floods11': {
        'encoder': 'terramind_optical',
        'decoder': 'seg_upernet',
        'task': 'segmentation',
        'target_score': 90.9,
        'priority': 1
    },
    'hlsburnscars': {
        'encoder': 'terramind_optical',
        'decoder': 'seg_upernet',
        'task': 'segmentation',
        'target_score': 83.6,
        'priority': 2
    },
    'mados': {
        'encoder': 'terramind_optical',
        'decoder': 'seg_upernet',
        'task': 'segmentation',
        'target_score': 75.6,
        'priority': 3
    },
    'fivebillionpixels': {
        'encoder': 'terramind_optical',
        'decoder': 'seg_upernet',
        'task': 'segmentation',
        'target_score': 69.2,
        'priority': 4
    },
    'spacenet7': {
        'encoder': 'terramind_optical',
        'decoder': 'seg_upernet',
        'task': 'segmentation',
        'target_score': 63.0,
        'priority': 5
    },
    'croptypemapping': {  # CTM-SS
        'encoder': 'terramind_optical',
        'decoder': 'seg_upernet',
        'task': 'segmentation',
        'target_score': 55.8,
        'priority': 6
    },
    'pastis': {
        'encoder': 'terramind_optical',
        'decoder': 'seg_upernet',
        'task': 'segmentation',
        'target_score': 43.1,
        'priority': 7
    },
    'dynamicen': {
        'encoder': 'terramind_optical',
        'decoder': 'seg_upernet',
        'task': 'segmentation',
        'target_score': 39.3,
        'priority': 8
    },
    'ai4smallfarms': {
        'encoder': 'terramind_optical',
        'decoder': 'seg_upernet',
        'task': 'segmentation',
        'target_score': 28.1,
        'priority': 9
    }
}


def check_dataset_available(dataset_name: str) -> bool:
    """Check if dataset is downloaded and ready."""
    try:
        # Check if config exists
        dataset_config = pangaea_dir / f"configs/dataset/{dataset_name}.yaml"
        if not dataset_config.exists():
            return False
        
        # Read config to check data path
        import yaml
        with open(dataset_config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check if data directory specified exists
        if 'data_root' in config:
            data_root = Path(config['data_root'])
            return data_root.exists()
        
        # Config exists but can't verify data
        return True
    except Exception as e:
        return False


def run_single_benchmark(dataset_name: str, config: dict, args) -> dict:
    """Run benchmark on a single dataset."""
    print(f"\n{'='*80}")
    print(f"Benchmarking: {dataset_name.upper()}")
    print(f"Target Score: {config['target_score']}%")
    print(f"{'='*80}\n")
    
    # Build command (Hydra config groups syntax with torchrun for distributed)
    cmd_parts = [
        "torchrun",
        "--nnodes=1",  # Single node
        "--nproc_per_node=1",  # Single GPU
        "pangaea/run.py",
        "--config-name=train",
        f"encoder={config['encoder']}",
        f"decoder={config['decoder']}",
        f"dataset={dataset_name}",
        f"task={config['task']}",
        "criterion=cross_entropy",
        "preprocessing=seg_default",
        "lr_scheduler=multi_step_lr",
        "use_wandb=False",
        "finetune=False",
        "train=True",
        f"work_dir={project_root / 'results' / 'pangaea_bench'}"
    ]
    
    # Add optional flags
    if args.batch_size:
        cmd_parts.append(f"--batch_size {args.batch_size}")
    
    if args.epochs:
        cmd_parts.append(f"--epochs {args.epochs}")
    
    if args.gpu:
        cmd_parts.append(f"--gpu {args.gpu}")
    
    # Run benchmark
    cmd = " ".join(cmd_parts)
    print(f"Command: {cmd}\n")
    
    if not args.dry_run:
        import subprocess
        result = subprocess.run(cmd, shell=True, capture_output=False)
        
        if result.returncode != 0:
            print(f"\n❌ Failed: {dataset_name}")
            return {
                'dataset': dataset_name,
                'success': False,
                'score': None,
                'target': config['target_score']
            }
        
        # Parse results (would need to extract from log files)
        # For now, just return success
        return {
            'dataset': dataset_name,
            'success': True,
            'score': None,  # Extract from results
            'target': config['target_score']
        }
    else:
        print("(Dry run - not executing)\n")
        return {
            'dataset': dataset_name,
            'success': None,
            'score': None,
            'target': config['target_score']
        }


def main():
    parser = argparse.ArgumentParser(
        description="Production Stage 1 Pangaea Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset selection
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=list(BENCHMARK_DATASETS.keys()) + ['all'],
        default=['all'],
        help='Datasets to benchmark'
    )
    
    parser.add_argument(
        '--priority',
        type=int,
        help='Run datasets up to this priority level (1=highest)'
    )
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--gpu', type=str, help='GPU device')
    
    # Execution control
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print commands without executing'
    )
    
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check dataset availability'
    )
    
    args = parser.parse_args()
    
    # Determine which datasets to run
    if 'all' in args.datasets:
        datasets_to_run = list(BENCHMARK_DATASETS.keys())
    else:
        datasets_to_run = args.datasets
    
    # Filter by priority if specified
    if args.priority:
        datasets_to_run = [
            d for d in datasets_to_run
            if BENCHMARK_DATASETS[d]['priority'] <= args.priority
        ]
    
    # Sort by priority
    datasets_to_run.sort(key=lambda d: BENCHMARK_DATASETS[d]['priority'])
    
    print(f"\n📋 Selected Datasets ({len(datasets_to_run)}):")
    for dataset in datasets_to_run:
        config = BENCHMARK_DATASETS[dataset]
        print(f"  {config['priority']}. {dataset:<20} → Target: {config['target_score']:.1f}%")
    print()
    
    # Check dataset availability
    if args.check_only:
        print("\n🔍 Checking dataset availability...\n")
        for dataset in datasets_to_run:
            available = check_dataset_available(dataset)
            status = "✓ Available" if available else "✗ Not found"
            print(f"  {dataset:<20} {status}")
        return
    
    # Run benchmarks
    results = []
    for dataset in datasets_to_run:
        config = BENCHMARK_DATASETS[dataset]
        result = run_single_benchmark(dataset, config, args)
        results.append(result)
    
    # Summary
    print("\n" + "="*80)
    print("📊 BENCHMARK SUMMARY")
    print("="*80)
    print(f"{'Dataset':<20} {'Target':<10} {'Achieved':<10} {'Status'}")
    print("-"*80)
    
    for result in results:
        dataset = result['dataset']
        target = result['target']
        score = result['score'] if result['score'] else 'N/A'
        
        if result['success'] is None:
            status = 'DRY RUN'
        elif result['success']:
            status = '✓ PASS' if (result['score'] and result['score'] >= target) else '? CHECK'
        else:
            status = '✗ FAIL'
        
        print(f"{dataset:<20} {target:.1f}%{'':<6} {score}{'':<6} {status}")
    
    print("="*80)
    
    # Save results
    if not args.dry_run:
        results_file = project_root / f"results/production_stage1_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        results_file.parent.mkdir(exist_ok=True)
        
        import json
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'config': 'Production Stage 1 (TerraMind + γ=0.7)',
                'results': results
            }, f, indent=2)
        
        print(f"\n✓ Results saved to: {results_file}")


if __name__ == "__main__":
    main()
