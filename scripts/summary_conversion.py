#!/usr/bin/env python3
"""Summary report for BigEarthNet v2 conversion."""

import json
from pathlib import Path

# Read logs
logs = [json.loads(line) for line in open('logs/benv2_ingest.jsonl')]

# Calculate statistics
total_processed = len(logs)
success_count = sum(1 for l in logs if l['ok'])
failed_count = sum(1 for l in logs if not l['ok'])
success_rate = (success_count / total_processed * 100) if total_processed > 0 else 0

# Split distribution
splits = {}
for l in logs:
    if l['ok']:
        splits[l['split']] = splits.get(l['split'], 0) + 1

# Size statistics
total_npz_mb = sum(l.get('bytes_npz', 0) for l in logs if l['ok']) / (1024**2)
total_deleted_mb = sum(l.get('deleted_s1_mb', 0) + l.get('deleted_s2_mb', 0) for l in logs if l['ok'])

# Output directory stats
npz_count = len(list(Path('data/tiles/benv2_catalog').glob('*.npz')))
json_count = len(list(Path('data/tiles/benv2_catalog').glob('*.json')))

print('='*70)
print('BigEarthNet v2 CONVERSION COMPLETE')
print('='*70)
print(f'Total patches attempted: {total_processed:,}')
print(f'Successfully converted:  {success_count:,} ({success_rate:.1f}%)')
print(f'Failed:                 {failed_count:,}')
print()
print('Split Distribution:')
train_pct = splits.get('train', 0)/success_count*100
val_pct = splits.get('val', 0)/success_count*100
test_pct = splits.get('test', 0)/success_count*100
print(f'  Training:   {splits.get("train", 0):>8,} patches ({train_pct:.1f}%)')
print(f'  Validation: {splits.get("val", 0):>8,} patches ({val_pct:.1f}%)')
print(f'  Test:       {splits.get("test", 0):>8,} patches ({test_pct:.1f}%)')
print()
print('Output Files:')
print(f'  NPZ files:  {npz_count:,}')
print(f'  JSON files: {json_count:,}')
print()
print('Storage:')
print(f'  Created:    {total_npz_mb/1024:.2f} GB (compressed tiles)')
print(f'  Freed:      {total_deleted_mb/1024:.2f} GB (deleted source data)')
print(f'  Net saving: {(total_deleted_mb - total_npz_mb)/1024:.2f} GB')
print()
print('Output location: data/tiles/benv2_catalog/')
print('Log file:        logs/benv2_ingest.jsonl')
print('='*70)
