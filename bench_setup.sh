#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
echo "=== installing requirements ==="
pip install -q -r requirements.txt
echo "=== verifying patch is present ==="
PYTHONPATH=. python -c "
from finetune.utils.training_utils import resolve_amp_dtype
import torch
assert resolve_amp_dtype('bfloat16') == (torch.bfloat16, True)
print('patch verified')
"
echo "=== running benchmark ==="
python benchmark_amp.py --repo-root . --output ./bench_results.json
echo ""
echo "=== bench_results.json ==="
cat ./bench_results.json
