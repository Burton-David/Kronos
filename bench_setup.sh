#!/usr/bin/env bash
# Driver invoked on the pod. Receives the GPU label as $1 (free-form, used
# only as a label inside the JSON output, e.g. "RTX 4090").
set -euo pipefail
cd "$(dirname "$0")"

GPU_LABEL="${1:-unknown}"

echo "=== installing requirements ==="
pip install -q -r requirements.txt
pip install -q yfinance

echo "=== verifying patch is present ==="
PYTHONPATH=. python -c "
from finetune.utils.training_utils import resolve_amp_dtype
import torch
assert resolve_amp_dtype('bfloat16') == (torch.bfloat16, True)
assert resolve_amp_dtype(None) == (torch.float32, False)
print('patch verified')
"

echo "=== running benchmark on ${GPU_LABEL} ==="
python benchmark_amp.py \
    --repo-root . \
    --gpu-label "${GPU_LABEL}" \
    --batches 10,25,50,100 \
    --seq-len 512 \
    --warmup 5 --iters 30 \
    --convergence-steps 200 \
    --convergence-batch 25 \
    --output ./bench_results.json

echo ""
echo "=== bench_results.json (truncated) ==="
python -c "
import json
r = json.load(open('./bench_results.json'))
print(json.dumps({
    'config': r['config'],
    'gpu': r['gpu'],
    'data': r['data'],
    'predictor_sweep_pairwise': {b: r['predictor_sweep'][b].get('pairwise') for b in r['predictor_sweep']},
    'tokenizer_sweep_pairwise': {b: r['tokenizer_sweep'][b].get('pairwise') for b in r['tokenizer_sweep']},
    'convergence_summary': {k: r['convergence'][k] for k in r['convergence'] if not k.endswith('_losses')},
}, indent=2))
"
