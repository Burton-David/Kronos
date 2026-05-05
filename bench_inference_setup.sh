#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
GPU_LABEL="${1:-unknown}"
echo "=== install ==="
pip install -q -r requirements.txt
pip install -q yfinance
echo "=== verify amp_dtype kwarg present ==="
python -c "
import inspect, sys
sys.path.insert(0, '.')
from model.kronos import KronosPredictor
sig = inspect.signature(KronosPredictor.__init__)
assert 'amp_dtype' in sig.parameters, sig
print('amp_dtype kwarg present')
"
echo "=== bench ==="
python benchmark_inference.py \
    --repo-root . --gpu-label "${GPU_LABEL}" \
    --pred-lens 10,30,60,120 --sample-count 5 \
    --max-context 512 --warmup 2 --iters 6 \
    --output ./bench_inference.json
echo ""
echo "=== summary ==="
python -c "
import json
r = json.load(open('./bench_inference.json'))
print(json.dumps({'config': r['config'], 'gpu': r['gpu'], 'data': r['data'],
    'sweep_pairwise': {k: r['sweep'][k].get('pairwise') for k in r['sweep']}}, indent=2))
"
