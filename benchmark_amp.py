"""
Wallclock benchmark for the bf16 AMP patch on the published Kronos-base +
Kronos-Tokenizer-base weights.

Mirrors the body of `finetune/train_predictor.py`'s training step (tokenize,
forward, loss, backward, clip-grad, optimizer.step, zero_grad) on a synthetic
OHLCV-shaped batch. Times FP32 and bf16 paths in the same process, on the
same random seed, with CUDA syncs around every measurement.

Usage:
    python scripts/benchmark_amp.py \\
        --tokenizer NeoQuasar/Kronos-Tokenizer-base \\
        --predictor NeoQuasar/Kronos-base \\
        --batch-size 50 --seq-len 512 \\
        --warmup 10 --iters 50 \\
        --output bench_results.json

Designed to run on a single CUDA GPU. Crashes loudly if CUDA isn't available
or autocast misbehaves; we never want to silently fall back to a CPU run.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def _import_kronos(repo_root: Path):
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "finetune"))
    from model.kronos import Kronos, KronosTokenizer  # noqa: E402
    from utils.training_utils import resolve_amp_dtype  # noqa: E402
    return Kronos, KronosTokenizer, resolve_amp_dtype


def _gpu_info() -> dict:
    if not torch.cuda.is_available():
        return {"cuda": False}
    p = torch.cuda.get_device_properties(0)
    return {
        "cuda": True,
        "name": p.name,
        "total_memory_gb": round(p.total_memory / (1024 ** 3), 2),
        "capability": f"{p.major}.{p.minor}",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def _build_inputs(batch_size: int, seq_len: int, device: torch.device, seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)
    # Synthetic OHLCV-shaped input: 6 feature columns matching Kronos's d_in.
    # Values are ~N(0, 1); the tokenizer's z-score normalisation expects this.
    batch_x = torch.randn(batch_size, seq_len, 6, generator=g).to(device, non_blocking=True)
    # Time-feature stamps: integer codes (minute/hour/weekday/day/month).
    batch_x_stamp = torch.randint(0, 12, (batch_size, seq_len, 5), generator=g).to(device, non_blocking=True)
    return batch_x, batch_x_stamp


def _time_steps(
    *,
    n_iters: int,
    tokenizer,
    predictor,
    optimizer,
    batch_x,
    batch_x_stamp,
    amp_dtype,
    amp_enabled,
) -> tuple[list[float], float, int]:
    """Run `n_iters` training steps and return (per-step times, last-loss, peak-bytes)."""
    times: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    last_loss = float("nan")
    for _ in range(n_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            with torch.no_grad():
                s0, s1 = tokenizer.encode(batch_x, half=True)
            token_in = [s0[:, :-1], s1[:, :-1]]
            token_out = [s0[:, 1:], s1[:, 1:]]
            logits = predictor(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
            loss, _, _ = predictor.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=3.0)
        optimizer.step()

        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        last_loss = float(loss.detach().item())

    peak_bytes = int(torch.cuda.max_memory_allocated())
    return times, last_loss, peak_bytes


def _summarise(times: list[float]) -> dict:
    sorted_t = sorted(times)
    n = len(sorted_t)
    return {
        "iters": n,
        "median_s": statistics.median(sorted_t),
        "mean_s": statistics.fmean(sorted_t),
        "stdev_s": statistics.pstdev(sorted_t) if n > 1 else 0.0,
        "min_s": sorted_t[0],
        "p90_s": sorted_t[int(n * 0.9)] if n >= 10 else sorted_t[-1],
        "max_s": sorted_t[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1] / "external" / "Kronos")
    parser.add_argument("--tokenizer", type=str, default="NeoQuasar/Kronos-Tokenizer-base")
    parser.add_argument("--predictor", type=str, default="NeoQuasar/Kronos-base")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--output", type=Path, default=Path("bench_results.json"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU.")

    Kronos, KronosTokenizer, resolve_amp_dtype = _import_kronos(args.repo_root)
    device = torch.device("cuda")

    print(f"Loading {args.tokenizer} ...")
    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer).to(device)
    tokenizer.eval()

    print(f"Loading {args.predictor} ...")
    predictor = Kronos.from_pretrained(args.predictor).to(device)
    predictor.train()

    n_params = sum(p.numel() for p in predictor.parameters())
    print(f"Predictor parameters: {n_params / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(predictor.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    torch.manual_seed(args.seed)
    batch_x, batch_x_stamp = _build_inputs(args.batch_size, args.seq_len, device, args.seed)

    results = {
        "config": {
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "warmup": args.warmup,
            "iters": args.iters,
            "seed": args.seed,
            "predictor_params_M": round(n_params / 1e6, 2),
            "tokenizer_repo": args.tokenizer,
            "predictor_repo": args.predictor,
        },
        "gpu": _gpu_info(),
    }

    for label, amp_dtype_str in [("fp32", None), ("bf16", "bfloat16")]:
        amp_dtype, amp_enabled = resolve_amp_dtype(amp_dtype_str)

        # Reset predictor weights to a known state before each phase so the FP32
        # and bf16 runs see the same starting point and equivalent losses.
        torch.manual_seed(args.seed)
        # Re-load predictor weights to wash out any state from the previous phase.
        predictor = Kronos.from_pretrained(args.predictor).to(device)
        predictor.train()
        optimizer = torch.optim.AdamW(predictor.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

        print(f"\n[{label}] warmup ({args.warmup} iters)...")
        _time_steps(
            n_iters=args.warmup,
            tokenizer=tokenizer,
            predictor=predictor,
            optimizer=optimizer,
            batch_x=batch_x,
            batch_x_stamp=batch_x_stamp,
            amp_dtype=amp_dtype,
            amp_enabled=amp_enabled,
        )

        print(f"[{label}] timed ({args.iters} iters)...")
        times, last_loss, peak_bytes = _time_steps(
            n_iters=args.iters,
            tokenizer=tokenizer,
            predictor=predictor,
            optimizer=optimizer,
            batch_x=batch_x,
            batch_x_stamp=batch_x_stamp,
            amp_dtype=amp_dtype,
            amp_enabled=amp_enabled,
        )

        summary = _summarise(times)
        steps_per_sec = 1.0 / summary["median_s"]
        results[label] = {
            **summary,
            "steps_per_sec_median": steps_per_sec,
            "tokens_per_sec_median": steps_per_sec * args.batch_size * args.seq_len,
            "peak_memory_gb": round(peak_bytes / (1024 ** 3), 3),
            "last_loss": last_loss,
        }
        print(
            f"[{label}] median {summary['median_s'] * 1000:.1f} ms/step  "
            f"({steps_per_sec:.2f} steps/sec)  "
            f"peak {results[label]['peak_memory_gb']:.2f} GB  "
            f"last_loss={last_loss:.4f}"
        )

    fp32 = results["fp32"]
    bf16 = results["bf16"]
    results["summary"] = {
        "speedup_x": round(fp32["median_s"] / bf16["median_s"], 3),
        "memory_ratio": round(bf16["peak_memory_gb"] / fp32["peak_memory_gb"], 3),
        "loss_rel_diff": round(abs(bf16["last_loss"] - fp32["last_loss"]) / max(abs(fp32["last_loss"]), 1e-6), 4),
    }

    print("\n=== Summary ===")
    print(f"speedup (fp32_med / bf16_med):     {results['summary']['speedup_x']:.3f}x")
    print(f"peak memory ratio (bf16 / fp32):   {results['summary']['memory_ratio']:.3f}")
    print(f"last-loss relative diff:           {results['summary']['loss_rel_diff']:.4f}")

    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
