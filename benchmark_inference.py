"""
Wallclock benchmark for the bf16 inference patch (KronosPredictor.amp_dtype).

Compares the public API surface — KronosPredictor.predict — under FP32 and
bf16, on real SPY 1-min bars from yfinance.

CLI:
    python benchmark_inference.py --repo-root . --gpu-label "RTX 4090" \\
        --pred-lens 10,30,60,120 --sample-count 5 --warmup 2 --iters 6 \\
        --output bench_inference.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _import_kronos(repo_root: Path):
    sys.path.insert(0, str(repo_root))
    from model.kronos import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402
    return Kronos, KronosTokenizer, KronosPredictor


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


def _ensure_yfinance() -> None:
    try:
        import yfinance  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "yfinance"])


def _fetch_spy() -> "pandas.DataFrame":
    _ensure_yfinance()
    import pandas as pd
    import yfinance as yf
    df = yf.download("SPY", interval="1m", period="7d", progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError("yfinance returned no SPY 1-min data")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().astype("float64")
    df.columns = ["open", "high", "low", "close", "volume"]
    df["amount"] = df["close"] * df["volume"]
    return df


def _summarise(times: list[float]) -> dict:
    s = sorted(times)
    return {
        "iters": len(s),
        "median_s": statistics.median(s),
        "mean_s": statistics.fmean(s),
        "stdev_s": statistics.pstdev(s) if len(s) > 1 else 0.0,
        "min_s": s[0],
        "max_s": s[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--gpu-label", required=True)
    parser.add_argument("--tokenizer", default="NeoQuasar/Kronos-Tokenizer-base")
    parser.add_argument("--predictor", default="NeoQuasar/Kronos-base")
    parser.add_argument("--pred-lens", default="10,30,60,120")
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument("--max-context", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA required.")

    Kronos, KronosTokenizer, KronosPredictor = _import_kronos(args.repo_root)
    pred_lens = [int(p) for p in args.pred_lens.split(",") if p.strip()]
    device = "cuda:0"

    print("[data] fetching SPY 1-min ...", flush=True)
    df = _fetch_spy()
    print(f"[data] got {len(df)} bars  {df.index[0]} → {df.index[-1]}", flush=True)

    # Use a single 480-bar history window for all runs (Kronos's default
    # max_context is 512; leaving headroom).
    history_len = 480
    if len(df) < history_len + max(pred_lens) + 16:
        sys.exit(f"yfinance only returned {len(df)} bars; bench needs more.")

    base_idx = len(df) - max(pred_lens) - history_len - 1
    df_in = df.iloc[base_idx:base_idx + history_len].copy()
    full_y_ts = df.index[base_idx + history_len:base_idx + history_len + max(pred_lens)]

    import pandas as pd
    x_ts = pd.Series(df_in.index.tz_localize(None) if df_in.index.tz else df_in.index)

    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer)
    base_predictor = Kronos.from_pretrained(args.predictor)

    results: dict[str, Any] = {
        "config": {
            "history_len": history_len,
            "pred_lens": pred_lens,
            "sample_count": args.sample_count,
            "max_context": args.max_context,
            "warmup": args.warmup,
            "iters": args.iters,
            "seed": args.seed,
        },
        "gpu": _gpu_info(),
        "data": {
            "n_bars": int(len(df)),
            "first_ts": str(df.index[0]),
            "last_ts": str(df.index[-1]),
            "history_window_first": str(df_in.index[0]),
            "history_window_last": str(df_in.index[-1]),
        },
        "sweep": {},
    }

    for pred_len in pred_lens:
        results["sweep"][pred_len] = {}
        y_ts = pd.Series(full_y_ts[:pred_len].tz_localize(None) if full_y_ts.tz else full_y_ts[:pred_len])
        for label, amp_dtype in (("fp32", None), ("bf16", "bfloat16")):
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            predictor = KronosPredictor(
                base_predictor, tokenizer,
                device=device, max_context=args.max_context, amp_dtype=amp_dtype,
            )
            try:
                # Warmup
                for _ in range(args.warmup):
                    predictor.predict(df_in, x_ts, y_ts, pred_len=pred_len,
                                      T=1.0, top_k=0, top_p=0.9,
                                      sample_count=args.sample_count, verbose=False)
                # Timed
                torch.cuda.reset_peak_memory_stats()
                times: list[float] = []
                last_pred_close = float("nan")
                for _ in range(args.iters):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    out = predictor.predict(df_in, x_ts, y_ts, pred_len=pred_len,
                                            T=1.0, top_k=0, top_p=0.9,
                                            sample_count=args.sample_count, verbose=False)
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - t0)
                    last_pred_close = float(out["close"].iloc[-1])
                peak = int(torch.cuda.max_memory_allocated())
                summ = _summarise(times)
                results["sweep"][pred_len][label] = {
                    **summ,
                    "peak_memory_gb": round(peak / (1024 ** 3), 3),
                    "last_pred_close": last_pred_close,
                    "ok": True,
                }
                print(
                    f"  pred_len={pred_len:>3} {label}: median {summ['median_s']*1000:.1f} ms  "
                    f"peak {peak/(1024**3):.2f} GB  close[-1]={last_pred_close:.4f}",
                    flush=True,
                )
            except torch.cuda.OutOfMemoryError as e:
                results["sweep"][pred_len][label] = {"ok": False, "error": "OOM", "msg": str(e)[:200]}
                print(f"  pred_len={pred_len:>3} {label}: OOM", flush=True)
            except Exception as e:
                results["sweep"][pred_len][label] = {"ok": False, "error": type(e).__name__, "msg": str(e)[:300]}
                print(f"  pred_len={pred_len:>3} {label}: {type(e).__name__}: {str(e)[:300]}", flush=True)
            finally:
                del predictor
                torch.cuda.empty_cache()
        # Pairwise summary
        s = results["sweep"][pred_len]
        if s.get("fp32", {}).get("ok") and s.get("bf16", {}).get("ok"):
            fp = s["fp32"]; bf = s["bf16"]
            s["pairwise"] = {
                "speedup_x": round(fp["median_s"] / bf["median_s"], 3),
                "memory_ratio": round(bf["peak_memory_gb"] / max(fp["peak_memory_gb"], 1e-6), 3),
                "close_rel_diff": round(abs(bf["last_pred_close"] - fp["last_pred_close"]) / max(abs(fp["last_pred_close"]), 1e-6), 4),
            }

    args.output.write_text(json.dumps(results, indent=2))
    print(f"\n[wrote] {args.output}", flush=True)


if __name__ == "__main__":
    main()
