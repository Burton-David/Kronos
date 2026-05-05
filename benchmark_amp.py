"""
Wallclock + correctness benchmark for the bf16 AMP patch.

Covers everything the patch touches and reports verbatim numbers — never
synthesises a result it didn't measure.

Phases:
  1. Data: pull a small SPY 1-min OHLCV sample via yfinance, z-score normalise
     per window, slice into seq_len windows. Requires internet on the pod.
  2. Predictor sweep: for each batch size, time the body of
     `finetune/train_predictor.py`'s step (tokenize -> forward -> loss ->
     backward -> clip-grad -> optimizer.step). FP32 then bf16. OOMs are
     recorded, never silently skipped.
  3. Tokenizer sweep: for each batch size, time the body of
     `finetune/train_tokenizer.py`'s step (forward -> recon+bsq loss ->
     backward -> clip -> optimizer.step). FP32 then bf16.
  4. Convergence trace: at the chosen batch size, run N successive predictor
     training steps under FP32 and bf16 starting from identical pretrained
     weights and identical batches. Record per-step losses. The PR uses these
     to claim that bf16 doesn't drift the loss curve.

CLI:
    python benchmark_amp.py \\
        --repo-root . \\
        --gpu-label "RTX 4090" \\
        --batches 10,25,50,100 \\
        --seq-len 512 \\
        --warmup 5 --iters 30 \\
        --convergence-steps 200 \\
        --convergence-batch 25 \\
        --output bench_results.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Imports from upstream Kronos (resolved at runtime once --repo-root is known).
# ---------------------------------------------------------------------------
def _import_kronos(repo_root: Path):
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "finetune"))
    from model.kronos import Kronos, KronosTokenizer  # noqa: E402
    from utils.training_utils import resolve_amp_dtype  # noqa: E402
    return Kronos, KronosTokenizer, resolve_amp_dtype


# ---------------------------------------------------------------------------
# Hardware metadata.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Real OHLCV: pull SPY 1-min bars via yfinance, z-score per window.
# ---------------------------------------------------------------------------
def _ensure_yfinance() -> None:
    try:
        import yfinance  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "yfinance"])


def _fetch_spy_1min(window_size: int, n_windows: int) -> tuple["pandas.DataFrame", torch.Tensor]:
    """
    Returns (raw_df, windowed_tensor).
    `windowed_tensor`: shape (n_windows, window_size, 6), dtype float32. Each
    window is z-score normalised independently (matching Kronos's per-window
    z-score convention from its dataset.py). Columns ordered [open, high, low,
    close, volume, amount] where amount = close * volume.
    """
    _ensure_yfinance()
    import pandas as pd
    import yfinance as yf

    n_bars_needed = window_size + n_windows + 100
    minutes_per_session = 390
    sessions_needed = max(2, math.ceil(n_bars_needed / minutes_per_session) + 2)
    period = f"{sessions_needed}d" if sessions_needed <= 7 else "7d"
    df = yf.download("SPY", interval="1m", period=period, progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError("yfinance returned no SPY 1-min data")
    # yfinance returns a multi-level column index; flatten.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().astype("float64")
    df["Amount"] = df["Close"] * df["Volume"]
    if len(df) < window_size + n_windows:
        raise RuntimeError(
            f"yfinance returned only {len(df)} bars; need {window_size + n_windows}"
        )

    arr = df.to_numpy()  # (T, 6)
    windows = []
    for i in range(n_windows):
        w = arr[i:i + window_size]
        mu = w.mean(axis=0, keepdims=True)
        sd = w.std(axis=0, keepdims=True)
        sd = sd + 1e-6
        w = (w - mu) / sd
        # Clip to mirror Kronos's `clip` config (default 5.0).
        w = w.clip(-5.0, 5.0)
        windows.append(w)
    arr_windowed = torch.tensor(windows, dtype=torch.float32)
    return df, arr_windowed


# ---------------------------------------------------------------------------
# Time-feature stamps. yfinance index gives us real datetimes; respect each
# embedding's vocab range exactly.
# ---------------------------------------------------------------------------
def _stamps_for_index(index, window_size: int, n_windows: int) -> torch.Tensor:
    import pandas as pd
    idx = pd.DatetimeIndex(index)
    # Per-feature vocabs in Kronos: minute=60, hour=24, weekday=7, day=32, month=13.
    minute = idx.minute.to_numpy()
    hour = idx.hour.to_numpy()
    weekday = idx.weekday.to_numpy()
    day = (idx.day - 1).to_numpy()  # Kronos's day vocab is 32; using 0..30
    month = (idx.month - 1).to_numpy()  # vocab 13; 0..11
    full = torch.tensor([minute, hour, weekday, day, month], dtype=torch.int64).T  # (T, 5)
    out = torch.empty(n_windows, window_size, 5, dtype=torch.int64)
    for i in range(n_windows):
        out[i] = full[i:i + window_size]
    return out


# ---------------------------------------------------------------------------
# Step bodies (faithful to the patched train scripts, minus DDP wrap).
# ---------------------------------------------------------------------------
def _predictor_step(tokenizer, predictor, optimizer, batch_x, batch_x_stamp, amp_dtype, amp_enabled):
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
        with torch.no_grad():
            s0, s1 = tokenizer.encode(batch_x, half=True)
        token_in = [s0[:, :-1], s1[:, :-1]]
        token_out = [s0[:, 1:], s1[:, 1:]]
        logits = predictor(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
        loss, _, _ = predictor.head.compute_loss(
            logits[0], logits[1], token_out[0], token_out[1]
        )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=3.0)
    optimizer.step()
    return float(loss.detach().item())


def _tokenizer_step(tokenizer_train, optimizer, batch_x, amp_dtype, amp_enabled):
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
        zs, bsq_loss, _, _ = tokenizer_train(batch_x)
        z_pre, z = zs
        recon_loss_pre = F.mse_loss(z_pre, batch_x)
        recon_loss_all = F.mse_loss(z, batch_x)
        loss = (recon_loss_pre + recon_loss_all + bsq_loss) / 2
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(tokenizer_train.parameters(), max_norm=2.0)
    optimizer.step()
    return float(loss.detach().item())


def _summarise(times: list[float]) -> dict:
    if not times:
        return {}
    s = sorted(times)
    return {
        "iters": len(s),
        "median_s": statistics.median(s),
        "mean_s": statistics.fmean(s),
        "stdev_s": statistics.pstdev(s) if len(s) > 1 else 0.0,
        "min_s": s[0],
        "p90_s": s[int(len(s) * 0.9)] if len(s) >= 10 else s[-1],
        "max_s": s[-1],
    }


# ---------------------------------------------------------------------------
# Sweep helpers.
# ---------------------------------------------------------------------------
def _sample_batch(arr_windowed: torch.Tensor, stamps_windowed: torch.Tensor, batch_size: int, device, seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = arr_windowed.shape[0]
    if n < batch_size:
        raise RuntimeError(f"only {n} windows available; need {batch_size}")
    idx = torch.randperm(n, generator=g)[:batch_size]
    return arr_windowed[idx].to(device, non_blocking=True), stamps_windowed[idx].to(device, non_blocking=True)


def _sweep_predictor(
    Kronos, KronosTokenizer, resolve_amp_dtype,
    tokenizer_repo: str, predictor_repo: str,
    arr_windowed, stamps_windowed,
    batches: list[int], warmup: int, iters: int, lr: float, seed: int,
) -> dict:
    device = torch.device("cuda")
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_repo).to(device).eval()
    out: dict[int, dict] = {}
    for B in batches:
        out[B] = {}
        for label, amp_dtype_str in (("fp32", None), ("bf16", "bfloat16")):
            amp_dtype, amp_enabled = resolve_amp_dtype(amp_dtype_str)
            torch.manual_seed(seed)
            predictor = Kronos.from_pretrained(predictor_repo).to(device).train()
            optimizer = torch.optim.AdamW(
                predictor.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1
            )
            try:
                batch_x, batch_x_stamp = _sample_batch(arr_windowed, stamps_windowed, B, device, seed)
                # Warmup
                for _ in range(warmup):
                    _predictor_step(tokenizer, predictor, optimizer, batch_x, batch_x_stamp, amp_dtype, amp_enabled)
                # Timed
                torch.cuda.reset_peak_memory_stats()
                times: list[float] = []
                last_loss = float("nan")
                for _ in range(iters):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    last_loss = _predictor_step(tokenizer, predictor, optimizer, batch_x, batch_x_stamp, amp_dtype, amp_enabled)
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - t0)
                peak = int(torch.cuda.max_memory_allocated())
                summ = _summarise(times)
                out[B][label] = {
                    **summ,
                    "steps_per_sec_median": 1.0 / summ["median_s"],
                    "peak_memory_gb": round(peak / (1024 ** 3), 3),
                    "last_loss": last_loss,
                    "ok": True,
                }
                print(f"  predictor B={B} {label}: {summ['median_s']*1000:.1f} ms/step  peak {peak/(1024**3):.2f} GB  last_loss={last_loss:.4f}", flush=True)
            except torch.cuda.OutOfMemoryError as e:
                out[B][label] = {"ok": False, "error": "OOM", "msg": str(e)[:200]}
                print(f"  predictor B={B} {label}: OOM", flush=True)
            except Exception as e:
                out[B][label] = {"ok": False, "error": type(e).__name__, "msg": str(e)[:200]}
                print(f"  predictor B={B} {label}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            finally:
                del predictor, optimizer
                torch.cuda.empty_cache()
        # Pairwise summary at this batch size.
        if out[B].get("fp32", {}).get("ok") and out[B].get("bf16", {}).get("ok"):
            fp = out[B]["fp32"]; bf = out[B]["bf16"]
            out[B]["pairwise"] = {
                "speedup_x": round(fp["median_s"] / bf["median_s"], 3),
                "memory_ratio": round(bf["peak_memory_gb"] / fp["peak_memory_gb"], 3),
                "loss_rel_diff": round(abs(bf["last_loss"] - fp["last_loss"]) / max(abs(fp["last_loss"]), 1e-6), 4),
            }
    return out


def _sweep_tokenizer(
    KronosTokenizer, resolve_amp_dtype,
    tokenizer_repo: str,
    arr_windowed,
    batches: list[int], warmup: int, iters: int, lr: float, seed: int,
) -> dict:
    device = torch.device("cuda")
    out: dict[int, dict] = {}
    for B in batches:
        out[B] = {}
        for label, amp_dtype_str in (("fp32", None), ("bf16", "bfloat16")):
            amp_dtype, amp_enabled = resolve_amp_dtype(amp_dtype_str)
            torch.manual_seed(seed)
            tok = KronosTokenizer.from_pretrained(tokenizer_repo).to(device).train()
            optimizer = torch.optim.AdamW(tok.parameters(), lr=lr, weight_decay=0.1)
            try:
                g = torch.Generator(device="cpu").manual_seed(seed)
                idx = torch.randperm(arr_windowed.shape[0], generator=g)[:B]
                batch_x = arr_windowed[idx].to(device, non_blocking=True)
                for _ in range(warmup):
                    _tokenizer_step(tok, optimizer, batch_x, amp_dtype, amp_enabled)
                torch.cuda.reset_peak_memory_stats()
                times: list[float] = []
                last_loss = float("nan")
                for _ in range(iters):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    last_loss = _tokenizer_step(tok, optimizer, batch_x, amp_dtype, amp_enabled)
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - t0)
                peak = int(torch.cuda.max_memory_allocated())
                summ = _summarise(times)
                out[B][label] = {
                    **summ,
                    "steps_per_sec_median": 1.0 / summ["median_s"],
                    "peak_memory_gb": round(peak / (1024 ** 3), 3),
                    "last_loss": last_loss,
                    "ok": True,
                }
                print(f"  tokenizer B={B} {label}: {summ['median_s']*1000:.1f} ms/step  peak {peak/(1024**3):.2f} GB  last_loss={last_loss:.4f}", flush=True)
            except torch.cuda.OutOfMemoryError as e:
                out[B][label] = {"ok": False, "error": "OOM", "msg": str(e)[:200]}
                print(f"  tokenizer B={B} {label}: OOM", flush=True)
            except Exception as e:
                out[B][label] = {"ok": False, "error": type(e).__name__, "msg": str(e)[:200]}
                print(f"  tokenizer B={B} {label}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            finally:
                del tok, optimizer
                torch.cuda.empty_cache()
        if out[B].get("fp32", {}).get("ok") and out[B].get("bf16", {}).get("ok"):
            fp = out[B]["fp32"]; bf = out[B]["bf16"]
            out[B]["pairwise"] = {
                "speedup_x": round(fp["median_s"] / bf["median_s"], 3),
                "memory_ratio": round(bf["peak_memory_gb"] / fp["peak_memory_gb"], 3),
                "loss_rel_diff": round(abs(bf["last_loss"] - fp["last_loss"]) / max(abs(fp["last_loss"]), 1e-6), 4),
            }
    return out


def _convergence_trace(
    Kronos, KronosTokenizer, resolve_amp_dtype,
    tokenizer_repo: str, predictor_repo: str,
    arr_windowed, stamps_windowed,
    n_steps: int, batch_size: int, lr: float, seed: int,
) -> dict:
    """
    Run `n_steps` predictor training steps for each of fp32 and bf16 starting
    from identical pretrained weights and the same per-step batches. Returns
    the per-step loss list for both, plus aggregate summary.
    """
    device = torch.device("cuda")
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_repo).to(device).eval()

    # Pre-sample N batches once so both phases see identical inputs.
    n = arr_windowed.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    cursor = 0
    for _ in range(n_steps):
        if cursor + batch_size > len(perm):
            cursor = 0
        idx = perm[cursor:cursor + batch_size]
        cursor += batch_size
        bx = arr_windowed[idx].to(device, non_blocking=True)
        bs = stamps_windowed[idx].to(device, non_blocking=True)
        batches.append((bx, bs))

    losses_per_phase: dict[str, list[float]] = {}
    for label, amp_dtype_str in (("fp32", None), ("bf16", "bfloat16")):
        amp_dtype, amp_enabled = resolve_amp_dtype(amp_dtype_str)
        torch.manual_seed(seed)
        predictor = Kronos.from_pretrained(predictor_repo).to(device).train()
        optimizer = torch.optim.AdamW(predictor.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
        losses: list[float] = []
        for step in range(n_steps):
            bx, bs = batches[step]
            ll = _predictor_step(tokenizer, predictor, optimizer, bx, bs, amp_dtype, amp_enabled)
            losses.append(ll)
            if (step + 1) % 50 == 0:
                print(f"    convergence {label} step {step+1}/{n_steps}: loss {ll:.4f}", flush=True)
        losses_per_phase[label] = losses
        del predictor, optimizer
        torch.cuda.empty_cache()

    fp = losses_per_phase["fp32"]
    bf = losses_per_phase["bf16"]
    diffs = [abs(b - f) / max(abs(f), 1e-6) for f, b in zip(fp, bf)]
    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "fp32_losses": fp,
        "bf16_losses": bf,
        "fp32_loss_first": fp[0],
        "fp32_loss_last": fp[-1],
        "bf16_loss_first": bf[0],
        "bf16_loss_last": bf[-1],
        "rel_diff_mean": float(statistics.fmean(diffs)),
        "rel_diff_median": float(statistics.median(diffs)),
        "rel_diff_max": float(max(diffs)),
        "rel_diff_at_last_step": diffs[-1],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--gpu-label", type=str, required=True, help="Free-form label for the GPU (e.g. 'RTX 4090').")
    parser.add_argument("--tokenizer", type=str, default="NeoQuasar/Kronos-Tokenizer-base")
    parser.add_argument("--predictor", type=str, default="NeoQuasar/Kronos-base")
    parser.add_argument("--batches", type=str, default="10,25,50,100",
                        help="Comma-separated batch sizes for the sweep")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--convergence-steps", type=int, default=200)
    parser.add_argument("--convergence-batch", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr-predictor", type=float, default=4e-5)
    parser.add_argument("--lr-tokenizer", type=float, default=2e-4)
    parser.add_argument("--n-windows", type=int, default=400)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA required.")

    Kronos, KronosTokenizer, resolve_amp_dtype = _import_kronos(args.repo_root)
    batches = [int(b) for b in args.batches.split(",") if b.strip()]

    print(f"[data] fetching SPY 1-min via yfinance ({args.n_windows} windows of {args.seq_len}) ...", flush=True)
    raw_df, arr_windowed = _fetch_spy_1min(args.seq_len, args.n_windows)
    stamps_windowed = _stamps_for_index(raw_df.index, args.seq_len, args.n_windows)
    print(f"[data] got {len(raw_df)} bars  windowed shape={tuple(arr_windowed.shape)}  date range {raw_df.index[0]} to {raw_df.index[-1]}", flush=True)

    results: dict[str, Any] = {
        "config": {
            "gpu_label": args.gpu_label,
            "batches": batches,
            "seq_len": args.seq_len,
            "warmup": args.warmup,
            "iters": args.iters,
            "convergence_steps": args.convergence_steps,
            "convergence_batch": args.convergence_batch,
            "seed": args.seed,
            "tokenizer_repo": args.tokenizer,
            "predictor_repo": args.predictor,
            "n_windows": args.n_windows,
        },
        "gpu": _gpu_info(),
        "data": {
            "source": "yfinance SPY 1m",
            "n_bars": int(len(raw_df)),
            "first_ts": str(raw_df.index[0]),
            "last_ts": str(raw_df.index[-1]),
        },
    }

    print("[predictor sweep] ...", flush=True)
    results["predictor_sweep"] = _sweep_predictor(
        Kronos, KronosTokenizer, resolve_amp_dtype,
        args.tokenizer, args.predictor,
        arr_windowed, stamps_windowed,
        batches=batches, warmup=args.warmup, iters=args.iters,
        lr=args.lr_predictor, seed=args.seed,
    )

    print("[tokenizer sweep] ...", flush=True)
    results["tokenizer_sweep"] = _sweep_tokenizer(
        KronosTokenizer, resolve_amp_dtype,
        args.tokenizer,
        arr_windowed,
        batches=batches, warmup=args.warmup, iters=args.iters,
        lr=args.lr_tokenizer, seed=args.seed,
    )

    print(f"[convergence] {args.convergence_steps} predictor steps at B={args.convergence_batch}, fp32 then bf16 ...", flush=True)
    results["convergence"] = _convergence_trace(
        Kronos, KronosTokenizer, resolve_amp_dtype,
        args.tokenizer, args.predictor,
        arr_windowed, stamps_windowed,
        n_steps=args.convergence_steps,
        batch_size=args.convergence_batch,
        lr=args.lr_predictor, seed=args.seed,
    )

    args.output.write_text(json.dumps(results, indent=2))
    print(f"\n[wrote] {args.output}", flush=True)
    # Concise summary on stdout — the JSON is the source of truth.
    print("\n=== convergence (predictor) ===")
    c = results["convergence"]
    print(f"  fp32: first {c['fp32_loss_first']:.4f} -> last {c['fp32_loss_last']:.4f}")
    print(f"  bf16: first {c['bf16_loss_first']:.4f} -> last {c['bf16_loss_last']:.4f}")
    print(f"  rel_diff mean={c['rel_diff_mean']:.4e} median={c['rel_diff_median']:.4e} max={c['rel_diff_max']:.4e}")


if __name__ == "__main__":
    main()
