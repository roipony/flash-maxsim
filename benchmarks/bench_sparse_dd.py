"""Benchmark sparse-output grad-D backward.

Compares FLASH_BWD_PATH={atomic,sparse_dd,auto} on KD/non-shared layouts where
each [Ld, d] document slab receives at most Lq non-zero rows.
"""
from __future__ import annotations

import gc
import json
import os
import statistics
import time

import torch
import torch.nn.functional as F


HERE = os.path.dirname(os.path.abspath(__file__))

KD_SHAPES = [
    ("kd-small B32 K8 Ld180", 32, 8, 32, 180, 128),
    ("colpali-neg B128 K8 Ld1030", 128, 8, 32, 1030, 128),
    ("colpali-neg B256 K16 Ld1030", 256, 16, 32, 1030, 128),
]
SHARED_SHAPES = [
    ("shared-smoke B64 Ld180", 64, 64, 32, 180, 128),
]
PATHS = ["atomic", "sparse_dd", "auto"]


def _reset():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def _time_step(fn, warmup=4, runs=10):
    try:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        _reset()
        ts = []
        for _ in range(runs):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        return {
            "median_ms": round(statistics.median(ts), 4),
            "std_ms": round(statistics.stdev(ts), 4) if len(ts) > 1 else 0.0,
            "peak_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 4),
        }
    except torch.cuda.OutOfMemoryError:
        _reset()
        return {"error": "OOM"}
    except Exception as exc:  # noqa: BLE001
        _reset()
        return {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}


def _make_kd_step(flash_maxsim_batched_train, Nq, K, Lq, Ld, d, path):
    def step():
        os.environ["FLASH_BWD_PATH"] = path
        torch.manual_seed(0)
        Q = F.normalize(
            torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float16), dim=-1,
        ).requires_grad_(True)
        D = F.normalize(
            torch.randn(Nq, K, Ld, d, device="cuda", dtype=torch.float16), dim=-1,
        ).requires_grad_(True)
        scores = flash_maxsim_batched_train(Q, D, shared_docs=False)
        scores.sum().backward()
    return step


def _make_shared_step(flash_maxsim_batched_train, Nq, B, Lq, Ld, d, path):
    def step():
        os.environ["FLASH_BWD_PATH"] = path
        torch.manual_seed(1)
        Q = F.normalize(
            torch.randn(Nq, Lq, d, device="cuda", dtype=torch.float16), dim=-1,
        ).requires_grad_(True)
        D = F.normalize(
            torch.randn(B, Ld, d, device="cuda", dtype=torch.float16), dim=-1,
        ).requires_grad_(True)
        scores = flash_maxsim_batched_train(Q, D, shared_docs=True)
        scores.sum().backward()
    return step


def main():
    from flash_maxsim import flash_maxsim_batched_train

    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}  torch={torch.__version__}", flush=True)
    out = {"gpu": gpu, "torch": torch.__version__, "paths": PATHS, "rows": []}

    for name, Nq, K, Lq, Ld, d in KD_SHAPES:
        print(f"\n=== {name} ===", flush=True)
        row = {"name": name, "layout": "nonshared", "Nq": Nq, "B": K, "Lq": Lq, "Ld": Ld, "d": d}
        for path in PATHS:
            r = _time_step(_make_kd_step(flash_maxsim_batched_train, Nq, K, Lq, Ld, d, path))
            row[path] = r
            print(f"  {path:>10s}: {r}", flush=True)
        out["rows"].append(row)

    for name, Nq, B, Lq, Ld, d in SHARED_SHAPES:
        print(f"\n=== {name} ===", flush=True)
        row = {"name": name, "layout": "shared", "Nq": Nq, "B": B, "Lq": Lq, "Ld": Ld, "d": d}
        for path in ("atomic", "sparse_dd"):
            r = _time_step(_make_shared_step(flash_maxsim_batched_train, Nq, B, Lq, Ld, d, path))
            row[path] = r
            print(f"  {path:>10s}: {r}", flush=True)
        out["rows"].append(row)

    out_path = os.path.join(HERE, f"bench_sparse_dd_{gpu.replace(' ', '_').replace('/', '_')}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
