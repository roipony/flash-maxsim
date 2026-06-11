"""Operator-design ablation: which design choice gives which speedup.

Reviewer R2 #4: justify each Flash-MaxSim design choice with a marginal
ablation, so the reader can see naive -> fused -> +query-chunking -> +CSR
as cumulative deltas, not as a single black-box "Flash".

Forward ablation (this script). The training-time CSR-vs-atomic ablation
already lives in tab:bwd_abl; here we isolate three forward variants on
fixed corpora:

  1. Naive einsum + max + sum (FP32, the textbook reduction)
  2. Fused forward, no query chunking (flash_maxsim_batched with
     query_chunk_size=None; one program per (q, b) pair)
  3. Fused forward + query chunking (the default flash_maxsim_batched with
     query_chunk_size=128; same kernel but Q tiled into chunks for better
     occupancy at large L_q)

Shapes cover textual (small L_q, the launch-bound regime), medium
(autotune-driven), and ColPali (compute-bound). FP16 inputs, FP32 accum
throughout. Output is latency, peak memory, and the cumulative deltas
(naive -> #2 -> #3).
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import torch  # noqa: E402
from flash_maxsim import flash_maxsim_batched  # noqa: E402

torch.set_float32_matmul_precision('high')
WARMUP, RUNS = 5, 30

SHAPES = [
    {"name": "textual",  "Lq": 32,   "Ld": 180,  "d": 128, "B": 1024},
    {"name": "long-doc", "Lq": 32,   "Ld": 1024, "d": 128, "B": 1024},
    {"name": "medium",   "Lq": 128,  "Ld": 1024, "d": 128, "B": 512},
    {"name": "visual",   "Lq": 512,  "Ld": 1024, "d": 128, "B": 256},
    {"name": "ColPali",  "Lq": 1024, "Ld": 1024, "d": 128, "B": 128},
]


def _bench(fn, runs=RUNS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); e.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def naive_einsum(Q, D):
    sim = torch.einsum('qid,bjd->qbij', Q.float(), D.float())
    return sim.max(dim=-1).values.sum(dim=-1)


def run_shape(s):
    Lq, Ld, d, B = s["Lq"], s["Ld"], s["d"], s["B"]
    Nq = 1
    Q = torch.randn(Nq, Lq, d, device='cuda', dtype=torch.float16)
    D = torch.randn(B, Ld, d, device='cuda', dtype=torch.float16)

    # 1. Naive einsum + max + sum (FP32 inside the timed region — like the user
    #    pasted on Stack Overflow)
    naive_ms, naive_peak = None, None
    try:
        torch.cuda.reset_peak_memory_stats()
        naive_ms = _bench(lambda: naive_einsum(Q, D))
        naive_peak = torch.cuda.max_memory_allocated() / 1e9
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()

    # 2. Fused forward, NO query chunking
    torch.cuda.reset_peak_memory_stats()
    fused_ms = _bench(lambda: flash_maxsim_batched(Q, D, query_chunk_size=None))
    fused_peak = torch.cuda.max_memory_allocated() / 1e9

    # 3. Fused + query chunking (default)
    torch.cuda.reset_peak_memory_stats()
    chunked_ms = _bench(lambda: flash_maxsim_batched(Q, D, query_chunk_size=128))
    chunked_peak = torch.cuda.max_memory_allocated() / 1e9

    naive_str = f"{naive_ms:.3f}ms" if naive_ms else "OOM"
    speedup_fused = (naive_ms / fused_ms) if naive_ms else None
    speedup_chunked = (naive_ms / chunked_ms) if naive_ms else None
    delta_chunking = fused_ms / chunked_ms  # >1 means chunking helps

    row = {
        "shape": s["name"], "Lq": Lq, "Ld": Ld, "d": d, "B": B,
        "naive_ms": round(naive_ms, 4) if naive_ms else None,
        "naive_peak_gb": round(naive_peak, 3) if naive_peak else None,
        "fused_no_chunk_ms": round(fused_ms, 4),
        "fused_no_chunk_peak_gb": round(fused_peak, 3),
        "fused_chunked_ms": round(chunked_ms, 4),
        "fused_chunked_peak_gb": round(chunked_peak, 3),
        "speedup_fused_vs_naive": round(speedup_fused, 2) if speedup_fused else None,
        "speedup_chunked_vs_naive": round(speedup_chunked, 2) if speedup_chunked else None,
        "delta_from_chunking": round(delta_chunking, 2),
    }
    sp_naive = f" -> fused {speedup_fused:.2f}x" if speedup_fused else ""
    sp_chunk = f", +chunking {delta_chunking:.2f}x" if delta_chunking else ""
    print(f"  {s['name']}: naive={naive_str}{sp_naive} (peak {fused_peak:.2f}GB){sp_chunk}",
          flush=True)
    return row


print(f"GPU: {torch.cuda.get_device_name()}", flush=True)
results = []
for s in SHAPES:
    print(f"\n=== {s['name']} (B={s['B']}, Lq={s['Lq']}, Ld={s['Ld']}, d={s['d']}) ===",
          flush=True)
    try:
        results.append(run_shape(s))
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
        results.append({"shape": s["name"], "error": str(e)[:200]})

gpu = torch.cuda.get_device_name()
out = os.path.join(HERE, f"bench_operator_design_{gpu.replace(' ', '_').replace('/', '_')}.json")
json.dump({"gpu": gpu, "warmup": WARMUP, "runs": RUNS, "results": results}, open(out, "w"), indent=2)
print(f"\nwrote {out}")
