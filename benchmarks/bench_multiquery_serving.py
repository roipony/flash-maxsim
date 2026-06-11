"""Multi-query serving sweep: Nq ∈ {1, 4, 8, 16, 32} × fixed B.

Reviewer R2 #8: address "does one program per (query token, doc) scale to
batched serving" by reporting throughput when the kernel scores Nq queries
against the same B-document corpus in a single launch. This is the realistic
operating point for a rerank service that buffers requests for a few ms
before dispatching.

Workloads cover textual (ColBERT-like, L_q=32, L_d=180, B=1024) and ColPali
(L_q=L_d=1024, B=128). For each Nq we report: per-call latency, throughput
in queries/s and (Nq*B)/s docs-scored, and peak GPU memory. Naive einsum is
the baseline at each cell where it fits.
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import torch  # noqa: E402
from flash_maxsim import flash_maxsim_batched  # noqa: E402

torch.set_float32_matmul_precision('high')
WARMUP, RUNS = 5, 30

CELLS = [
    {"name": "textual",  "Lq": 32,   "Ld": 180,  "d": 128, "B": 1024},
    {"name": "long-doc", "Lq": 32,   "Ld": 1024, "d": 128, "B": 1024},
    {"name": "ColPali",  "Lq": 1024, "Ld": 1024, "d": 128, "B": 128},
]
NQS = [1, 4, 8, 16, 32]


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
    """Reference: [Nq, B, Lq, Ld] tensor + max + sum, all in FP32."""
    sim = torch.einsum('qid,bjd->qbij', Q.float(), D.float())
    return sim.max(dim=-1).values.sum(dim=-1)


def run_cell(c, Nq):
    Lq, Ld, d, B = c["Lq"], c["Ld"], c["d"], c["B"]
    Q = torch.randn(Nq, Lq, d, device='cuda', dtype=torch.float16)
    D = torch.randn(B, Ld, d, device='cuda', dtype=torch.float16)

    # Flash
    def f_flash():
        return flash_maxsim_batched(Q, D)
    torch.cuda.reset_peak_memory_stats()
    t_flash = _bench(f_flash)
    peak_flash = torch.cuda.max_memory_allocated() / 1e9

    # Naive einsum (FP32 cast inside the timed region, like a textbook impl)
    naive_ms, peak_naive, naive_status = None, None, "OOM"
    try:
        def f_naive():
            return naive_einsum(Q, D)
        torch.cuda.reset_peak_memory_stats()
        t = _bench(f_naive)
        naive_ms = t
        peak_naive = torch.cuda.max_memory_allocated() / 1e9
        naive_status = "ok"
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()

    qps_flash = 1000.0 / t_flash * Nq
    docs_per_s = qps_flash * B
    speedup = (naive_ms / t_flash) if naive_ms else None
    row = {
        "cell": c["name"], "Nq": Nq, "Lq": Lq, "Ld": Ld, "B": B,
        "flash_ms": round(t_flash, 4),
        "flash_peak_gb": round(peak_flash, 3),
        "naive_ms": round(naive_ms, 4) if naive_ms else None,
        "naive_peak_gb": round(peak_naive, 3) if peak_naive else None,
        "naive_status": naive_status,
        "qps_flash": round(qps_flash, 1),
        "docs_per_s_flash": int(docs_per_s),
        "speedup_over_naive": round(speedup, 2) if speedup else None,
    }
    sp = f" ({speedup:.1f}x)" if speedup else ""
    naive_str = f"{naive_ms:.2f}ms" if naive_ms else "OOM"
    print(f"  Nq={Nq:>2}: flash={t_flash:.2f}ms ({qps_flash:.0f} QPS, "
          f"{docs_per_s/1000:.0f}K docs/s, {peak_flash:.2f}GB) vs naive={naive_str}{sp}",
          flush=True)
    return row


results = []
for c in CELLS:
    print(f"\n=== {c['name']} (B={c['B']}, Lq={c['Lq']}, Ld={c['Ld']}, d={c['d']}) ===", flush=True)
    for nq in NQS:
        try:
            results.append(run_cell(c, nq))
        except Exception as e:
            print(f"  Nq={nq} FAILED: {type(e).__name__}: {e}", flush=True)
            results.append({"cell": c["name"], "Nq": nq, "error": str(e)[:200]})

gpu = torch.cuda.get_device_name()
out = os.path.join(HERE, f"bench_multiquery_serving_{gpu.replace(' ', '_').replace('/', '_')}.json")
json.dump({"gpu": gpu, "warmup": WARMUP, "runs": RUNS, "results": results}, open(out, "w"), indent=2)
print(f"\nwrote {out}")
