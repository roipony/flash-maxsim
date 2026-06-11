"""Perf comparison: in-one-shot kernel (spills at d > 512) vs split-d.

Sweeps d ∈ {128, 256, 384, 512, 768, 1024, 2048} at the ColPali shape
(B=128, Lq=Ld=1024). For each d we report:
  - In-one-shot kernel (current flash_maxsim_batched): ms + peak GB
  - Split-d kernel (flash_maxsim_splitd._launch_fwd_splitd): ms + peak GB
  - Speedup (in-one-shot / split-d), > 1 means split-d is faster

Goal: split-d should match the in-one-shot kernel at small d (no perf
tax for the inner-loop overhead) AND flatten the d > 512 cliff.
"""
import json, os, sys, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from flash_maxsim import flash_maxsim_batched
from flash_maxsim.flash_maxsim_splitd import _launch_fwd_splitd
from flash_maxsim.flash_maxsim import _default_lengths

torch.set_float32_matmul_precision('high')
WARMUP, RUNS = 3, 10
SHAPES = [
    ("textual",  1024, 32,   180),
    ("ColPali",  128,  1024, 1024),
]
D_VALUES = [128, 256, 384, 512, 768, 1024, 2048]


def _bench(fn, runs=RUNS, warmup=WARMUP):
    for _ in range(warmup):
        fn(); torch.cuda.synchronize()
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(runs):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); e.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2], torch.cuda.max_memory_allocated() / 1e9


gpu = torch.cuda.get_device_name()
print(f"GPU: {gpu}", flush=True)
out = {"gpu": gpu, "rows": []}
for name, B, Lq, Ld in SHAPES:
    print(f"\n=== {name} (B={B}, Lq={Lq}, Ld={Ld}) ===", flush=True)
    for d in D_VALUES:
        row = {"shape": name, "B": B, "Lq": Lq, "Ld": Ld, "d": d}
        Q = F.normalize(torch.randn(1, Lq, d, device='cuda', dtype=torch.float16), dim=-1)
        D = F.normalize(torch.randn(B,  Ld, d, device='cuda', dtype=torch.float16), dim=-1)
        try:
            base_ms, base_peak = _bench(lambda: flash_maxsim_batched(Q, D))
            row["inshot_ms"] = round(base_ms, 3); row["inshot_peak_gb"] = round(base_peak, 3)
        except Exception as e:
            row["inshot_ms"] = None; row["inshot_err"] = type(e).__name__
        try:
            lengths = _default_lengths(B, Ld, Q.device, None)
            split_ms, split_peak = _bench(lambda: _launch_fwd_splitd(
                Q.contiguous(), D.contiguous(), lengths, 1, B, Lq, Ld, d,
                shared_docs=True, save_argmax=False,
            )[0])
            row["splitd_ms"] = round(split_ms, 3); row["splitd_peak_gb"] = round(split_peak, 3)
        except Exception as e:
            row["splitd_ms"] = None; row["splitd_err"] = type(e).__name__
        if row.get("inshot_ms") and row.get("splitd_ms"):
            row["speedup_splitd_vs_inshot"] = round(row["inshot_ms"] / row["splitd_ms"], 2)
            mark = " ★" if row["speedup_splitd_vs_inshot"] > 1.1 else ""
            print(f"  d={d:>4d}: inshot={row['inshot_ms']:>7.3f}ms  splitd={row['splitd_ms']:>7.3f}ms  "
                  f"speedup={row['speedup_splitd_vs_inshot']:.2f}x{mark}", flush=True)
        else:
            print(f"  d={d:>4d}: {row}", flush=True)
        out["rows"].append(row)
        del Q, D; torch.cuda.empty_cache()

p = os.path.join(HERE, f"bench_splitd_perf_{gpu.replace(' ', '_').replace('/', '_')}.json")
json.dump(out, open(p, "w"), indent=2)
print(f"\nwrote {p}", flush=True)
