"""torch.compile(mode="max-autotune-no-cudagraphs") at the paper's shapes.

Reviewer W5/Q5: the canonical compile-MA (graphs ON) numbers are in
Tab. 1 / App. A; the graphs-off flavour removes the CUDA-graph private
pool but loses the autotuner's strongest mode. This script measures
ONLY the no-cudagraphs flavour (one method per process — no allocator
cross-contamination per the checklist) at:
  - the five tab:fwd shapes at B=1K
  - ColPali B=10K and B=20K (does it still OOM without the pool?)

Output: bench_compile_ma_nocg_<GPU>.json
"""
import json, os, sys
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True

d = 128
CELLS = [
    ("textual",  32,   300,  1000),
    ("long-doc", 32,   1024, 1000),
    ("medium",   128,  1024, 1000),
    ("visual",   512,  1024, 1000),
    ("ColPali",  1024, 1024, 1000),
    ("ColPali",  1024, 1024, 10000),
    ("ColPali",  1024, 1024, 20000),
]


def naive(Q, D):
    return torch.einsum("qd,bld->bql", Q, D).max(2).values.sum(1)


def main():
    gpu = torch.cuda.get_device_name()
    print(f"GPU: {gpu}  torch {torch.__version__}", flush=True)
    rows = []
    for label, Lq, Ld, B in CELLS:
        print(f"=== {label} Lq={Lq} Ld={Ld} B={B} ===", flush=True)
        Q = F.normalize(torch.randn(Lq, d, device='cuda', dtype=torch.float16), dim=-1)
        D = F.normalize(torch.randn(B, Ld, d, device='cuda', dtype=torch.float16), dim=-1)
        fn = torch.compile(naive, mode="max-autotune-no-cudagraphs", dynamic=False)
        row = {"label": label, "Lq": Lq, "Ld": Ld, "B": B}
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            for _ in range(5):
                fn(Q, D); torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            ts = []
            for _ in range(10):
                s.record(); fn(Q, D); e.record(); torch.cuda.synchronize()
                ts.append(s.elapsed_time(e))
            ts.sort()
            row.update(ms=round(ts[len(ts)//2], 3),
                       peak_gb=round(torch.cuda.max_memory_allocated()/1e9, 3), ok=True)
            print(f"  compile-MA-nocg: {row['ms']} ms, peak {row['peak_gb']} GB", flush=True)
        except torch.cuda.OutOfMemoryError as ex:
            row.update(ms=None, peak_gb=None, ok=False, error=str(ex)[:120])
            print(f"  OOM: {str(ex)[:100]}", flush=True)
        rows.append(row)
        del Q, D, fn
        torch.cuda.empty_cache()
        torch._dynamo.reset()
    op = os.path.join(HERE, f"bench_compile_ma_nocg_{gpu.replace(' ', '_').replace('/', '_')}.json")
    json.dump({"gpu": gpu, "torch": torch.__version__,
               "mode": "max-autotune-no-cudagraphs", "rows": rows}, open(op, "w"), indent=2)
    print("wrote", op, flush=True)


if __name__ == '__main__':
    main()
