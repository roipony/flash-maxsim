"""H100 head-to-head vs raphaelsty's PR #224 numbers.

Measures Flash with both backward paths at the contrastive-training shapes
wins or comes close:
  - ColBERT in-batch train: B=512, Lq=32, Ld=180  
  - ColBERT B=128 train: B=128, Lq=32, Ld=180  (both kernels slower than
  - ColPali in-batch train: B=64, Lq=Ld=1024  (our existing strong shape;
                                               check we didn't regress)
  - ColPali B=128 train: should still 'unlock'

Each shape is timed with:
  - FLASH_BWD_PATH=invgrid (the old default: CSR backward)
  - FLASH_BWD_PATH=atomic  (the proposed default for short Lq)
  - naive autograd (the baseline used by raphaelsty's report)

Reports step latency (median of 5 timed runs), peak GPU memory, and a memory
ratio vs the naive baseline. Numbers should be directly comparable to
raphaelsty's table on H100.
"""
import json, os, sys, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from flash_maxsim import flash_maxsim_batched_train

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
d = 128


def make(B, Lq, Ld):
    Q = F.normalize(torch.randn(B, Lq, d, device='cuda', dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(B, Ld, d, device='cuda', dtype=torch.float16), dim=-1)
    return Q, D


def step_flash(Q, D):
    Q.grad = None; D.grad = None
    flash_maxsim_batched_train(Q, D, shared_docs=True).sum().backward()


def step_naive(Q, D):
    """Truly-naive autograd through einsum (the einsum baseline raphaelsty used)."""
    Q.grad = None; D.grad = None
    Qf = Q.float(); Df = D.float()
    s = torch.einsum("nqd,bld->nbql", Qf, Df).max(dim=3).values.sum(dim=2)
    s.sum().backward()


def measure(fn, args, warmup=3, n=5):
    try:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        for _ in range(warmup):
            fn(*args); torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(n):
            start.record(); fn(*args); end.record(); torch.cuda.synchronize()
            ts.append(start.elapsed_time(end))
        ts.sort()
        return {"ms": round(float(ts[len(ts) // 2]), 2),
                "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3)}
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        torch.cuda.empty_cache()
        return {"ms": None, "peak_gb": None, "error": type(e).__name__}


SHAPES = [
    ("ColBERT B=128",  128,   32,  180),
    ("ColBERT B=512",  512,   32,  180),
    ("ColBERT B=1024", 1024,  32,  180),
    ("ColPali B=32",   32,    1024, 1024),
    ("ColPali B=64",   64,    1024, 1024),
    ("ColPali B=128",  128,   1024, 1024),  # paper "unlocked" shape
]

gpu = torch.cuda.get_device_name()
print(f"GPU: {gpu}")
out = {"gpu": gpu, "d": d, "rows": []}

for name, B, Lq, Ld in SHAPES:
    print(f"\n=== {name} (B={B}, Lq={Lq}, Ld={Ld}) ===")
    row = {"name": name, "B": B, "Lq": Lq, "Ld": Ld}

    # naive (raphaelsty's einsum baseline)
    try:
        Qn, Dn = make(B, Lq, Ld); Qn.requires_grad_(True); Dn.requires_grad_(True)
        row["naive"] = measure(step_naive, [Qn, Dn])
        del Qn, Dn; torch.cuda.empty_cache()
    except Exception as e:
        row["naive"] = {"error": type(e).__name__}
    print(f"  naive:   {row['naive']}")

    # flash with CSR (invgrid) backward
    os.environ["FLASH_BWD_PATH"] = "invgrid"
    try:
        Qc, Dc = make(B, Lq, Ld); Qc.requires_grad_(True); Dc.requires_grad_(True)
        row["flash_csr"] = measure(step_flash, [Qc, Dc])
        del Qc, Dc; torch.cuda.empty_cache()
    except Exception as e:
        row["flash_csr"] = {"error": type(e).__name__}
    print(f"  flash_csr:    {row['flash_csr']}")

    # flash with atomic backward
    os.environ["FLASH_BWD_PATH"] = "atomic"
    try:
        Qa, Da = make(B, Lq, Ld); Qa.requires_grad_(True); Da.requires_grad_(True)
        row["flash_atomic"] = measure(step_flash, [Qa, Da])
        del Qa, Da; torch.cuda.empty_cache()
    except Exception as e:
        row["flash_atomic"] = {"error": type(e).__name__}
    print(f"  flash_atomic: {row['flash_atomic']}")

    # Speedups vs naive, memory ratios vs naive
    def ratio(numer, denom):
        if numer and numer.get("ms") and denom and denom.get("ms"):
            return round(denom["ms"] / numer["ms"], 2)
        return None
    def memr(numer, denom):
        if numer and numer.get("peak_gb") and denom and denom.get("peak_gb"):
            return round(denom["peak_gb"] / numer["peak_gb"], 1)
        return None
    row["csr_speedup_vs_naive"]    = ratio(row.get("flash_csr"), row.get("naive"))
    row["atomic_speedup_vs_naive"] = ratio(row.get("flash_atomic"), row.get("naive"))
    row["csr_mem_ratio_vs_naive"]    = memr(row.get("flash_csr"), row.get("naive"))
    row["atomic_mem_ratio_vs_naive"] = memr(row.get("flash_atomic"), row.get("naive"))
    out["rows"].append(row)
    print(f"  CSR    spd={row['csr_speedup_vs_naive']}x mem={row['csr_mem_ratio_vs_naive']}x")
    print(f"  ATOMIC spd={row['atomic_speedup_vs_naive']}x mem={row['atomic_mem_ratio_vs_naive']}x")

p = os.path.join(HERE, f"bench_beat_lik_{gpu.replace(' ', '_').replace('/', '_')}.json")
json.dump(out, open(p, "w"), indent=2)
print(f"\nwrote {p}")