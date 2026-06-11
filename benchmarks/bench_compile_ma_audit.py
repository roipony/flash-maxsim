"""Re-audit the App. A compile-MA numbers with the standard
torch.compile(max-autotune) protocol (CUDA graphs ON).

Background: App. A of the paper claims:
    ColPali B=1K  : Flash 1.7 ms / compile-MA 3.8 ms / eager 6.6 ms  (2.3x)
    ColPali B=10K : compile-MA ~36 ms (fits on 80GB) / Flash 16.3 ms
These cells have no checked-in JSON-of-record in the audit map.
This script regenerates them on A100-80GB with the canonical
mode='max-autotune' (CUDA-graph capture enabled, which is the default
production-strength compile-MA configuration that App. A is comparing
against — distinct from our scaling-figure bench which had to
disable CUDA graphs for protocol + bug reasons).

Protocol:
- ColPali shape: Lq=Ld=1024, d=128, FP16 source tensors with .float()
  precast OUTSIDE the timed region (matching App. A's 'matched-FP32 with
  TF32 on' baseline).
- 5 warmup + 10 timed CUDA-event medians per cell.
- Each function gets its own try/except so compile-MA failures don't
  hide eager / Flash numbers.
- Two B values: 1000, 10000 (the two cells App. A cites explicitly).
"""
import json, os, sys, torch, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from flash_maxsim import flash_maxsim

torch.set_float32_matmul_precision('high')  # TF32 ON (paper's matched protocol)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

Lq = Ld = 1024
d = 128


def make(B):
    Q = F.normalize(torch.randn(Lq, d, device='cuda', dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(B, Ld, d, device='cuda', dtype=torch.float16), dim=-1)
    return Q, D


def naive_fp32_precast(Qf, Df):
    return torch.einsum("qd,bld->bql", Qf, Df).max(2).values.sum(1)


# Canonical compile-MA (CUDA graphs enabled). This is what App. A's
# claimed numbers are supposed to be measured against.
compile_fp32_ma_cudagraphs = torch.compile(
    naive_fp32_precast, mode="max-autotune", dynamic=False,
)


def measure(fn, args, label, warmup=5, n=10):
    try:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        for _ in range(warmup):
            _ = fn(*args); torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(n):
            start.record(); _ = fn(*args); end.record(); torch.cuda.synchronize()
            ts.append(start.elapsed_time(end))
        ts.sort()
        return {
            "ms": round(float(ts[len(ts) // 2]), 3),
            "ms_p20": round(float(ts[len(ts) * 2 // 10]), 3),
            "ms_p80": round(float(ts[len(ts) * 8 // 10]), 3),
            "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        }
    except (torch.cuda.OutOfMemoryError, RuntimeError, AssertionError) as e:
        torch.cuda.empty_cache()
        return {"ms": None, "peak_gb": None, "error": f"{type(e).__name__}: {str(e)[:120]}"}


gpu = torch.cuda.get_device_name()
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {gpu} ({vram:.1f} GB VRAM)")
print(f"torch={torch.__version__}")

out = {"gpu": gpu, "vram_gb": round(vram, 1), "Lq": Lq, "Ld": Ld, "d": d,
       "torch": torch.__version__,
       "torch_compile_mode": "max-autotune (CUDA graphs ENABLED, dynamic=False)",
       "rows": []}

for B in [1000, 10000]:
    print(f"\n=== B={B} ===")
    Q, D = make(B)
    Qf, Df = Q.float(), D.float()

    fl = measure(flash_maxsim, [Q, D], "flash")
    print(f"  flash:           {fl}")

    # eager FP32 with TF32 (the 'naive' baseline App. A cites)
    n32 = measure(naive_fp32_precast, [Qf, Df], "eager_fp32")
    print(f"  eager_fp32(TF32):{n32}")
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    # compile-MA with CUDA graphs (the App. A protocol)
    cma = measure(compile_fp32_ma_cudagraphs, [Qf, Df], "compile_ma")
    print(f"  compile-MA:      {cma}")
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    out["rows"].append({"B": B, "flash": fl, "eager_fp32_tf32": n32, "compile_ma": cma})
    del Q, D, Qf, Df; torch.cuda.empty_cache()

p = os.path.join(HERE, f"bench_compile_ma_audit_{gpu.replace(' ', '_').replace('/', '_')}.json")
json.dump(out, open(p, "w"), indent=2)
print(f"\nwrote {p}")
