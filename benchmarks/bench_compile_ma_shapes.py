"""Per-shape compile-MA audit for tab:fwd column expansion.

Runs the same canonical max-autotune (CUDA graphs ON) protocol as
bench_compile_ma_audit.py, but across the 5 tab:fwd canonical shapes
at B=1K instead of just ColPali at B=1K and B=10K. Output is appended
to data/bench_compile_ma_shapes_<gpu>.json so tab:fwd can gain a
compile-MA speedup column.
"""
import json, os, sys, torch, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from flash_maxsim import flash_maxsim

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# tab:fwd canonical shapes (Lq, Ld) at d=128, B=1000, Nq=1
SHAPES = [
    ("textual",  32,   300),
    ("long-doc", 32,   1024),
    ("medium",   128,  1024),
    ("visual",   512,  1024),
    ("ColPali",  1024, 1024),
]
B = 1000
d = 128


def make(Lq, Ld):
    Q = F.normalize(torch.randn(Lq, d, device='cuda', dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(B, Ld, d, device='cuda', dtype=torch.float16), dim=-1)
    return Q, D


def naive_fp32_precast(Qf, Df):
    return torch.einsum("qd,bld->bql", Qf, Df).max(2).values.sum(1)


compile_fp32_ma = torch.compile(naive_fp32_precast, mode="max-autotune",
                                 dynamic=False)


def measure(fn, args, warmup=5, n=10):
    try:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        for _ in range(warmup):
            _ = fn(*args); torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(n):
            s.record(); _ = fn(*args); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        ts.sort()
        return {"ms": round(float(ts[len(ts) // 2]), 3),
                "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3)}
    except (torch.cuda.OutOfMemoryError, RuntimeError, AssertionError) as ex:
        torch.cuda.empty_cache()
        return {"ms": None, "peak_gb": None, "error": f"{type(ex).__name__}"}


gpu = torch.cuda.get_device_name()
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {gpu} ({vram:.1f} GB)\ntorch={torch.__version__}\n")

out = {"gpu": gpu, "vram_gb": round(vram, 1), "B": B, "d": d,
       "torch": torch.__version__,
       "torch_compile_mode": "max-autotune (CUDA graphs ENABLED, dynamic=False)",
       "rows": []}

for name, Lq, Ld in SHAPES:
    print(f"=== {name} Lq={Lq} Ld={Ld} ===")
    Q, D = make(Lq, Ld)
    Qf, Df = Q.float(), D.float()
    fl  = measure(flash_maxsim, [Q, D]);                       print(f"  flash:           {fl}")
    n32 = measure(naive_fp32_precast, [Qf, Df]);               print(f"  eager_fp32:      {n32}")
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    cma = measure(compile_fp32_ma, [Qf, Df]);                  print(f"  compile-MA:      {cma}")
    out["rows"].append({"shape": name, "Lq": Lq, "Ld": Ld,
                        "flash": fl, "eager_fp32_tf32": n32, "compile_ma": cma})
    del Q, D, Qf, Df; torch.cuda.empty_cache()

p = os.path.join(HERE, f"bench_compile_ma_shapes_{gpu.replace(' ', '_').replace('/', '_')}.json")
json.dump(out, open(p, "w"), indent=2)
print(f"\nwrote {p}")
