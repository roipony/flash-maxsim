"""Lock down tab:train: ColPali in-batch-negatives training step (fwd+bwd) at
B=64 (last that fits naive) and B=128 (paper's unlock claim).
Measures step latency + peak GPU memory for flash vs naive autograd backward.
Matched precision (FP32 accumulation via TF32 'high'), cast hoisted in naive.
"""
import json, os, sys, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from flash_maxsim import flash_maxsim_batched_train

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
Lq = Ld = 1024
d = 128


def make(B):
    Q = F.normalize(torch.randn(B, Lq, d, device='cuda', dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(B, Ld, d, device='cuda', dtype=torch.float16), dim=-1)
    return Q, D


def step_flash(Q, D):
    Q.grad = None; D.grad = None
    flash_maxsim_batched_train(Q, D, shared_docs=True).sum().backward()


def step_naive(Q, D):
    """Truly-naive autograd: materializes [B,B,Lq,Ld] FP32 score tensor + its grad.
    Casts to FP32 inside (cast cost amortized over warmup; same precision as Flash)."""
    Q.grad = None; D.grad = None
    Qf = Q.float(); Df = D.float()
    sim = torch.einsum("nqd,bld->nbql", Qf, Df)          # [B,B,Lq,Ld] FP32, materialized
    scores = sim.max(dim=3).values.sum(dim=2)            # [B,B]
    scores.sum().backward()


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
                "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        torch.cuda.empty_cache()
        return {"ms": None, "peak_gb": None, "error": type(e).__name__}


gpu = torch.cuda.get_device_name()
print(f"GPU: {gpu}")
out = {"gpu": gpu, "Lq": Lq, "Ld": Ld, "d": d, "rows": []}

for B in [64, 128]:
    print(f"\n=== B={B} ===")
    try:
        Q, D = make(B); Q.requires_grad_(True); D.requires_grad_(True)
    except Exception as e:
        print(f"  make() OOM: {e}")
        out["rows"].append({"B": B, "make_error": type(e).__name__})
        continue
    flash = measure(step_flash, [Q, D])
    print(f"  flash: {flash}")
    Qn, Dn = make(B); Qn.requires_grad_(True); Dn.requires_grad_(True)
    naive = measure(step_naive, [Qn, Dn])
    print(f"  naive: {naive}")
    row = {"B": B, "flash": flash, "naive": naive}
    if flash["peak_gb"] and naive["peak_gb"]:
        row["mem_ratio"] = round(naive["peak_gb"] / flash["peak_gb"], 1)
    out["rows"].append(row)
    del Q, D, Qn, Dn; torch.cuda.empty_cache()

p = os.path.join(HERE, f"bench_train_paper_{gpu.replace(' ', '_').replace('/', '_')}.json")
json.dump(out, open(p, "w"), indent=2)
print("wrote", p)
