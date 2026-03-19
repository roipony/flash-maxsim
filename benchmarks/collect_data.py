"""Collect benchmark data for visualization. Run on each GPU and save JSON.

Usage:
    python benchmarks/collect_data.py
    # produces benchmarks/results_<GPU_NAME>.json
"""
import json, time, torch, torch.nn.functional as F

assert torch.cuda.is_available(), "CUDA required"
gpu_name = torch.cuda.get_device_name()
print(f"GPU: {gpu_name}\n")

from flash_maxsim import flash_maxsim, flash_maxsim_batched, flash_maxsim_train
from flash_maxsim import flash_maxsim_int8, quantize_int8

def bench(fn, *a, warmup=10, n=50):
    for _ in range(warmup): fn(*a)
    torch.cuda.synchronize()
    t = []
    for _ in range(n):
        torch.cuda.synchronize(); s = time.perf_counter(); fn(*a)
        torch.cuda.synchronize(); t.append((time.perf_counter()-s)*1000)
    t.sort(); return t[len(t)//2]

def make(B, Lq, Ld, d):
    Q = F.normalize(torch.randn(Lq, d, device='cuda', dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(B, Ld, d, device='cuda', dtype=torch.float16), dim=-1)
    return Q, D

def naive_fp32(Q, D):
    return torch.einsum('qd,bld->bql', Q.float(), D.float()).max(2).values.sum(1)

results = {"gpu": gpu_name, "speedup": [], "memory": [], "int8": [], "batched": []}

# ── Speedup sweep ──
print("Speedup sweep...")
for B, Lq, Ld, d, label in [
    (1000, 32,   300,  128, "ColBERT\n1K docs"),
    (500,  32,   1024, 128, "ColPali text\n500 docs"),
    (1000, 1024, 1024, 128, "ColPali image\n1K docs"),
    (5000, 1024, 1024, 128, "ColPali image\n5K docs"),
]:
    Q, D = make(B, Lq, Ld, d)
    wup = 5 if Lq >= 1024 else 10
    iters = 20 if Lq >= 1024 else 50
    n = bench(naive_fp32, Q, D, warmup=wup, n=iters)
    f = bench(flash_maxsim, Q, D, warmup=wup, n=iters)
    print(f"  {label.replace(chr(10),' ')}: naive={n:.2f}ms  flash={f:.2f}ms  {n/f:.1f}x")
    results["speedup"].append({"label": label, "naive_ms": round(n, 3), "flash_ms": round(f, 3), "speedup": round(n/f, 1)})
    del Q, D; torch.cuda.empty_cache()

# ── Memory sweep ──
print("\nMemory sweep...")
for Nq, B, Lq, Ld, d, label in [
    (1,  500,  32,   1024, 128, "1q x 500p\nLd=1024"),
    (1,  1000, 1024, 1024, 128, "1q x 1Kp\nLq=Ld=1024"),
    (10, 1000, 1024, 1024, 128, "10q x 1Kp\nLq=Ld=1024"),
]:
    Q = F.normalize(torch.randn(Nq, Lq, d, device='cuda', dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(B, Ld, d, device='cuda', dtype=torch.float16), dim=-1)
    sim_gb = Nq * B * Lq * Ld * 4 / 1e9

    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    try:
        _ = torch.einsum('nqd,bld->nbql', Q.float(), D.float()).max(3).values.sum(2)
        torch.cuda.synchronize()
        naive_gb = (torch.cuda.max_memory_allocated() - base) / 1e9
        del _
    except RuntimeError:
        naive_gb = sim_gb
    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    _ = flash_maxsim_batched(Q, D, shared_docs=True); torch.cuda.synchronize()
    flash_gb = (torch.cuda.max_memory_allocated() - base) / 1e9
    del _; torch.cuda.empty_cache()

    ratio = max(1, naive_gb / max(flash_gb, 0.001))
    print(f"  {label.replace(chr(10),' ')}: naive={naive_gb:.2f}GB  flash={flash_gb:.4f}GB  {ratio:.0f}x")
    results["memory"].append({"label": label, "naive_gb": round(naive_gb, 3), "flash_gb": round(flash_gb, 4), "ratio": round(ratio, 0)})
    del Q, D; torch.cuda.empty_cache()

# ── INT8 sweep ──
print("\nINT8 sweep...")
for B, Lq, Ld, d, label in [
    (1000, 32, 300,  128, "ColBERT 1K"),
    (5000, 32, 300,  128, "ColBERT 5K"),
    (500,  32, 1024, 128, "ColPali 500"),
]:
    Q, D = make(B, Lq, Ld, d)
    Dq, s, m = quantize_int8(D)
    nf = bench(naive_fp32, Q, D, warmup=5, n=30)
    fi = bench(flash_maxsim_int8, Q, Dq, s, m, warmup=5, n=30)
    print(f"  {label}: naive_fp32={nf:.2f}ms  flash_q8={fi:.2f}ms  {nf/fi:.1f}x")
    results["int8"].append({"label": label, "naive_ms": round(nf, 3), "flash_ms": round(fi, 3), "speedup": round(nf/fi, 1)})
    del Q, D, Dq, s, m; torch.cuda.empty_cache()

# ── Batched sweep ──
print("\nBatched sweep...")
for Nq, B, Lq, Ld, d, label in [
    (100, 1000, 32,   300,  128, "100q x 1Kd\nColBERT"),
    (50,  500,  32,   1024, 128, "50q x 500d\nColPali"),
    (100, 100,  1024, 1024, 128, "100q x 100d\nLq=Ld=1024"),
]:
    Q = F.normalize(torch.randn(Nq, Lq, d, device='cuda', dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(B, Ld, d, device='cuda', dtype=torch.float16), dim=-1)
    def naive_loop():
        return torch.stack([naive_fp32(Q[i], D) for i in range(Nq)])
    nl = bench(naive_loop, warmup=3, n=10)
    fb = bench(flash_maxsim_batched, Q, D, None, True, warmup=5, n=20)
    tp = Nq * B / (fb / 1000)
    print(f"  {label.replace(chr(10),' ')}: naive={nl:.1f}ms  flash={fb:.2f}ms  {nl/fb:.1f}x  ({tp/1e6:.1f}M pairs/s)")
    results["batched"].append({"label": label, "naive_ms": round(nl, 2), "flash_ms": round(fb, 3), "speedup": round(nl/fb, 1), "pairs_per_sec": round(tp/1e6, 1)})
    del Q, D; torch.cuda.empty_cache()

# ── Save ──
gpu_tag = gpu_name.replace(" ", "_").replace("/", "-")
out = f"benchmarks/results_{gpu_tag}.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out}")
