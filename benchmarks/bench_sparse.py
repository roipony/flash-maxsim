"""Benchmark sparse MaxSim vs dense. Run: python benchmarks/bench_sparse.py"""
import time, torch, torch.nn.functional as F

assert torch.cuda.is_available(), "CUDA required"
print(f"GPU: {torch.cuda.get_device_name()}\n")

from flash_maxsim import flash_maxsim, flash_maxsim_sparse, maxsim_sparse_naive

def bench(fn, *a, warmup=10, n=50):
    for _ in range(warmup): fn(*a)
    torch.cuda.synchronize()
    t = []
    for _ in range(n):
        torch.cuda.synchronize(); s = time.perf_counter(); fn(*a)
        torch.cuda.synchronize(); t.append((time.perf_counter()-s)*1000)
    t.sort(); return t[len(t)//2]

def make_mask(B, Lq, keep_frac):
    importance = torch.randn(B, Lq, device='cuda')
    k = max(1, int(Lq * keep_frac))
    topk_idx = importance.topk(k, dim=1).indices
    mask = torch.zeros(B, Lq, device='cuda', dtype=torch.bool)
    mask.scatter_(1, topk_idx, True)
    return mask

# ── Correctness check ──
print("=" * 65)
print("CORRECTNESS")
print("=" * 65)
for B, Lq, Ld, d, keep in [(50, 32, 300, 128, 0.5), (20, 1024, 1024, 128, 0.5)]:
    Q = F.normalize(torch.randn(Lq, d, device='cuda', dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(B, Ld, d, device='cuda', dtype=torch.float16), dim=-1)
    mask = make_mask(B, Lq, keep)
    ref = maxsim_sparse_naive(Q, D, mask)
    out = flash_maxsim_sparse(Q, D, mask)
    err = (ref - out).abs().max().item()
    print(f"  B={B}, Lq={Lq}, Ld={Ld}, keep={keep}: err={err:.2e} {'OK' if err < 1.0 else 'FAIL'}")

# ── Speedup: sparse vs dense ──
print(f"\n{'=' * 65}")
print("SPARSE vs DENSE FLASH-MAXSIM")
print("=" * 65)

for B, Lq, Ld, d, label in [
    (1000, 32,   300,  128, "Textual (Lq=32, Ld=300)"),
    (500,  32,   1024, 128, "Long-doc (Lq=32, Ld=1024)"),
    (1000, 1024, 1024, 128, "Visual (Lq=1024, Ld=1024)"),
    (5000, 1024, 1024, 128, "Visual large (Lq=1024, B=5K)"),
]:
    Q = F.normalize(torch.randn(Lq, d, device='cuda', dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(B, Ld, d, device='cuda', dtype=torch.float16), dim=-1)

    # Dense (full flash_maxsim)
    wup = 5 if Lq >= 512 else 10
    iters = 20 if Lq >= 512 else 40
    t_dense = bench(flash_maxsim, Q, D, warmup=wup, n=iters)

    print(f"\n  {label}")
    print(f"    Dense (100%):  {t_dense:.3f} ms")

    for keep in [0.75, 0.5, 0.25]:
        mask = make_mask(B, Lq, keep)
        t_sparse = bench(flash_maxsim_sparse, Q, D, mask, warmup=wup, n=iters)
        speedup = t_dense / t_sparse if t_sparse > 0 else 0
        pct = int(keep * 100)
        print(f"    Sparse ({pct}%):  {t_sparse:.3f} ms  ({speedup:.2f}x vs dense)")

# ── Sparse vs Naive ──
print(f"\n{'=' * 65}")
print("SPARSE FLASH vs NAIVE (50% tokens)")
print("=" * 65)

for B, Lq, Ld, d, label in [
    (1000, 32,   300,  128, "Textual"),
    (500,  32,   1024, 128, "Long-doc"),
    (1000, 1024, 1024, 128, "Visual"),
]:
    Q = F.normalize(torch.randn(Lq, d, device='cuda', dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(B, Ld, d, device='cuda', dtype=torch.float16), dim=-1)
    mask = make_mask(B, Lq, 0.5)

    wup = 5 if Lq >= 512 else 10
    iters = 20 if Lq >= 512 else 40

    t_naive = bench(maxsim_sparse_naive, Q, D, mask, warmup=wup, n=iters)
    t_flash = bench(flash_maxsim_sparse, Q, D, mask, warmup=wup, n=iters)
    print(f"  {label:15s}: naive={t_naive:.2f}ms  flash_sparse={t_flash:.2f}ms  {t_naive/t_flash:.1f}x")

print(f"\n{'=' * 65}")
print("DONE")
print("=" * 65)
