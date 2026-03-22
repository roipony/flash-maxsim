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

def make_indices(B, Lq, K):
    """Simulate per-doc importance: each doc picks K different tokens."""
    importance = torch.randn(B, Lq, device='cuda')
    return importance.topk(K, dim=1).indices.to(torch.int32)  # [B, K]

# ── Correctness ──
print("=" * 65)
print("CORRECTNESS")
print("=" * 65)
for B, Lq, Ld, d, keep in [(50, 32, 300, 128, 0.5), (20, 1024, 1024, 128, 0.5), (100, 64, 512, 128, 0.25)]:
    Q = F.normalize(torch.randn(Lq, d, device='cuda', dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(B, Ld, d, device='cuda', dtype=torch.float16), dim=-1)
    K = max(1, int(Lq * keep))
    indices = make_indices(B, Lq, K)
    ref = maxsim_sparse_naive(Q, D, indices)
    out = flash_maxsim_sparse(Q, D, indices)
    err = (ref - out).abs().max().item()
    rank_ok = (ref.argsort(descending=True)[:10] == out.argsort(descending=True)[:10]).all()
    print(f"  B={B}, Lq={Lq}, K={K}: err={err:.2e} rank={'OK' if rank_ok else 'FAIL'}")

# ── Sparse vs Dense flash_maxsim ──
print(f"\n{'=' * 65}")
print("SPARSE vs DENSE FLASH-MAXSIM")
print("=" * 65)

for B, Lq, Ld, d, label in [
    (1000, 32,   300,  128, "Textual (Lq=32, Ld=300)"),
    (1000, 128,  300,  128, "Medium (Lq=128, Ld=300)"),
    (1000, 256,  300,  128, "Long query (Lq=256, Ld=300)"),
    (1000, 128,  1024, 128, "Medium+long doc (Lq=128, Ld=1024)"),
    (1000, 256,  1024, 128, "Long query+doc (Lq=256, Ld=1024)"),
    (1000, 1024, 1024, 128, "Visual (Lq=1024, Ld=1024)"),
    (5000, 1024, 1024, 128, "Visual large (Lq=1024, B=5K)"),
]:
    Q = F.normalize(torch.randn(Lq, d, device='cuda', dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(B, Ld, d, device='cuda', dtype=torch.float16), dim=-1)

    wup = 5 if Lq >= 512 else 10
    iters = 20 if Lq >= 512 else 40
    t_dense = bench(flash_maxsim, Q, D, warmup=wup, n=iters)

    print(f"\n  {label}")
    print(f"    Dense (100%):      {t_dense:.3f} ms")

    for keep in [0.75, 0.5, 0.25]:
        K = max(1, int(Lq * keep))
        indices = make_indices(B, Lq, K)
        t_sparse = bench(flash_maxsim_sparse, Q, D, indices, warmup=wup, n=iters)
        speedup = t_dense / t_sparse if t_sparse > 0 else 0
        pct = int(keep * 100)
        print(f"    Sparse ({pct:2d}%, K={K:4d}): {t_sparse:.3f} ms  ({speedup:.2f}x vs dense)")

# ── Sparse flash vs naive sparse ──
print(f"\n{'=' * 65}")
print("SPARSE FLASH vs NAIVE SPARSE (50%)")
print("=" * 65)

for B, Lq, Ld, d, label in [
    (1000, 32,   300,  128, "Textual"),
    (500,  32,   1024, 128, "Long-doc"),
    (1000, 1024, 1024, 128, "Visual"),
]:
    Q = F.normalize(torch.randn(Lq, d, device='cuda', dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(B, Ld, d, device='cuda', dtype=torch.float16), dim=-1)
    K = Lq // 2
    indices = make_indices(B, Lq, K)

    wup = 5 if Lq >= 512 else 10
    iters = 20 if Lq >= 512 else 40
    t_naive = bench(maxsim_sparse_naive, Q, D, indices, warmup=wup, n=iters)
    t_flash = bench(flash_maxsim_sparse, Q, D, indices, warmup=wup, n=iters)
    print(f"  {label:15s}: naive={t_naive:.2f}ms  flash={t_flash:.2f}ms  {t_naive/t_flash:.1f}x")

print(f"\n{'=' * 65}")
print("DONE")
print("=" * 65)
