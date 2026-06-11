"""Benchmark sparse MaxSim vs dense. Run: python benchmarks/bench_sparse.py"""
import time, torch, torch.nn.functional as F

assert torch.cuda.is_available(), "CUDA required"
print(f"GPU: {torch.cuda.get_device_name()}\n")

from flash_maxsim import flash_maxsim
from flash_maxsim.flash_maxsim_sparse import flash_maxsim_sparse, maxsim_sparse_naive
from utils import bench_interleaved, compile_high_precision
maxsim_sparse_compiled = compile_high_precision(maxsim_sparse_naive)

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

    t_dense = bench_interleaved([flash_maxsim], [[Q, D]]).item()

    print(f"\n  {label}")
    print(f"    Dense (100%):      {t_dense:.3f} ms")

    for keep in [0.75, 0.5, 0.25]:
        K = max(1, int(Lq * keep))
        indices = make_indices(B, Lq, K)
        t_sparse = bench_interleaved([flash_maxsim_sparse], [[Q, D, indices]]).item()
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
    t_naive, t_compiled, t_flash = bench_interleaved([maxsim_sparse_naive, maxsim_sparse_compiled, flash_maxsim_sparse],
                                                     [[Q, D, indices]])
    print(f"  {label:15s}: naive={t_naive:.2f}ms compiled={t_compiled:.2f}ms flash={t_flash:.2f}ms  "
          f"naive_speedup={t_naive/t_flash:.1f}x compiled_speedup={t_compiled/t_flash:.1f}x")

print(f"\n{'=' * 65}")
print("DONE")
print("=" * 65)
