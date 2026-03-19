"""Benchmark sparse MaxSim vs dense. Run: python benchmarks/bench_sparse.py"""
import time, torch, torch.nn.functional as F

assert torch.cuda.is_available(), "CUDA required"
print(f"GPU: {torch.cuda.get_device_name()}\n")

from flash_maxsim import flash_maxsim, flash_maxsim_sparse, prepare_sparse

def bench(fn, *a, warmup=10, n=50):
    for _ in range(warmup): fn(*a)
    torch.cuda.synchronize()
    t = []
    for _ in range(n):
        torch.cuda.synchronize(); s = time.perf_counter(); fn(*a)
        torch.cuda.synchronize(); t.append((time.perf_counter()-s)*1000)
    t.sort(); return t[len(t)//2]

# ── Correctness ──
print("=" * 65)
print("CORRECTNESS (sparse vs dense on kept tokens)")
print("=" * 65)
for B, Lq, Ld, d, keep in [(50, 32, 300, 128, 0.5), (20, 1024, 1024, 128, 0.5), (100, 64, 512, 128, 0.25)]:
    Q = F.normalize(torch.randn(Lq, d, device='cuda', dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(B, Ld, d, device='cuda', dtype=torch.float16), dim=-1)
    k = max(1, int(Lq * keep))

    # Dense with truncated Q (ground truth for sorted Q)
    Q_trunc = Q[:k]
    ref = flash_maxsim(Q_trunc, D)

    # Sparse with uniform query_lengths
    qlens = torch.full((B,), k, device='cuda', dtype=torch.int32)
    out = flash_maxsim_sparse(Q, D, qlens)

    err = (ref - out).abs().max().item()
    print(f"  B={B}, Lq={Lq}, keep={keep}: err={err:.2e} {'OK' if err < 0.01 else 'FAIL'}")

# ── Speedup: sparse vs dense ──
print(f"\n{'=' * 65}")
print("SPARSE vs DENSE (Q shared, per-doc query_lengths)")
print("=" * 65)

for B, Lq, Ld, d, label in [
    (1000, 32,   300,  128, "Textual (Lq=32, Ld=300)"),
    (500,  32,   1024, 128, "Long-doc (Lq=32, Ld=1024)"),
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
        k = max(1, int(Lq * keep))
        qlens = torch.full((B,), k, device='cuda', dtype=torch.int32)
        t_sparse = bench(flash_maxsim_sparse, Q, D, qlens, warmup=wup, n=iters)
        speedup = t_dense / t_sparse if t_sparse > 0 else 0
        pct = int(keep * 100)
        print(f"    Sparse ({pct:2d}%):      {t_sparse:.3f} ms  ({speedup:.2f}x vs dense)")

# ── With prepare_sparse (realistic: per-doc importance) ──
print(f"\n{'=' * 65}")
print("REALISTIC: prepare_sparse + flash_maxsim_sparse")
print("=" * 65)

for B, Lq, Ld, d, label in [
    (1000, 1024, 1024, 128, "Visual (Lq=1024, Ld=1024)"),
    (5000, 1024, 1024, 128, "Visual large (Lq=1024, B=5K)"),
]:
    Q = F.normalize(torch.randn(Lq, d, device='cuda', dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(B, Ld, d, device='cuda', dtype=torch.float16), dim=-1)
    importance = torch.randn(B, Lq, device='cuda')

    wup = 5; iters = 20
    t_dense = bench(flash_maxsim, Q, D, warmup=wup, n=iters)

    print(f"\n  {label}")
    print(f"    Dense:             {t_dense:.3f} ms")

    for keep in [0.5, 0.25]:
        Q_sorted, qlens, order = prepare_sparse(Q, importance, keep_ratio=keep)
        # Time only the kernel (prepare_sparse is one-time preprocessing)
        t_sparse = bench(flash_maxsim_sparse, Q_sorted, D, qlens, warmup=wup, n=iters)
        avg_ql = qlens.float().mean().item()
        speedup = t_dense / t_sparse
        pct = int(keep * 100)
        print(f"    Sparse ({pct:2d}%):      {t_sparse:.3f} ms  ({speedup:.2f}x)  avg_ql={avg_ql:.0f}/{Lq}")

print(f"\n{'=' * 65}")
print("DONE")
print("=" * 65)
