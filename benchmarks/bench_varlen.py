"""Benchmark flash_maxsim_varlen vs naive and padded approaches.

Usage: python benchmarks/bench_varlen.py
"""
import time, torch, torch.nn.functional as F
import random

assert torch.cuda.is_available(), "CUDA required"
print(f"GPU: {torch.cuda.get_device_name()}\n")

from flash_maxsim import maxsim_naive
from flash_maxsim.flash_maxsim_varlen import flash_maxsim_varlen, pack_pairs
from utils import bench_interleaved, compile_high_precision

random.seed(42)


# ── Correctness ──
print("=" * 65)
print("CORRECTNESS: varlen vs naive")
print("=" * 65)

for N, lq_range, ld_range, d, label in [
    (10,  (8, 32),   (100, 500),  128, "small text"),
    (50,  (16, 64),  (300, 1024), 128, "mixed"),
    (100, (32, 128), (512, 2048), 128, "long"),
    (20,  (8, 32),   (100, 300),  96,  "d=96"),
]:
    q_embs = [F.normalize(torch.randn(random.randint(*lq_range), d, device='cuda', dtype=torch.float16), dim=-1) for _ in range(N)]
    d_embs = [F.normalize(torch.randn(random.randint(*ld_range), d, device='cuda', dtype=torch.float16), dim=-1) for _ in range(N)]

    ref = torch.stack([maxsim_naive(q, dd.unsqueeze(0)).squeeze(0) for q, dd in zip(q_embs, d_embs)])
    Q_pk, D_pk, cu_q, cu_d, mlq, mld = pack_pairs(q_embs, d_embs)
    out = flash_maxsim_varlen(Q_pk, D_pk, cu_q, cu_d, mlq, mld)
    err = (ref - out).abs().max().item()
    rank_ok = (ref.argsort(descending=True)[:5] == out.argsort(descending=True)[:5]).all()
    print(f"  {label:20s} N={N:3d}: err={err:.2e}  rank={'OK' if rank_ok else 'FAIL'}")

# ── Speed: varlen vs padded vs naive ──
print(f"\n{'=' * 65}")
print("SPEED: varlen vs flash_pairs vs naive loop")
print("=" * 65)

for N, lq_range, ld_range, d, label in [
    (100,  (8, 32),    (100, 500),   128, "100 text pairs"),
    (500,  (8, 32),    (100, 500),   128, "500 text pairs"),
    (100,  (16, 64),   (300, 1024),  128, "100 mixed pairs"),
    (500,  (16, 64),   (300, 1024),  128, "500 mixed pairs"),
    (100,  (32, 128),  (512, 2048),  128, "100 long pairs"),
    (500,  (32, 128),  (512, 2048),  128, "500 long pairs"),
    (1000, (8, 32),    (100, 500),   128, "1000 text pairs"),
    (1000, (16, 64),   (300, 1024),  128, "1000 mixed pairs"),
]:
    q_embs = [F.normalize(torch.randn(random.randint(*lq_range), d, device='cuda', dtype=torch.float16), dim=-1) for _ in range(N)]
    d_embs = [F.normalize(torch.randn(random.randint(*ld_range), d, device='cuda', dtype=torch.float16), dim=-1) for _ in range(N)]

    # Pre-pack (not timed)
    Q_pk, D_pk, cu_q, cu_d, mlq, mld = pack_pairs(q_embs, d_embs)

    # Varlen (kernel only — packing already done)

    # Varlen (with packing — full e2e)
    def varlen_e2e():
        Qp, Dp, cq, cd, mq, md = pack_pairs(q_embs, d_embs)
        return flash_maxsim_varlen(Qp, Dp, cq, cd, mq, md)

    # Naive loop
    def naive_loop():
        return torch.stack([maxsim_naive(q, dd.unsqueeze(0)).squeeze(0) for q, dd in zip(q_embs, d_embs)])
    naive_compiled = compile_high_precision(naive_loop)

    t_varlen, t_varlen_e2e, t_naive, t_compiled = bench_interleaved([flash_maxsim_varlen, varlen_e2e, naive_loop, naive_compiled], [[Q_pk, D_pk, cu_q, cu_d, mlq, mld], [], [], []])

    print(f"  {label:25s}: naive={t_naive:8.2f}ms compiled={t_compiled:8.2f}ms varlen_e2e={t_varlen_e2e:8.2f}ms  varlen_kernel={t_varlen:7.3f}ms  naive/kernel={t_naive/t_varlen:6.0f}x  compiled/kernel={t_compiled/t_varlen:6.0f}x")

# ── Memory comparison ──
print(f"\n{'=' * 65}")
print("MEMORY: varlen (packed) vs padded")
print("=" * 65)

for N, lq_range, ld_range, d, label in [
    (500,  (8, 64),   (100, 2048),  128, "500 pairs, mixed lengths"),
    (1000, (8, 64),   (100, 2048),  128, "1000 pairs, mixed lengths"),
]:
    q_embs = [F.normalize(torch.randn(random.randint(*lq_range), d, device='cuda', dtype=torch.float16), dim=-1) for _ in range(N)]
    d_embs = [F.normalize(torch.randn(random.randint(*ld_range), d, device='cuda', dtype=torch.float16), dim=-1) for _ in range(N)]

    # Packed size
    Q_pk, D_pk, cu_q, cu_d, mlq, mld = pack_pairs(q_embs, d_embs)
    packed_mb = (Q_pk.nbytes + D_pk.nbytes) / 1e6

    # Padded size
    max_lq = max(q.shape[0] for q in q_embs)
    max_ld = max(dd.shape[0] for dd in d_embs)
    padded_mb = (N * max_lq * d * 2 + N * max_ld * d * 2) / 1e6

    avg_lq = sum(q.shape[0] for q in q_embs) / N
    avg_ld = sum(dd.shape[0] for dd in d_embs) / N

    print(f"  {label:40s}: packed={packed_mb:.1f}MB  padded={padded_mb:.1f}MB  {padded_mb/packed_mb:.1f}x less  avg_lq={avg_lq:.0f}/{max_lq}  avg_ld={avg_ld:.0f}/{max_ld}")

    del Q_pk, D_pk, q_embs, d_embs; torch.cuda.empty_cache()

print(f"\n{'=' * 65}")
print("DONE")
print("=" * 65)
