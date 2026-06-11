"""Diagnose ColPali-uniform varlen anomaly: N=500 naive=5.95ms but N=1000 naive=4.26ms."""
import sys, time, torch
import torch.nn.functional as F
sys.path.insert(0, '.')
from flash_maxsim import flash_maxsim_varlen, pack_pairs

device = 'cuda'
D_DIM = 128
DTYPE = torch.float16

def bench(fn, warmup=10, runs=30):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2], times[0], times[-1]

gpu = torch.cuda.get_device_name(0)
print(f"GPU: {gpu}\n")

lq_lo, lq_hi = 512, 1024
ld_lo, ld_hi = 256, 1024

print("="*90)
print(" ColPali-uniform varlen diagnostic")
print("="*90)

for N in [100, 200, 500, 750, 1000, 1500, 2000, 3000, 5000, 10000]:
    torch.manual_seed(42 + N)
    q_lens = torch.randint(lq_lo, lq_hi + 1, (N,)).tolist()
    d_lens = torch.randint(ld_lo, ld_hi + 1, (N,)).tolist()
    max_lq = max(q_lens)
    max_ld = max(d_lens)
    avg_lq = sum(q_lens) / N
    avg_ld = sum(d_lens) / N
    total_d = sum(d_lens)
    padded_d = N * max_ld
    d_saved = 1 - total_d / padded_d

    sim_matrix_mb = N * max_lq * max_ld * 2 / (1024**2)
    padded_q_mb = N * max_lq * D_DIM * 2 / (1024**2)
    padded_d_mb = N * max_ld * D_DIM * 2 / (1024**2)

    print(f"\n--- N={N:,} ---")
    print(f"  max_lq={max_lq}  avg_lq={avg_lq:.0f}  max_ld={max_ld}  avg_ld={avg_ld:.0f}")
    print(f"  d_saved={d_saved*100:.1f}%  sim_matrix={sim_matrix_mb:.0f}MB  Q_pad={padded_q_mb:.0f}MB  D_pad={padded_d_mb:.0f}MB")

    try:
        q_embs = [F.normalize(torch.randn(l, D_DIM, dtype=DTYPE, device=device), dim=-1) for l in q_lens]
        d_embs = [F.normalize(torch.randn(l, D_DIM, dtype=DTYPE, device=device), dim=-1) for l in d_lens]
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"  OOM creating embeddings")
        continue

    # Pack for varlen
    Q_pk, D_pk, cu_q, cu_d, mq, md = pack_pairs(q_embs, d_embs)

    # Varlen kernel
    vl_med, vl_min, vl_max = bench(lambda: flash_maxsim_varlen(Q_pk, D_pk, cu_q, cu_d, mq, md))
    print(f"  varlen:  median={vl_med:.2f}ms  min={vl_min:.2f}  max={vl_max:.2f}")

    # Padded batched - bmm
    try:
        Q_bat = torch.zeros(N, max_lq, D_DIM, dtype=DTYPE, device=device)
        D_bat = torch.zeros(N, max_ld, D_DIM, dtype=DTYPE, device=device)
        for i in range(N):
            Q_bat[i, :q_lens[i]] = q_embs[i]
            D_bat[i, :d_lens[i]] = d_embs[i]

        def naive_bmm():
            S = torch.bmm(Q_bat, D_bat.transpose(1, 2))
            return S.max(dim=2).values.sum(dim=1)

        def naive_einsum():
            S = torch.einsum('nqd,nld->nql', Q_bat, D_bat)
            return S.max(dim=2).values.sum(dim=1)

        bmm_med, bmm_min, bmm_max = bench(naive_bmm)
        ein_med, ein_min, ein_max = bench(naive_einsum)
        print(f"  bmm:     median={bmm_med:.2f}ms  min={bmm_min:.2f}  max={bmm_max:.2f}  speedup={bmm_med/vl_med:.1f}x")
        print(f"  einsum:  median={ein_med:.2f}ms  min={ein_min:.2f}  max={ein_max:.2f}  speedup={ein_med/vl_med:.1f}x")

        del Q_bat, D_bat
    except torch.cuda.OutOfMemoryError:
        print(f"  naive: OOM")
        torch.cuda.empty_cache()

    # Also test flash_maxsim (padded, not varlen) for reference
    from flash_maxsim import flash_maxsim
    try:
        D_padded = torch.zeros(N, max_ld, D_DIM, dtype=DTYPE, device=device)
        for i in range(N):
            D_padded[i, :d_lens[i]] = d_embs[i]
        # Use first query as representative (Lq=1024)
        Q_single = F.normalize(torch.randn(1024, D_DIM, dtype=DTYPE, device=device), dim=-1)
        flash_med, flash_min, flash_max = bench(lambda: flash_maxsim(Q_single, D_padded))
        print(f"  flash_padded: median={flash_med:.2f}ms  (single Q, padded D)")
        del D_padded, Q_single
    except torch.cuda.OutOfMemoryError:
        print(f"  flash_padded: OOM")
        torch.cuda.empty_cache()

    del q_embs, d_embs, Q_pk, D_pk
    torch.cuda.empty_cache()

print("\nDone.")
