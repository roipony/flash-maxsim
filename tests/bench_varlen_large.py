"""Focused varlen benchmark at larger N to see true speedup scaling."""
import sys, time, torch
import torch.nn.functional as F
sys.path.insert(0, '.')
from flash_maxsim import flash_maxsim_varlen, pack_pairs

device = 'cuda'
D_DIM = 128
DTYPE = torch.float16

def bench(fn, warmup=10, runs=20):
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
    return times[len(times) // 2]

gpu = torch.cuda.get_device_name(0)
print(f"GPU: {gpu}\n")

def run_varlen(regime, lq_range, ld_range, n_list, ld_dist='uniform'):
    lq_lo, lq_hi = lq_range
    ld_lo, ld_hi = ld_range

    def _sample_lens(n, lo, hi, dist):
        if dist == 'skewed':
            avg = lo + (hi - lo) * 0.15
            lens = torch.poisson(torch.full((n,), avg)).int()
            return torch.clamp(lens, lo, hi).tolist()
        return torch.randint(lo, hi + 1, (n,)).tolist()

    dist_label = f'skewed(avg≈{ld_lo + (ld_hi-ld_lo)*0.15:.0f})' if ld_dist == 'skewed' else f'U[{ld_lo},{ld_hi}]'
    print(f"{'='*80}")
    print(f" {regime}  Lq~U[{lq_lo},{lq_hi}]  Ld~{dist_label}")
    print(f"{'='*80}")
    print(f"  {'N':>9}  {'naive_ms':>9}  {'varlen_ms':>10}  {'speedup':>8}  {'D_saved':>8}  {'mem_naive':>10}  {'mem_varlen':>10}")
    print(f"  {'-'*75}")

    for N in n_list:
        torch.manual_seed(42 + N)
        q_lens = torch.randint(lq_lo, lq_hi + 1, (N,)).tolist()
        d_lens = _sample_lens(N, ld_lo, ld_hi, ld_dist)
        max_lq = max(q_lens)
        max_ld = max(d_lens)
        avg_ld = sum(d_lens) / N

        try:
            q_embs = [F.normalize(torch.randn(l, D_DIM, dtype=DTYPE, device=device), dim=-1) for l in q_lens]
            d_embs = [F.normalize(torch.randn(l, D_DIM, dtype=DTYPE, device=device), dim=-1) for l in d_lens]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  {N:>9,}  OOM creating embeddings")
            continue

        # Pack for varlen
        Q_pk, D_pk, cu_q, cu_d, mq, md = pack_pairs(q_embs, d_embs)

        total_d = sum(d_lens)
        padded_d = N * max_ld
        d_saved = 1 - total_d / padded_d

        # Sim matrix size for padded naive: N * max_lq * max_ld * 2 bytes (fp16)
        sim_matrix_bytes = N * max_lq * max_ld * 2
        sim_matrix_mb = sim_matrix_bytes / (1024**2)

        # Padded batched naive
        naive_ms = None
        try:
            Q_bat = torch.zeros(N, max_lq, D_DIM, dtype=DTYPE, device=device)
            D_bat = torch.zeros(N, max_ld, D_DIM, dtype=DTYPE, device=device)
            for i in range(N):
                Q_bat[i, :q_lens[i]] = q_embs[i]
                D_bat[i, :d_lens[i]] = d_embs[i]

            def naive_batched():
                S = torch.bmm(Q_bat, D_bat.transpose(1, 2))
                return S.max(dim=2).values.sum(dim=1)

            naive_ms = bench(naive_batched)
            del Q_bat, D_bat
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()

        # Varlen kernel
        varlen_ms = bench(lambda: flash_maxsim_varlen(Q_pk, D_pk, cu_q, cu_d, mq, md))

        sp_str = f"{naive_ms/varlen_ms:.1f}x" if naive_ms else "—"
        n_str = f"{naive_ms:.2f}" if naive_ms else "OOM"

        print(f"  {N:>9,}  {n_str:>9}  {varlen_ms:>10.2f}  {sp_str:>8}  {d_saved*100:>7.0f}%  {sim_matrix_mb:>9.0f}MB  {0:>10}")

        del q_embs, d_embs, Q_pk, D_pk
        torch.cuda.empty_cache()
    print()

# ColBERT uniform
run_varlen('ColBERT-uniform', (20, 32), (50, 300),
           [1_000, 5_000, 10_000, 20_000, 50_000, 100_000])

# ColPali uniform
run_varlen('ColPali-uniform', (512, 1024), (256, 1024),
           [100, 500, 1_000, 2_000, 5_000, 10_000])

# ColBERT skewed
run_varlen('ColBERT-skewed', (20, 32), (5, 300),
           [1_000, 5_000, 10_000, 20_000, 50_000, 100_000], ld_dist='skewed')

# ColPali skewed
run_varlen('ColPali-skewed', (512, 1024), (50, 1024),
           [500, 1_000, 2_000, 5_000, 10_000], ld_dist='skewed')

print("Done.")
