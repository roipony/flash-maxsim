"""bench_full.py — Flash-MaxSim benchmark suite.

Two methods:  torch_naive (fp16 cuBLAS)  vs  flash_maxsim

Regimes:
  ColBERT         Lq=32,   Ld=180   — standard text retrieval
  ColBERT-LongD   Lq=32,   Ld=1024  — long documents
  ColPali         Lq=1024, Ld=1024  — vision-language
  ColPali-ShortD  Lq=1024, Ld=512

Extras:  Batch-Q, Varlen, Packed-D

Metrics: latency (ms), peak GPU memory (MB), speedup, correctness.

Usage:
    python tests/bench_full.py          # full suite
    python tests/bench_full.py --quick  # fewer B values
"""

import sys, time, gc, argparse
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')

from flash_maxsim import (
    flash_maxsim, flash_maxsim_batched, maxsim_naive,
    flash_maxsim_varlen, pack_pairs,
    flash_maxsim_packed, pack_docs,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
device   = 'cuda'
D_DIM    = 128
DTYPE    = torch.float16
N_WARMUP = 5
N_RUNS   = 20
RANK_K   = 20

parser = argparse.ArgumentParser()
parser.add_argument('--quick', action='store_true')
args, _ = parser.parse_known_args()

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _bench(fn):
    for _ in range(N_WARMUP):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(N_RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def _peak_mb(fn):
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(device) / 1e6


def _is_oom(exc):
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    s = str(exc).lower()
    if isinstance(exc, RuntimeError) and 'out of memory' in s:
        return True
    if 'outofresources' in type(exc).__name__ or 'out of resource' in s:
        return True
    return False


def _check(ref, out):
    r, o = ref.reshape(-1), out.reshape(-1)
    err = (r - o).abs().max().item()
    k = min(RANK_K, r.shape[0])
    top_ref = set(r.argsort(descending=True)[:k].tolist())
    top_out = set(o.argsort(descending=True)[:k].tolist())
    overlap = len(top_ref & top_out) / k
    return 'OK' if (err < 0.15 and overlap >= 0.9) else 'MISMATCH'


def run_safe(fn):
    """Returns (ms, peak_mb) or ('OOM', None)."""
    try:
        ms = _bench(fn)
        mb = _peak_mb(fn)
        return ms, mb
    except Exception as e:
        if _is_oom(e):
            torch.cuda.empty_cache()
            gc.collect()
            return 'OOM', None
        raise


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------

def run_regime(regime, Lq, Ld, B_list):
    print()
    print('=' * 75)
    print(f' {regime}  Lq={Lq}  Ld={Ld}  d={D_DIM}')
    print('=' * 75)
    print(f"  {'B':>8}  {'method':<14}  {'ms':>8}  {'MB':>7}  "
          f"{'speedup':>8}  {'mem_save':>9}  {'correct'}")
    print(f"  {'-'*67}")

    for B in B_list:
        try:
            Q = F.normalize(torch.randn(Lq, D_DIM, dtype=DTYPE, device=device), dim=-1)
            D = F.normalize(torch.randn(B, Ld, D_DIM, dtype=DTYPE, device=device), dim=-1)
        except Exception as e:
            if _is_oom(e):
                torch.cuda.empty_cache()
                print(f"  {B:>8,}  alloc OOM")
                continue
            raise

        # naive
        n_ms, n_mb = run_safe(lambda: maxsim_naive(Q, D))
        ref_out = maxsim_naive(Q, D) if n_ms != 'OOM' else None
        n_ms_s = f'{n_ms:8.2f}' if isinstance(n_ms, float) else '     OOM'
        n_mb_s = f'{n_mb:7.0f}' if n_mb else '    OOM'
        print(f"  {B:>8,}  {'torch_naive':<14}  {n_ms_s}  {n_mb_s}  "
              f"{'ref':>8}  {'ref':>9}  ref")

        # flash
        f_ms, f_mb = run_safe(lambda: flash_maxsim(Q, D))
        if isinstance(f_ms, float) and ref_out is not None:
            cor = _check(ref_out, flash_maxsim(Q, D))
        elif f_ms == 'OOM':
            cor = 'OOM'
        else:
            cor = '—'

        sp = f'{n_ms/f_ms:7.2f}x' if isinstance(n_ms, float) and isinstance(f_ms, float) else '      —'
        ms = f'{n_mb - f_mb:6.0f}MB' if isinstance(n_mb, float) and isinstance(f_mb, float) else '      —'
        f_ms_s = f'{f_ms:8.2f}' if isinstance(f_ms, float) else '     OOM'
        f_mb_s = f'{f_mb:7.0f}' if f_mb else '    OOM'
        print(f"  {B:>8,}  {'flash':<14}  {f_ms_s}  {f_mb_s}  "
              f"{sp:>8}  {ms:>9}  {cor}")

        del Q, D
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Batch-Q
# ---------------------------------------------------------------------------

def run_batchq(regime, Lq, Ld, NQ, b_list):
    print()
    print('=' * 75)
    print(f' Batch-Q  {regime}  Lq={Lq}  Ld={Ld}  NQ={NQ}')
    print('=' * 75)
    print(f"  {'B':>8}  {'serial_ms':>10}  {'batched_ms':>11}  "
          f"{'vs_serial':>10}  {'correct'}")
    print(f"  {'-'*55}")

    for B in b_list:
        try:
            D = F.normalize(torch.randn(B, Ld, D_DIM, dtype=DTYPE, device=device), dim=-1)
            Qb = F.normalize(torch.randn(NQ, Lq, D_DIM, dtype=DTYPE, device=device), dim=-1)
        except Exception as e:
            if _is_oom(e):
                torch.cuda.empty_cache()
                print(f"  {B:>8,}  OOM")
                continue
            raise

        def serial_fn():
            return torch.stack([flash_maxsim(Qb[i], D) for i in range(NQ)])

        s_ms, _ = run_safe(serial_fn)
        b_ms, b_mb = run_safe(lambda: flash_maxsim_batched(Qb, D))

        if isinstance(s_ms, float) and isinstance(b_ms, float):
            ref = serial_fn()
            out = flash_maxsim_batched(Qb, D)
            cor = _check(ref, out)
            vs = f'{s_ms/b_ms:9.2f}x'
        else:
            cor = 'OOM' if b_ms == 'OOM' else '—'
            vs = '        —'

        s_ms_s = f'{s_ms:10.2f}' if isinstance(s_ms, float) else '       OOM'
        b_ms_s = f'{b_ms:11.2f}' if isinstance(b_ms, float) else '        OOM'
        print(f"  {B:>8,}  {s_ms_s}  {b_ms_s}  {vs}  {cor}")

        del D, Qb
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Varlen
# ---------------------------------------------------------------------------

def run_varlen(regime, lq_range, ld_range, n_pairs_list, ld_dist='uniform'):
    """ld_dist: 'uniform' or 'skewed' (Poisson-like, realistic)."""
    lq_lo, lq_hi = lq_range
    ld_lo, ld_hi = ld_range

    def _sample_lens(n, lo, hi, dist):
        if dist == 'skewed':
            # Poisson-like: most docs short, few long (like HotpotQA)
            avg = lo + (hi - lo) * 0.15  # avg near 15th percentile
            lens = torch.poisson(torch.full((n,), avg)).int()
            lens = torch.clamp(lens, lo, hi)
            return lens.tolist()
        return torch.randint(lo, hi + 1, (n,)).tolist()

    dist_label = f'Ld~Poisson(avg≈{ld_lo + (ld_hi-ld_lo)*0.15:.0f},max={ld_hi})' if ld_dist == 'skewed' \
        else f'Ld~U[{ld_lo},{ld_hi}]'
    print()
    print('=' * 75)
    print(f' Varlen  {regime}  Lq~U[{lq_lo},{lq_hi}]  {dist_label}')
    print('=' * 75)

    # Correctness on small sample
    CORR_N = 50
    torch.manual_seed(42)
    cq = torch.randint(lq_lo, lq_hi + 1, (CORR_N,)).tolist()
    cd = _sample_lens(CORR_N, ld_lo, ld_hi, ld_dist)
    cq_embs = [F.normalize(torch.randn(l, D_DIM, dtype=DTYPE, device=device), dim=-1) for l in cq]
    cd_embs = [F.normalize(torch.randn(l, D_DIM, dtype=DTYPE, device=device), dim=-1) for l in cd]
    Q_pk, D_pk, cu_q, cu_d, mq, md = pack_pairs(cq_embs, cd_embs)
    ref = torch.stack([flash_maxsim(cq_embs[i], cd_embs[i].unsqueeze(0)).squeeze() for i in range(CORR_N)])
    out = flash_maxsim_varlen(Q_pk, D_pk, cu_q, cu_d, mq, md)
    err = (ref - out).abs().max().item()
    cor = 'OK' if err < 0.1 else 'MISMATCH'
    print(f"  correctness (N={CORR_N}): max_err={err:.4f} {cor}")
    del cq_embs, cd_embs, Q_pk, D_pk, cu_q, cu_d, ref, out
    torch.cuda.empty_cache()

    print(f"  {'N':>9}  {'naive_ms':>9}  {'varlen_ms':>10}  {'speedup':>8}  "
          f"{'D_saved':>8}  {'correct'}")
    print(f"  {'-'*58}")

    for N in n_pairs_list:
        torch.manual_seed(42 + N)
        q_lens = torch.randint(lq_lo, lq_hi + 1, (N,)).tolist()
        d_lens = _sample_lens(N, ld_lo, ld_hi, ld_dist)
        max_lq = max(q_lens)
        max_ld = max(d_lens)

        try:
            q_embs = [F.normalize(torch.randn(l, D_DIM, dtype=DTYPE, device=device), dim=-1) for l in q_lens]
            d_embs = [F.normalize(torch.randn(l, D_DIM, dtype=DTYPE, device=device), dim=-1) for l in d_lens]
        except Exception as e:
            if _is_oom(e):
                torch.cuda.empty_cache()
                print(f"  {N:>9,}  OOM")
                continue
            raise

        Q_packed, D_packed, cu_q, cu_d, mq, md = pack_pairs(q_embs, d_embs)
        total_d = sum(d_lens)
        mem_save = f'{(1 - total_d / (N * max_ld)) * 100:.0f}%'

        # Naive baseline: pad all pairs into batched tensors, one bmm call
        try:
            Q_bat = torch.zeros(N, max_lq, D_DIM, dtype=DTYPE, device=device)
            D_bat = torch.zeros(N, max_ld, D_DIM, dtype=DTYPE, device=device)
            for i in range(N):
                Q_bat[i, :q_lens[i]] = q_embs[i]
                D_bat[i, :d_lens[i]] = d_embs[i]

            def naive_batched():
                S = torch.bmm(Q_bat, D_bat.transpose(1, 2))  # [N, max_lq, max_ld]
                return S.max(dim=2).values.sum(dim=1)         # [N]

            naive_ms, _ = run_safe(naive_batched)
            del Q_bat, D_bat
        except Exception as e:
            if _is_oom(e):
                naive_ms = 'OOM'
                torch.cuda.empty_cache()
            else:
                raise

        del q_embs, d_embs

        # Varlen kernel
        vl_ms, vl_mb = run_safe(lambda: flash_maxsim_varlen(Q_packed, D_packed, cu_q, cu_d, mq, md))

        sp = f'{naive_ms/vl_ms:7.1f}x' if isinstance(naive_ms, float) and isinstance(vl_ms, float) else '      —'
        n_s = f'{naive_ms:9.2f}' if isinstance(naive_ms, float) else '      OOM'
        v_s = f'{vl_ms:10.2f}' if isinstance(vl_ms, float) else '       OOM'
        print(f"  {N:>9,}  {n_s}  {v_s}  {sp}  {mem_save:>8}  {cor}")

        del Q_packed, D_packed, cu_q, cu_d
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Packed-D
# ---------------------------------------------------------------------------

def run_packed(regime, Lq, ld_range, B_list):
    ld_lo, ld_hi = ld_range
    print()
    print('=' * 75)
    print(f' Packed-D  {regime}  Lq={Lq}  Ld~U[{ld_lo},{ld_hi}]')
    print('=' * 75)
    print(f"  {'B':>8}  {'padded_ms':>10}  {'packed_ms':>10}  "
          f"{'speedup':>8}  {'D_saved':>8}  {'correct'}")
    print(f"  {'-'*58}")

    for B in B_list:
        torch.manual_seed(42)
        d_lens = torch.randint(ld_lo, ld_hi + 1, (B,)).tolist()
        max_ld = max(d_lens)

        try:
            d_embs = [F.normalize(torch.randn(l, D_DIM, dtype=DTYPE, device=device), dim=-1) for l in d_lens]
            Q = F.normalize(torch.randn(Lq, D_DIM, dtype=DTYPE, device=device), dim=-1)
        except Exception as e:
            if _is_oom(e):
                torch.cuda.empty_cache()
                print(f"  {B:>8,}  OOM")
                continue
            raise

        D_packed, cu_d, mld = pack_docs(d_embs)
        del d_embs
        torch.cuda.empty_cache()

        # Padded D
        try:
            D_padded = torch.zeros(B, max_ld, D_DIM, dtype=DTYPE, device=device)
            off = 0
            for i, l in enumerate(d_lens):
                D_padded[i, :l] = D_packed[off:off+l]
                off += l
            doc_lengths_padded = torch.tensor(d_lens, device=device, dtype=torch.int32)
        except Exception as e:
            if _is_oom(e):
                D_padded = None
                torch.cuda.empty_cache()
            else:
                raise

        pad_ms = None
        if D_padded is not None:
            pad_ms, _ = run_safe(lambda: flash_maxsim(Q, D_padded, doc_lengths=doc_lengths_padded))

        pk_ms, _ = run_safe(lambda: flash_maxsim_packed(Q, D_packed, cu_d))

        # Correctness
        if isinstance(pad_ms, float) and isinstance(pk_ms, float):
            ref = flash_maxsim(Q, D_padded, doc_lengths=doc_lengths_padded)
            out = flash_maxsim_packed(Q, D_packed, cu_d)
            cor = _check(ref, out)
        else:
            cor = 'OOM' if pk_ms == 'OOM' else '—'

        total_d = sum(d_lens)
        mem_save = f'{(1 - total_d / (B * max_ld)) * 100:.0f}%'

        sp = f'{pad_ms/pk_ms:7.2f}x' if isinstance(pad_ms, float) and isinstance(pk_ms, float) else '      —'
        pad_s = f'{pad_ms:10.2f}' if isinstance(pad_ms, float) else '       OOM'
        pk_s = f'{pk_ms:10.2f}' if isinstance(pk_ms, float) else '       OOM'
        print(f"  {B:>8,}  {pad_s}  {pk_s}  {sp:>8}  {mem_save:>8}  {cor}")

        del Q, D_packed, cu_d
        if D_padded is not None:
            del D_padded, doc_lengths_padded
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

print()
print('=' * 75)
print(f' Flash-MaxSim Benchmark Suite')
print(f' Device : {torch.cuda.get_device_name(0)}')
print(f' d={D_DIM}  dtype={DTYPE}  warmup={N_WARMUP}  runs={N_RUNS}')
try:
    import triton; print(f' Triton : {triton.__version__}')
except: pass
print('=' * 75)

if args.quick:
    COLBERT_B    = [1_000, 10_000, 100_000]
    COLBERT_LD_B = [100, 1_000, 10_000]
    COLPALI_B    = [100, 500, 1_000, 5_000]
    VARLEN_N     = [1_000, 5_000]
    BATCHQ_B_CB  = [10, 100, 1_000, 10_000]
    BATCHQ_B_CP  = [10, 50, 100, 500]
    PACKED_B_CB  = [1_000, 10_000]
    PACKED_B_CP  = [100, 500]
else:
    COLBERT_B    = [1_000, 10_000, 100_000, 500_000]
    COLBERT_LD_B = [100, 1_000, 10_000, 50_000]
    COLPALI_B    = [100, 500, 1_000, 5_000, 10_000]
    VARLEN_N     = [1_000, 5_000, 20_000]
    BATCHQ_B_CB  = [10, 50, 100, 500, 1_000, 5_000, 10_000]
    BATCHQ_B_CP  = [10, 50, 100, 500, 1_000]
    PACKED_B_CB  = [1_000, 10_000, 100_000]
    PACKED_B_CP  = [100, 500, 1_000]

# Core
run_regime('ColBERT',        Lq=32,   Ld=180,  B_list=COLBERT_B)
run_regime('ColBERT-LongD',  Lq=32,   Ld=1024, B_list=COLBERT_LD_B)
run_regime('ColPali',        Lq=1024, Ld=1024, B_list=COLPALI_B)
run_regime('ColPali-ShortD', Lq=1024, Ld=512,  B_list=COLPALI_B)

# Batch-Q (sweep B — batched wins big at small B, parity at large B)
run_batchq('ColBERT',  Lq=32,   Ld=180,  NQ=16, b_list=BATCHQ_B_CB)
run_batchq('ColPali',  Lq=1024, Ld=1024, NQ=4,  b_list=BATCHQ_B_CP)

# Varlen (uniform)
run_varlen('ColBERT',  lq_range=(20, 32),     ld_range=(50, 300),   n_pairs_list=VARLEN_N)
run_varlen('ColPali',  lq_range=(512, 1024),  ld_range=(256, 1024), n_pairs_list=[500, 2_000])

# Varlen (skewed — realistic: most docs short, few long)
VARLEN_SKEW_N = [1_000, 5_000, 20_000, 100_000] if not args.quick else [1_000, 5_000]
run_varlen('ColBERT-skewed',  lq_range=(20, 32), ld_range=(5, 300),
           n_pairs_list=VARLEN_SKEW_N, ld_dist='skewed')
run_varlen('ColPali-skewed',  lq_range=(512, 1024), ld_range=(50, 1024),
           n_pairs_list=[500, 2_000, 10_000] if not args.quick else [500, 2_000],
           ld_dist='skewed')

# Packed-D
run_packed('ColBERT',  Lq=32,   ld_range=(50, 300),   B_list=PACKED_B_CB)
run_packed('ColPali',  Lq=1024, ld_range=(256, 1024),  B_list=PACKED_B_CP)

print()
print('Done.')
