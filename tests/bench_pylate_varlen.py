"""Inference-only varlen: pylate vs flash-maxsim on three patterns.

Patterns:
  (1) Pairwise: list of (Q_i, D_i) pairs, all different lengths.
      pylate `colbert_scores_pairwise` (Python loop) vs our `flash_maxsim_varlen`.
  (2) Many-to-many: Nq queries × B shared docs with masks.
      pylate `colbert_scores` (padded + multiply by mask) vs our
      `flash_maxsim_batched(..., doc_lengths, query_lengths)`.
  (3) Single query × many docs (retrieval): 1 query vs B variable-length docs.
      pylate `colbert_scores` (mask) vs our `flash_maxsim(Q, D, doc_lengths=...)`.
"""
import sys, time, gc, torch
import torch.nn.functional as F

sys.path.insert(0, '/path/to/data')
sys.path.insert(0, '/path/to/data')

from flash_maxsim import (
    flash_maxsim, flash_maxsim_batched, flash_maxsim_varlen, pack_pairs,
)
from pylate.scores import colbert_scores, colbert_scores_pairwise

device = 'cuda'
DTYPE = torch.float16


def bench(fn, warmup=3, runs=5):
    for _ in range(warmup): fn()
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    ts = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return ts[len(ts)//2]


def varlen_lists(B, lq_range, ld_range, d=128, ld_dist='uniform', seed=0):
    torch.manual_seed(seed)
    if ld_dist == 'skewed':
        lo, hi = ld_range
        avg = lo + (hi - lo) * 0.15
        lens_d = torch.poisson(torch.full((B,), float(avg))).int().clamp(lo, hi).tolist()
    else:
        lens_d = torch.randint(ld_range[0], ld_range[1] + 1, (B,)).tolist()
    lens_q = torch.randint(lq_range[0], lq_range[1] + 1, (B,)).tolist()
    q_list = [F.normalize(torch.randn(l, d, dtype=DTYPE, device=device), dim=-1) for l in lens_q]
    d_list = [F.normalize(torch.randn(l, d, dtype=DTYPE, device=device), dim=-1) for l in lens_d]
    return q_list, d_list, lens_q, lens_d


def pad_to_max(embs, max_len, d, dtype=DTYPE):
    """Pad list of [li, d] to [N, max_len, d] with zeros."""
    N = len(embs)
    out = torch.zeros(N, max_len, d, dtype=dtype, device=device)
    mask = torch.zeros(N, max_len, dtype=dtype, device=device)
    for i, e in enumerate(embs):
        out[i, :e.shape[0]] = e
        mask[i, :e.shape[0]] = 1.0
    return out, mask


print(f"GPU: {torch.cuda.get_device_name(0)}\n")

# ---------------------------------------------------------------------------
# (1) Pairwise — pylate Python loop vs flash varlen one-shot
# ---------------------------------------------------------------------------
print("=" * 95)
print(" (1) Pairwise:  pylate.colbert_scores_pairwise   vs   flash_maxsim_varlen")
print("=" * 95)
print(f"  {'regime':<32} {'B':>6}  {'pyt_ms':>9}  {'flash_ms':>9}  {'speedup':>8}  {'err':>9}")

CASES_PAIR = [
    ("ColBERT skewed (HotpotQA-like)",  500,  (20, 32), (5, 175), 'skewed'),
    ("ColBERT skewed",                 5000,  (20, 32), (5, 175), 'skewed'),
    ("ColBERT uniform",                1000,  (32, 32), (50, 300), 'uniform'),
    ("ColPali uniform",                 500,  (512, 1024), (256, 1024), 'uniform'),
]
for tag, B, lqr, ldr, dist in CASES_PAIR:
    q_list, d_list, lens_q, lens_d = varlen_lists(B, lqr, ldr, ld_dist=dist)

    # pylate: stack to padded [B, max_lq, d] / [B, max_ld, d] (it expects same tensor shape)
    max_lq, max_ld = max(lens_q), max(lens_d)
    Q_pad, _ = pad_to_max(q_list, max_lq, 128)
    D_pad, _ = pad_to_max(d_list, max_ld, 128)

    pyt_ms = bench(lambda: colbert_scores_pairwise(Q_pad, D_pad))

    # flash: pack
    Q_pk, D_pk, cu_q, cu_d, mq, md = pack_pairs(q_list, d_list)
    flash_ms = bench(lambda: flash_maxsim_varlen(Q_pk, D_pk, cu_q, cu_d, mq, md))

    s_pyt = colbert_scores_pairwise(Q_pad, D_pad)
    s_fl  = flash_maxsim_varlen(Q_pk, D_pk, cu_q, cu_d, mq, md)
    err = (s_pyt.float() - s_fl.float()).abs().max().item() / max(s_pyt.abs().max().item(), 1e-9)
    print(f"  {tag:<32} {B:>6,}  {pyt_ms:>7.2f}ms  {flash_ms:>7.2f}ms  {pyt_ms/flash_ms:>6.1f}×  {err:>9.2e}")

# ---------------------------------------------------------------------------
# (2) Many-to-many varlen — Nq × B with masks
# ---------------------------------------------------------------------------
print()
print("=" * 95)
print(" (2) Many-to-many varlen:  pylate.colbert_scores(qmask, dmask)   vs   flash_maxsim_batched")
print("=" * 95)
print(f"  {'regime':<32} {'Nq×B':>10}  {'pyt_ms':>9}  {'flash_ms':>9}  {'speedup':>8}  {'err':>9}")

CASES_MANY = [
    ("ColBERT uniform",  64, 64, (24, 32), (100, 180)),
    ("ColBERT skewed",  128, 128, (24, 32), (5, 175)),
    ("ColPali (small)",   8,  8, (512, 1024), (256, 1024)),
    ("ColPali (B=32)",   32, 32, (512, 1024), (256, 1024)),
]
for tag, Nq, B, lqr, ldr in CASES_MANY:
    torch.manual_seed(Nq * 1000 + B)
    lens_q = torch.randint(lqr[0], lqr[1] + 1, (Nq,)).tolist()
    lens_d = torch.randint(ldr[0], ldr[1] + 1, (B,)).tolist()
    q_list = [F.normalize(torch.randn(l, 128, dtype=DTYPE, device=device), dim=-1) for l in lens_q]
    d_list = [F.normalize(torch.randn(l, 128, dtype=DTYPE, device=device), dim=-1) for l in lens_d]
    max_lq, max_ld = max(lens_q), max(lens_d)
    Q_pad, q_mask = pad_to_max(q_list, max_lq, 128)
    D_pad, d_mask = pad_to_max(d_list, max_ld, 128)
    q_lens_t = torch.tensor(lens_q, dtype=torch.int32, device=device)
    d_lens_t = torch.tensor(lens_d, dtype=torch.int32, device=device)

    pyt_ms = bench(lambda: colbert_scores(Q_pad, D_pad, q_mask, d_mask))
    flash_ms = bench(lambda: flash_maxsim_batched(
        Q_pad, D_pad, doc_lengths=d_lens_t, query_lengths=q_lens_t, shared_docs=True))

    s_pyt = colbert_scores(Q_pad, D_pad, q_mask, d_mask)
    s_fl  = flash_maxsim_batched(Q_pad, D_pad,
                                  doc_lengths=d_lens_t, query_lengths=q_lens_t, shared_docs=True)
    err = (s_pyt.float() - s_fl.float()).abs().max().item() / max(s_pyt.abs().max().item(), 1e-9)
    print(f"  {tag:<32} {f'{Nq}×{B}':>10}  {pyt_ms:>7.2f}ms  {flash_ms:>7.2f}ms  "
          f"{pyt_ms/flash_ms:>6.1f}×  {err:>9.2e}")

# ---------------------------------------------------------------------------
# (3) Single query × many docs (retrieval)
# ---------------------------------------------------------------------------
print()
print("=" * 95)
print(" (3) 1q × B docs (retrieval):  pylate vs flash_maxsim")
print("=" * 95)
print(f"  {'regime':<32} {'B':>8}  {'pyt_ms':>9}  {'flash_ms':>9}  {'speedup':>8}  {'err':>9}")

CASES_ONE = [
    ("ColBERT skewed",      10_000,  32, (5, 175)),
    ("ColBERT skewed",      100_000, 32, (5, 175)),
    ("ColPali uniform",       1_000, 1024, (256, 1024)),
]
for tag, B, Lq, ldr in CASES_ONE:
    torch.manual_seed(B)
    Q = F.normalize(torch.randn(Lq, 128, dtype=DTYPE, device=device), dim=-1)
    Q_for_pyt = Q.unsqueeze(0)  # [1, Lq, d] for pylate's einsum
    lens_d = torch.randint(ldr[0], ldr[1] + 1, (B,)).tolist()
    max_ld = max(lens_d)
    d_list = [F.normalize(torch.randn(l, 128, dtype=DTYPE, device=device), dim=-1) for l in lens_d]
    D_pad, d_mask = pad_to_max(d_list, max_ld, 128)
    d_lens_t = torch.tensor(lens_d, dtype=torch.int32, device=device)

    try:
        pyt_ms = bench(lambda: colbert_scores(Q_for_pyt, D_pad, documents_mask=d_mask))
    except torch.cuda.OutOfMemoryError:
        pyt_ms = float('nan')
    flash_ms = bench(lambda: flash_maxsim(Q, D_pad, doc_lengths=d_lens_t))

    if pyt_ms == pyt_ms:
        s_pyt = colbert_scores(Q_for_pyt, D_pad, documents_mask=d_mask).squeeze(0)
        s_fl  = flash_maxsim(Q, D_pad, doc_lengths=d_lens_t)
        err = (s_pyt.float() - s_fl.float()).abs().max().item() / max(s_pyt.abs().max().item(), 1e-9)
        sp_str = f"{pyt_ms/flash_ms:>6.1f}×"
        pyt_str = f"{pyt_ms:>7.2f}ms"
    else:
        err = float('nan')
        sp_str = "    OOM"
        pyt_str = f"{'OOM':>9}"
    print(f"  {tag:<32} {B:>8,}  {pyt_str}  {flash_ms:>7.2f}ms  {sp_str}  {err:>9.2e}")

print("\nDone.")
