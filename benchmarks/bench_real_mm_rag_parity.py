"""End-to-end scoring parity on REAL-MM-RAG with ColPali and GVE encoders.

The paper currently shows end-to-end retrieval parity on BEIR with
ColBERTv2 (tab:beir_ndcg). The headline motivation is ColPali, but no
end-to-end ColPali table exists. This bench fills that gap: load the
pre-computed ColPali and GVE doc/query embeddings for the four
REAL-MM-RAG subsets, score with two backends, and report:

  - max |Δ| in scores (vs FP32 reference)
  - mean |Δ|
  - Spearman ρ over rankings
  - top-K ranking overlap (K = 5, 10, 20)
  - nDCG@10 if qrels are available (row-i query <-> row-i page)

Backends:
  - naive einsum, chunked over docs to bound memory, FP32 accumulation
    -> the FP32 reference
  - flash_maxsim_batched -> the kernel under test

Output: bench_real_mm_rag_parity_<GPU>.json next to this script.
"""
from __future__ import annotations
import gc, json, os, time
from pathlib import Path
import torch
import torch.nn.functional as F

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
EMBED_ROOT = Path('/path/to/data')

SUBSETS = [
    ('finreport',  '2.7K'),
    ('finslides',  '2.3K'),
    ('techreport', '1.7K'),
    ('techslides', '2K'),
]
ENCODERS = ['colpali', 'gve']

DEV = 'cuda'


def naive_scores_fp32_chunked(Q, D, target_tile_gb=8.0):
    """FP32 reference: einsum + max + sum, double-chunked over queries
    and docs so the [chunk_q, chunk_b, Lq, Ld] sim tile stays under
    `target_tile_gb`. Auto-sizes the chunks from the input shapes —
    works for both ColPali (Nq=100, Lq=50) and GVE (Nq~1K, Lq~110).
    """
    Nq, Lq, dim = Q.shape
    B, Ld, _ = D.shape
    bytes_per_score_cell = Lq * Ld * 4
    cells_budget = int(target_tile_gb * (1024**3) / bytes_per_score_cell)
    # Pick chunks: prefer to keep all queries together when feasible,
    # else split queries evenly so chunk_q*chunk_b ~ cells_budget.
    if Nq * 64 <= cells_budget:
        chunk_q = Nq
        chunk_b = max(1, min(B, cells_budget // Nq))
    else:
        chunk_q = max(1, int((cells_budget) ** 0.5))
        chunk_b = max(1, cells_budget // chunk_q)
    print(f"    chunk_q={chunk_q}, chunk_b={chunk_b}  "
          f"(sim tile ~ {chunk_q * chunk_b * bytes_per_score_cell / 1e9:.2f} GB)", flush=True)
    scores = torch.empty(Nq, B, device=Q.device, dtype=torch.float32)
    Q32 = Q.to(torch.float32)
    for qs in range(0, Nq, chunk_q):
        qe = min(qs + chunk_q, Nq)
        Qc = Q32[qs:qe]
        for ds in range(0, B, chunk_b):
            de = min(ds + chunk_b, B)
            Dc = D[ds:de].to(torch.float32)
            S = torch.einsum('nqd,bld->nbql', Qc, Dc)
            scores[qs:qe, ds:de] = S.max(dim=-1).values.sum(dim=-1)
            del Dc, S
    return scores


def flash_scores(Q, D):
    """flash_maxsim_batched: same operator, FP16 in / FP32 accum / no
    materialised similarity tensor."""
    from flash_maxsim import flash_maxsim_batched
    return flash_maxsim_batched(Q.to(torch.float16), D.to(torch.float16))


def topk_overlap(scores_a, scores_b, k):
    """Mean over queries of |topk(a) ∩ topk(b)| / k."""
    ta = scores_a.topk(k, dim=1).indices.cpu().numpy()
    tb = scores_b.topk(k, dim=1).indices.cpu().numpy()
    agree = 0
    for ra, rb in zip(ta, tb):
        agree += len(set(ra.tolist()) & set(rb.tolist()))
    return agree / (ta.shape[0] * k)


def spearman_per_query(scores_a, scores_b):
    """Mean Spearman ρ over queries — fast vectorised version."""
    # rank scores in [0, B)
    ra = scores_a.argsort(dim=1).argsort(dim=1).float()
    rb = scores_b.argsort(dim=1).argsort(dim=1).float()
    # Pearson on ranks = Spearman
    ra_c = ra - ra.mean(dim=1, keepdim=True)
    rb_c = rb - rb.mean(dim=1, keepdim=True)
    num = (ra_c * rb_c).sum(dim=1)
    den = torch.sqrt((ra_c**2).sum(dim=1) * (rb_c**2).sum(dim=1)).clamp_min(1e-12)
    rho = (num / den).mean().item()
    return rho


def ndcg_at_k(scores, qrels, k):
    """Compute mean nDCG@k. qrels[i] = relevant doc index for query i (one per query)."""
    if qrels is None:
        return None
    Nq, B = scores.shape
    topk_idx = scores.topk(k, dim=1).indices.cpu().numpy()  # (Nq, k)
    import math
    ndcgs = []
    for i in range(Nq):
        rel = qrels[i]
        # binary gain: 1 if in topk, 0 otherwise
        if rel in topk_idx[i]:
            pos = int((topk_idx[i] == rel).nonzero()[0][0])
            dcg = 1.0 / math.log2(pos + 2)
        else:
            dcg = 0.0
        # idcg with single relevant doc = 1 / log2(2) = 1
        ndcgs.append(dcg)
    return sum(ndcgs) / len(ndcgs)


def run_one(encoder, dataset, n_str):
    p = EMBED_ROOT / encoder / f'{dataset}_{n_str}.pt'
    if not p.exists():
        return {'encoder': encoder, 'dataset': dataset, 'error': f'missing: {p}'}
    print(f"\n=== {encoder} / {dataset} ===", flush=True)
    d = torch.load(p, map_location='cpu', weights_only=False)
    Q = d['query_embeddings'].to(DEV)   # (Nq, Lq, dim)
    D = d['doc_embeddings'].to(DEV)     # (B, Ld, dim)
    q_lens = d.get('query_lengths')
    Nq, Lq, dim = Q.shape
    B, Ld, _ = D.shape
    print(f"  Q: {tuple(Q.shape)} {Q.dtype}   D: {tuple(D.shape)} {D.dtype}", flush=True)

    # If query_lengths is provided, zero out padded query tokens so they
    # don't contribute to the score on either backend (matches the
    # reference pipeline).
    if q_lens is not None:
        q_lens = q_lens.to(DEV).long()
        pos = torch.arange(Lq, device=DEV).unsqueeze(0)
        mask = (pos < q_lens.unsqueeze(1)).to(Q.dtype).unsqueeze(-1)
        Q = Q * mask

    # naive FP32 reference (double-chunked over queries and docs)
    t0 = time.time()
    scores_naive = naive_scores_fp32_chunked(Q, D, target_tile_gb=8.0)
    torch.cuda.synchronize()
    t_naive = time.time() - t0
    print(f"  naive FP32: {t_naive:.2f}s   "
          f"peak={torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)

    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    # flash
    t0 = time.time()
    scores_flash = flash_scores(Q, D).to(torch.float32)
    torch.cuda.synchronize()
    t_flash = time.time() - t0
    print(f"  flash (FP16 in / FP32 acc):  {t_flash:.2f}s   "
          f"peak={torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)

    diff = (scores_naive - scores_flash).abs()
    parity = {
        'max_abs_delta':  float(diff.max()),
        'mean_abs_delta': float(diff.mean()),
        'spearman_rho':   spearman_per_query(scores_naive, scores_flash),
        'top5_overlap':   topk_overlap(scores_naive, scores_flash, 5),
        'top10_overlap':  topk_overlap(scores_naive, scores_flash, 10),
        'top20_overlap':  topk_overlap(scores_naive, scores_flash, 20),
    }
    print(f"  parity: max|Δ|={parity['max_abs_delta']:.3e}  "
          f"mean|Δ|={parity['mean_abs_delta']:.3e}  "
          f"ρ={parity['spearman_rho']:.6f}  "
          f"top10_overlap={parity['top10_overlap']:.4f}", flush=True)

    # nDCG dropped: the .pt files were prepared by an earlier pipeline that
    # subsampled queries, so the natural row-i query <-> row-i page qrel
    # does not hold. Parity (max|Δ|, ρ, top-K overlap) is the right
    # measure for the kernel-equivalence claim regardless.

    return {
        'encoder': encoder, 'dataset': dataset, 'n_str': n_str,
        'Nq': Nq, 'B': B, 'Lq': Lq, 'Ld': Ld, 'dim': dim,
        'naive_wall_s': round(t_naive, 3),
        'flash_wall_s': round(t_flash, 3),
        'parity': parity,
        'model_meta': str(d.get('model', '')),
        'dataset_meta': str(d.get('dataset', '')),
    }


def main():
    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}  torch={torch.__version__}", flush=True)
    import flash_maxsim
    print(f"flash_maxsim {flash_maxsim.__version__}\n", flush=True)

    results = []
    for encoder in ENCODERS:
        for dataset, n_str in SUBSETS:
            try:
                r = run_one(encoder, dataset, n_str)
                results.append(r)
            except Exception as e:
                import traceback; traceback.print_exc()
                results.append({'encoder': encoder, 'dataset': dataset,
                                'error': f'{type(e).__name__}: {e}'})
            gc.collect(); torch.cuda.empty_cache()

    out = {'gpu': gpu, 'torch_version': torch.__version__,
           'flash_maxsim_version': flash_maxsim.__version__,
           'results': results}
    op = HERE / f'bench_real_mm_rag_parity_{gpu.replace(" ", "_").replace("/", "_")}.json'
    op.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {op}", flush=True)


if __name__ == '__main__':
    main()
