"""Two-stage Top-K MaxSim: INT8 coarse scan → FP16 precision re-score.

For large N (ColPali N>10K), computing full FP16 MaxSim over all docs is
compute-bound. This module eliminates most of that work:

  Stage 1 — INT8 coarse scan (all N docs, fast):
      Approximate MaxSim via per-token affine INT8 scoring.
      INT8 flash_maxsim is ~4-8x faster than FP16 (smaller tensors, faster
      GEMM) and the ranking error is small (<1% recall@100 in practice).

  Stage 2 — FP16 precision re-score (top n_candidates only):
      Re-score the top n_candidates (default 4k) with exact FP16 flash.
      At n_candidates=200, N'=200 docs take <1ms regardless of original N.

Total cost: ~INT8_time + negligible  vs  full FP16 scan
Expected speedup at N=50K ColPali: ~20-25x (94ms → ~4-5ms)

Usage:
    from flash_maxsim import flash_maxsim_topk, quantize_int8

    # Pre-quantize once (offline), reuse per query
    D_int8, scales, mins = quantize_int8(D)

    scores, indices = flash_maxsim_topk(Q, D, k=100,
                                        D_int8=D_int8, scales=scales, mins=mins)
"""
import torch
from .flash_maxsim import flash_maxsim, _default_lengths
from .flash_maxsim_quant import flash_maxsim_int8, quantize_int8


def flash_maxsim_topk(
    Q: torch.Tensor,
    D: torch.Tensor,
    k: int,
    doc_lengths=None,
    n_candidates: int = None,
    D_int8: torch.Tensor = None,
    scales: torch.Tensor = None,
    mins: torch.Tensor = None,
) -> tuple:
    """Approximate top-K MaxSim via two-stage INT8→FP16 filtering.

    Args:
        Q:            [Lq, d] float16 query embeddings
        D:            [N, Ld, d] float16 document embeddings
        k:            number of final top-K results to return
        doc_lengths:  [N] int32 actual token counts per doc (None = all Ld)
        n_candidates: docs to re-score in stage 2 (default: min(k*4, N))
        D_int8:       pre-quantized [N, Ld, d] uint8 — pass to avoid
                      re-quantizing per query (strongly recommended at large N)
        scales:       [N, Ld, 1] float16 (from quantize_int8)
        mins:         [N, Ld, 1] float16 (from quantize_int8)

    Returns:
        (top_k_scores [k], top_k_indices [k])
        Indices are into the original D tensor.
    """
    assert Q.dim() == 2, "flash_maxsim_topk: Q must be 2D [Lq, d]"
    N = D.shape[0]
    k = min(k, N)

    if n_candidates is None:
        n_candidates = min(k * 4, N)
    n_candidates = min(n_candidates, N)

    # No benefit from two-stage if we'd rescore almost everything
    if n_candidates >= int(N * 0.8):
        scores = flash_maxsim(Q, D, doc_lengths)
        top_k_scores, top_k_idx = scores.topk(k)
        return top_k_scores, top_k_idx

    # ------------------------------------------------------------------
    # Stage 1: INT8 coarse scan over all N documents
    # ------------------------------------------------------------------
    if D_int8 is None:
        # Quantize on the fly — expensive; pass pre-quantized for production
        D_int8, scales, mins = quantize_int8(D)

    coarse_scores = flash_maxsim_int8(Q, D_int8, scales, mins, doc_lengths)
    # coarse_scores: [N] float32

    # Select top n_candidates by approximate score
    _, cand_idx = coarse_scores.topk(n_candidates)   # [n_candidates], GPU tensor

    # ------------------------------------------------------------------
    # Stage 2: FP16 exact re-score on candidates only
    # ------------------------------------------------------------------
    D_cand = D[cand_idx]   # [n_candidates, Ld, d] — contiguous gather copy
    lengths_cand = doc_lengths[cand_idx] if doc_lengths is not None else None

    fine_scores = flash_maxsim(Q, D_cand, lengths_cand)   # [n_candidates]

    # Map back to original indices
    _, top_k_local = fine_scores.topk(k)
    top_k_idx    = cand_idx[top_k_local]
    top_k_scores = fine_scores[top_k_local]

    return top_k_scores, top_k_idx
