"""Sparse MaxSim: importance-based query token pruning.

Sort query tokens by global importance, then use per-doc query_lengths
to stop early. Q stays shared (not expanded per doc) — zero overhead.

Usage:
    from flash_maxsim import flash_maxsim_sparse, maxsim_sparse_naive

    # importance: [B, Lq] — per-doc importance scores for query tokens
    # Sort Q by global importance (most important first)
    global_imp = importance.mean(dim=0)
    order = global_imp.argsort(descending=True)
    Q_sorted = Q[order]

    # Per doc: keep top 50%
    k = Lq // 2
    query_lengths = torch.full((B,), k, device='cuda', dtype=torch.int32)

    scores = flash_maxsim_sparse(Q_sorted, D, query_lengths)
"""

import torch
import triton
import triton.language as tl
from .flash_maxsim import _next_pow2, _get_configs, _prune_configs


# ---------------------------------------------------------------------------
# Kernel: same as dense but with per-doc query_lengths
# ---------------------------------------------------------------------------

@triton.autotune(configs=_get_configs(), key=["Lq", "Ld", "d_pad"],
                 prune_configs_by={"early_config_prune": _prune_configs})
@triton.jit
def _sparse_maxsim_kernel(
    Q_ptr, D_ptr, d_lengths_ptr, q_lengths_ptr, scores_ptr,
    B: tl.constexpr,
    Lq: tl.constexpr, Ld: tl.constexpr, d: tl.constexpr, d_pad: tl.constexpr,
    stride_q_l, stride_q_d,
    stride_d_b, stride_d_l, stride_d_d,
    BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr,
):
    doc_id = tl.program_id(0)
    doc_len = tl.load(d_lengths_ptr + doc_id).to(tl.int32)
    q_len = tl.load(q_lengths_ptr + doc_id).to(tl.int32)

    k_off = tl.arange(0, d_pad)
    k_mask = k_off < d
    score_acc = tl.zeros([], dtype=tl.float32)

    # Q is shared [Lq, d] — same pointer for all docs
    # But q_len varies per doc, so we skip trailing query blocks
    for q_start in tl.static_range(0, Lq, BLOCK_Q):
        q_off = q_start + tl.arange(0, BLOCK_Q)
        q_valid = q_off < q_len  # per-doc cutoff

        Q_block = tl.load(
            Q_ptr + q_off[:, None] * stride_q_l + k_off[None, :] * stride_q_d,
            mask=q_valid[:, None] & k_mask[None, :], other=0.0,
        ).to(tl.float16)

        m = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)

        for d_start in range(0, Ld, BLOCK_D):
            d_off = d_start + tl.arange(0, BLOCK_D)
            d_valid = d_off < doc_len

            D_block = tl.load(
                D_ptr + doc_id * stride_d_b + d_off[:, None] * stride_d_l + k_off[None, :] * stride_d_d,
                mask=d_valid[:, None] & k_mask[None, :], other=0.0,
            ).to(tl.float16)

            S = tl.dot(Q_block, tl.trans(D_block))
            S = tl.where(d_valid[None, :], S, float("-inf"))
            m = tl.maximum(m, tl.max(S, axis=1))

        m = tl.where(q_valid, m, 0.0)
        score_acc += tl.sum(m)

    tl.store(scores_ptr + doc_id, score_acc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def flash_maxsim_sparse(
    Q: torch.Tensor,
    D: torch.Tensor,
    query_lengths: torch.Tensor,
    doc_lengths=None,
) -> torch.Tensor:
    """Sparse MaxSim with per-doc query token counts.

    Q should be sorted by importance (most important tokens first).
    query_lengths[i] = how many query tokens to use for doc i.

    Args:
        Q: [Lq, d] query embeddings (sorted by importance)
        D: [B, Ld, d] document embeddings
        query_lengths: [B] int — number of query tokens per doc
        doc_lengths: optional [B] int — number of doc tokens per doc

    Returns:
        scores: [B] float32
    """
    assert Q.dim() == 2 and D.dim() == 3
    Lq, d = Q.shape
    B, Ld, _ = D.shape

    Q = Q.contiguous().half()
    D = D.contiguous().half()
    query_lengths = query_lengths.to(torch.int32).contiguous()
    scores = torch.empty(B, device=Q.device, dtype=torch.float32)

    if doc_lengths is None:
        doc_lengths = torch.full((B,), Ld, device=D.device, dtype=torch.int32)
    else:
        doc_lengths = doc_lengths.to(torch.int32).contiguous()

    d_pad = _next_pow2(d)

    _sparse_maxsim_kernel[(B,)](
        Q, D, doc_lengths, query_lengths, scores,
        B, Lq, Ld, d, d_pad,
        Q.stride(0), Q.stride(1),
        D.stride(0), D.stride(1), D.stride(2),
    )
    return scores


def prepare_sparse(Q, importance, keep_ratio=0.5):
    """Sort Q by global importance and compute query_lengths.

    Args:
        Q: [Lq, d] query embeddings
        importance: [B, Lq] per-doc importance scores
        keep_ratio: fraction of tokens to keep (0.0 to 1.0)

    Returns:
        Q_sorted: [Lq, d] — sorted by global importance (most important first)
        query_lengths: [B] int32 — per-doc token counts
        order: [Lq] — sort indices (for mapping back)
    """
    Lq = Q.shape[0]
    B = importance.shape[0]

    # Global importance = mean across docs
    global_imp = importance.mean(dim=0)  # [Lq]
    order = global_imp.argsort(descending=True)  # [Lq]
    Q_sorted = Q[order]  # most important first

    # Per-doc: reorder importance, then find per-doc cutoff
    imp_sorted = importance[:, order]  # [B, Lq]
    # Cumulative sum of "is this token in top-k for this doc?"
    k = max(1, int(Lq * keep_ratio))
    # Per-doc top-k in original order → after global sort, count how many fall in first K positions
    topk_mask = torch.zeros_like(importance, dtype=torch.bool)
    topk_idx = importance.topk(k, dim=1).indices
    topk_mask.scatter_(1, topk_idx, True)
    topk_reordered = topk_mask[:, order]  # [B, Lq]

    # query_lengths[i] = max position of an active token + 1
    # (since tokens are globally sorted, active tokens cluster near the front)
    active_positions = topk_reordered.float() * torch.arange(Lq, device=Q.device).float()
    query_lengths = (active_positions.max(dim=1).values + 1).to(torch.int32)
    # Clamp to at least 1
    query_lengths = query_lengths.clamp(min=1, max=Lq)

    return Q_sorted, query_lengths, order


def maxsim_sparse_naive(
    Q: torch.Tensor,
    D: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Reference: sparse MaxSim with boolean mask.

    Args:
        Q: [Lq, d], D: [B, Ld, d], mask: [B, Lq] bool -> [B]
    """
    S = torch.einsum('qd,bld->bql', Q.float(), D.float())  # [B, Lq, Ld]
    max_sim = S.max(dim=2).values  # [B, Lq]
    max_sim = max_sim * mask.float()  # [B, Lq]
    return max_sim.sum(dim=1)  # [B]
