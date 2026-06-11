"""Sparse MaxSim: per-doc query token selection via index buffer.

Each doc selects K query tokens (same K for all docs, different tokens per doc).
The kernel uses an index buffer [B, K] to gather from shared Q [Lq, d].
Q is loaded via indirect indexing — no per-doc Q expansion needed.

Usage:
    from flash_maxsim import flash_maxsim_sparse

    # importance: [B, Lq] per-doc importance scores
    K = Lq // 2  # keep 50%
    indices = importance.topk(K, dim=1).indices  # [B, K]
    indices = indices.sort(dim=1).values  # sort for better memory access

    scores = flash_maxsim_sparse(Q, D, indices)
"""

import torch
import triton
import triton.language as tl
from .flash_maxsim import _next_pow2


# ---------------------------------------------------------------------------
# Kernel: one program per doc, gather Q via index buffer
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_Q": 16, "BLOCK_D": 16}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_Q": 16, "BLOCK_D": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_Q": 16, "BLOCK_D": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_Q": 64, "BLOCK_D": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 64, "BLOCK_D": 128}, num_warps=8, num_stages=2),
    ],
    key=["K", "Ld", "d_pad"],
)
@triton.jit
def _sparse_maxsim_kernel(
    Q_ptr, D_ptr, idx_ptr, d_lengths_ptr, scores_ptr,
    K: tl.constexpr, Ld: tl.constexpr, d: tl.constexpr, d_pad: tl.constexpr,
    stride_q_l, stride_q_d,
    stride_d_b, stride_d_l, stride_d_d,
    stride_idx_b, stride_idx_k,
    BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr,
):
    doc_id = tl.program_id(0)
    doc_len = tl.load(d_lengths_ptr + doc_id).to(tl.int32)

    k_off = tl.arange(0, d_pad)
    k_mask = k_off < d
    score_acc = tl.zeros([], dtype=tl.float32)

    for q_start in tl.static_range(0, K, BLOCK_Q):
        q_off = q_start + tl.arange(0, BLOCK_Q)
        q_valid = q_off < K

        # Load indices for this doc's selected query tokens
        q_indices = tl.load(
            idx_ptr + doc_id * stride_idx_b + q_off * stride_idx_k,
            mask=q_valid, other=0,
        ).to(tl.int32)

        # Gather Q from shared buffer using indices
        Q_block = tl.load(
            Q_ptr + q_indices[:, None] * stride_q_l + k_off[None, :] * stride_q_d,
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
    indices: torch.Tensor,
    doc_lengths=None,
) -> torch.Tensor:
    """Sparse MaxSim with per-doc query token selection.

    Args:
        Q: [Lq, d] query embeddings (shared across all docs)
        D: [B, Ld, d] document embeddings
        indices: [B, K] int — which query tokens to use per doc (same K for all)
        doc_lengths: optional [B] int — variable doc lengths

    Returns:
        scores: [B] float32
    """
    assert Q.dim() == 2 and D.dim() == 3 and indices.dim() == 2
    Lq, d = Q.shape
    B, Ld, _ = D.shape
    _, K = indices.shape
    assert indices.shape[0] == B

    Q = Q.contiguous().half()
    D = D.contiguous().half()
    indices = indices.to(torch.int32).contiguous()
    scores = torch.empty(B, device=Q.device, dtype=torch.float32)

    if doc_lengths is None:
        doc_lengths = torch.full((B,), Ld, device=D.device, dtype=torch.int32)
    else:
        doc_lengths = doc_lengths.to(torch.int32).contiguous()

    d_pad = _next_pow2(d)

    _sparse_maxsim_kernel[(B,)](
        Q, D, indices, doc_lengths, scores,
        K, Ld, d, d_pad,
        Q.stride(0), Q.stride(1),
        D.stride(0), D.stride(1), D.stride(2),
        indices.stride(0), indices.stride(1),
    )
    return scores


def maxsim_sparse_naive(
    Q: torch.Tensor,
    D: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Reference: sparse MaxSim with index buffer.

    Args:
        Q: [Lq, d], D: [B, Ld, d], indices: [B, K] int -> [B]
    """
    B, K = indices.shape
    # Gather Q per doc
    Q_sel = Q[indices]  # [B, K, d]
    S = torch.einsum('bkd,bld->bkl', Q_sel.float(), D.float())  # [B, K, Ld]
    return S.max(dim=2).values.sum(dim=1)  # [B]
