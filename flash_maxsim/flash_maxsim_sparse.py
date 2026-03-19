"""Sparse MaxSim: per-doc query token masking.

Given a mask M[B, Lq] where M[i,j]=True means "compute query token j for doc i",
this computes MaxSim only for active tokens — skipping ~50% of computation.

The trick: compact active query tokens per doc into a dense buffer,
then run a kernel with per-doc query lengths so trailing blocks are never entered.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel: one program per doc, dynamic query length
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_Q": 16, "BLOCK_D": 16}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_Q": 16, "BLOCK_D": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_Q": 16, "BLOCK_D": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 128}, num_warps=8, num_stages=2),
    ],
    key=["max_K", "Ld", "d_pad"],
)
@triton.jit
def _sparse_maxsim_kernel(
    Q_ptr, D_ptr, q_lengths_ptr, d_lengths_ptr, scores_ptr,
    max_K: tl.constexpr, Ld: tl.constexpr, d: tl.constexpr, d_pad: tl.constexpr,
    stride_q_b, stride_q_l, stride_q_d,
    stride_d_b, stride_d_l, stride_d_d,
    BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr,
):
    doc_id = tl.program_id(0)
    num_active = tl.load(q_lengths_ptr + doc_id).to(tl.int32)
    doc_len = tl.load(d_lengths_ptr + doc_id).to(tl.int32)

    k_off = tl.arange(0, d_pad)
    k_mask = k_off < d
    score_acc = tl.zeros([], dtype=tl.float32)

    for q_start in tl.static_range(0, max_K, BLOCK_Q):
        q_off = q_start + tl.arange(0, BLOCK_Q)
        q_valid = q_off < num_active

        Q_block = tl.load(
            Q_ptr + doc_id * stride_q_b + q_off[:, None] * stride_q_l + k_off[None, :] * stride_q_d,
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

def _next_pow2(x):
    return 1 << (x - 1).bit_length()


def flash_maxsim_sparse(
    Q: torch.Tensor,
    D: torch.Tensor,
    mask: torch.Tensor,
    doc_lengths=None,
) -> torch.Tensor:
    """Sparse MaxSim with per-doc query token masking.

    Args:
        Q: [Lq, d] query embeddings
        D: [B, Ld, d] document embeddings
        mask: [B, Lq] bool — True means compute this query token for this doc
        doc_lengths: optional [B] int tensor for variable doc lengths

    Returns:
        scores: [B] float32 — sparse MaxSim scores
    """
    assert Q.dim() == 2 and D.dim() == 3 and mask.dim() == 2
    Lq, d = Q.shape
    B, Ld, _ = D.shape
    assert mask.shape == (B, Lq)

    # ── Compact Q per doc: active tokens first ──
    # Sort indices so True comes before False (descending)
    sorted_idx = mask.float().argsort(dim=1, descending=True, stable=True)  # [B, Lq]
    active_counts = mask.sum(dim=1).int()  # [B]
    max_K = int(active_counts.max().item())
    # Round up to next BLOCK_Q for kernel alignment
    max_K_padded = max(max_K, 16)

    # Gather Q tokens per doc into compact buffer [B, max_K, d]
    Q_exp = Q.unsqueeze(0).expand(B, -1, -1)  # [B, Lq, d]
    idx_exp = sorted_idx[:, :max_K_padded, None].expand(-1, -1, d)  # [B, max_K, d]
    Q_compact = torch.gather(Q_exp, 1, idx_exp).contiguous().half()  # [B, max_K, d]

    D = D.contiguous().half()
    scores = torch.empty(B, device=Q.device, dtype=torch.float32)

    if doc_lengths is None:
        doc_lengths = torch.full((B,), Ld, device=D.device, dtype=torch.int32)
    else:
        doc_lengths = doc_lengths.to(torch.int32).contiguous()

    d_pad = _next_pow2(d)

    _sparse_maxsim_kernel[(B,)](
        Q_compact, D, active_counts.contiguous(), doc_lengths, scores,
        max_K_padded, Ld, d, d_pad,
        Q_compact.stride(0), Q_compact.stride(1), Q_compact.stride(2),
        D.stride(0), D.stride(1), D.stride(2),
    )
    return scores


def maxsim_sparse_naive(
    Q: torch.Tensor,
    D: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Reference implementation for sparse MaxSim.

    Args:
        Q: [Lq, d], D: [B, Ld, d], mask: [B, Lq] bool -> [B]
    """
    # S[b, q, l] = Q[q] · D[b, l]
    S = torch.einsum('qd,bld->bql', Q.float(), D.float())  # [B, Lq, Ld]
    max_sim = S.max(dim=2).values  # [B, Lq]
    # Zero out masked positions
    max_sim = max_sim * mask.float()  # [B, Lq]
    return max_sim.sum(dim=1)  # [B]
