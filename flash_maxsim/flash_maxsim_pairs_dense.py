"""Dense pairs MaxSim — direct grid-per-pair kernel, no varlen packing.

Computes scores[b] = MaxSim(Q[b], D[b]) for b in 0..B over equal-length
pair-batched tensors:

    Q : [B, Lq, d]   D : [B, Ld, d]   ->   scores : [B]

This is the equal-length-pair fast path. The general (varlen) entry
point `flash_maxsim_pairs(q_embs, d_embs)` packs lists into the varlen
kernel, which is the right shape when each pair has a different length
but pays a list->cu_seqlens packing tax at every call. For the common
"already-batched, all-same-length" case (reranker scoring B pairs of
the same shape), this kernel skips the packing tax entirely:

  - Grid is (B,) — one Triton program per pair.
  - Each program loads Q[b] and D[b] for the SAME b.
  - Q tiles by BLOCK_Q, D tiles by BLOCK_D — same online-max recurrence
    as the standard forward kernel.
  - Output is one FP32 scalar per pair.

No autotune (deterministic config). No d_pad pre-computation needed at
the caller — the kernel uses the same `nextpow2(d)` constexpr trick the
in-shot kernel does.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .flash_maxsim import _next_pow2


@triton.jit
def _maxsim_pairs_dense_kernel(
    Q_ptr, D_ptr, lengths_ptr, q_lengths_ptr, scores_ptr,
    Lq: tl.constexpr, Ld, d: tl.constexpr, d_pad: tl.constexpr,
    stride_q_b, stride_q_l, stride_q_d,
    stride_d_b, stride_d_l, stride_d_d,
    use_q_lengths: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """One program per pair. pid = b indexes BOTH Q[b] and D[b]."""
    b = tl.program_id(0).to(tl.int64)

    doc_len = tl.load(lengths_ptr + b).to(tl.int32)
    q_len = tl.load(q_lengths_ptr + b).to(tl.int32) if use_q_lengths else Lq

    k_off = tl.arange(0, d_pad)
    k_mask = k_off < d
    score_acc = tl.zeros([], dtype=tl.float32)

    for q_start in tl.static_range(0, Lq, BLOCK_Q):
        q_off = q_start + tl.arange(0, BLOCK_Q)
        q_valid = q_off < q_len

        Q_block = tl.load(
            Q_ptr + b * stride_q_b + q_off[:, None] * stride_q_l + k_off[None, :] * stride_q_d,
            mask=q_valid[:, None] & k_mask[None, :], other=0.0,
        ).to(tl.float16)

        m = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)

        for d_start in range(0, Ld, BLOCK_D):
            d_off = d_start + tl.arange(0, BLOCK_D)
            d_valid = d_off < doc_len

            D_block = tl.load(
                D_ptr + b * stride_d_b + d_off[:, None] * stride_d_l + k_off[None, :] * stride_d_d,
                mask=d_valid[:, None] & k_mask[None, :], other=0.0,
            ).to(tl.float16)

            S = tl.dot(Q_block, tl.trans(D_block))
            S = tl.where(d_valid[None, :], S, float("-inf"))
            m = tl.maximum(m, tl.max(S, axis=1))

        m = tl.where(q_valid, m, 0.0)
        score_acc += tl.sum(m)

    tl.store(scores_ptr + b, score_acc)


def flash_maxsim_pairs_dense(
    Q: torch.Tensor, D: torch.Tensor,
    q_lengths: torch.Tensor | None = None,
    doc_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dense pairs MaxSim. Q: [B, Lq, d], D: [B, Ld, d] -> scores: [B].

    Args:
        Q: [B, Lq, d] query embeddings (fp16/bf16/fp32). One row per pair.
        D: [B, Ld, d] doc embeddings (same B and d as Q). One row per pair.
        q_lengths: optional [B] int32; per-pair valid Q length (else Lq).
        doc_lengths: optional [B] int32; per-pair valid D length (else Ld).

    Returns:
        scores: [B] fp32 — one score per (Q[b], D[b]) pair.
    """
    assert Q.ndim == 3 and D.ndim == 3, f"Q.ndim={Q.ndim}, D.ndim={D.ndim} — both must be 3D"
    assert Q.shape[0] == D.shape[0], f"B mismatch: Q.shape[0]={Q.shape[0]}, D.shape[0]={D.shape[0]}"
    assert Q.shape[2] == D.shape[2], f"d mismatch: Q.shape[2]={Q.shape[2]}, D.shape[2]={D.shape[2]}"
    B, Lq, d = Q.shape
    _, Ld, _ = D.shape

    d_pad = _next_pow2(d)
    scores = torch.empty(B, device=Q.device, dtype=torch.float32)

    # Deterministic config. Block sizes scale with sequence length so the
    # per-outer-iter reduction overhead amortises at long Lq/Ld:
    #   Lq<=32   : BLOCK_Q = max(16, Lq)        (no padding for ColBERT)
    #   Lq<=128  : BLOCK_Q = 32                 (ColBERT-long)
    #   Lq>128   : BLOCK_Q = 64                 (ColPali)
    #   Ld<=256  : BLOCK_D = 64                 (ColBERT)
    #   Ld>256   : BLOCK_D = 128                (ColPali / long-doc)
    if Lq <= 32:
        BLOCK_Q = max(16, Lq)
    elif Lq <= 128:
        BLOCK_Q = 32
    else:
        BLOCK_Q = 64
    BLOCK_D = 64 if Ld <= 256 else 128
    num_warps = 4
    num_stages = 2

    # Lengths buffers (use ones if not provided).
    if doc_lengths is None:
        doc_lengths = torch.full((B,), Ld, device=Q.device, dtype=torch.int32)
    use_q_lengths = q_lengths is not None
    q_lengths_t = q_lengths if use_q_lengths else doc_lengths  # dummy if unused

    _maxsim_pairs_dense_kernel[(B,)](
        Q, D, doc_lengths, q_lengths_t, scores,
        Lq, Ld, d, d_pad,
        Q.stride(0), Q.stride(1), Q.stride(2),
        D.stride(0), D.stride(1), D.stride(2),
        1 if use_q_lengths else 0,
        BLOCK_Q=BLOCK_Q, BLOCK_D=BLOCK_D,
        num_warps=num_warps, num_stages=num_stages,
    )
    return scores
