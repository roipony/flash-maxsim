"""Sparse-output grad-D backward for batched MaxSim.

The standard atomic dD path writes into a dense [N_flat, Ld, d] gradient through
atomics. For KD/non-shared layouts, each (query, doc) slab has only Lq winners,
so most Ld rows are zero. This module builds the unique touched destinations
from the saved argmax and launches one destination-owned program per touched row.
The returned gradient is still dense, matching PyTorch autograd's contract.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .flash_maxsim import _next_pow2


def build_sparse_dd_tables(
    argmax: torch.Tensor,
    Nq: int,
    B: int,
    Lq: int,
    Ld: int,
    shared_docs: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build sorted sparse destination tables for grad-D.

    Returns:
      perm: int64[nnz], original flat entry ids sorted by destination.
      seg_start: int32[U], first sorted entry per unique destination.
      seg_count: int32[U], entry count per unique destination.
      uniq_db: int32[U], flat document batch index for each destination.
      uniq_j: int32[U], document token row for each destination.
    """
    device = argmax.device
    assert argmax.is_cuda
    assert argmax.shape == (Nq * B, Lq), (
        f"expected argmax [{Nq * B}, {Lq}], got {tuple(argmax.shape)}"
    )

    nnz = Nq * B * Lq
    entry = torch.arange(nnz, device=device, dtype=torch.int64)
    pair = entry // int(Lq)
    j = argmax.reshape(-1).to(torch.int64)

    if shared_docs:
        db = pair // int(Nq)
    else:
        db = pair

    key = db * int(Ld) + j
    perm = torch.argsort(key)
    key_sorted = key.index_select(0, perm)
    uniq, counts = torch.unique_consecutive(key_sorted, return_counts=True)

    seg_start = torch.empty_like(counts)
    if counts.numel() > 0:
        seg_start[0] = 0
        seg_start[1:] = torch.cumsum(counts, dim=0)[:-1]

    uniq_db = (uniq // int(Ld)).to(torch.int32)
    uniq_j = (uniq % int(Ld)).to(torch.int32)
    return (
        perm.contiguous(),
        seg_start.to(torch.int32).contiguous(),
        counts.to(torch.int32).contiguous(),
        uniq_db.contiguous(),
        uniq_j.contiguous(),
    )


@triton.jit
def _maxsim_bwd_dD_sparse_kernel(
    perm_ptr,
    seg_start_ptr,
    seg_count_ptr,
    uniq_db_ptr,
    uniq_j_ptr,
    Q_ptr,
    grad_s_ptr,
    grad_D_ptr,
    Nq: tl.constexpr,
    B: tl.constexpr,
    Lq: tl.constexpr,
    Ld,
    d: tl.constexpr,
    d_pad: tl.constexpr,
    stride_d_b,
    stride_d_l,
    stride_d_d,
    stride_q_n,
    stride_q_l,
    stride_q_d,
    stride_gs_n,
    stride_gs_b,
    shared_docs: tl.constexpr,
):
    pid = tl.program_id(0)

    seg_start = tl.load(seg_start_ptr + pid)
    seg_count = tl.load(seg_count_ptr + pid)
    db = tl.load(uniq_db_ptr + pid)
    j = tl.load(uniq_j_ptr + pid)

    k = tl.arange(0, d_pad)
    km = k < d
    acc = tl.zeros([d_pad], dtype=tl.float32)

    for off in range(0, seg_count):
        entry = tl.load(perm_ptr + seg_start + off)
        pair = entry // Lq
        lq_idx = entry % Lq

        if shared_docs:
            doc_id = pair // Nq
            q_idx = pair % Nq
        else:
            q_idx = pair // B
            doc_id = pair % B

        gs = tl.load(grad_s_ptr + q_idx * stride_gs_n + doc_id * stride_gs_b).to(tl.float32)
        qv = tl.load(
            Q_ptr + q_idx * stride_q_n + lq_idx * stride_q_l + k * stride_q_d,
            mask=km,
            other=0.0,
        ).to(tl.float32)
        acc += gs * qv

    db_i64 = tl.cast(db, tl.int64)
    j_i64 = tl.cast(j, tl.int64)
    tl.store(
        grad_D_ptr + db_i64 * stride_d_b + j_i64 * stride_d_l + k * stride_d_d,
        acc,
        mask=km,
    )


def _launch_bwd_dD_sparse(
    Q: torch.Tensor,
    argmax: torch.Tensor,
    grad_scores: torch.Tensor,
    Nq: int,
    B: int,
    Lq: int,
    Ld: int,
    d: int,
    shared_docs: bool,
) -> torch.Tensor:
    """Return dense FP32 grad_D while launching only touched destination rows."""
    assert Q.is_cuda and argmax.is_cuda and grad_scores.is_cuda
    d_pad = _next_pow2(d)
    n_flat = B if shared_docs else Nq * B
    grad_D = torch.zeros((n_flat, Ld, d), device=Q.device, dtype=torch.float32)

    perm, seg_start, seg_count, uniq_db, uniq_j = build_sparse_dd_tables(
        argmax, Nq, B, Lq, Ld, shared_docs
    )
    U = int(seg_count.numel())
    if U == 0:
        return grad_D

    _maxsim_bwd_dD_sparse_kernel[(U,)](
        perm,
        seg_start,
        seg_count,
        uniq_db,
        uniq_j,
        Q,
        grad_scores,
        grad_D,
        Nq,
        B,
        Lq,
        Ld,
        d,
        d_pad,
        grad_D.stride(0),
        grad_D.stride(1),
        grad_D.stride(2),
        Q.stride(0),
        Q.stride(1),
        Q.stride(2),
        grad_scores.stride(0),
        grad_scores.stride(1),
        1 if shared_docs else 0,
    )
    return grad_D


def should_use_sparse_dd_auto(Nq: int, B: int, Lq: int, Ld: int, shared_docs: bool) -> bool:
    """Conservative auto-dispatch gate for the sparse-output dD path."""
    if shared_docs:
        return False
    if Ld < 128:
        return False
    return Nq * B * Lq >= 4096
