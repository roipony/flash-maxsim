"""Fused MaxSim + softmax-CE + backward training step.

This module intentionally does not expose a torch.autograd.Function. The
public API below is a training-step helper: it computes the scalar contrastive
loss and writes gradients directly into ``Q.grad`` and ``D.grad``.

MVP scope:
  - fixed-length shared-doc batches only: Q [Nq, Lq, d], D [Nd, Ld, d]
  - fp16/bf16 CUDA inputs
  - diagonal labels only
  - Lq <= 32, d <= 256
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .flash_maxsim import _next_pow2


@triton.jit
def _maxsim_contrastive_step_kernel(
    Q_ptr, D_ptr, grad_Q_ptr, grad_D_ptr, loss_ptr,
    Nq: tl.constexpr, Nd: tl.constexpr,
    Lq: tl.constexpr, Ld: tl.constexpr,
    d: tl.constexpr, d_pad: tl.constexpr,
    stride_q_n, stride_q_l, stride_q_d,
    stride_d_b, stride_d_l, stride_d_d,
    stride_gq_n, stride_gq_l, stride_gq_d,
    stride_gd_b, stride_gd_l, stride_gd_d,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    q_idx = tl.program_id(0)
    if q_idx >= Nq:
        return

    q_off = tl.arange(0, BLOCK_Q)
    k = tl.arange(0, d_pad)
    d_off_blk = tl.arange(0, BLOCK_D)
    q_mask = q_off < Lq
    k_mask = k < d

    Q_block = tl.load(
        Q_ptr + q_idx * stride_q_n + q_off[:, None] * stride_q_l + k[None, :] * stride_q_d,
        mask=q_mask[:, None] & k_mask[None, :],
        other=0.0,
    ).to(tl.float16)

    # Pass 1: stream all docs, compute scores, and online softmax statistics.
    row_max = tl.full([], float("-inf"), dtype=tl.float32)
    row_sum = tl.zeros([], dtype=tl.float32)
    target_score = tl.zeros([], dtype=tl.float32)

    for b in range(Nd):
        m = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)

        for d_start in range(0, Ld, BLOCK_D):
            d_off = d_start + d_off_blk
            d_mask = d_off < Ld
            D_block = tl.load(
                D_ptr + b * stride_d_b + d_off[:, None] * stride_d_l + k[None, :] * stride_d_d,
                mask=d_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float16)

            S = tl.dot(Q_block, tl.trans(D_block))
            S = tl.where(d_mask[None, :], S, float("-inf"))
            m = tl.maximum(m, tl.max(S, axis=1))

        m = tl.where(q_mask, m, 0.0)
        score = tl.sum(m)
        target_score = tl.where(b == q_idx, score, target_score)

        new_max = tl.maximum(row_max, score)
        row_sum = row_sum * tl.exp(row_max - new_max) + tl.exp(score - new_max)
        row_max = new_max

    loss_i = tl.log(row_sum) + row_max - target_score
    tl.atomic_add(loss_ptr, loss_i / Nq)

    # Pass 2: re-stream all docs, recompute MaxSim winners, and backprop.
    grad_q = tl.zeros([BLOCK_Q, d_pad], dtype=tl.float32)
    inv_nq = 1.0 / Nq

    for b in range(Nd):
        m = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
        m_idx = tl.full([BLOCK_Q], 0, dtype=tl.int32)

        for d_start in range(0, Ld, BLOCK_D):
            d_off = d_start + d_off_blk
            d_mask = d_off < Ld
            D_block = tl.load(
                D_ptr + b * stride_d_b + d_off[:, None] * stride_d_l + k[None, :] * stride_d_d,
                mask=d_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float16)

            S = tl.dot(Q_block, tl.trans(D_block))
            S = tl.where(d_mask[None, :], S, float("-inf"))
            tile_max = tl.max(S, axis=1)
            tile_argmax = tl.argmax(S, axis=1).to(tl.int32) + d_start
            update = tile_max > m
            m_idx = tl.where(update, tile_argmax, m_idx)
            m = tl.maximum(m, tile_max)

        m = tl.where(q_mask, m, 0.0)
        score = tl.sum(m)
        prob = tl.exp(score - row_max) / row_sum
        gs = (prob - tl.where(b == q_idx, 1.0, 0.0)) * inv_nq

        winner_D = tl.load(
            D_ptr + b * stride_d_b + m_idx[:, None] * stride_d_l + k[None, :] * stride_d_d,
            mask=q_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        grad_q += gs * winner_D

        tl.atomic_add(
            grad_D_ptr + b * stride_gd_b + m_idx[:, None] * stride_gd_l + k[None, :] * stride_gd_d,
            gs * Q_block.to(tl.float32),
            sem="relaxed",
            mask=q_mask[:, None] & k_mask[None, :],
        )

    old_q = tl.load(
        grad_Q_ptr + q_idx * stride_gq_n + q_off[:, None] * stride_gq_l + k[None, :] * stride_gq_d,
        mask=q_mask[:, None] & k_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    tl.store(
        grad_Q_ptr + q_idx * stride_gq_n + q_off[:, None] * stride_gq_l + k[None, :] * stride_gq_d,
        old_q + grad_q,
        mask=q_mask[:, None] & k_mask[None, :],
    )


def _validate_inputs(Q: torch.Tensor, D: torch.Tensor, labels: torch.Tensor | None) -> None:
    if Q.dim() != 3:
        raise ValueError(f"Q must be [Nq, Lq, d], got shape {tuple(Q.shape)}")
    if D.dim() != 3:
        raise ValueError(f"D must be [Nd, Ld, d], got shape {tuple(D.shape)}")
    if Q.device.type != "cuda" or D.device.type != "cuda":
        raise ValueError("Q and D must be CUDA tensors")
    if Q.device != D.device:
        raise ValueError(f"Q and D must be on the same device, got {Q.device} and {D.device}")
    if Q.dtype not in (torch.float16, torch.bfloat16) or D.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Q and D must be fp16 or bf16, got {Q.dtype} and {D.dtype}")
    if Q.dtype != D.dtype:
        raise ValueError(f"Q and D must have the same dtype for the MVP, got {Q.dtype} and {D.dtype}")
    if Q.shape[2] != D.shape[2]:
        raise ValueError(f"Q and D embedding dims must match, got {Q.shape[2]} and {D.shape[2]}")

    Nq, Lq, d = Q.shape
    Nd = D.shape[0]
    if Nd < Nq:
        raise ValueError(f"diagonal labels require Nd >= Nq, got Nq={Nq}, Nd={Nd}")
    if Lq > 32:
        raise ValueError(f"MVP kernel supports Lq <= 32, got Lq={Lq}")
    if d > 256:
        raise ValueError(f"MVP kernel supports d <= 256, got d={d}")
    if _next_pow2(d) != d:
        # tl.dot prefers power-of-two K tiles and the existing project shapes
        # are d=16/128. Keep the MVP honest instead of hiding a slow fallback.
        raise ValueError(f"MVP kernel requires power-of-two d, got d={d}")

    if labels is not None:
        if labels.shape != (Nq,):
            raise ValueError(f"labels must have shape [{Nq}], got {tuple(labels.shape)}")
        if labels.dtype != torch.int64:
            raise ValueError(f"labels must be int64, got {labels.dtype}")
        expected = torch.arange(Nq, device=labels.device, dtype=torch.int64)
        if not torch.equal(labels, expected):
            raise ValueError("MVP supports diagonal labels only: labels must equal arange(Nq)")


def colbert_contrastive_step(
    Q: torch.Tensor,
    D: torch.Tensor,
    labels: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run a fused contrastive MaxSim training step.

    Args:
        Q: query embeddings [Nq, Lq, d], fp16/bf16 CUDA.
        D: document embeddings [Nd, Ld, d], fp16/bf16 CUDA.
        labels: optional int64 [Nq]. MVP accepts only diagonal arange labels.

    Returns:
        Scalar fp32 loss tensor. The tensor is not connected to autograd.

    Side effect:
        Accumulates gradients into ``Q.grad`` and ``D.grad``. If either grad is
        ``None``, it is allocated and zero-initialized before the kernel launch.
    """
    _validate_inputs(Q, D, labels)
    Nq, Lq, d = Q.shape
    Nd, Ld, _ = D.shape

    Q_c = Q.contiguous()
    D_c = D.contiguous()

    if Q.grad is None:
        Q.grad = torch.zeros_like(Q)
    if D.grad is None:
        D.grad = torch.zeros_like(D)
    if Q.grad.shape != Q.shape or Q.grad.dtype != Q.dtype or Q.grad.device != Q.device:
        raise ValueError("Q.grad, when present, must match Q shape, dtype, and device")
    if D.grad.shape != D.shape or D.grad.dtype != D.dtype or D.grad.device != D.device:
        raise ValueError("D.grad, when present, must match D shape, dtype, and device")

    loss = torch.zeros((), device=Q.device, dtype=torch.float32)
    d_pad = _next_pow2(d)
    block_q = 32
    # Keep the score tile close to the existing forward kernel while limiting
    # the [BLOCK_Q, BLOCK_D] accumulator footprint in this larger step kernel.
    block_d = 64 if Ld >= 64 else triton.next_power_of_2(Ld)

    _maxsim_contrastive_step_kernel[(Nq,)](
        Q_c, D_c, Q.grad, D.grad, loss,
        Nq, Nd, Lq, Ld, d, d_pad,
        Q_c.stride(0), Q_c.stride(1), Q_c.stride(2),
        D_c.stride(0), D_c.stride(1), D_c.stride(2),
        Q.grad.stride(0), Q.grad.stride(1), Q.grad.stride(2),
        D.grad.stride(0), D.grad.stride(1), D.grad.stride(2),
        block_q, block_d,
        num_warps=8, num_stages=3,
    )
    return loss
