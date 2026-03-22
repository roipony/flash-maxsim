"""Fused INT8 quantized MaxSim kernel."""

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Quantization utilities
# ---------------------------------------------------------------------------

def quantize_int8(D: torch.Tensor):
    """Per-token affine INT8 quantization.

    Args:
        D: [B, Ld, d] float/half tensor.

    Returns:
        D_uint8: [B, Ld, d] uint8
        scales:  [B, Ld, 1] float16
        mins:    [B, Ld, 1] float16
    """
    assert D.dim() == 3
    Df = D.float()
    d_min = Df.min(dim=-1, keepdim=True).values
    d_max = Df.max(dim=-1, keepdim=True).values
    scales = ((d_max - d_min) / 255.0).clamp_min(1e-8)
    D_uint8 = ((Df - d_min) / scales).round().clamp(0, 255).to(torch.uint8)
    return D_uint8, scales.half(), d_min.half()


def dequantize_int8(D_uint8: torch.Tensor, scales: torch.Tensor, mins: torch.Tensor) -> torch.Tensor:
    """Reconstruct float tensors from per-token affine INT8 encoding."""
    return D_uint8.float() * scales.float() + mins.float()


# ---------------------------------------------------------------------------
# Fused INT8 MaxSim kernel
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_Q": 16, "BLOCK_D": 16}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_Q": 16, "BLOCK_D": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 16, "BLOCK_D": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 128}, num_warps=8, num_stages=2),
    ],
    key=["Lq", "Ld", "d_pad"],
)
@triton.jit
def _maxsim_int8_kernel(
    Q_ptr, D_ptr, scales_ptr, mins_ptr, lengths_ptr, scores_ptr,
    Lq: tl.constexpr, Ld: tl.constexpr, d: tl.constexpr, d_pad: tl.constexpr,
    stride_d_b, stride_d_l, stride_d_d,
    stride_q_l, stride_q_d,
    stride_s_b, stride_s_l,
    BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr,
):
    doc_id = tl.program_id(0)
    doc_len = tl.load(lengths_ptr + doc_id).to(tl.int32)

    k_off = tl.arange(0, d_pad)
    k_mask = k_off < d
    score_acc = tl.zeros([], dtype=tl.float32)

    for q_start in tl.static_range(0, Lq, BLOCK_Q):
        q_off = q_start + tl.arange(0, BLOCK_Q)
        q_valid = q_off < Lq

        Q_block = tl.load(
            Q_ptr + q_off[:, None] * stride_q_l + k_off[None, :] * stride_q_d,
            mask=q_valid[:, None] & k_mask[None, :], other=0.0,
        ).to(tl.float16)

        m = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)

        for d_start in tl.static_range(0, Ld, BLOCK_D):
            d_off = d_start + tl.arange(0, BLOCK_D)
            d_valid = d_off < doc_len

            D_raw = tl.load(
                D_ptr + doc_id * stride_d_b + d_off[:, None] * stride_d_l + k_off[None, :] * stride_d_d,
                mask=d_valid[:, None] & k_mask[None, :], other=0,
            )
            sc = tl.load(scales_ptr + doc_id * stride_s_b + d_off * stride_s_l, mask=d_valid, other=1.0).to(tl.float16)
            mn = tl.load(mins_ptr + doc_id * stride_s_b + d_off * stride_s_l, mask=d_valid, other=0.0).to(tl.float16)

            D_block = D_raw.to(tl.float16) * sc[:, None] + mn[:, None]

            S = tl.dot(Q_block, tl.trans(D_block))
            S = tl.where(d_valid[None, :], S, float("-inf"))
            m = tl.maximum(m, tl.max(S, axis=1))

        m = tl.where(q_valid, m, 0.0)
        score_acc += tl.sum(m)

    tl.store(scores_ptr + doc_id, score_acc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def flash_maxsim_int8(
    Q: torch.Tensor,
    D_uint8: torch.Tensor,
    scales: torch.Tensor,
    mins: torch.Tensor,
    doc_lengths=None,
) -> torch.Tensor:
    """Fused dequant+MaxSim. Q: [Lq, d], D_uint8: [B, Ld, d] -> [B]."""
    assert Q.dim() == 2 and D_uint8.dim() == 3 and Q.shape[1] == D_uint8.shape[2]
    Lq, d = Q.shape
    B, Ld, _ = D_uint8.shape

    if scales.dim() == 3:
        scales = scales.squeeze(-1)
    if mins.dim() == 3:
        mins = mins.squeeze(-1)

    if doc_lengths is None:
        doc_lengths = torch.full((B,), Ld, device=D_uint8.device, dtype=torch.int32)
    else:
        doc_lengths = doc_lengths.to(torch.int32).contiguous()

    Q = Q.contiguous().half()
    D_uint8 = D_uint8.contiguous()
    scales = scales.contiguous().half()
    mins = mins.contiguous().half()
    scores = torch.empty(B, device=Q.device, dtype=torch.float32)

    from .flash_maxsim import _next_pow2
    d_pad = _next_pow2(d)
    _maxsim_int8_kernel[(B,)](
        Q, D_uint8, scales, mins, doc_lengths, scores,
        Lq, Ld, d, d_pad,
        D_uint8.stride(0), D_uint8.stride(1), D_uint8.stride(2),
        Q.stride(0), Q.stride(1),
        scales.stride(0), scales.stride(1),
    )
    return scores
