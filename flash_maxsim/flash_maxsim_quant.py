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


def quantize_int8_symmetric(D: torch.Tensor):
    """Per-token symmetric INT8 quantization (no zero-point offset).

    Maps D to signed int8 [-127, 127] centered at zero.
    Dequant is just D_int8 * scale — no add needed.
    This makes the fused kernel faster (one fewer op per element).

    Args:
        D: [B, Ld, d] float/half tensor.

    Returns:
        D_int8: [B, Ld, d] int8 (signed)
        scales: [B, Ld, 1] float16
    """
    assert D.dim() == 3
    Df = D.float()
    absmax = Df.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scales = absmax / 127.0
    D_int8 = (Df / scales).round().clamp(-127, 127).to(torch.int8)
    return D_int8, scales.half()


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
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_Q": 64, "BLOCK_D": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 64, "BLOCK_D": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_Q": 128, "BLOCK_D": 64}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_Q": 128, "BLOCK_D": 64}, num_warps=8, num_stages=2),
    ],
    key=["Lq", "d_pad"],
)
@triton.jit
def _maxsim_int8_kernel(
    Q_ptr, D_ptr, scales_ptr, mins_ptr, lengths_ptr, scores_ptr,
    Nq, B,
    Lq: tl.constexpr, Ld, d: tl.constexpr, d_pad: tl.constexpr,
    stride_q_n, stride_q_l, stride_q_d,
    stride_d_b, stride_d_l, stride_d_d,
    stride_s_b, stride_s_l,
    BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    doc_id = pid // Nq
    q_idx = pid % Nq
    if doc_id >= B:
        return

    doc_len = tl.load(lengths_ptr + doc_id).to(tl.int32)

    k_off = tl.arange(0, d_pad)
    k_mask = k_off < d
    score_acc = tl.zeros([], dtype=tl.float32)

    doc_id64 = tl.cast(doc_id, tl.int64)

    for q_start in tl.static_range(0, Lq, BLOCK_Q):
        q_off = q_start + tl.arange(0, BLOCK_Q)
        q_valid = q_off < Lq

        Q_block = tl.load(
            Q_ptr + q_idx * stride_q_n + q_off[:, None] * stride_q_l + k_off[None, :] * stride_q_d,
            mask=q_valid[:, None] & k_mask[None, :], other=0.0,
        ).to(tl.float16)

        # Precompute sum of Q per query token for the min correction:
        # Q · (D*sc + mn) = (Q · D)*sc + sum(Q)*mn
        Q_sum = tl.sum(Q_block.to(tl.float32), axis=1)  # [BLOCK_Q]

        m = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)

        for d_start in range(0, Ld, BLOCK_D):
            d_off = d_start + tl.arange(0, BLOCK_D)
            d_valid = d_off < doc_len

            D_raw = tl.load(
                D_ptr + doc_id64 * stride_d_b + d_off[:, None] * stride_d_l + k_off[None, :] * stride_d_d,
                mask=d_valid[:, None] & k_mask[None, :], other=0,
            )
            sc = tl.load(scales_ptr + doc_id64 * stride_s_b + d_off * stride_s_l, mask=d_valid, other=1.0).to(tl.float32)
            mn = tl.load(mins_ptr + doc_id64 * stride_s_b + d_off * stride_s_l, mask=d_valid, other=0.0).to(tl.float32)

            # Scale + min correction after dot (not per-element dequant):
            # Q · (D_raw * sc + mn) = (Q · D_raw) * sc + sum(Q) * mn
            S = tl.dot(Q_block, tl.trans(D_raw.to(tl.float16))).to(tl.float32)
            S = S * sc[None, :] + Q_sum[:, None] * mn[None, :]
            S = tl.where(d_valid[None, :], S, float("-inf"))
            m = tl.maximum(m, tl.max(S, axis=1))

        m = tl.where(q_valid, m, 0.0)
        score_acc += tl.sum(m)

    tl.store(scores_ptr + q_idx * B + doc_id, score_acc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def flash_maxsim_int8(
    Q: torch.Tensor,
    D_uint8: torch.Tensor,
    scales: torch.Tensor,
    mins: torch.Tensor,
    doc_lengths=None,
    query_chunk_size: int = 128,
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

    from .flash_maxsim import _next_pow2
    d_pad = _next_pow2(d)

    # Q chunking for occupancy (same pattern as FP16 kernel)
    if query_chunk_size is not None and Lq > query_chunk_size:
        C = query_chunk_size
        Nq = (Lq + C - 1) // C
        if Lq % C != 0:
            Q = torch.nn.functional.pad(Q, (0, 0, 0, Nq * C - Lq))
        Q_chunked = Q.view(Nq, C, d)
        actual_Lq = C
    else:
        Q_chunked = Q.unsqueeze(0)
        Nq = 1
        actual_Lq = Lq

    scores = torch.empty(Nq, B, device=Q.device, dtype=torch.float32)

    _maxsim_int8_kernel[(Nq * B,)](
        Q_chunked, D_uint8, scales, mins, doc_lengths, scores,
        Nq, B, actual_Lq, Ld, d, d_pad,
        Q_chunked.stride(0), Q_chunked.stride(1), Q_chunked.stride(2),
        D_uint8.stride(0), D_uint8.stride(1), D_uint8.stride(2),
        scales.stride(0), scales.stride(1),
    )
    return scores.squeeze(0) if Nq == 1 else scores.sum(dim=0)


# ---------------------------------------------------------------------------
# Symmetric INT8 kernel — scale only, no zero-point add
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_Q": 16, "BLOCK_D": 16}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_Q": 16, "BLOCK_D": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 16, "BLOCK_D": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_Q": 64, "BLOCK_D": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 64, "BLOCK_D": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_Q": 128, "BLOCK_D": 64}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_Q": 128, "BLOCK_D": 64}, num_warps=8, num_stages=2),
    ],
    key=["Lq", "d_pad"],
)
@triton.jit
def _maxsim_int8_sym_kernel(
    Q_ptr, D_ptr, scales_ptr, lengths_ptr, scores_ptr,
    Nq, B,
    Lq: tl.constexpr, Ld, d: tl.constexpr, d_pad: tl.constexpr,
    stride_q_n, stride_q_l, stride_q_d,
    stride_d_b, stride_d_l, stride_d_d,
    stride_s_b, stride_s_l,
    BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Symmetric INT8: dequant = D_int8 * scale (no add). One fewer op per element."""
    pid = tl.program_id(0)
    doc_id = pid // Nq
    q_idx = pid % Nq
    if doc_id >= B:
        return

    doc_len = tl.load(lengths_ptr + doc_id).to(tl.int32)

    k_off = tl.arange(0, d_pad)
    k_mask = k_off < d
    score_acc = tl.zeros([], dtype=tl.float32)

    doc_id64 = tl.cast(doc_id, tl.int64)

    for q_start in tl.static_range(0, Lq, BLOCK_Q):
        q_off = q_start + tl.arange(0, BLOCK_Q)
        q_valid = q_off < Lq

        Q_block = tl.load(
            Q_ptr + q_idx * stride_q_n + q_off[:, None] * stride_q_l + k_off[None, :] * stride_q_d,
            mask=q_valid[:, None] & k_mask[None, :], other=0.0,
        ).to(tl.float16)

        m = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)

        for d_start in range(0, Ld, BLOCK_D):
            d_off = d_start + tl.arange(0, BLOCK_D)
            d_valid = d_off < doc_len

            D_raw = tl.load(
                D_ptr + doc_id64 * stride_d_b + d_off[:, None] * stride_d_l + k_off[None, :] * stride_d_d,
                mask=d_valid[:, None] & k_mask[None, :], other=0,
            )
            sc = tl.load(scales_ptr + doc_id64 * stride_s_b + d_off * stride_s_l,
                         mask=d_valid, other=1.0).to(tl.float16)

            D_block = D_raw.to(tl.float16)
            S = tl.dot(Q_block, tl.trans(D_block)) * sc[None, :]
            S = tl.where(d_valid[None, :], S, float("-inf"))
            m = tl.maximum(m, tl.max(S, axis=1))

        m = tl.where(q_valid, m, 0.0)
        score_acc += tl.sum(m)

    tl.store(scores_ptr + q_idx * B + doc_id, score_acc)


def flash_maxsim_int8_symmetric(
    Q: torch.Tensor,
    D_int8: torch.Tensor,
    scales: torch.Tensor,
    doc_lengths=None,
    query_chunk_size: int = 128,
) -> torch.Tensor:
    """Symmetric INT8 fused dequant+MaxSim. Faster than affine (no add).

    Use with quantize_int8_symmetric().
    Q: [Lq, d], D_int8: [B, Ld, d] int8 -> [B].
    """
    assert Q.dim() == 2 and D_int8.dim() == 3 and Q.shape[1] == D_int8.shape[2]
    Lq, d = Q.shape
    B, Ld, _ = D_int8.shape

    if scales.dim() == 3:
        scales = scales.squeeze(-1)

    if doc_lengths is None:
        doc_lengths = torch.full((B,), Ld, device=D_int8.device, dtype=torch.int32)
    else:
        doc_lengths = doc_lengths.to(torch.int32).contiguous()

    Q = Q.contiguous().half()
    D_int8 = D_int8.contiguous()
    scales = scales.contiguous().half()

    from .flash_maxsim import _next_pow2
    d_pad = _next_pow2(d)

    # Q chunking for occupancy (same pattern as FP16 kernel)
    if query_chunk_size is not None and Lq > query_chunk_size:
        C = query_chunk_size
        Nq = (Lq + C - 1) // C
        if Lq % C != 0:
            Q = torch.nn.functional.pad(Q, (0, 0, 0, Nq * C - Lq))
        Q_chunked = Q.view(Nq, C, d)
        actual_Lq = C
    else:
        Q_chunked = Q.unsqueeze(0)
        Nq = 1
        actual_Lq = Lq

    scores = torch.empty(Nq, B, device=Q.device, dtype=torch.float32)

    _maxsim_int8_sym_kernel[(Nq * B,)](
        Q_chunked, D_int8, scales, doc_lengths, scores,
        Nq, B, actual_Lq, Ld, d, d_pad,
        Q_chunked.stride(0), Q_chunked.stride(1), Q_chunked.stride(2),
        D_int8.stride(0), D_int8.stride(1), D_int8.stride(2),
        scales.stride(0), scales.stride(1),
    )
    return scores.squeeze(0) if Nq == 1 else scores.sum(dim=0)


# ---------------------------------------------------------------------------
# True INT8×INT8 kernel — both operands INT8, uses INT8 tensor cores (2x FP16)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_Q": 16, "BLOCK_D": 16}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_Q": 16, "BLOCK_D": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_Q": 64, "BLOCK_D": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 64, "BLOCK_D": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_Q": 128, "BLOCK_D": 64}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_Q": 128, "BLOCK_D": 64}, num_warps=8, num_stages=2),
    ],
    key=["Lq", "d_pad"],
)
@triton.jit
def _maxsim_int8x8_kernel(
    Q_ptr, D_ptr, q_scales_ptr, d_scales_ptr, lengths_ptr, scores_ptr,
    Nq, B,
    Lq: tl.constexpr, Ld, d: tl.constexpr, d_pad: tl.constexpr,
    stride_q_n, stride_q_l, stride_q_d,
    stride_d_b, stride_d_l, stride_d_d,
    stride_ds_b, stride_ds_l,
    BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """True INT8×INT8 MaxSim. Both Q and D are int8, uses INT8 tensor cores.

    dot(Q_int8, D_int8) → int32, then dequant: S_real = S_int32 * sq * sd
    A100 INT8 tensor cores: 624 TOPS (2x FP16's 312 TFLOPS).
    """
    pid = tl.program_id(0)
    doc_id = pid // Nq
    q_idx = pid % Nq
    if doc_id >= B:
        return

    doc_len = tl.load(lengths_ptr + doc_id).to(tl.int32)
    doc_id64 = tl.cast(doc_id, tl.int64)

    k_off = tl.arange(0, d_pad)
    k_mask = k_off < d
    score_acc = tl.zeros([], dtype=tl.float32)

    for q_start in tl.static_range(0, Lq, BLOCK_Q):
        q_off = q_start + tl.arange(0, BLOCK_Q)
        q_valid = q_off < Lq

        Q_block = tl.load(
            Q_ptr + q_idx * stride_q_n + q_off[:, None] * stride_q_l + k_off[None, :] * stride_q_d,
            mask=q_valid[:, None] & k_mask[None, :], other=0,
        )  # int8

        q_sc = tl.load(q_scales_ptr + q_idx * Lq + q_off, mask=q_valid, other=1.0).to(tl.float32)

        m = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)

        for d_start in range(0, Ld, BLOCK_D):
            d_off = d_start + tl.arange(0, BLOCK_D)
            d_valid = d_off < doc_len

            D_block = tl.load(
                D_ptr + doc_id64 * stride_d_b + d_off[:, None] * stride_d_l + k_off[None, :] * stride_d_d,
                mask=d_valid[:, None] & k_mask[None, :], other=0,
            )  # int8

            d_sc = tl.load(
                d_scales_ptr + doc_id64 * stride_ds_b + d_off * stride_ds_l,
                mask=d_valid, other=1.0,
            ).to(tl.float32)

            # INT8×INT8 dot → INT32 tensor cores
            S_int32 = tl.dot(Q_block, tl.trans(D_block))

            # Dequant: S_real[q,d] = S_int32[q,d] * scale_q[q] * scale_d[d]
            S = S_int32.to(tl.float32) * q_sc[:, None] * d_sc[None, :]
            S = tl.where(d_valid[None, :], S, float("-inf"))
            m = tl.maximum(m, tl.max(S, axis=1))

        m = tl.where(q_valid, m, 0.0)
        score_acc += tl.sum(m)

    tl.store(scores_ptr + q_idx * B + doc_id, score_acc)


def quantize_query_int8(Q: torch.Tensor):
    """Per-token symmetric INT8 quantization for queries.

    Args:
        Q: [Lq, d] float/half tensor.

    Returns:
        Q_int8:  [Lq, d] int8
        q_scales: [Lq] float16
    """
    Qf = Q.float()
    absmax = Qf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    q_scales = (absmax / 127.0).half().squeeze(-1)
    Q_int8 = (Qf / absmax * 127.0).round().clamp(-127, 127).to(torch.int8)
    return Q_int8, q_scales


def flash_maxsim_int8x8(
    Q: torch.Tensor,
    D_int8: torch.Tensor,
    d_scales: torch.Tensor,
    Q_int8: torch.Tensor = None,
    q_scales: torch.Tensor = None,
    doc_lengths=None,
    query_chunk_size: int = 128,
) -> torch.Tensor:
    """True INT8×INT8 MaxSim — both operands INT8, uses INT8 tensor cores.

    D should be pre-quantized at index time with quantize_int8_symmetric().
    Q can be pre-quantized with quantize_query_int8(), or passed as FP16
    (quantized on the fly — cheap for a single query).

    A100 INT8 tensor cores: 624 TOPS (2x FP16's 312 TFLOPS).

    Args:
        Q:        [Lq, d] float16 query (used if Q_int8 not provided)
        D_int8:   [B, Ld, d] int8 pre-quantized docs
        d_scales: [B, Ld] float16 per-token doc scales
        Q_int8:   [Lq, d] int8 pre-quantized query (optional)
        q_scales: [Lq] float16 per-token query scales (optional)
        doc_lengths: optional [B] int32
        query_chunk_size: chunk Q for occupancy (default 128)

    Returns:
        scores: [B] float32
    """
    if Q_int8 is None or q_scales is None:
        Q_int8, q_scales = quantize_query_int8(Q)

    assert Q_int8.dim() == 2 and D_int8.dim() == 3 and Q_int8.shape[1] == D_int8.shape[2]
    Lq, d = Q_int8.shape
    B, Ld, _ = D_int8.shape

    if d_scales.dim() == 3:
        d_scales = d_scales.squeeze(-1)

    if doc_lengths is None:
        doc_lengths = torch.full((B,), Ld, device=D_int8.device, dtype=torch.int32)
    else:
        doc_lengths = doc_lengths.to(torch.int32).contiguous()

    from .flash_maxsim import _next_pow2
    d_pad = _next_pow2(d)

    # Q chunking
    if query_chunk_size is not None and Lq > query_chunk_size:
        C = query_chunk_size
        Nq = (Lq + C - 1) // C
        if Lq % C != 0:
            Q_int8 = torch.nn.functional.pad(Q_int8, (0, 0, 0, Nq * C - Lq))
            q_scales = torch.nn.functional.pad(q_scales, (0, Nq * C - Lq))
        Q_chunked = Q_int8.view(Nq, C, d)
        q_scales_chunked = q_scales.view(Nq, C)
        actual_Lq = C
    else:
        Q_chunked = Q_int8.unsqueeze(0)
        q_scales_chunked = q_scales.unsqueeze(0)
        Nq = 1
        actual_Lq = Lq

    Q_chunked = Q_chunked.contiguous()
    D_int8 = D_int8.contiguous()
    q_scales_chunked = q_scales_chunked.contiguous()
    d_scales = d_scales.contiguous().half()
    scores = torch.empty(Nq, B, device=Q_int8.device, dtype=torch.float32)

    _maxsim_int8x8_kernel[(Nq * B,)](
        Q_chunked, D_int8, q_scales_chunked, d_scales, doc_lengths, scores,
        Nq, B, actual_Lq, Ld, d, d_pad,
        Q_chunked.stride(0), Q_chunked.stride(1), Q_chunked.stride(2),
        D_int8.stride(0), D_int8.stride(1), D_int8.stride(2),
        d_scales.stride(0), d_scales.stride(1),
    )
    return scores.squeeze(0) if Nq == 1 else scores.sum(dim=0)
