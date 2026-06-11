"""Benchmark kernel optimizations: L2 cache for Q, varlen packing.

Compares current kernel vs optimized versions.

Usage: python benchmarks/bench_optimizations.py
"""
import torch, torch.nn.functional as F

assert torch.cuda.is_available(), "CUDA required"
print(f"GPU: {torch.cuda.get_device_name()}\n")

import triton
import triton.language as tl
from flash_maxsim import flash_maxsim
from flash_maxsim.flash_maxsim import _next_pow2, _get_configs, _prune_configs
from utils import bench_interleaved, compile_high_precision

# ── Optimization 1: Swizzled grid for L2 cache reuse ──
# Current: pid = q_idx * B + doc_idx (docs are contiguous)
# Swizzled: group docs into clusters so nearby pids share Q in L2

@triton.autotune(configs=_get_configs(), key=["Lq", "Ld", "d_pad"],
                 prune_configs_by={"early_config_prune": _prune_configs})
@triton.jit
def _maxsim_l2_kernel(
    Q_ptr, D_ptr, lengths_ptr, scores_ptr,
    B: tl.constexpr,
    Lq: tl.constexpr, Ld: tl.constexpr, d: tl.constexpr, d_pad: tl.constexpr,
    stride_q_l, stride_q_d,
    stride_d_b, stride_d_l, stride_d_d,
    BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr,
):
    doc_idx = tl.program_id(0)
    if doc_idx >= B:
        return

    doc_len = tl.load(lengths_ptr + doc_idx).to(tl.int32)
    k_off = tl.arange(0, d_pad)
    k_mask = k_off < d
    score_acc = tl.zeros([], dtype=tl.float32)

    # Load Q blocks OUTSIDE the D loop — keep Q in registers/L1
    for q_start in tl.static_range(0, Lq, BLOCK_Q):
        q_off = q_start + tl.arange(0, BLOCK_Q)
        q_valid = q_off < Lq

        # Load Q tile (this stays in registers across D iterations)
        Q_block = tl.load(
            Q_ptr + q_off[:, None] * stride_q_l + k_off[None, :] * stride_q_d,
            mask=q_valid[:, None] & k_mask[None, :], other=0.0,
        ).to(tl.float16)

        m = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)

        for d_start in range(0, Ld, BLOCK_D):
            d_off = d_start + tl.arange(0, BLOCK_D)
            d_valid = d_off < doc_len

            D_block = tl.load(
                D_ptr + doc_idx * stride_d_b + d_off[:, None] * stride_d_l + k_off[None, :] * stride_d_d,
                mask=d_valid[:, None] & k_mask[None, :], other=0.0,
            ).to(tl.float16)

            S = tl.dot(Q_block, tl.trans(D_block))
            S = tl.where(d_valid[None, :], S, float("-inf"))
            m = tl.maximum(m, tl.max(S, axis=1))

        m = tl.where(q_valid, m, 0.0)
        score_acc += tl.sum(m)

    tl.store(scores_ptr + doc_idx, score_acc)


def flash_maxsim_l2(Q, D, doc_lengths=None):
    """Flash-MaxSim with L2 cache optimization for Q."""
    Lq, d = Q.shape
    B, Ld, _ = D.shape
    Q = Q.contiguous().half()
    D = D.contiguous().half()
    if doc_lengths is None:
        doc_lengths = torch.full((B,), Ld, device=D.device, dtype=torch.int32)
    scores = torch.empty(B, device=Q.device, dtype=torch.float32)
    d_pad = _next_pow2(d)
    _maxsim_l2_kernel[(B,)](
        Q, D, doc_lengths, scores,
        B, Lq, Ld, d, d_pad,
        Q.stride(0), Q.stride(1),
        D.stride(0), D.stride(1), D.stride(2),
    )
    return scores


# ── Optimization 2: Varlen packed D ──
# Pack all doc tokens contiguously, use cu_seqlens for boundaries

@triton.autotune(configs=_get_configs(), key=["Lq", "max_Ld", "d_pad"],
                 prune_configs_by={"early_config_prune": _prune_configs})
@triton.jit
def _maxsim_varlen_kernel(
    Q_ptr, D_packed_ptr, cu_seqlens_ptr, scores_ptr,
    B: tl.constexpr,
    Lq: tl.constexpr, max_Ld: tl.constexpr, d: tl.constexpr, d_pad: tl.constexpr,
    stride_q_l, stride_q_d,
    stride_d_t, stride_d_d,
    BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr,
):
    doc_idx = tl.program_id(0)
    if doc_idx >= B:
        return

    # Variable length from cu_seqlens
    start = tl.load(cu_seqlens_ptr + doc_idx).to(tl.int32)
    end = tl.load(cu_seqlens_ptr + doc_idx + 1).to(tl.int32)
    doc_len = end - start

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

        for d_start in range(0, max_Ld, BLOCK_D):
            d_off = d_start + tl.arange(0, BLOCK_D)
            d_valid = d_off < doc_len

            # Load from packed buffer using offset
            D_block = tl.load(
                D_packed_ptr + (start + d_off[:, None]) * stride_d_t + k_off[None, :] * stride_d_d,
                mask=d_valid[:, None] & k_mask[None, :], other=0.0,
            ).to(tl.float16)

            S = tl.dot(Q_block, tl.trans(D_block))
            S = tl.where(d_valid[None, :], S, float("-inf"))
            m = tl.maximum(m, tl.max(S, axis=1))

        m = tl.where(q_valid, m, 0.0)
        score_acc += tl.sum(m)

    tl.store(scores_ptr + doc_idx, score_acc)


def pack_varlen(D, doc_lengths=None):
    """Pack D into contiguous buffer with cu_seqlens."""
    B, Ld, d = D.shape
    if doc_lengths is None:
        doc_lengths = torch.full((B,), Ld, device=D.device, dtype=torch.int32)
    cu_seqlens = torch.zeros(B + 1, device=D.device, dtype=torch.int32)
    cu_seqlens[1:] = doc_lengths.cumsum(0)
    total = cu_seqlens[-1].item()
    D_packed = torch.zeros(total, d, device=D.device, dtype=D.dtype)
    for i in range(B):
        L = doc_lengths[i].item()
        D_packed[cu_seqlens[i]:cu_seqlens[i]+L] = D[i, :L]
    return D_packed, cu_seqlens


def flash_maxsim_varlen(Q, D_packed, cu_seqlens, max_Ld):
    """Flash-MaxSim with varlen packed D."""
    Lq, d = Q.shape
    B = cu_seqlens.shape[0] - 1
    Q = Q.contiguous().half()
    D_packed = D_packed.contiguous().half()
    scores = torch.empty(B, device=Q.device, dtype=torch.float32)
    d_pad = _next_pow2(d)
    _maxsim_varlen_kernel[(B,)](
        Q, D_packed, cu_seqlens, scores,
        B, Lq, max_Ld, d, d_pad,
        Q.stride(0), Q.stride(1),
        D_packed.stride(0), D_packed.stride(1),
    )
    return scores


def naive_maxsim(Q, D):
    return torch.einsum('qd,bld->bql', Q.float(), D.float()).max(2).values.sum(1)

compiled_maxsim = compile_high_precision(naive_maxsim)

print("=" * 65)
print("CORRECTNESS")
print("=" * 65)

for B, Lq, Ld, d in [(100, 32, 300, 128), (50, 1024, 1024, 128)]:
    Q = F.normalize(torch.randn(Lq, d, device='cuda', dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(B, Ld, d, device='cuda', dtype=torch.float16), dim=-1)

    ref = naive_maxsim(Q, D)
    cur = flash_maxsim(Q, D)
    l2 = flash_maxsim_l2(Q, D)

    # Varlen
    lengths = torch.full((B,), Ld, device='cuda', dtype=torch.int32)
    D_packed, cu_seqlens = pack_varlen(D, lengths)
    vl = flash_maxsim_varlen(Q, D_packed, cu_seqlens, Ld)

    err_cur = (ref - cur).abs().max().item()
    err_l2 = (ref - l2).abs().max().item()
    err_vl = (ref - vl).abs().max().item()
    print(f"  B={B}, Lq={Lq}, Ld={Ld}: current={err_cur:.2e}  l2={err_l2:.2e}  varlen={err_vl:.2e}")

print(f"\n{'=' * 65}")
print("SPEED COMPARISON")
print("=" * 65)

for B, Lq, Ld, d, label in [
    (1000, 32,   300,  128, "Textual (Lq=32, Ld=300)"),
    (500,  32,   1024, 128, "Long-doc (Lq=32, Ld=1024)"),
    (1000, 1024, 1024, 128, "Visual (Lq=1024, Ld=1024)"),
    (5000, 1024, 1024, 128, "Visual large (B=5K)"),
]:
    Q = F.normalize(torch.randn(Lq, d, device='cuda', dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(B, Ld, d, device='cuda', dtype=torch.float16), dim=-1)
    lengths = torch.full((B,), Ld, device='cuda', dtype=torch.int32)
    D_packed, cu_seqlens = pack_varlen(D, lengths)

    t_naive, t_compiled, t_cur, t_l2, t_vl = bench_interleaved([naive_maxsim, compiled_maxsim, flash_maxsim,
                                                          flash_maxsim_l2, flash_maxsim_varlen],
                                                         [[Q, D], [Q, D], [Q, D], [Q, D],
                                                          [Q, D_packed, cu_seqlens, Ld]])

    print(f"\n  {label}")
    print(f"    Naive:     {t_naive:.3f} ms")
    print(f"    Compiled:     {t_compiled:.3f} ms")
    print(f"    Current:   {t_cur:.3f} ms  ({t_naive/t_cur:.1f}x vs naive {t_compiled/t_cur:.1f}x vs compiled)")
    print(f"    L2-opt:    {t_l2:.3f} ms  ({t_naive/t_l2:.1f}x vs naive, {t_compiled/t_l2:.1f}x vs compiled, "
          f"{t_cur/t_l2:.2f}x vs current)")
    print(f"    Varlen:    {t_vl:.3f} ms  ({t_naive/t_vl:.1f}x vs naive, {t_compiled/t_vl:.1f}x vs compiled,"
          f" {t_cur/t_vl:.2f}x vs current)")

    del D_packed, cu_seqlens; torch.cuda.empty_cache()

# ── Variable length benefit ──
print(f"\n{'=' * 65}")
print("VARIABLE LENGTH BENEFIT (mixed doc lengths)")
print("=" * 65)

for B, Lq, max_Ld, d, label in [
    (1000, 32, 1024, 128, "Mixed docs (max_Ld=1024)"),
    (1000, 1024, 1024, 128, "Visual mixed (max_Ld=1024)"),
]:
    Q = F.normalize(torch.randn(Lq, d, device='cuda', dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(B, max_Ld, d, device='cuda', dtype=torch.float16), dim=-1)

    # Mix of lengths: 20% short (128), 60% medium (512), 20% full (1024)
    lengths = torch.zeros(B, device='cuda', dtype=torch.int32)
    lengths[:B//5] = 128
    lengths[B//5:4*B//5] = 512
    lengths[4*B//5:] = max_Ld

    wup = 5 if Lq >= 512 else 10
    iters = 20 if Lq >= 512 else 40

    # t_cur - current with doc_lengths, t_vl - varlen packed, t_padded - current without lengths
    D_packed, cu_seqlens = pack_varlen(D, lengths)
    t_cur, t_vl, t_padded = bench_interleaved([flash_maxsim, flash_maxsim_varlen, flash_maxsim],
                              [[Q, D, lengths], [Q, D_packed, cu_seqlens, max_Ld], [Q, D]])

    mem_padded = B * max_Ld * d * 2 / 1e6
    mem_packed = D_packed.nbytes / 1e6

    print(f"\n  {label}")
    print(f"    Padded (no lengths): {t_padded:.3f} ms  mem={mem_padded:.0f}MB")
    print(f"    Current (lengths):   {t_cur:.3f} ms")
    print(f"    Varlen packed:       {t_vl:.3f} ms  mem={mem_packed:.0f}MB ({mem_padded/mem_packed:.1f}x less)")

    del D_packed, cu_seqlens; torch.cuda.empty_cache()

print(f"\n{'=' * 65}")
print("DONE")
print("=" * 65)
