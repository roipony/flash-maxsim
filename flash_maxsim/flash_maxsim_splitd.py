"""Split-d forward kernel: tile the embedding dimension `d` so the kernel
runs at fat-embedding shapes (d = 384, 512, 768, 1024, 2048, ...).

Why this exists. The standard `_maxsim_fwd_kernel` (in `flash_maxsim.py`)
assumes the entire d-vector fits in a register tile via `d_pad =
next_pow2(d)`. At d ≥ 768 on A100 (or d ≥ 1024 on H100), the SRAM budget
for the Q + D operand tiles crosses the device limit. Triton compiles
the kernel anyway, but spills to local memory — `bench_fat_embeddings.py`
measured this as a 3-18× perf cliff at d > 512 on both arches, and a
60× off-roofline gap at d=2048 on A100.

The fix is the standard split-K matmul pattern: introduce a `BLOCK_K`
inner loop that tiles the `d` axis. Each iteration loads
`[BLOCK_Q, BLOCK_K]` from Q and `[BLOCK_D, BLOCK_K]` from D and
accumulates `S += Q_k @ D_k.T` in FP32. The row-max and sum reductions
that finish the kernel are unchanged.

This kernel is functionally identical to `_maxsim_fwd_kernel` (same
inputs, same outputs, same argmax layout) — the dispatcher in
`_launch_fwd_splitd` below picks this version when `d > _SPLITD_THRESH`.
For small d (≤ 256), the in-one-shot kernel stays faster (no inner-loop
overhead, full-width tensor cores). For large d, this kernel runs at
roofline instead of spilling.

Backward note. Backward dQ/dD/CSR kernels load whole rows from Q and D
(one d-vector at a time per program). A single d-vector at d=4096 fp16
is 8 KB — register-friendly. Backward likely doesn't need splitting up
to d=4096 or so. If it does, the same pattern applies.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .flash_maxsim import _next_pow2, _default_lengths


# ---------------------------------------------------------------------------
# Per-arch heuristic table for split-d launch config.
# Same pattern as _bwd_unified_config in flash_maxsim_batched_train.py:
# pick (BLOCK_Q, BLOCK_D, BLOCK_K, num_warps, num_stages) from a hand-tuned
# table keyed on (arch, d_bucket). No Triton autotune trial overhead.
# Numbers will be filled in once bench_splitd_correctness.py + the perf
# sweep both pass. Starting values are conservative defaults chosen to
# keep SRAM budget under control.
# ---------------------------------------------------------------------------

# Threshold: switch to split-d when d > this. 512 is the safe default —
# measured by bench_splitd_perf.py on both A100 and H100, split-d is at
# parity or faster than in-one-shot at every shape we tested for d >= 512,
# whereas at d in {256, 384} the inner-loop overhead loses vs in-one-shot
# on ColPali shape (textual is fine either way). The killer-feature region
# is d >= 768 where in-one-shot starts spilling and split-d wins by
# 1.2x to 4.5x.
_SPLITD_THRESH = 512


_SPLITD_ARCH_CACHE: dict[int, str] = {}

def _splitd_arch(device_index: int) -> str:
    if device_index not in _SPLITD_ARCH_CACHE:
        name = torch.cuda.get_device_properties(device_index).name
        if any(x in name for x in ("H100", "H200", "B100", "B200")):
            _SPLITD_ARCH_CACHE[device_index] = "hopper"
        elif any(x in name for x in ("A100", "A800")):
            _SPLITD_ARCH_CACHE[device_index] = "ampere"
        else:
            _SPLITD_ARCH_CACHE[device_index] = "other"
    return _SPLITD_ARCH_CACHE[device_index]


def _splitd_config(d: int, Lq: int, Ld: int, device_index: int = 0):
    """Return (BLOCK_Q, BLOCK_D, BLOCK_K, num_warps, num_stages) for the
    split-d forward kernel. Conservative defaults; tuned tables to follow
    after the empirical sweep."""
    arch = _splitd_arch(device_index)
    # BLOCK_K is the d-axis tile size. 64 keeps SRAM safe at any d; bumping
    # to 128 gives better tensor-core utilisation at the cost of SRAM.
    if arch == "hopper":
        if d >= 1024:
            return (64, 64, 64, 4, 3)
        return (64, 64, 128, 4, 3)
    elif arch == "ampere":
        # Ampere SRAM budget (164 KB) is tighter than H100. (32, 32, 64) was
        # the safe default, but bench_splitd_perf measured a 3x regression
        # at d=1024 ColPali — the small block hurts when the d-tile loop is
        # already long (16 iterations at d=1024 / BLOCK_K=64). Bigger
        # blocks + larger BLOCK_K at d >= 1024 reduce inner-loop overhead
        # at the cost of slightly higher SRAM (still under budget).
        if d >= 1024:
            return (64, 64, 128, 4, 2)
        return (64, 64, 64, 4, 2)
    else:
        return (32, 32, 64, 4, 2)


# ---------------------------------------------------------------------------
# Split-d forward kernel.
# ---------------------------------------------------------------------------

@triton.jit
def _maxsim_fwd_kernel_splitd(
    Q_ptr, D_ptr, lengths_ptr, q_lengths_ptr, scores_ptr, argmax_ptr,
    Nq, B,
    Lq: tl.constexpr, Ld, d: tl.constexpr,
    stride_q_n, stride_q_l, stride_q_d,
    stride_d_b, stride_d_l, stride_d_d,
    stride_s_n, stride_s_b,
    shared_docs: tl.constexpr,
    save_argmax: tl.constexpr,
    use_q_lengths: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Same external contract as `_maxsim_fwd_kernel` in flash_maxsim.py
    but with an inner BLOCK_K loop over the embedding dim. No d_pad
    constexpr — d-axis bounds are runtime-checked per tile."""
    pid = tl.program_id(0)
    if shared_docs:
        doc_idx = pid // Nq
        q_idx = pid % Nq
    else:
        q_idx = pid // B
        doc_idx = pid % B
    if q_idx >= Nq:
        return

    d_batch = tl.cast(doc_idx if shared_docs else q_idx * B + doc_idx, tl.int64)
    doc_len = tl.load(lengths_ptr + d_batch).to(tl.int32)
    q_len = tl.load(q_lengths_ptr + q_idx).to(tl.int32) if use_q_lengths else Lq

    score_acc = tl.zeros([], dtype=tl.float32)

    for q_start in tl.static_range(0, Lq, BLOCK_Q):
        q_off = q_start + tl.arange(0, BLOCK_Q)
        q_valid = q_off < q_len

        m = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
        m_idx = tl.full([BLOCK_Q], 0, dtype=tl.int32)

        for d_start in range(0, Ld, BLOCK_D):
            d_off = d_start + tl.arange(0, BLOCK_D)
            d_valid = d_off < doc_len

            # Accumulate Q·Dᵀ across BLOCK_K slices of the embedding dim.
            S = tl.zeros([BLOCK_Q, BLOCK_D], dtype=tl.float32)
            for k_start in range(0, d, BLOCK_K):
                k_off = k_start + tl.arange(0, BLOCK_K)
                k_mask = k_off < d
                Q_tile = tl.load(
                    Q_ptr + q_idx * stride_q_n
                          + q_off[:, None] * stride_q_l
                          + k_off[None, :] * stride_q_d,
                    mask=q_valid[:, None] & k_mask[None, :], other=0.0,
                ).to(tl.float16)
                D_tile = tl.load(
                    D_ptr + d_batch * stride_d_b
                          + d_off[:, None] * stride_d_l
                          + k_off[None, :] * stride_d_d,
                    mask=d_valid[:, None] & k_mask[None, :], other=0.0,
                ).to(tl.float16)
                # FP32 accumulation across d-tiles (same as the in-one-shot
                # kernel's single tl.dot, just split across iterations).
                S += tl.dot(Q_tile, tl.trans(D_tile))

            S = tl.where(d_valid[None, :], S, float("-inf"))
            tile_max = tl.max(S, axis=1)
            if save_argmax:
                tile_argmax = tl.argmax(S, axis=1).to(tl.int32) + d_start
                update = tile_max > m
                m_idx = tl.where(update, tile_argmax, m_idx)
            m = tl.maximum(m, tile_max)

        m = tl.where(q_valid, m, 0.0)
        score_acc += tl.sum(m)

        if save_argmax:
            tl.store(argmax_ptr + pid * Lq + q_off, m_idx, mask=q_valid)

    tl.store(scores_ptr + q_idx * stride_s_n + doc_idx * stride_s_b, score_acc)


def _launch_fwd_splitd(Q, D, lengths, Nq, B, Lq, Ld, d, shared_docs,
                       save_argmax, q_lengths=None):
    """Drop-in launcher for the split-d forward.

    Returns (scores, argmax) in the same layout as `_launch_fwd`. The
    dispatcher in the public API decides whether to call this or the
    in-one-shot path based on `d > _SPLITD_THRESH`.
    """
    # Correctness-wise this kernel works at any d (it's just split-K matmul).
    # The d > _SPLITD_THRESH guard belongs in the dispatcher, not here —
    # callers that bypass the dispatcher (e.g. unit tests) can run it at
    # any d.
    BLOCK_Q, BLOCK_D, BLOCK_K, num_warps, num_stages = _splitd_config(
        d, Lq, Ld, Q.device.index or 0,
    )
    scores = torch.empty(Nq, B, device=Q.device, dtype=torch.float32)
    # Grid: one program per (q_idx, doc_idx) pair. The kernel decodes pid
    # via `doc_idx = pid // Nq; q_idx = pid % Nq` for shared_docs (doc-major
    # so the same D doc is consecutive across the Nq programs that touch it,
    # keeping it L1-hot), and via `q_idx = pid // B; doc_idx = pid % B` for
    # non-shared. Grid size is Nq*B either way. The earlier bug was using
    # grid=(B,) for shared_docs, which left scores[q_idx>0] uninitialised
    # and produced garbage when Q-chunking (in flash_maxsim_batched) passes
    # Nq>1 chunked queries.
    total = Nq * B
    if save_argmax:
        argmax = torch.empty(total, Lq, device=Q.device, dtype=torch.int32)
    else:
        argmax = torch.empty(1, device=Q.device, dtype=torch.int32)

    use_q_lengths = q_lengths is not None
    if not use_q_lengths:
        q_lengths = torch.empty(1, device=Q.device, dtype=torch.int32)

    grid = (total,)
    _maxsim_fwd_kernel_splitd[grid](
        Q, D, lengths, q_lengths, scores, argmax,
        Nq, B,
        Lq, Ld, d,
        Q.stride(0), Q.stride(1), Q.stride(2),
        D.stride(0), D.stride(1), D.stride(2),
        scores.stride(0), scores.stride(1),
        1 if shared_docs else 0,
        1 if save_argmax else 0,
        1 if use_q_lengths else 0,
        BLOCK_Q=BLOCK_Q, BLOCK_D=BLOCK_D, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )
    return scores, argmax


if __name__ == "__main__":
    # Correctness check: split-d output must match a naive FP32 einsum
    # reference within FP16-cast noise (~1e-4 at ColPali shape).
    import torch.nn.functional as F
    torch.manual_seed(0)
    device = "cuda"

    def naive(Q, D):
        sim = torch.einsum('qid,bjd->qbij', Q.float(), D.float())
        return sim.max(dim=-1).values.sum(dim=-1)

    for d in [384, 512, 768, 1024, 2048]:
        Nq, B, Lq, Ld = 1, 32, 128, 256
        Q = F.normalize(torch.randn(Nq, Lq, d, device=device, dtype=torch.float16), dim=-1)
        D = F.normalize(torch.randn(B,  Ld, d, device=device, dtype=torch.float16), dim=-1)
        lengths = torch.full((B,), Ld, device=device, dtype=torch.int32)
        ref = naive(Q, D)
        scores, _ = _launch_fwd_splitd(
            Q, D, lengths, Nq, B, Lq, Ld, d,
            shared_docs=True, save_argmax=False,
        )
        max_abs = (ref - scores).abs().max().item()
        print(f"  d={d:>4d}: scores match: {torch.allclose(ref, scores, atol=1e-2, rtol=1e-2)}  max|Δ|={max_abs:.3e}")
