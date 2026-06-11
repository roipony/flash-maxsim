"""Batched MaxSim with autograd for contrastive and distillation training.

Computes `scores[Nq, B]` from `Q[Nq, Lq, d]` and either:
  - `D[B, Ld, d]`         when `shared_docs=True`  (contrastive in-batch negs)
  - `D[Nq, B, Ld, d]`     when `shared_docs=False` (knowledge-distillation)

Unlike dense reference paths that materialize `[Nq, B, Lq, Ld]`, this
implementation reuses the fused MaxSim forward with saved argmax indices and
computes the backward from the sparse winner structure.

Forward:
  - reuses `_maxsim_fwd_kernel(..., save_argmax=True)` — one kernel launch.

Backward (multiple kernel launches):
  - dQ: per-(q_idx, lq_idx) CTA gathers the winning doc rows and reduces over B.
  - dD: dispatched by shape/mode —
      * small / non-shared  → FP32 atomic scatter kernel
      * large shared-docs   → inverse-grid CSR grouped reduction (atomic-free)

Variable-length handling:
  If `query_lengths` is provided, padded query positions are zeroed in the
  saved `Q` buffer and post-masked in `grad_Q`, so gradients are exact even
  when the user pads queries with non-zero values.

Precision:
  Math runs in FP16 tensor cores with FP32 accumulation internally. FP32 and
  BF16 inputs are accepted; the *kernel* downcasts to FP16 inline on every
  tile load (see flash_maxsim.py `.to(tl.float16)` at the Q/D-block loads),
  so the saved Q/D tensors stay in the caller's input dtype. Gradients are
  returned in the caller's input dtype. Precision is FP16-bounded inside the
  kernel — for BF16 inputs that means matmul accuracy is closer to FP16
  (~10-bit mantissa) than to the user's BF16 (~7-bit), strictly an
  *improvement*; exact-bf16 numerics are not guaranteed but the FP32
  accumulator and final cast keep gradients well within bf16 noise.
"""
import torch
import triton
import triton.language as tl

from .flash_maxsim import (
    _next_pow2, _launch_fwd, _default_lengths,
)


# ---------------------------------------------------------------------------
# Backward kernel: dQ
# Grid: (Nq * Lq,)  one CTA per (query, query-token)
#   - For each doc b (shared or per-query), read argmax and the winning D row,
#     accumulate `grad_score * D[winning_row]` into grad_Q[q_idx, lq_idx, :].
# ---------------------------------------------------------------------------

@triton.jit
def _maxsim_bwd_dQ_batched_kernel(
    D_ptr, argmax_ptr, grad_s_ptr, grad_Q_ptr,
    Nq: tl.constexpr, B: tl.constexpr, Lq: tl.constexpr, d: tl.constexpr, d_pad: tl.constexpr,
    stride_d_b, stride_d_l, stride_d_d,
    stride_gs_n, stride_gs_b,
    stride_gq_n, stride_gq_l, stride_gq_d,
    shared_docs: tl.constexpr,
):
    pid = tl.program_id(0)
    q_idx = pid // Lq
    lq_idx = pid % Lq
    if q_idx >= Nq:
        return

    k = tl.arange(0, d_pad)
    km = k < d
    acc = tl.zeros([d_pad], dtype=tl.float32)

    for b in range(B):
        # pair_idx matches the forward's CTA ordering so argmax layout aligns.
        if shared_docs:
            pair_idx = b * Nq + q_idx          # fwd: pid = doc_idx * Nq + q_idx
            d_batch = b                         # D is [B, Ld, d]
        else:
            pair_idx = q_idx * B + b            # fwd: pid = q_idx * B + doc_idx
            d_batch = q_idx * B + b             # D is [Nq*B, Ld, d] (flattened)

        gs = tl.load(grad_s_ptr + q_idx * stride_gs_n + b * stride_gs_b).to(tl.float32)
        j = tl.load(argmax_ptr + pair_idx * Lq + lq_idx)
        j_i64 = tl.cast(j, tl.int64)
        db_i64 = tl.cast(d_batch, tl.int64)
        v = tl.load(
            D_ptr + db_i64 * stride_d_b + j_i64 * stride_d_l + k * stride_d_d,
            mask=km, other=0.0,
        ).to(tl.float32)
        acc += gs * v

    # Stride-safe store — don't assume contiguous grad_Q layout.
    tl.store(grad_Q_ptr + q_idx * stride_gq_n + lq_idx * stride_gq_l + k * stride_gq_d,
             acc.to(grad_Q_ptr.dtype.element_ty), mask=km)


# ---------------------------------------------------------------------------
# Backward kernel: dD
# Grid: (B,) shared_docs or (Nq*B,) non-shared.  For shared, every (q, lq)
# argmax that matches `j` scatters a contribution; we atomic_add across Nq*Lq.
# ---------------------------------------------------------------------------

@triton.jit
def _maxsim_bwd_dD_batched_kernel(
    Q_ptr, argmax_ptr, grad_s_ptr, grad_D_ptr,
    Nq: tl.constexpr, B: tl.constexpr, Lq: tl.constexpr, Ld, d: tl.constexpr, d_pad: tl.constexpr,
    stride_d_b, stride_d_l, stride_d_d,
    stride_q_n, stride_q_l, stride_q_d,
    stride_gs_n, stride_gs_b,
    shared_docs: tl.constexpr,
):
    pid = tl.program_id(0)
    if shared_docs:
        doc_id = pid
        if doc_id >= B:
            return
    else:
        q_idx_outer = pid // B
        doc_id = pid % B
        if q_idx_outer >= Nq:
            return

    k = tl.arange(0, d_pad)
    km = k < d
    db_i64 = tl.cast(doc_id, tl.int64)
    # For non-shared, the "D batch index" in the [Nq*B, Ld, d] view is q_idx_outer*B + doc_id.
    if not shared_docs:
        db_i64 = tl.cast(q_idx_outer * B + doc_id, tl.int64)

    for q_idx in range(Nq):
        # In non-shared mode, only the owner-query contributes to this doc.
        # Triton 3.5 doesn't support `continue` — use a guard instead.
        owns_doc = shared_docs or (q_idx == q_idx_outer)
        if owns_doc:
            gs = tl.load(grad_s_ptr + q_idx * stride_gs_n + doc_id * stride_gs_b).to(tl.float32)

            if shared_docs:
                pair_idx = doc_id * Nq + q_idx
            else:
                pair_idx = q_idx * B + doc_id

            for lq_idx in range(Lq):
                j = tl.load(argmax_ptr + pair_idx * Lq + lq_idx)
                j_i64 = tl.cast(j, tl.int64)
                qv = tl.load(
                    Q_ptr + q_idx * stride_q_n + lq_idx * stride_q_l + k * stride_q_d,
                    mask=km, other=0.0,
                ).to(tl.float32)
                # FP32 atomic — much faster than FP16 atomic on Ampere, and avoids
                # precision loss on hot doc positions where many queries collide.
                tl.atomic_add(
                    grad_D_ptr + db_i64 * stride_d_b + j_i64 * stride_d_l + k * stride_d_d,
                    gs * qv, mask=km,
                )


# ---------------------------------------------------------------------------
# Backward kernel: unified dQ+dD (single-launch, Q-register-hoisted)
# Grid: (Nq*Lq,)  one program per (query, query-token), same as dQ.
#   - Loads Q[q_idx, lq_idx, :] ONCE into a register and reuses across docs.
#   - For each doc b: loads gs and argmax j, then loads D[b, j, :] ONCE,
#     uses it for dQ accumulation AND multiplies by hoisted Q for dD atomic.
#   - Single kernel launch instead of two; ~50% fewer kernel-launch overheads,
#     and avoids reloading D for the dD pass.
# ---------------------------------------------------------------------------

# Per-arch heuristic table for atomic_unified launch config. Replaces
# @triton.autotune (which costs first-call autotune trials + non-determinism).
# Numbers swept empirically across all ColBERT/ColPali contrastive shapes
# on H100 and A100 — see benchmarks/bench_config_sweep_*.json. The table
# below covers every measured shape within ~2% of the per-shape optimum
# and generalises to unseen shapes by bucketing on grid size (Nq*Lq).
# Pattern inspired by Flash-KMeans's _heuristic_euclid_config
# (svg-project/flash-kmeans/.../assign_euclid_triton.py:568-598).
_BWD_UNIFIED_ARCH_CACHE: dict[int, str] = {}

def _bwd_unified_arch(device_index: int) -> str:
    if device_index not in _BWD_UNIFIED_ARCH_CACHE:
        name = torch.cuda.get_device_properties(device_index).name
        if any(x in name for x in ("H100", "H200", "B100", "B200")):
            _BWD_UNIFIED_ARCH_CACHE[device_index] = "hopper"
        elif any(x in name for x in ("A100", "A800")):
            _BWD_UNIFIED_ARCH_CACHE[device_index] = "ampere"
        else:
            _BWD_UNIFIED_ARCH_CACHE[device_index] = "other"
    return _BWD_UNIFIED_ARCH_CACHE[device_index]


def _bwd_unified_config(Nq: int, Lq: int, device_index: int = 0) -> tuple[int, int]:
    """Return (num_warps, num_stages) for the unified backward kernel.

    Buckets on grid size Nq*Lq (= number of programs launched). On H100
    every measured shape preferred num_warps=1 + deep pipeline. On A100
    the grid-size bucket matters: small grids need more warps per program
    for ILP, large grids tolerate fewer warps and benefit from deeper
    pipelining.
    """
    arch = _bwd_unified_arch(device_index)
    grid = Nq * Lq
    if arch == "hopper":
        # Sweep: (1, 3) is fastest on 5 of 6 shapes; (1, 2) within 1% on the 6th.
        return (1, 3)
    elif arch == "ampere":
        if grid <= 8192:                    # tiny grid (e.g. ColBERT B<=256, Lq=32)
            return (4, 3)
        elif grid <= 65536:                 # medium (ColBERT B=512/1024, ColPali B=32)
            return (2, 3)
        else:                               # large (ColPali B>=64, Lq=1024)
            return (2, 2)
    else:
        return (4, 2)                       # conservative default


@triton.jit
def _maxsim_bwd_unified_kernel(
    Q_ptr, D_ptr, argmax_ptr, grad_s_ptr,
    grad_Q_ptr, grad_D_ptr,
    Nq: tl.constexpr, B: tl.constexpr, Lq: tl.constexpr, Ld,
    d: tl.constexpr, d_pad: tl.constexpr,
    stride_d_b, stride_d_l, stride_d_d,
    stride_q_n, stride_q_l, stride_q_d,
    stride_gs_n, stride_gs_b,
    stride_gq_n, stride_gq_l, stride_gq_d,
    shared_docs: tl.constexpr,
):
    pid = tl.program_id(0)
    q_idx = pid // Lq
    lq_idx = pid % Lq
    if q_idx >= Nq:
        return

    k = tl.arange(0, d_pad)
    km = k < d

    # Hoist Q[q_idx, lq_idx, :] into a register — used unchanged for every doc
    # in this query token's loop. This is the key unified-backward trick: dD's
    # atomic_add reuses qv that we already needed for dQ's read of D.
    qv = tl.load(
        Q_ptr + q_idx * stride_q_n + lq_idx * stride_q_l + k * stride_q_d,
        mask=km, other=0.0,
    ).to(tl.float32)

    acc_q = tl.zeros([d_pad], dtype=tl.float32)

    for b in range(B):
        if shared_docs:
            pair_idx = b * Nq + q_idx
            d_batch = b
        else:
            pair_idx = q_idx * B + b
            d_batch = q_idx * B + b

        gs = tl.load(grad_s_ptr + q_idx * stride_gs_n + b * stride_gs_b).to(tl.float32)
        j = tl.load(argmax_ptr + pair_idx * Lq + lq_idx)
        j_i64 = tl.cast(j, tl.int64)
        db_i64 = tl.cast(d_batch, tl.int64)
        # Load D[d_batch, j, :] — same row used for dQ accumulation AND as the
        # atomic-add target for dD. We pay one global load, two uses.
        dv = tl.load(
            D_ptr + db_i64 * stride_d_b + j_i64 * stride_d_l + k * stride_d_d,
            mask=km, other=0.0,
        ).to(tl.float32)
        acc_q += gs * dv
        # dD: scatter gs*qv into grad_D[d_batch, j, :]. FP32 atomic.
        tl.atomic_add(
            grad_D_ptr + db_i64 * stride_d_b + j_i64 * stride_d_l + k * stride_d_d,
            gs * qv, mask=km,
        )

    tl.store(grad_Q_ptr + q_idx * stride_gq_n + lq_idx * stride_gq_l + k * stride_gq_d,
             acc_q.to(grad_Q_ptr.dtype.element_ty), mask=km)


# ---------------------------------------------------------------------------
# INT8-D variant of the unified backward kernel.
# Identical to `_maxsim_bwd_unified_kernel` except D arrives as INT8 +
# per-token FP16 scales; dequant is fused inline (one multiply per row
# load). Used by flash_maxsim_int8_train.py Phase 2 to skip the
# pre-dequant transient FP16 buffer — saved INT8 D stays as the only D
# representation through the entire backward, halving the D footprint.
# ---------------------------------------------------------------------------

@triton.jit
def _maxsim_bwd_unified_int8d_kernel(
    Q_ptr, D_int8_ptr, d_scales_ptr, argmax_ptr, grad_s_ptr,
    grad_Q_ptr, grad_D_ptr,
    Nq: tl.constexpr, B: tl.constexpr, Lq: tl.constexpr, Ld,
    d: tl.constexpr, d_pad: tl.constexpr,
    stride_d_b, stride_d_l, stride_d_d,                # D_int8: [B*?, Ld, d]
    stride_s_b, stride_s_l,                            # d_scales: [B*?, Ld] (or [..., Ld, 1] squeezed)
    stride_q_n, stride_q_l, stride_q_d,
    stride_gs_n, stride_gs_b,
    stride_gq_n, stride_gq_l, stride_gq_d,
    stride_gd_b, stride_gd_l, stride_gd_d,             # grad_D may differ from D in stride
    shared_docs: tl.constexpr,
):
    pid = tl.program_id(0)
    q_idx = pid // Lq
    lq_idx = pid % Lq
    if q_idx >= Nq:
        return

    k = tl.arange(0, d_pad)
    km = k < d

    qv = tl.load(
        Q_ptr + q_idx * stride_q_n + lq_idx * stride_q_l + k * stride_q_d,
        mask=km, other=0.0,
    ).to(tl.float32)

    acc_q = tl.zeros([d_pad], dtype=tl.float32)

    for b in range(B):
        if shared_docs:
            pair_idx = b * Nq + q_idx
            d_batch = b
        else:
            pair_idx = q_idx * B + b
            d_batch = q_idx * B + b

        gs = tl.load(grad_s_ptr + q_idx * stride_gs_n + b * stride_gs_b).to(tl.float32)
        j = tl.load(argmax_ptr + pair_idx * Lq + lq_idx)
        j_i64 = tl.cast(j, tl.int64)
        db_i64 = tl.cast(d_batch, tl.int64)
        # Load D[d_batch, j, :] as INT8, dequant via per-token scale.
        dv_int8 = tl.load(
            D_int8_ptr + db_i64 * stride_d_b + j_i64 * stride_d_l + k * stride_d_d,
            mask=km, other=0,
        )
        scale = tl.load(d_scales_ptr + db_i64 * stride_s_b + j_i64 * stride_s_l).to(tl.float32)
        dv = dv_int8.to(tl.float32) * scale
        acc_q += gs * dv
        tl.atomic_add(
            grad_D_ptr + db_i64 * stride_gd_b + j_i64 * stride_gd_l + k * stride_gd_d,
            gs * qv, mask=km,
        )

    tl.store(grad_Q_ptr + q_idx * stride_gq_n + lq_idx * stride_gq_l + k * stride_gq_d,
             acc_q.to(grad_Q_ptr.dtype.element_ty), mask=km)


# ---------------------------------------------------------------------------
# Autograd Function
# ---------------------------------------------------------------------------

class _FlashMaxSimBatchedFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, D, shared_docs: bool, doc_lengths, query_lengths):
        assert Q.dim() == 3, "Q must be [Nq, Lq, d]"
        Nq, Lq, d = Q.shape
        if shared_docs:
            assert D.dim() == 3, "D must be [B, Ld, d] when shared_docs=True"
            B, Ld, _ = D.shape
            D_flat = D
        else:
            assert D.dim() == 4, "D must be [Nq, B, Ld, d] when shared_docs=False"
            _, B, Ld, _ = D.shape
            D_flat = D.reshape(Nq * B, Ld, d)

        # Save original dtypes so backward can return gradients in user's dtype
        # (kernel math runs in FP16; we cast back on the way out).
        orig_Q_dtype = Q.dtype
        orig_D_dtype = D.dtype

        # --- Phase-0 memory cleanup (Jun 2026) ---
        # Previously we did `.half().clone()` on Q and `.half()` on D before
        # save_for_backward. That doubled Q+D in memory whenever the user
        # passed bf16/fp32, because the user's tensor stayed alive AND we
        # held a fp16 copy for the backward. The forward kernel already
        # downcasts to fp16 inline on every Q/D tile load (see
        # `flash_maxsim.py` `.to(tl.float16)` at the `tl.load` sites), so
        # the saved-for-backward copy can stay in the user's native dtype.
        # Saves ~1080 MB on colpali-neg B256 n16 Ld1030 (KD-4D layout).
        # Q_c needs to be contiguous (kernels use strided loads but expect
        # the standard last-dim-contiguous layout) and cloned ONLY when we
        # mutate it for the q_mask path below — otherwise we'd modify the
        # user's tensor in place.
        total = B if shared_docs else Nq * B
        lengths = _default_lengths(total, Ld, D.device, doc_lengths)
        q_lens = query_lengths.to(torch.int32).contiguous() if query_lengths is not None else None

        # --- Variable-length correctness ---
        # Forward masks padded query positions in the score reduction, so their
        # contribution to `scores` is 0. Backward, however, reads argmax and Q
        # for *all* (q, l) indices — if the user padded Q with non-zero values
        # (pylate's `colbert_scores` does that: it applies masks post-einsum),
        # padded rows would leak garbage into grad_D. Zero them in the SAVED Q
        # so the backward kernels see exact zeros for padded positions.
        q_mask = None
        if q_lens is not None:
            # Need a writable copy for the in-place mask multiply.
            Q_c = Q.contiguous().clone()
            q_mask = (torch.arange(Lq, device=Q.device, dtype=q_lens.dtype)
                      < q_lens.unsqueeze(-1))        # [Nq, Lq] bool
            Q_c *= q_mask.unsqueeze(-1).to(Q_c.dtype)
        else:
            # No masking → no mutation needed; saving the user's tensor
            # (or a contig view of it) is fine. autograd's save_for_backward
            # holds a reference, not a copy.
            Q_c = Q.contiguous()
        D_c = D_flat.contiguous()

        scores, argmax = _launch_fwd(
            Q_c, D_c, lengths, Nq, B, Lq, Ld, d, shared_docs, True, q_lengths=q_lens,
        )
        ctx.save_for_backward(Q_c, D_c, argmax)
        ctx.shared_docs = shared_docs
        ctx.shape = (Nq, B, Lq, Ld, d)
        ctx.q_mask = q_mask                          # None if no varlen
        ctx.orig_Q_dtype = orig_Q_dtype
        ctx.orig_D_dtype = orig_D_dtype
        return scores   # [Nq, B] FP32

    @staticmethod
    def backward(ctx, grad_scores):
        Q, D, argmax = ctx.saved_tensors
        Nq, B, Lq, Ld, d = ctx.shape
        shared_docs = ctx.shared_docs
        grad_scores = grad_scores.contiguous().float()
        d_pad = _next_pow2(d)

        # Three backward paths:
        #   - "invgrid"        : atomic-free CSR scatter into grad_D (fastest at
        #                        large Nq*B*Lq; pays a few hundred MB transient
        #                        for the CSR build, mitigated by int32 indices).
        #   - "atomic"         : two-kernel atomic scatter (legacy; slow).
        #   - "atomic_unified" : fused dQ+dD with Q-hoisting; same
        #                        memory as plain atomic but ~2x faster because
        #                        it amortises one D load + one Q register across
        #                        both grad accumulations.
        #   - "sparse_dd"      : sort touched dD destinations and write only
        #                        non-zero rows (target: non-shared KD layout).
        # `FLASH_BWD_PATH={auto,atomic,atomic_unified,invgrid,sparse_dd}` env var
        # overrides. Default is deterministic: atomic_unified for the
        # shared-docs (in-batch-negatives) layout and plain atomic for KD
        # (shared_docs=False). sparse_dd is opt-in while benchmarked.
        import os as _os
        _force = _os.environ.get("FLASH_BWD_PATH", "auto")
        if _force == "atomic":
            mode = "atomic"
        elif _force == "atomic_unified":
            mode = "atomic_unified" if shared_docs else "atomic"
        elif _force == "invgrid":
            mode = "invgrid" if shared_docs else "atomic"
        elif _force in ("sparse", "sparse_dd"):
            mode = "sparse_dd"
        else:
            # Default: atomic_unified for shared_docs (fused dQ+dD with Q
            # register hoisting — fast and SRAM-light). Plain atomic for
            # KD/non-shared until sparse_dd has a demonstrated crossover.
            mode = "atomic_unified" if shared_docs else "atomic"

        if mode == "atomic_unified":
            # Single fused kernel — writes BOTH grad_Q and grad_D in one launch.
            # Config from the per-arch heuristic (no autotune trial overhead).
            grad_D_fp32 = torch.zeros(D.shape, device=D.device, dtype=torch.float32)
            grad_Q = torch.zeros_like(Q)
            nw, ns = _bwd_unified_config(Nq, Lq, Q.device.index or 0)
            _maxsim_bwd_unified_kernel[(Nq * Lq,)](
                Q, D, argmax, grad_scores,
                grad_Q, grad_D_fp32,
                Nq, B, Lq, Ld, d, d_pad,
                D.stride(0), D.stride(1), D.stride(2),
                Q.stride(0), Q.stride(1), Q.stride(2),
                grad_scores.stride(0), grad_scores.stride(1),
                grad_Q.stride(0), grad_Q.stride(1), grad_Q.stride(2),
                1,  # shared_docs (mode only chosen when shared_docs=True)
                num_warps=nw, num_stages=ns,
            )
        else:
            # Two-kernel paths share the same dQ launch.
            grad_Q = torch.zeros_like(Q)
            _maxsim_bwd_dQ_batched_kernel[(Nq * Lq,)](
                D, argmax, grad_scores, grad_Q,
                Nq, B, Lq, d, d_pad,
                D.stride(0), D.stride(1), D.stride(2),
                grad_scores.stride(0), grad_scores.stride(1),
                grad_Q.stride(0), grad_Q.stride(1), grad_Q.stride(2),
                1 if shared_docs else 0,
            )
            if mode == "invgrid":
                grad_D_fp32 = torch.zeros(D.shape, device=D.device, dtype=torch.float32)
                from .flash_maxsim_invgrid_proposal import (
                    build_inverse_csr, _maxsim_bwd_dD_invgrid_kernel,
                )
                row_ptr, col_idx = build_inverse_csr(argmax, Nq, B, Lq, Ld, shared_docs)
                n_dest = B * Ld  # shared_docs only here
                _maxsim_bwd_dD_invgrid_kernel[(n_dest,)](
                    Q, grad_scores, grad_D_fp32,
                    row_ptr, col_idx,
                    Nq, B, Lq, Ld, d, d_pad,
                    grad_D_fp32.stride(0), grad_D_fp32.stride(1), grad_D_fp32.stride(2),
                    Q.stride(0), Q.stride(1), Q.stride(2),
                    grad_scores.stride(0), grad_scores.stride(1),
                    1,  # shared_docs
                )
            elif mode == "sparse_dd":
                from .flash_maxsim_sparse_dd import _launch_bwd_dD_sparse
                grad_D_fp32 = _launch_bwd_dD_sparse(
                    Q, argmax, grad_scores, Nq, B, Lq, Ld, d, shared_docs,
                )
            else:
                # mode == "atomic" — legacy plain atomic scatter.
                grad_D_fp32 = torch.zeros(D.shape, device=D.device, dtype=torch.float32)
                grid_size = B if shared_docs else Nq * B
                _maxsim_bwd_dD_batched_kernel[(grid_size,)](
                    Q, argmax, grad_scores, grad_D_fp32,
                    Nq, B, Lq, Ld, d, d_pad,
                    grad_D_fp32.stride(0), grad_D_fp32.stride(1), grad_D_fp32.stride(2),
                    Q.stride(0), Q.stride(1), Q.stride(2),
                    grad_scores.stride(0), grad_scores.stride(1),
                    1 if shared_docs else 0,
                )
        # Post-mask grad_Q for padded query positions (dQ kernel writes
        # non-zero values at padded slots — the true gradient there is 0
        # because the forward masks them out of the score sum).
        if ctx.q_mask is not None:
            grad_Q *= ctx.q_mask.unsqueeze(-1).to(grad_Q.dtype)

        # Cast back to caller's input dtype (kernels work in FP16 internally).
        grad_Q = grad_Q.to(ctx.orig_Q_dtype)
        grad_D = grad_D_fp32.to(ctx.orig_D_dtype)

        # Reshape grad_D back to user shape for non-shared case
        if not shared_docs:
            grad_D = grad_D.reshape(Nq, B, Ld, d)

        return grad_Q, grad_D, None, None, None


def flash_maxsim_batched_train(
    Q: torch.Tensor, D: torch.Tensor,
    shared_docs: bool = True,
    doc_lengths: torch.Tensor | None = None,
    query_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Batched MaxSim with autograd backward.

    Args:
        Q: [Nq, Lq, d]           (query batch)
        D: [B, Ld, d]            when shared_docs=True (contrastive)
           [Nq, B, Ld, d]        when shared_docs=False (knowledge distillation)
        doc_lengths: optional per-doc lengths  (shape [B] or [Nq*B])
        query_lengths: optional per-query lengths  (shape [Nq])

    Returns:
        scores [Nq, B]  (FP32)
    """
    return _FlashMaxSimBatchedFn.apply(Q, D, shared_docs, doc_lengths, query_lengths)
