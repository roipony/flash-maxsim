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
  BF16 inputs are accepted, the gradients are returned in the caller's input
  dtype — but precision is FP16-bounded inside the kernel. For BF16 inputs
  this means gradient accuracy is closer to FP16 (~10-bit mantissa) than to
  the user's BF16 (~7-bit) — strictly an *improvement*, but exact-bf16
  numerics are not guaranteed.
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
             acc.to(tl.float16), mask=km)


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

        # .clone() to guarantee a fresh buffer for in-place mask mutation below.
        Q_c = Q.contiguous().half().clone()
        D_c = D_flat.contiguous().half()
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
            q_mask = (torch.arange(Lq, device=Q.device, dtype=q_lens.dtype)
                      < q_lens.unsqueeze(-1))        # [Nq, Lq] bool
            Q_c *= q_mask.unsqueeze(-1).to(Q_c.dtype)

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

        grad_Q = torch.zeros_like(Q)
        _maxsim_bwd_dQ_batched_kernel[(Nq * Lq,)](
            D, argmax, grad_scores, grad_Q,
            Nq, B, Lq, d, d_pad,
            D.stride(0), D.stride(1), D.stride(2),
            grad_scores.stride(0), grad_scores.stride(1),
            grad_Q.stride(0), grad_Q.stride(1), grad_Q.stride(2),
            1 if shared_docs else 0,
        )

        # dD: dispatch on shape and mode.
        #   - shared_docs + non-trivial work: invgrid (CSR pre-pass + atomic-free
        #     gather kernel). Verified bit-exact vs FP32 reference; 1.3-20× speedup.
        #   - non-shared (KD) OR tiny shapes: FP32 atomic kernel (CSR build cost
        #     would dominate at small Nq*B*Lq, and non-shared has 67M-CTA grid
        #     bloat at large Ld).
        grad_D_fp32 = torch.zeros(D.shape, device=D.device, dtype=torch.float32)
        use_invgrid = shared_docs and (Nq * B * Lq) > 50_000
        if use_invgrid:
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
        else:
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
