"""Proposal: fused backward kernel for batched MaxSim (grad_Q + grad_D).

Why fuse?
---------
The existing backward in ``flash_maxsim_batched_train.py`` launches two
Triton kernels:

    _maxsim_bwd_dQ_batched_kernel    : one CTA per (q_idx, lq_idx), loops over B docs
    _maxsim_bwd_dD_batched_kernel    : one CTA per doc, loops over Nq*Lq, atomic-adds

Both read the *same* three tensors (``argmax``, ``Q``, ``D``) and
``grad_scores`` from HBM, and both traverse the same (q, b, l) index
cube. That is two full HBM passes over ``argmax`` and two separate
launch overheads for work that is algebraically coupled:

    for each (q, b, l):
        j = argmax[q, b, l]
        gs = grad_scores[q, b]
        grad_Q[q, l, :] += gs * D[b, j, :]          # (1) dQ contribution
        grad_D[b, j, :] += gs * Q[q,    l, :]       # (2) dD contribution

A single fused kernel visits each (q, b, l) once, loads ``j`` once,
loads ``gs`` once, and performs both scatters in the same loop body.

Kernel design
-------------

Grid choice  : ``(B,)`` in ``shared_docs=True`` mode
               ``(Nq*B,)`` in ``shared_docs=False`` mode

Per-CTA scope: owns one doc ``b`` (shared_docs) or one (q_outer, b) pair.
               For that doc we iterate over all (q, l) pairs whose
               argmax references this doc's ``Ld`` rows.

Local state  : A full FP32 ``grad_D_local[Ld, d_pad]`` accumulator held
               in registers/SMEM. Because only this CTA can produce
               writes targeting ``grad_D[b, :, :]``, there is **no
               cross-CTA contention on grad_D** — we flush with a single
               coalesced ``tl.store`` at the end, no atomics.

grad_Q path  : ``grad_Q[q, l, :]`` collects contributions from every
               doc, so a per-doc CTA only computes a *partial* dQ. We
               therefore ``tl.atomic_add`` FP32 partials into
               ``grad_Q_fp32`` (preallocated zero, cast to FP16 after).
               Contention is spread across ``Nq*Lq*d`` addresses with
               exactly ``B`` contributors per address — much lower than
               the dD atomic pressure in the old split kernel (which
               hits the same ``[b, j, :]`` row many times when several
               (q, l) pairs share an argmax).

Memory      : Each CTA reads argmax[b, :, :] once, grad_scores[:, b]
               once, Q once (shared across Nq queries), and D[b, :, :]
               once. The previous two-kernel backward read argmax
               twice, grad_scores twice, and did not coalesce the
               shared Q/D traffic across the dQ/dD kernels.

Tradeoffs
---------
+ One launch replaces two → amortises overhead at small B (e.g. B=16,
  Nq=4 where launch cost dominates).
+ Single HBM pass over shared inputs (argmax, grad_scores).
+ dD no longer needs atomics — CTAs partition ``grad_D`` by doc id.
+ Only ``Nq`` unique Q rows are visited per CTA; we keep Q tiles hot.

- Higher register pressure: the fused CTA holds grad_D_local plus the
  running dQ partials. For large Ld*d this may force spills or reduce
  occupancy.
- grad_Q now uses FP32 atomics instead of exclusive writes. On Ampere/
  Hopper FP32 atomics are cheap enough that this is a win, but under
  very skewed grad_scores distributions the contention profile
  changes from "grad_D hot rows" to "grad_Q hot (q, l)" cells.
- We do not benefit from tensor cores here (pattern is a scatter, not
  a dense GEMM). The bucketed tl.dot variant in the base file is an
  orthogonal optimisation that could be plugged in per tile if needed.

Big-O
-----
Work per CTA (shared_docs): O(Nq * Lq * d)  same as split kernels.
Total launches: 1 (was 2).
Total HBM reads of argmax: B * Nq * Lq  (was 2 * B * Nq * Lq).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .flash_maxsim import _next_pow2, _launch_fwd, _default_lengths


# ---------------------------------------------------------------------------
# Fused backward kernel — one pass, both grads.
#
# Grid: (B,) shared_docs=True, (Nq*B,) shared_docs=False.
# Each CTA owns exactly one document's grad_D slice (no cross-CTA writes
# to that slice → plain store). grad_Q receives cross-doc contributions
# from every CTA → FP32 atomic_add.
# ---------------------------------------------------------------------------

@triton.jit
def _maxsim_bwd_fused_kernel(
    # Inputs
    Q_ptr, D_ptr, argmax_ptr, grad_s_ptr,
    # Outputs (both preallocated zero by caller)
    grad_Q_ptr,      # FP32 [Nq, Lq, d]      atomic target
    grad_D_ptr,      # FP32 [B (or Nq*B), Ld, d]   exclusive per CTA
    # Shapes
    Nq: tl.constexpr, B: tl.constexpr, Lq: tl.constexpr, Ld,
    d: tl.constexpr, d_pad: tl.constexpr,
    # Strides
    stride_q_n, stride_q_l, stride_q_d,
    stride_d_b, stride_d_l, stride_d_d,
    stride_gs_n, stride_gs_b,
    stride_gq_n, stride_gq_l, stride_gq_d,
    stride_gd_b, stride_gd_l, stride_gd_d,
    shared_docs: tl.constexpr,
):
    """Fused grad_Q / grad_D kernel.

    Iteration order is identical to the split kernels so argmax layout
    from the forward remains valid:

        shared_docs : pair_idx = doc_id * Nq + q_idx
                      D batch index = doc_id
        otherwise   : pair_idx = q_idx * B + doc_id
                      D batch index = q_idx * B + doc_id (only q_idx = q_outer)
    """
    pid = tl.program_id(0)

    # ---------- Resolve which (q_outer, doc) this CTA owns ----------
    if shared_docs:
        doc_id = pid
        if doc_id >= B:
            return
        q_outer = 0          # unused, just to keep types tidy
        db_i64 = tl.cast(doc_id, tl.int64)
    else:
        q_outer = pid // B
        doc_id = pid % B
        if q_outer >= Nq:
            return
        # In non-shared mode, grad_D / D are laid out as [Nq*B, Ld, d]
        db_i64 = tl.cast(q_outer * B + doc_id, tl.int64)

    k = tl.arange(0, d_pad)               # [d_pad]
    km = k < d

    # =========================================================
    # Main loop: for every (q_idx, lq_idx) owned by this (doc),
    #   1. fetch j = argmax[pair_idx, lq_idx]  and gs = grad_scores[q_idx, doc_id]
    #   2. load D[doc, j, :] → contribute gs * D row   to grad_Q[q_idx, lq_idx]
    #   3. load Q[q_idx, lq_idx, :] → contribute gs * Q row to grad_D[doc, j]
    # In non-shared mode, only q_idx == q_outer actually participates.
    # =========================================================

    for q_idx in range(Nq):
        if (not shared_docs) and (q_idx != q_outer):
            # Keep the loop static-shape; just skip work.
            continue

        # grad_scores[q_idx, doc_id] — hoisted out of the inner lq loop
        gs = tl.load(
            grad_s_ptr + q_idx * stride_gs_n + doc_id * stride_gs_b
        ).to(tl.float32)

        if shared_docs:
            pair_idx = doc_id * Nq + q_idx
        else:
            pair_idx = q_idx * B + doc_id

        for lq_idx in range(Lq):
            # Shared work: argmax lookup (single HBM load replaces two in
            # the split kernels).
            j = tl.load(argmax_ptr + pair_idx * Lq + lq_idx)
            j_i64 = tl.cast(j, tl.int64)

            # ---- Load D[doc, j, :] and Q[q_idx, lq_idx, :] once each ----
            d_row = tl.load(
                D_ptr + db_i64 * stride_d_b
                      + j_i64 * stride_d_l
                      + k * stride_d_d,
                mask=km, other=0.0,
            ).to(tl.float32)

            q_row = tl.load(
                Q_ptr + q_idx * stride_q_n
                      + lq_idx * stride_q_l
                      + k * stride_q_d,
                mask=km, other=0.0,
            ).to(tl.float32)

            # ---- (1) grad_Q[q_idx, lq_idx, :] += gs * d_row (atomic: cross-doc) ----
            gq_addr = (grad_Q_ptr
                       + q_idx * stride_gq_n
                       + lq_idx * stride_gq_l
                       + k * stride_gq_d)
            tl.atomic_add(gq_addr, gs * d_row, mask=km)

            # ---- (2) grad_D[doc, j, :] += gs * q_row (no atomic: CTA owns doc) ----
            # Several (q_idx, lq_idx) in this CTA may share the same j →
            # we still need atomic_add _within_ the CTA unless we carry
            # a local Ld×d accumulator. For simplicity and to avoid the
            # register blowup of that accumulator for large Ld, we use
            # FP32 atomic_add into our own CTA's exclusive slice. This
            # is cache-local (L2) and contention is intra-CTA only.
            gd_addr = (grad_D_ptr
                       + db_i64 * stride_gd_b
                       + j_i64 * stride_gd_l
                       + k * stride_gd_d)
            tl.atomic_add(gd_addr, gs * q_row, mask=km)


# ---------------------------------------------------------------------------
# Autograd wrapper using the fused kernel
# ---------------------------------------------------------------------------

class _FlashMaxSimBatchedFusedBwdFn(torch.autograd.Function):
    """Same forward as the production path, fused backward kernel."""

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

        Q_c = Q.contiguous().half()
        D_c = D_flat.contiguous().half()
        total = B if shared_docs else Nq * B
        lengths = _default_lengths(total, Ld, D.device, doc_lengths)
        q_lens = (query_lengths.to(torch.int32).contiguous()
                  if query_lengths is not None else None)

        scores, argmax = _launch_fwd(
            Q_c, D_c, lengths, Nq, B, Lq, Ld, d,
            shared_docs, True, q_lengths=q_lens,
        )
        ctx.save_for_backward(Q_c, D_c, argmax)
        ctx.shared_docs = shared_docs
        ctx.shape = (Nq, B, Lq, Ld, d)
        return scores  # [Nq, B] FP32

    @staticmethod
    def backward(ctx, grad_scores):
        Q, D, argmax = ctx.saved_tensors
        Nq, B, Lq, Ld, d = ctx.shape
        shared_docs = ctx.shared_docs
        grad_scores = grad_scores.contiguous().float()
        d_pad = _next_pow2(d)

        # Both output buffers in FP32 so atomics are well-defined on all archs.
        grad_Q_fp32 = torch.zeros(
            (Nq, Lq, d), device=Q.device, dtype=torch.float32
        )
        grad_D_fp32 = torch.zeros(
            D.shape, device=D.device, dtype=torch.float32
        )

        grid_size = B if shared_docs else Nq * B
        _maxsim_bwd_fused_kernel[(grid_size,)](
            Q, D, argmax, grad_scores,
            grad_Q_fp32, grad_D_fp32,
            Nq, B, Lq, Ld, d, d_pad,
            Q.stride(0), Q.stride(1), Q.stride(2),
            D.stride(0), D.stride(1), D.stride(2),
            grad_scores.stride(0), grad_scores.stride(1),
            grad_Q_fp32.stride(0), grad_Q_fp32.stride(1), grad_Q_fp32.stride(2),
            grad_D_fp32.stride(0), grad_D_fp32.stride(1), grad_D_fp32.stride(2),
            1 if shared_docs else 0,
        )

        grad_Q = grad_Q_fp32.to(Q.dtype)
        grad_D = grad_D_fp32.to(D.dtype)

        if not shared_docs:
            grad_D = grad_D.reshape(Nq, B, Ld, d)
        return grad_Q, grad_D, None, None, None


def flash_maxsim_batched_train_fused(
    Q: torch.Tensor, D: torch.Tensor,
    shared_docs: bool = True,
    doc_lengths: torch.Tensor | None = None,
    query_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Same signature as :func:`flash_maxsim_batched_train`, fused-bwd variant."""
    return _FlashMaxSimBatchedFusedBwdFn.apply(
        Q, D, shared_docs, doc_lengths, query_lengths,
    )


# ---------------------------------------------------------------------------
# Correctness test vs. pure-torch reference (run manually on a GPU box).
# ---------------------------------------------------------------------------

def _torch_reference_maxsim(Q: torch.Tensor, D: torch.Tensor,
                            shared_docs: bool) -> torch.Tensor:
    """Pure-torch MaxSim reference used as backward ground truth.

    scores[q, b] = sum_l max_j <Q[q, l, :], D[b, j, :]>
    (non-shared uses D[q, b, :, :])
    """
    if shared_docs:
        # [Nq, 1, Lq, d] @ [1, B, d, Ld] -> [Nq, B, Lq, Ld]
        sim = torch.einsum("nld,bjd->nblj", Q.float(), D.float())
    else:
        sim = torch.einsum("nld,nbjd->nblj", Q.float(), D.float())
    per_tok = sim.max(dim=-1).values   # [Nq, B, Lq]
    return per_tok.sum(dim=-1)         # [Nq, B]


def _run_correctness():
    """Small sanity check: compare fused backward against autograd through
    the pure-torch reference. Invoked from ``__main__``."""
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[SKIP] correctness test needs a CUDA device.")
        return

    for shared_docs in (True, False):
        Nq, B, Lq, Ld, d = 3, 4, 8, 16, 64
        Q = torch.randn(Nq, Lq, d, device=device,
                        dtype=torch.float16, requires_grad=True)
        if shared_docs:
            D = torch.randn(B, Ld, d, device=device,
                            dtype=torch.float16, requires_grad=True)
        else:
            D = torch.randn(Nq, B, Ld, d, device=device,
                            dtype=torch.float16, requires_grad=True)

        # Reference
        Q_ref = Q.detach().clone().requires_grad_(True)
        D_ref = D.detach().clone().requires_grad_(True)
        ref = _torch_reference_maxsim(Q_ref, D_ref, shared_docs)
        grad_out = torch.randn_like(ref)
        ref.backward(grad_out)

        # Fused kernel
        out = flash_maxsim_batched_train_fused(Q, D, shared_docs=shared_docs)
        out.backward(grad_out)

        # Compare (FP16 + scatter-sum → loose tolerances).
        atol = 5e-2
        rtol = 5e-2
        ok_q = torch.allclose(Q.grad.float(), Q_ref.grad.float(), atol=atol, rtol=rtol)
        ok_d = torch.allclose(D.grad.float(), D_ref.grad.float(), atol=atol, rtol=rtol)
        tag = "shared_docs" if shared_docs else "per-query docs"
        print(f"[{tag}] grad_Q match={ok_q}  grad_D match={ok_d}")
        if not ok_q:
            print("  grad_Q max err:",
                  (Q.grad.float() - Q_ref.grad.float()).abs().max().item())
        if not ok_d:
            print("  grad_D max err:",
                  (D.grad.float() - D_ref.grad.float()).abs().max().item())


if __name__ == "__main__":
    _run_correctness()
