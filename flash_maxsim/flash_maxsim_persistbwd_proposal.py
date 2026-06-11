"""Persistent-range backward for grad_D — proposal (v0.2.1 candidate).

The existing `_maxsim_bwd_dD_batched_kernel` uses grid `(B,)` (or `(Nq*B,)`
non-shared): each CTA owns one doc and walks the full `Nq*Lq` step space
serially, issuing one FP32 atomic_add per step into `grad_D[b, argmax, :]`.

At ColBERT scale (B=1024, Nq=1024, Lq=32, Ld=180, d=128) that's 32 K serial
iterations per CTA on ~108 SMs — most SMs idle after a short ramp, and Q
is re-read from HBM by each doc independently.

This kernel inverts the parallelism:

    Grid:   (B, n_step_chunks)        where n_step_chunks = ceil(Nq*Lq / STEP_TILE)
            (Nq*B, n_step_chunks)     non-shared mode

    Per CTA:
        1. Load STEP_TILE contiguous rows of Q (for this doc's step range) into
           registers once — `STEP_TILE x d_pad` FP16.
        2. Loop the STEP_TILE rows: read argmax[pair, lq], read gs, atomic_add
           `gs * Q[q,lq,:]` into `grad_D[b, argmax, :]`.

Cross-CTA contention for a given (b, j) destination is still possible (two
different step chunks may land on the same argmax), but destinations are
spread across ~Ld rows per doc and atomics are FP32 — well within hardware
throughput.

Trade-offs:
- Register pressure: STEP_TILE=64, d=128 FP16 -> 16 KB of Q per CTA. Fine.
- STEP_TILE dictates the parallel-vs-loop balance. At STEP_TILE=64,
  n_step_chunks = 32768 / 64 = 512 for ColBERT scale: grid = (1024, 512) =
  524 K CTAs -> completely saturates the GPU, whereas (1024,) leaves most
  SMs idle.
- Expected Q HBM traffic: old kernel streams Q once per doc (B docs) for a
  total of B * (Nq*Lq*d) loads. New kernel also walks every (q, lq) once
  per doc, but the STEP_TILE-sized Q tile is held in registers through the
  inner loop, so HBM reads are the same *logical* count — the win is that
  the loop body itself does no repeated HBM fetch of Q within the tile and
  each tile's load is coalesced and cached across the warps of the CTA.

If Q traffic ever needs further reduction (e.g. for B >> Nq), one can swap
the grid semantics to `(n_step_chunks, B)` and share Q across docs via
SMEM; that is a follow-up.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .flash_maxsim import _next_pow2, _launch_fwd, _default_lengths
from .flash_maxsim_batched_train import _maxsim_bwd_dQ_batched_kernel


# ---------------------------------------------------------------------------
# Persistent-range grad_D kernel
# ---------------------------------------------------------------------------

@triton.jit
def _maxsim_bwd_dD_persistent_kernel(
    Q_ptr, argmax_ptr, grad_s_ptr, grad_D_ptr,
    Nq: tl.constexpr, B: tl.constexpr, Lq: tl.constexpr, Ld, d: tl.constexpr, d_pad: tl.constexpr,
    stride_d_b, stride_d_l, stride_d_d,
    stride_q_n, stride_q_l, stride_q_d,
    stride_gs_n, stride_gs_b,
    shared_docs: tl.constexpr,
    STEP_TILE: tl.constexpr,
):
    """Backward for grad_D, split by (doc, step_chunk).

    Grid:
        pid_doc: 0..B-1        (shared_docs)   or 0..Nq*B-1 (non-shared)
        pid_chunk: 0..n_step_chunks-1
    """
    pid_doc = tl.program_id(0)
    pid_chunk = tl.program_id(1)

    # Decode doc_id and (optional) outer q index.
    if shared_docs:
        doc_id = pid_doc
        if doc_id >= B:
            return
        # For shared_docs, every CTA covers steps in [0, Nq*Lq).
        total_steps = Nq * Lq
        q_idx_outer = 0  # unused
    else:
        q_idx_outer = pid_doc // B
        doc_id = pid_doc % B
        if q_idx_outer >= Nq:
            return
        # Non-shared: only Lq steps per (q, doc) pair.
        total_steps = Lq

    # Step range this CTA owns.
    step_start = pid_chunk * STEP_TILE
    if step_start >= total_steps:
        return
    step_off = step_start + tl.arange(0, STEP_TILE)        # [S]
    step_mask = step_off < total_steps

    # Decode q_idx / lq_idx for each step in the tile.
    if shared_docs:
        q_idx_vec = step_off // Lq                         # [S]
        lq_idx_vec = step_off % Lq                         # [S]
        pair_idx_vec = doc_id * Nq + q_idx_vec             # [S]
    else:
        q_idx_vec = tl.full([STEP_TILE], q_idx_outer, dtype=tl.int32)
        lq_idx_vec = step_off                              # [S]
        pair_idx_vec = q_idx_outer * B + doc_id            # scalar, will broadcast

    # Destination batch index in grad_D (flattened view).
    db_i64 = tl.cast(doc_id, tl.int64)
    if not shared_docs:
        db_i64 = tl.cast(q_idx_outer * B + doc_id, tl.int64)

    # Channel range (one CTA covers all d channels).
    k = tl.arange(0, d_pad)
    km = k < d

    # --------------------------------------------------------------------
    # Load the full [STEP_TILE, d_pad] Q tile ONCE into registers.
    # For STEP_TILE=64, d_pad=128, FP16: 64*128*2 = 16 KB of registers/SMEM.
    # Out-of-range steps are masked to zero so they contribute nothing if
    # they accidentally hit a valid j (step_mask guards atomics separately).
    # --------------------------------------------------------------------
    Q_tile_ptr = (Q_ptr
                  + q_idx_vec[:, None] * stride_q_n
                  + lq_idx_vec[:, None] * stride_q_l
                  + k[None, :] * stride_q_d)
    q_tile = tl.load(
        Q_tile_ptr,
        mask=step_mask[:, None] & km[None, :],
        other=0.0,
    ).to(tl.float32)                                        # [S, d_pad]

    # Load argmax for this step tile.
    arg_off = pair_idx_vec * Lq + lq_idx_vec                # [S]
    a = tl.load(argmax_ptr + arg_off, mask=step_mask, other=0)   # [S] int32

    # Load grad_scores for this tile.
    if shared_docs:
        gs_off = q_idx_vec * stride_gs_n + doc_id * stride_gs_b
        gs = tl.load(grad_s_ptr + gs_off, mask=step_mask, other=0.0).to(tl.float32)
    else:
        gs_scalar = tl.load(
            grad_s_ptr + q_idx_outer * stride_gs_n + doc_id * stride_gs_b
        ).to(tl.float32)
        gs = tl.full([STEP_TILE], 0.0, dtype=tl.float32) + gs_scalar

    # weighted Q: [S, d_pad], computed once, reused in the atomic loop.
    weighted = q_tile * gs[:, None]                         # [S, d_pad]

    # --------------------------------------------------------------------
    # Emit one atomic_add per step. Unrolled over the (small, constexpr)
    # STEP_TILE; Triton compiles this into STEP_TILE vector atomics, each
    # of width d_pad.
    #
    # We intentionally do NOT bucket by j here — profiling showed a bucket
    # tl.dot is slower than direct FP32 atomics when the routing matrix
    # is ~99% zero (Nq*Lq >> Ld).
    # --------------------------------------------------------------------
    for s in tl.static_range(STEP_TILE):
        valid_s = step_mask[s]
        if valid_s:
            j_s = tl.cast(a[s], tl.int64)
            dst = (grad_D_ptr
                   + db_i64 * stride_d_b
                   + j_s * stride_d_l
                   + k * stride_d_d)
            tl.atomic_add(dst, weighted[s, :], mask=km)


# ---------------------------------------------------------------------------
# STEP_TILE heuristic
# ---------------------------------------------------------------------------

def _pick_step_tile(total_steps: int, d_pad: int) -> int:
    """Pick STEP_TILE so CTA register pressure stays bounded and the grid
    fully saturates the GPU. d_pad=128 -> 64 is the sweet spot (16 KB FP16
    Q tile; ~512 chunks at Nq*Lq=32 K -> 500+ K CTAs total at B=1024).
    """
    # Don't pick a tile larger than the work itself.
    if total_steps <= 16:
        return 16
    if total_steps <= 32:
        return 32
    # d_pad=128 -> 64 rows = 16 KB FP16 per CTA, plenty of register room.
    # d_pad=256 -> drop to 32 rows to keep register pressure reasonable.
    if d_pad <= 128:
        return 64
    if d_pad <= 256:
        return 32
    return 16


# ---------------------------------------------------------------------------
# Drop-in autograd function with the new backward
# ---------------------------------------------------------------------------

class _FlashMaxSimPersistBwdFn(torch.autograd.Function):
    """Identical to `_FlashMaxSimBatchedFn` but uses the persistent-range
    dD kernel in the backward. Forward and dQ path are unchanged.
    """

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
        q_lens = query_lengths.to(torch.int32).contiguous() if query_lengths is not None else None

        scores, argmax = _launch_fwd(
            Q_c, D_c, lengths, Nq, B, Lq, Ld, d, shared_docs, True, q_lengths=q_lens,
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

        # grad_Q — reuse the existing kernel (not the target of this proposal).
        grad_Q = torch.zeros_like(Q)
        _maxsim_bwd_dQ_batched_kernel[(Nq * Lq,)](
            D, argmax, grad_scores, grad_Q,
            Nq, B, Lq, d, d_pad,
            D.stride(0), D.stride(1), D.stride(2),
            grad_scores.stride(0), grad_scores.stride(1),
            1 if shared_docs else 0,
        )

        # grad_D — persistent-range kernel.
        grad_D_fp32 = torch.zeros(D.shape, device=D.device, dtype=torch.float32)
        total_steps = (Nq * Lq) if shared_docs else Lq
        STEP_TILE = _pick_step_tile(total_steps, d_pad)
        n_step_chunks = triton.cdiv(total_steps, STEP_TILE)
        grid_docs = B if shared_docs else Nq * B
        grid = (grid_docs, n_step_chunks)

        _maxsim_bwd_dD_persistent_kernel[grid](
            Q, argmax, grad_scores, grad_D_fp32,
            Nq, B, Lq, Ld, d, d_pad,
            grad_D_fp32.stride(0), grad_D_fp32.stride(1), grad_D_fp32.stride(2),
            Q.stride(0), Q.stride(1), Q.stride(2),
            grad_scores.stride(0), grad_scores.stride(1),
            1 if shared_docs else 0,
            STEP_TILE,
            num_warps=8,
            num_stages=2,
        )
        grad_D = grad_D_fp32.to(D.dtype)

        if not shared_docs:
            grad_D = grad_D.reshape(Nq, B, Ld, d)

        return grad_Q, grad_D, None, None, None


def flash_maxsim_batched_train_persistbwd(
    Q: torch.Tensor, D: torch.Tensor,
    shared_docs: bool = True,
    doc_lengths: torch.Tensor | None = None,
    query_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Drop-in replacement for `flash_maxsim_batched_train` using the
    persistent-range dD backward. See module docstring.
    """
    return _FlashMaxSimPersistBwdFn.apply(Q, D, shared_docs, doc_lengths, query_lengths)


# ---------------------------------------------------------------------------
# Correctness test (run with: python -m flash_maxsim.flash_maxsim_persistbwd_proposal)
# ---------------------------------------------------------------------------

def _torch_reference_maxsim(Q, D, shared_docs=True):
    """Dense torch reference: scores[q, b] = sum_l max_j (Q[q,l] . D[b,j])."""
    if shared_docs:
        # Q: [Nq, Lq, d], D: [B, Ld, d]
        # sim: [Nq, B, Lq, Ld]
        sim = torch.einsum("qld,bkd->qblk", Q.float(), D.float())
        max_per_l, _ = sim.max(dim=-1)        # [Nq, B, Lq]
        return max_per_l.sum(dim=-1)          # [Nq, B]
    else:
        # Q: [Nq, Lq, d], D: [Nq, B, Ld, d]
        sim = torch.einsum("qld,qbkd->qblk", Q.float(), D.float())
        max_per_l, _ = sim.max(dim=-1)
        return max_per_l.sum(dim=-1)


def _run_correctness(shared_docs: bool, Nq=3, B=5, Lq=7, Ld=13, d=64,
                     atol_score=5e-3, atol_grad=5e-3):
    torch.manual_seed(0)
    device = "cuda"

    Q = torch.randn(Nq, Lq, d, device=device, dtype=torch.float16, requires_grad=True)
    if shared_docs:
        D = torch.randn(B, Ld, d, device=device, dtype=torch.float16, requires_grad=True)
    else:
        D = torch.randn(Nq, B, Ld, d, device=device, dtype=torch.float16, requires_grad=True)

    # --- reference via torch (dense) ---
    Q_ref = Q.detach().clone().requires_grad_(True)
    D_ref = D.detach().clone().requires_grad_(True)
    scores_ref = _torch_reference_maxsim(Q_ref, D_ref, shared_docs=shared_docs)
    grad_out = torch.randn_like(scores_ref)
    scores_ref.backward(grad_out)

    # --- kernel under test ---
    scores_kern = flash_maxsim_batched_train_persistbwd(Q, D, shared_docs=shared_docs)
    scores_kern.backward(grad_out.clone())

    # Forward comparison.
    score_err = (scores_kern.float() - scores_ref.float()).abs().max().item()
    gq_err = (Q.grad.float() - Q_ref.grad.float()).abs().max().item()
    gd_err = (D.grad.float() - D_ref.grad.float()).abs().max().item()

    tag = "shared_docs" if shared_docs else "non-shared"
    print(f"[{tag}] scores max|err|={score_err:.4e}  "
          f"grad_Q max|err|={gq_err:.4e}  grad_D max|err|={gd_err:.4e}")

    assert score_err < atol_score, f"scores mismatch {score_err}"
    assert gq_err < atol_grad, f"grad_Q mismatch {gq_err}"
    assert gd_err < atol_grad, f"grad_D mismatch {gd_err}"


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available — skipping correctness test.")
        raise SystemExit(0)

    _run_correctness(shared_docs=True)
    _run_correctness(shared_docs=False)
    # Slightly bigger shape to exercise multi-chunk grid (n_step_chunks > 1).
    _run_correctness(shared_docs=True, Nq=4, B=6, Lq=16, Ld=24, d=64,
                     atol_score=1e-2, atol_grad=1e-2)
    print("All correctness checks passed.")
