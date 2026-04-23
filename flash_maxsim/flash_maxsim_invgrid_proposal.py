"""Atomic-free backward dD via inverse-grid (CSR) gather.

Proposal: replace the `atomic_add`-based `_maxsim_bwd_dD_batched_kernel` with a
destination-owned gather. Instead of iterating over sources `(q, l)` and
atomically writing into `grad_D[b, argmax[q,b,l], :]`, we iterate over
destinations `(b, j)` and each program gathers all sources that mapped to it.

Construction.
  The saved `argmax` has layout `[total_pairs, Lq]` int32, where
  `total_pairs = B` (with `pair_idx = doc_id * Nq + q_idx`) when
  `shared_docs=True`, and `total_pairs = Nq*B` (with `pair_idx = q_idx*B + b`)
  when `shared_docs=False`.  See `_maxsim_bwd_dD_batched_kernel` in
  `flash_maxsim_batched_train.py`.

  For the shared-docs case the destination `(b, j)` receives a contribution
  from every `(q, l)` such that `argmax[b*Nq + q, l] == j`.  So per `(b, j)`
  the "source list" is a subset of `{0..Nq-1} x {0..Lq-1}` of total size
  bounded by `Nq*Lq`.

CSR layout.
  We build two int32 tensors over the FLAT destination index
  `dest = b*Ld + j` (range `[0, B*Ld)`):

    row_ptr  : [B*Ld + 1]     prefix sums of per-destination contribution counts
    col_idx  : [Nq*B*Lq]      packed source (q_idx, lq_idx) pairs, encoded as
                              `q_idx * Lq + lq_idx` (fits since Nq*Lq < 2^31)

  For `shared_docs=False` we use `dest = (q*B + b) * Ld + j` (range
  `[0, Nq*B*Ld)`), and `col_idx` only needs `Lq` entries per `(q, b)` pair
  (total `Nq*B*Lq`, same as above).  In this case the encoded source is just
  `lq_idx` because `q_idx` is uniquely determined by the destination row.

Memory cost.
  `col_idx` is `Nq*B*Lq` int32 = 4 * Nq * B * Lq bytes.  For Nq=B=64, Lq=32
  that is 0.5 MiB; for Nq=128, B=256, Lq=64 it is 8 MiB — tiny next to the
  activations.  `row_ptr` is `B*Ld + 1` (or `Nq*B*Ld + 1`) int32.  Building
  CSR is two linear passes over `argmax`: one `scatter_add_` to count, one
  `argsort` or stable bucket-fill for the payload (both run in pure PyTorch).

Handling `shared_docs=False` (KD case).
  The pair layout differs (`q_idx*B + b` rather than `b*Nq + q_idx`) and every
  `(q, b)` pair owns an independent `[Ld, d]` slab of `grad_D`.  The CSR
  structure is built with a `dest` key that includes `q_idx`, so each
  destination still has a clean source list.  The kernel unpacks
  `q_idx, b, j` from its program id and proceeds identically.

Load balance.
  Per-destination contribution count is in `[0, Nq*Lq]`.  In practice argmax
  is heavily skewed: many `(b, j)` slots receive 0 contributions (cheap early
  exit), most get 1-2 (register accumulation dominates, one coalesced write),
  and a few hot slots receive many.  The grid is still `B*Ld` (resp.
  `Nq*B*Ld`) programs — same as the current scatter kernel — so SM occupancy
  is unchanged; only the work per program varies.  For extreme skew we could
  later split hot slots across multiple CTAs with a second reduction pass,
  but the simple version below is enough to beat FP32 atomics.

Zero- and one-contribution slots.
  Zero contributions: program loads `row_ptr[dest]`, sees `start == end`,
  writes a zero row and exits.  Cost is one `tl.store` of a `[d_pad]` zero
  vector — still coalesced, no wasted global reads.
  One contribution: program loads a single `(q, l)` pair, fetches `gs * Q`,
  writes it.  No reduction overhead, identical to a direct scatter but with
  *zero* atomics and no cross-CTA interference.

The kernel below keeps things simple on purpose: one program per destination,
one `d_pad` accumulator, a plain `for` loop over the source segment.  We
accept that hot slots serialize across their segment; the win is that we
*never* touch atomic memory and we write each `grad_D[dest, :]` exactly once.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .flash_maxsim import _next_pow2, _launch_fwd, _default_lengths


# ---------------------------------------------------------------------------
# CSR builder (pure PyTorch; runs once per backward call).
# ---------------------------------------------------------------------------

def build_inverse_csr(
    argmax: torch.Tensor,   # [total_pairs, Lq] int32 (or int64)
    Nq: int, B: int, Lq: int, Ld: int,
    shared_docs: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (row_ptr, col_idx) mapping destination slot -> source list.

    Shared-docs layout:
        pair_idx = b*Nq + q_idx
        dest     = b*Ld + j                 range [0, B*Ld)
        col      = q_idx*Lq + lq_idx        range [0, Nq*Lq)

    Non-shared (KD) layout:
        pair_idx = q_idx*B + b
        dest     = (q*B + b)*Ld + j         range [0, Nq*B*Ld)
        col      = lq_idx                   range [0, Lq)
    """
    device = argmax.device
    arg = argmax.to(torch.int64)

    if shared_docs:
        total_pairs = B
        n_dest = B * Ld
        # pair_idx = b*Nq + q_idx  →  b = pair_idx // Nq, q_idx = pair_idx % Nq
        pair_ids = torch.arange(total_pairs * Nq, device=device)  # wait: argmax is [B, Nq*?]
        # argmax layout per _launch_fwd: flat pair_idx with total = B, not B*Nq.
        # Actually: for shared_docs save_argmax produces [B*Nq, Lq] since pair_idx
        # ranges over doc_id*Nq + q_idx.  Flatten that:
        assert arg.shape[0] == B * Nq, f"expected argmax [{B*Nq}, {Lq}], got {tuple(arg.shape)}"
        flat_pair = torch.arange(B * Nq, device=device)
        b_of_pair = flat_pair // Nq          # [B*Nq]
        q_of_pair = flat_pair % Nq           # [B*Nq]
        # dest[p, l] = b_of_pair[p]*Ld + argmax[p, l]
        dest = b_of_pair[:, None] * Ld + arg        # [B*Nq, Lq]
        # col[p, l] = q_of_pair[p]*Lq + l
        lq_range = torch.arange(Lq, device=device)
        col = q_of_pair[:, None] * Lq + lq_range[None, :]     # [B*Nq, Lq]
    else:
        assert arg.shape[0] == Nq * B
        n_dest = Nq * B * Ld
        flat_pair = torch.arange(Nq * B, device=device)
        q_of_pair = flat_pair // B
        b_of_pair = flat_pair % B
        # dest row base is (q*B + b)*Ld
        row_base = (q_of_pair * B + b_of_pair) * Ld          # [Nq*B]
        dest = row_base[:, None] + arg                       # [Nq*B, Lq]
        lq_range = torch.arange(Lq, device=device)
        col = lq_range[None, :].expand(Nq * B, Lq).contiguous()  # [Nq*B, Lq]

    dest_flat = dest.reshape(-1).to(torch.int64)             # [nnz]
    col_flat = col.reshape(-1).to(torch.int32)               # [nnz]
    nnz = dest_flat.numel()

    # Stable sort by destination → contiguous segments per slot.
    order = torch.argsort(dest_flat, stable=True)
    dest_sorted = dest_flat[order]
    col_sorted = col_flat[order]

    # row_ptr via bincount + cumsum.
    counts = torch.bincount(dest_sorted, minlength=n_dest)   # [n_dest]
    row_ptr = torch.zeros(n_dest + 1, dtype=torch.int32, device=device)
    row_ptr[1:] = counts.to(torch.int32).cumsum(dim=0)

    return row_ptr.contiguous(), col_sorted.contiguous()


# ---------------------------------------------------------------------------
# Inverse-grid dD kernel.  One program per destination slot.
# ---------------------------------------------------------------------------

@triton.jit
def _maxsim_bwd_dD_invgrid_kernel(
    Q_ptr, grad_s_ptr, grad_D_ptr,
    row_ptr_ptr, col_idx_ptr,
    Nq: tl.constexpr, B: tl.constexpr, Lq: tl.constexpr, Ld,
    d: tl.constexpr, d_pad: tl.constexpr,
    stride_d_b, stride_d_l, stride_d_d,
    stride_q_n, stride_q_l, stride_q_d,
    stride_gs_n, stride_gs_b,
    shared_docs: tl.constexpr,
):
    """Grid: (n_dest,)  with n_dest = B*Ld (shared) or Nq*B*Ld (non-shared).

    Each program:
      1. decodes (q_idx_outer, b, j) from its pid,
      2. reads its CSR segment [row_ptr[dest], row_ptr[dest+1]),
      3. for each packed source, unpacks (q_idx, lq_idx), loads gs[q,b] and
         Q[q, lq, :], accumulates `gs * Q` in an FP32 register tile,
      4. stores the result once into grad_D (no atomics, no collisions).
    """
    pid = tl.program_id(0)

    # Decode destination.
    if shared_docs:
        doc_id = pid // Ld
        j = pid % Ld
        if doc_id >= B:
            return
        db_i64 = tl.cast(doc_id, tl.int64)
    else:
        # pid = ((q*B + b)*Ld + j)
        j = pid % Ld
        qb = pid // Ld
        q_idx_outer = qb // B
        doc_id = qb % B
        if q_idx_outer >= Nq:
            return
        db_i64 = tl.cast(q_idx_outer * B + doc_id, tl.int64)

    j_i64 = tl.cast(j, tl.int64)

    k = tl.arange(0, d_pad)
    km = k < d

    # Load CSR segment bounds.
    seg_start = tl.load(row_ptr_ptr + pid)
    seg_end = tl.load(row_ptr_ptr + pid + 1)

    acc = tl.zeros([d_pad], dtype=tl.float32)

    # Scan the source list (typically short: 0, 1, or a handful of entries).
    for cursor in range(seg_start, seg_end):
        col = tl.load(col_idx_ptr + cursor)
        if shared_docs:
            q_idx = col // Lq
            lq_idx = col % Lq
        else:
            # q is fixed (= q_idx_outer); col holds lq_idx directly.
            q_idx = q_idx_outer
            lq_idx = col

        gs = tl.load(grad_s_ptr + q_idx * stride_gs_n + doc_id * stride_gs_b).to(tl.float32)
        qv = tl.load(
            Q_ptr + q_idx * stride_q_n + lq_idx * stride_q_l + k * stride_q_d,
            mask=km, other=0.0,
        ).to(tl.float32)
        acc += gs * qv

    tl.store(
        grad_D_ptr + db_i64 * stride_d_b + j_i64 * stride_d_l + k * stride_d_d,
        acc, mask=km,
    )


# ---------------------------------------------------------------------------
# Autograd function with the invgrid dD path.
# ---------------------------------------------------------------------------

@triton.jit
def _maxsim_bwd_dQ_batched_kernel(
    D_ptr, argmax_ptr, grad_s_ptr, grad_Q_ptr,
    Nq: tl.constexpr, B: tl.constexpr, Lq: tl.constexpr, d: tl.constexpr, d_pad: tl.constexpr,
    stride_d_b, stride_d_l, stride_d_d,
    stride_gs_n, stride_gs_b,
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
        v = tl.load(
            D_ptr + db_i64 * stride_d_b + j_i64 * stride_d_l + k * stride_d_d,
            mask=km, other=0.0,
        ).to(tl.float32)
        acc += gs * v

    tl.store(grad_Q_ptr + q_idx * Lq * d + lq_idx * d + k, acc.to(tl.float16), mask=km)


class _FlashMaxSimInvGridFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, D, shared_docs: bool, doc_lengths, query_lengths):
        assert Q.dim() == 3, "Q must be [Nq, Lq, d]"
        Nq, Lq, d = Q.shape
        if shared_docs:
            assert D.dim() == 3
            B, Ld, _ = D.shape
            D_flat = D
        else:
            assert D.dim() == 4
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
        return scores

    @staticmethod
    def backward(ctx, grad_scores):
        Q, D, argmax = ctx.saved_tensors
        Nq, B, Lq, Ld, d = ctx.shape
        shared_docs = ctx.shared_docs
        grad_scores = grad_scores.contiguous().float()
        d_pad = _next_pow2(d)

        # grad_Q — unchanged from the batched version.
        grad_Q = torch.zeros_like(Q)
        _maxsim_bwd_dQ_batched_kernel[(Nq * Lq,)](
            D, argmax, grad_scores, grad_Q,
            Nq, B, Lq, d, d_pad,
            D.stride(0), D.stride(1), D.stride(2),
            grad_scores.stride(0), grad_scores.stride(1),
            1 if shared_docs else 0,
        )

        # grad_D — atomic-free CSR gather.
        row_ptr, col_idx = build_inverse_csr(argmax, Nq, B, Lq, Ld, shared_docs)
        n_dest = (B if shared_docs else Nq * B) * Ld
        grad_D_fp32 = torch.zeros(D.shape, device=D.device, dtype=torch.float32)

        _maxsim_bwd_dD_invgrid_kernel[(n_dest,)](
            Q, grad_scores, grad_D_fp32,
            row_ptr, col_idx,
            Nq, B, Lq, Ld, d, d_pad,
            grad_D_fp32.stride(0), grad_D_fp32.stride(1), grad_D_fp32.stride(2),
            Q.stride(0), Q.stride(1), Q.stride(2),
            grad_scores.stride(0), grad_scores.stride(1),
            1 if shared_docs else 0,
        )
        grad_D = grad_D_fp32.to(D.dtype)
        if not shared_docs:
            grad_D = grad_D.reshape(Nq, B, Ld, d)
        return grad_Q, grad_D, None, None, None


def flash_maxsim_invgrid_train(
    Q: torch.Tensor, D: torch.Tensor,
    shared_docs: bool = True,
    doc_lengths: torch.Tensor | None = None,
    query_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Same API as flash_maxsim_batched_train but uses the atomic-free dD path."""
    return _FlashMaxSimInvGridFn.apply(Q, D, shared_docs, doc_lengths, query_lengths)


# ---------------------------------------------------------------------------
# Correctness test (requires CUDA; the user will run this).
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from .flash_maxsim_batched_train import flash_maxsim_batched_train

    torch.manual_seed(0)
    device = "cuda"

    def _check(shared_docs: bool):
        Nq, B, Lq, Ld, d = 4, 8, 16, 24, 64
        if shared_docs:
            D = torch.randn(B, Ld, d, device=device, dtype=torch.float16, requires_grad=True)
        else:
            D = torch.randn(Nq, B, Ld, d, device=device, dtype=torch.float16, requires_grad=True)
        Q = torch.randn(Nq, Lq, d, device=device, dtype=torch.float16, requires_grad=True)

        # Reference: existing (atomic) backward.
        Qr = Q.detach().clone().requires_grad_()
        Dr = D.detach().clone().requires_grad_()
        scores_ref = flash_maxsim_batched_train(Qr, Dr, shared_docs=shared_docs)
        g = torch.randn_like(scores_ref)
        scores_ref.backward(g)

        # New: invgrid backward.
        Qn = Q.detach().clone().requires_grad_()
        Dn = D.detach().clone().requires_grad_()
        scores_new = flash_maxsim_invgrid_train(Qn, Dn, shared_docs=shared_docs)
        scores_new.backward(g)

        print(f"[shared_docs={shared_docs}]")
        print(f"  scores match : {torch.allclose(scores_ref, scores_new, atol=1e-3, rtol=1e-3)}")
        print(f"  grad_Q match : {torch.allclose(Qr.grad, Qn.grad, atol=1e-2, rtol=1e-2)}")
        print(f"  grad_D match : {torch.allclose(Dr.grad, Dn.grad, atol=1e-2, rtol=1e-2)}")
        print(f"  grad_D maxerr: {(Dr.grad.float() - Dn.grad.float()).abs().max().item():.4e}")

    _check(shared_docs=True)
    _check(shared_docs=False)
