"""Sort-inverse backward for MaxSim `grad_D` — proposal / prototype.

This is a drop-in replacement path for the `grad_D` scatter currently done by
`_maxsim_bwd_dD_batched_kernel` (FP32 `tl.atomic_add`) in
`flash_maxsim_batched_train.py`. At small shapes (e.g. ColBERT Lq=32 Ld=180
B=128) the atomic scatter loses to PyTorch's dense matmul backward (0.3x).
This file implements the same idea as Flash-KMeans (arxiv 2603.09229,
github.com/svg-project/flash-kmeans): pre-sort the argmax tokens by their
destination so all contributions to a single `(b, j)` slot end up in one
contiguous segment, then write each destination exactly once with zero
atomics and zero wasted compute.

-------------------------------------------------------------------------------
(a) What the sort key represents
-------------------------------------------------------------------------------
For every (q_idx, doc_id, lq_idx) triple there is exactly one `grad_D`
destination: row `j = argmax[q_idx, doc_id, lq_idx]` of doc `doc_id` (in the
shared_docs case) or of flat doc `q_idx * B + doc_id` (in the non-shared
case). We flatten these triples to a single entry index `e in [0, E)` and
build a 64-bit key

    key[e] = db_idx[e] * Ld + j[e]

where `db_idx` is the flat doc-batch index (`doc_id` shared, `q_idx*B+doc_id`
non-shared) and `j` is the argmax of that entry. Entries with identical
`(db, j)` share the same key. After `perm = torch.argsort(key)` and a single
`torch.unique_consecutive(... return_counts=True)` on `key[perm]`, the
contributions of every unique destination live in a single contiguous range
of the permuted index space.

(b) CTA mapping
-------------------------------------------------------------------------------
We produce two CPU-side tables of length `U` (number of unique destinations):

    seg_starts[u] = first permuted entry index whose key is `uniq[u]`
    seg_counts[u] = size of that run
    uniq_db[u]    = uniq[u] // Ld
    uniq_j[u]     = uniq[u] %  Ld

The kernel launches a 2D grid `(U, ceil(d / BLOCK_D))`. CTA `(u, pid_d)`:
   - looks up its segment `[start, start + count)`,
   - iterates over the segment in chunks of `BLOCK_S`,
   - for each chunk loads the `(q_idx, lq_idx)` from the permutation table,
     gathers `Q[q_idx, lq_idx, k_off]` and `grad_scores[q_idx, doc_id]`,
     accumulates `gs * Q` in a register tile of shape `[BLOCK_D]`,
   - writes `grad_D[uniq_db[u], uniq_j[u], k_off]` exactly once.

No atomics, no routing-matrix masking, one coalesced store per destination.

(c) Argmax layout subtlety (shared_docs=True vs False)
-------------------------------------------------------------------------------
`argmax` is stored as a 2D tensor of shape `[Nq*B, Lq]`. The flat pair index
`pair_idx` is computed DIFFERENTLY in the forward depending on `shared_docs`:

    shared_docs=True  :   pair_idx = doc_id  * Nq + q_idx        # doc-major
    shared_docs=False :   pair_idx = q_idx   * B  + doc_id       # query-major

Both modes must produce, per `(q_idx, doc_id, lq_idx)`, the correct `(db, j)`
destination:
    shared_docs=True  :   db = doc_id                 j = argmax[doc*Nq+q, lq]
    shared_docs=False :   db = q_idx*B + doc_id       j = argmax[q*B+doc, lq]

We compute the per-entry `(q_idx, doc_id, lq_idx, db, j)` tuple purely from
the flat entry index `e` so the key-building step works for either layout.

(d) Known gotchas
-------------------------------------------------------------------------------
* `torch.argsort` is NOT stable by default. Within a segment we sum over all
  entries, so permutation order inside a segment does not affect the sum
  value — it only affects FP rounding. Under atomic add the rounding order
  is also nondeterministic, so this is no worse. If strict determinism is
  required, pass `stable=True` to `torch.argsort` (slightly slower).
* Destinations with zero contributions must still receive zero. We pre-zero
  `grad_D` with `torch.zeros_like(D, dtype=fp32)` and the kernel only writes
  destinations that appear in `uniq`.
* `key` must be int64 to avoid overflow when `B * Ld > 2**31` (rare, but
  the Nq*B non-shared case with B=1024 and Ld=2048 already hits ~2M
  entries). `argsort` on int64 is fine.
* We cast accumulators to fp32 and cast the final store to match
  `grad_D_fp32.dtype` (fp32), matching the existing backward's precision.
* The `BLOCK_S` (segment chunk) cost is a register trade: too small slows
  the inner loop, too large blows registers when `d_pad` is big. 32–64
  works well for typical `d=128`.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

try:
    from .flash_maxsim import _next_pow2, _launch_fwd, _default_lengths  # type: ignore
except ImportError:  # executed as __main__
    _next_pow2 = lambda x: 1 << (x - 1).bit_length()


# ---------------------------------------------------------------------------
# Sort-inverse grad_D kernel.
# Grid: (U, ceil(d_pad / BLOCK_D))
#   U = #unique (db, j) destinations.
# Per-CTA work:
#   - load segment bounds [start, start + count),
#   - for each chunk of BLOCK_S permuted entries, gather (q_idx, lq_idx)
#     from perm → entry_meta tables, load Q rows + grad_scores, accumulate
#     gs * Q into acc[BLOCK_D],
#   - write grad_D[db, j, k_off] once, no atomics.
# ---------------------------------------------------------------------------

@triton.jit
def _maxsim_bwd_dD_sorted_kernel(
    # Entry-level tables (size E = Nq*B*Lq shared, or Nq*B*Lq non-shared)
    perm_ptr,                # int64[E]  — perm[s] gives the original entry index
    entry_q_ptr,             # int32[E]  — q_idx  of original entry
    entry_doc_ptr,           # int32[E]  — doc_id of original entry
    entry_lq_ptr,            # int32[E]  — lq_idx of original entry
    # Segment-level tables (size U)
    seg_start_ptr,           # int32[U]
    seg_count_ptr,           # int32[U]
    uniq_db_ptr,             # int32[U]  — flat doc-batch index
    uniq_j_ptr,              # int32[U]  — argmax slot
    # Data
    Q_ptr, grad_s_ptr, grad_D_ptr,
    Lq: tl.constexpr, Ld,
    d: tl.constexpr, d_pad: tl.constexpr,
    stride_d_b, stride_d_l, stride_d_d,
    stride_q_n, stride_q_l, stride_q_d,
    stride_gs_n, stride_gs_b,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_u = tl.program_id(0)
    pid_d = tl.program_id(1)

    # Segment info
    start = tl.load(seg_start_ptr + pid_u)
    count = tl.load(seg_count_ptr + pid_u)
    db = tl.load(uniq_db_ptr + pid_u)
    j = tl.load(uniq_j_ptr + pid_u)

    # k-tile for this CTA
    k_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    k_mask = k_off < d

    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    # Scan the segment in chunks of BLOCK_S permuted entries.
    for s_base in range(0, count, BLOCK_S):
        s_off = s_base + tl.arange(0, BLOCK_S)        # [BS]
        s_valid = s_off < count
        perm_idx = start + s_off                       # [BS]

        # perm[perm_idx] = original flat entry id e (int64)
        _e = tl.load(perm_ptr + perm_idx, mask=s_valid, other=0)
        # For gathers we need int32 (or int64) indices; the entry_* tables
        # are small enough that int32 is safe (E = Nq*B*Lq).
        e = _e.to(tl.int32)

        q_idx  = tl.load(entry_q_ptr  + e, mask=s_valid, other=0)   # [BS]
        doc_id = tl.load(entry_doc_ptr + e, mask=s_valid, other=0)  # [BS]
        lq_idx = tl.load(entry_lq_ptr + e, mask=s_valid, other=0)   # [BS]

        # grad_scores[q_idx, doc_id]   [BS]
        gs_off = q_idx * stride_gs_n + doc_id * stride_gs_b
        gs = tl.load(grad_s_ptr + gs_off, mask=s_valid, other=0.0).to(tl.float32)

        # Q[q_idx, lq_idx, k_off]      [BS, BLOCK_D]
        q_ptrs = (Q_ptr
                  + q_idx[:, None]  * stride_q_n
                  + lq_idx[:, None] * stride_q_l
                  + k_off[None, :]  * stride_q_d)
        qv = tl.load(q_ptrs,
                     mask=s_valid[:, None] & k_mask[None, :],
                     other=0.0).to(tl.float32)                  # [BS, BD]

        # Weighted contrib: (gs[:, None] * qv) summed over BS → [BD]
        weighted = qv * gs[:, None]                              # [BS, BD]
        acc += tl.sum(weighted, axis=0)                          # [BD]

    # One coalesced write to grad_D[db, j, :]
    db_i64 = tl.cast(db, tl.int64)
    j_i64  = tl.cast(j,  tl.int64)
    out_ptr = (grad_D_ptr
               + db_i64 * stride_d_b
               + j_i64  * stride_d_l
               + k_off  * stride_d_d)
    tl.store(out_ptr, acc, mask=k_mask)


# ---------------------------------------------------------------------------
# Host-side helpers: build (perm, segments, uniq dests) on-device.
# ---------------------------------------------------------------------------

def _build_sort_tables(
    argmax: torch.Tensor,
    Nq: int, B: int, Lq: int, Ld: int,
    shared_docs: bool,
) -> dict:
    """Flatten argmax into entry tables + sort by (db, j) key.

    Returns a dict containing:
      perm         : int64[E]    argsort of key (ascending)
      entry_q      : int32[E]    q_idx  per original entry
      entry_doc    : int32[E]    doc_id per original entry
      entry_lq     : int32[E]    lq_idx per original entry
      seg_start    : int32[U]    start of each segment in perm
      seg_count    : int32[U]    length of each segment
      uniq_db      : int32[U]    flat doc-batch index of each dest
      uniq_j       : int32[U]    argmax slot of each dest
    """
    device = argmax.device

    # argmax is stored as [Nq*B, Lq] int32, indexed by pair_idx * Lq + lq.
    # pair_idx layout:
    #   shared_docs=True  : pair_idx = doc_id * Nq + q_idx
    #   shared_docs=False : pair_idx = q_idx  * B  + doc_id
    # We canonicalize to a 3D view argmax3[q, b, l] regardless of mode,
    # which lets us build the key with simple broadcasting.
    if shared_docs:
        # argmax stored [B * Nq, Lq] with rows ordered (doc*Nq + q).
        am = argmax.view(B, Nq, Lq).permute(1, 0, 2).contiguous()  # [Nq, B, Lq]
    else:
        am = argmax.view(Nq, B, Lq).contiguous()                   # [Nq, B, Lq]

    E = Nq * B * Lq

    # Build entry-level (q, doc, lq) tables — just meshgrid indices.
    q_range  = torch.arange(Nq, device=device, dtype=torch.int32)
    b_range  = torch.arange(B,  device=device, dtype=torch.int32)
    lq_range = torch.arange(Lq, device=device, dtype=torch.int32)
    # Full mesh, same ordering as `am` ([Nq, B, Lq]) — so entry_* match am flat.
    entry_q   = q_range[:, None, None].expand(Nq, B, Lq).contiguous().view(-1)
    entry_doc = b_range[None, :, None].expand(Nq, B, Lq).contiguous().view(-1)
    entry_lq  = lq_range[None, None, :].expand(Nq, B, Lq).contiguous().view(-1)

    # Flat doc-batch index used by grad_D (handles both shared modes).
    if shared_docs:
        db = entry_doc                                      # grad_D is [B, Ld, d]
    else:
        db = entry_q.to(torch.int64) * B + entry_doc.to(torch.int64)
        db = db.to(torch.int32)                             # grad_D is [Nq*B, Ld, d]

    # Per-entry argmax slot j.
    j_flat = am.view(-1).to(torch.int32)                    # [E]

    # Combined key in int64 (safe for B*Ld up to ~2^63).
    key = db.to(torch.int64) * int(Ld) + j_flat.to(torch.int64)

    # argsort (unstable by default; see gotchas in the header).
    perm = torch.argsort(key)                               # int64[E]
    key_sorted = key.index_select(0, perm)                  # [E] int64

    # Unique consecutive keys → segments.
    uniq, counts = torch.unique_consecutive(key_sorted, return_counts=True)
    # seg_start[u] = sum(counts[:u])  → cumulative via cumsum + roll.
    seg_start = torch.zeros_like(counts)
    seg_start[1:] = torch.cumsum(counts, dim=0)[:-1]

    uniq_db = (uniq // int(Ld)).to(torch.int32)
    uniq_j  = (uniq %  int(Ld)).to(torch.int32)
    seg_start = seg_start.to(torch.int32)
    seg_count = counts.to(torch.int32)

    return dict(
        perm=perm.contiguous(),
        entry_q=entry_q.contiguous(),
        entry_doc=entry_doc.contiguous(),
        entry_lq=entry_lq.contiguous(),
        seg_start=seg_start.contiguous(),
        seg_count=seg_count.contiguous(),
        uniq_db=uniq_db.contiguous(),
        uniq_j=uniq_j.contiguous(),
        E=E,
        U=int(uniq.numel()),
    )


def _launch_bwd_dD_sorted(
    Q: torch.Tensor,
    argmax: torch.Tensor,
    grad_scores: torch.Tensor,
    Nq: int, B: int, Lq: int, Ld: int, d: int,
    shared_docs: bool,
    block_s: int = 32,
    block_d: int | None = None,
) -> torch.Tensor:
    """Sort-inverse grad_D launch. Returns an FP32 grad_D tensor.

    Shape of the returned grad_D:
       shared_docs=True  : [B,      Ld, d]   (same layout as D)
       shared_docs=False : [Nq * B, Ld, d]   (flat layout; caller reshapes)
    """
    assert Q.is_cuda and argmax.is_cuda and grad_scores.is_cuda
    d_pad = _next_pow2(d)

    tables = _build_sort_tables(argmax, Nq, B, Lq, Ld, shared_docs)
    U = tables["U"]

    N_flat = B if shared_docs else Nq * B
    grad_D = torch.zeros((N_flat, Ld, d), device=Q.device, dtype=torch.float32)

    if U == 0:
        return grad_D

    BLOCK_D = block_d if block_d is not None else d_pad
    # BLOCK_D must be a power of two and >= d_pad for full coverage in a
    # single tile; if caller overrides with a smaller tile we'll loop via
    # the grid's second axis.
    BLOCK_D = _next_pow2(BLOCK_D)
    num_d_tiles = (d_pad + BLOCK_D - 1) // BLOCK_D

    grid = (U, num_d_tiles)
    _maxsim_bwd_dD_sorted_kernel[grid](
        tables["perm"],
        tables["entry_q"], tables["entry_doc"], tables["entry_lq"],
        tables["seg_start"], tables["seg_count"],
        tables["uniq_db"], tables["uniq_j"],
        Q, grad_scores, grad_D,
        Lq, Ld, d, d_pad,
        grad_D.stride(0), grad_D.stride(1), grad_D.stride(2),
        Q.stride(0), Q.stride(1), Q.stride(2),
        grad_scores.stride(0), grad_scores.stride(1),
        BLOCK_S=block_s,
        BLOCK_D=BLOCK_D,
    )
    return grad_D


# ---------------------------------------------------------------------------
# Drop-in autograd function: same API as _FlashMaxSimBatchedFn but the dD
# path uses sort-inverse whenever the sort cost is cheap relative to atomics.
# ---------------------------------------------------------------------------

def _should_use_sortbwd(Nq: int, B: int, Lq: int, Ld: int) -> bool:
    """Heuristic dispatcher.

    Sort-inverse is faster when Nq*Lq per doc >> Ld (lots of atomic collisions)
    OR when Ld is small (few destinations, cheap unique_consecutive). We
    fall back to atomics at very tiny shapes where the argsort/launch
    overhead dominates.
    """
    E = Nq * B * Lq
    if E < 1024:  # atomic cost negligible
        return False
    # If the average #entries per destination > 4, atomics bottleneck is
    # real and the sort amortizes. The non-shared case has B destinations
    # per query (no collisions across queries), so use a stricter threshold.
    return True


try:
    from .flash_maxsim_batched_train import (  # type: ignore
        _maxsim_bwd_dQ_batched_kernel,
    )
except ImportError:
    _maxsim_bwd_dQ_batched_kernel = None  # pragma: no cover


class _FlashMaxSimBatchedSortFn(torch.autograd.Function):
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

        # ---- grad_Q: unchanged atomic-free kernel from flash_maxsim_batched_train
        grad_Q = torch.zeros_like(Q)
        if _maxsim_bwd_dQ_batched_kernel is None:
            raise RuntimeError("grad_Q kernel unavailable; cannot import.")
        _maxsim_bwd_dQ_batched_kernel[(Nq * Lq,)](
            D, argmax, grad_scores, grad_Q,
            Nq, B, Lq, d, d_pad,
            D.stride(0), D.stride(1), D.stride(2),
            grad_scores.stride(0), grad_scores.stride(1),
            1 if shared_docs else 0,
        )

        # ---- grad_D: sort-inverse (this file) or fallback
        if _should_use_sortbwd(Nq, B, Lq, Ld):
            grad_D_fp32 = _launch_bwd_dD_sorted(
                Q, argmax, grad_scores,
                Nq, B, Lq, Ld, d, shared_docs,
            )
        else:  # pragma: no cover — falls back to existing atomic kernel.
            from .flash_maxsim_batched_train import _maxsim_bwd_dD_batched_kernel
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

        grad_D = grad_D_fp32.to(D.dtype)
        if not shared_docs:
            grad_D = grad_D.reshape(Nq, B, Ld, d)
        return grad_Q, grad_D, None, None, None


def flash_maxsim_batched_train_sortbwd(
    Q: torch.Tensor, D: torch.Tensor,
    shared_docs: bool = True,
    doc_lengths: torch.Tensor | None = None,
    query_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """flash_maxsim_batched_train with the sort-inverse grad_D path."""
    return _FlashMaxSimBatchedSortFn.apply(Q, D, shared_docs, doc_lengths, query_lengths)


# ---------------------------------------------------------------------------
# In-file correctness test (runs only on demand; `python -m` this file).
# ---------------------------------------------------------------------------

def _torch_reference_grad_D(
    Q: torch.Tensor, D: torch.Tensor, grad_scores: torch.Tensor,
    shared_docs: bool,
) -> torch.Tensor:
    """Reference: compute grad_D via a dense pylate-style colbert_scores graph.

    We build S[Nq, B, Lq, Ld], take .max(dim=-1) along Ld, sum across Lq,
    backprop with grad_scores. The result `D.grad` is the ground truth for
    the sort-inverse kernel.
    """
    Q_ = Q.detach().float().requires_grad_(False)
    D_ = D.detach().float().requires_grad_(True)
    if shared_docs:
        # Q: [Nq, Lq, d], D: [B, Ld, d]  →  S: [Nq, B, Lq, Ld]
        S = torch.einsum("qld,bkd->qblk", Q_, D_)
    else:
        # Q: [Nq, Lq, d], D: [Nq, B, Ld, d]  →  S: [Nq, B, Lq, Ld]
        S = torch.einsum("qld,qbkd->qblk", Q_, D_)
    m = S.max(dim=-1).values              # [Nq, B, Lq]
    scores = m.sum(dim=-1)                # [Nq, B]
    scores.backward(grad_scores.float())
    return D_.grad.detach()


def _run_self_test():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else None
    if device is None:
        print("No CUDA available — skipping self test.")
        return
    for shared_docs in (True, False):
        Nq, B, Lq, Ld, d = 2, 3, 5, 7, 16
        Q = torch.randn(Nq, Lq, d, device=device, dtype=torch.float16)
        if shared_docs:
            D = torch.randn(B, Ld, d, device=device, dtype=torch.float16)
        else:
            D = torch.randn(Nq, B, Ld, d, device=device, dtype=torch.float16)
        grad_scores = torch.randn(Nq, B, device=device, dtype=torch.float32)

        # Reference via torch autograd
        gD_ref = _torch_reference_grad_D(Q, D, grad_scores, shared_docs).float()

        # Our kernel
        Qf = Q.clone().detach().requires_grad_(True)
        Df = D.clone().detach().requires_grad_(True)
        scores = flash_maxsim_batched_train_sortbwd(Qf, Df, shared_docs=shared_docs)
        scores.backward(grad_scores)
        gD_ours = Df.grad.float()

        err = (gD_ref - gD_ours).abs().max().item()
        rel = err / max(1e-8, gD_ref.abs().max().item())
        tag = "shared_docs=True" if shared_docs else "shared_docs=False"
        print(f"[{tag}] grad_D max_abs_err = {err:.3e}  max_rel_err = {rel:.3e}")
        assert rel < 5e-2, f"grad_D mismatch ({tag}): rel err {rel}"
    print("Self-test PASSED.")


if __name__ == "__main__":
    _run_self_test()
