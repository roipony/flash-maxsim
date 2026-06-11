"""Pure-PyTorch backward for flash-maxsim batched training.

Why this file exists
--------------------
The Triton `_maxsim_bwd_dD_batched_kernel` in `flash_maxsim_batched_train.py`
uses `tl.atomic_add` to scatter `grad_scores * Q` into `grad_D` at positions
selected by the saved `argmax`. At small shapes (ColBERT Ld=180, B=128) the
atomics serialize on the few hot positions in the document and our Triton
backward ends up ~3.3x slower than PyTorch's dense einsum backward.

PyTorch's `scatter_add_` / `index_add_` are highly optimized CUDA primitives
(batched segment-reduction with bucket-sort under the hood on recent
versions). They may be a strictly better fallback for the "small batch"
regime than any hand-written atomic kernel — and they don't require a
custom kernel at all.

Math (shared_docs=True)
-----------------------
Inputs:
    Q            : [Nq, Lq, d]       (fp16 in practice)
    D            : [B,  Ld, d]
    argmax       : [Nq*B, Lq] int32, layout argmax[pid, lq] where
                   pid = doc_id * Nq + q_idx   (shared_docs)
                   pid = q_idx  * B  + doc_id  (non-shared)
    grad_scores  : [Nq, B] fp32

Outputs (shared_docs=True):
    grad_D[b, argmax[b*Nq+q, l], :] += grad_scores[q, b] * Q[q, l, :]
    grad_Q[q, l, :]                 += sum_b  grad_scores[q, b] * D[b, argmax[b*Nq+q, l], :]

Non-shared (shared_docs=False): D is [Nq, B, Ld, d], a separate set of docs
per query. The autograd function flattens it to [Nq*B, Ld, d] internally.
We keep that convention here.

Memory cost of the `contribs` intermediate
------------------------------------------
To avoid an inner Python loop over `b`, the simplest formulation builds

    contribs = grad_scores[:, :, None, None] * Q[:, None, :, :]   # [Nq, B, Lq, d]

and then scatters it into `grad_D`. Storage cost (fp32, 4 bytes):

    bytes(contribs) = Nq * B * Lq * d * 4

Concrete shapes:

    | Setting              | Nq  | B    | Lq  | d   | contribs size |
    |----------------------|-----|------|-----|-----|---------------|
    | ColBERT small        |  8  |  128 |  32 | 128 |     16 MiB    |
    | ColBERT medium       |  8  |  128 | 180 | 128 |     90 MiB    |
    | ColBERT big-batch    |  8  | 1024 | 180 | 128 |    720 MiB    |
    | ColBERT huge-batch   |  8  | 4096 | 180 | 128 |    2.8 GiB    |
    | ColPali typical      |  8  |   32 | 512 | 128 |     64 MiB    |
    | ColPali large doc    |  8  |   32 |1024 | 128 |    128 MiB    |
    | ColPali big-batch    |  8  |  256 |1024 | 128 |      1 GiB    |

Rules of thumb:
  * <= 256 MiB : comfortable; materialize `contribs` and use scatter_add_.
  * 256 MiB - 1 GiB : still OK on an 80 GiB GPU, watch peak activation budget.
  * > 1 GiB : tile over `b` (process `b_tile` docs at a time) or skip
              `contribs` entirely and use `index_add_` inside a small
              per-query loop (Nq is typically <= 8).

At ColBERT B=1024, Lq=180, d=128, contribs is 720 MiB which is acceptable
on an 80 GiB card (the forward activations dwarf it), but at B=4096 it
becomes the dominant activation and we should tile.

ColPali (B=32) is fine at any reasonable Lq — contribs is always < 200 MiB.
"""

from __future__ import annotations

from typing import Tuple

import torch


# ---------------------------------------------------------------------------
# Pure-PyTorch backward
# ---------------------------------------------------------------------------


def _resolve_argmax_layout(
    argmax: torch.Tensor, Nq: int, B: int, Lq: int, shared_docs: bool
) -> torch.Tensor:
    """Return argmax reshaped to [Nq, B, Lq] regardless of input layout.

    Forward stores argmax with CTA ordering:
        shared_docs=True  : pid = doc_id * Nq + q_idx  -> shape [B, Nq, Lq]
        shared_docs=False : pid = q_idx  * B  + doc_id -> shape [Nq, B, Lq]
    """
    argmax = argmax.view(-1, Lq)
    if shared_docs:
        # [B*Nq, Lq] laid out as (doc, query) -> reorder to (query, doc)
        am = argmax.view(B, Nq, Lq).permute(1, 0, 2).contiguous()  # [Nq, B, Lq]
    else:
        am = argmax.view(Nq, B, Lq).contiguous()
    return am.to(torch.long)


def torch_backward(
    Q: torch.Tensor,
    D: torch.Tensor,
    argmax: torch.Tensor,
    grad_scores: torch.Tensor,
    shared_docs: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch backward using scatter_add_ / gather.

    Args:
        Q            : [Nq, Lq, d]
        D            : [B, Ld, d]            if shared_docs else [Nq*B, Ld, d] or [Nq, B, Ld, d]
        argmax       : [Nq*B, Lq] int (any integer dtype)
        grad_scores  : [Nq, B]
        shared_docs  : controls argmax pid layout (see module docstring)

    Returns:
        grad_Q : [Nq, Lq, d]   same dtype as Q
        grad_D : same shape and dtype as input D

    Strategy:
      * Materialize `contribs = gs * Q` of shape [Nq, B, Lq, d].
      * For grad_D: scatter_add_ along the Ld axis using `argmax` as the index.
      * For grad_Q: gather D rows at `argmax`, multiply by gs, sum over B.
    """
    Nq, Lq, d = Q.shape

    # Normalize D shape to [*, Ld, d] where * is either B (shared) or Nq*B.
    orig_D_shape = D.shape
    if shared_docs:
        assert D.dim() == 3, "D must be [B, Ld, d] when shared_docs=True"
        B, Ld, _ = D.shape
        D_flat = D
    else:
        if D.dim() == 4:
            Nq_, B, Ld, _ = D.shape
            assert Nq_ == Nq
            D_flat = D.reshape(Nq * B, Ld, d)
        else:
            # Already flattened [Nq*B, Ld, d]
            NB, Ld, _ = D.shape
            assert NB % Nq == 0
            B = NB // Nq
            D_flat = D

    # Promote to fp32 for numerically clean reduction; cast back at the end.
    Q_f = Q.float()
    D_f = D_flat.float()
    gs = grad_scores.float()  # [Nq, B]

    # argmax laid out as [Nq, B, Lq].
    am = _resolve_argmax_layout(argmax, Nq, B, Lq, shared_docs)  # [Nq, B, Lq] long

    # -----------------------------------------------------------------
    # grad_Q[q, l, :] = sum_b gs[q, b] * D_flat[ b_eff(q,b), am[q,b,l], : ]
    # -----------------------------------------------------------------
    if shared_docs:
        # D_flat shape [B, Ld, d].  We need D_flat[b, am[q,b,l], :] for all q,b,l.
        # Expand D to match query axis via gather:
        #   idx[q, b, l, k] = am[q, b, l]           (broadcast over k)
        # gather from D_exp of shape [Nq, B, Ld, d] along dim=2.
        idx = am.unsqueeze(-1).expand(Nq, B, Lq, d)                # [Nq,B,Lq,d]
        D_exp = D_f.unsqueeze(0).expand(Nq, B, Ld, d)              # [Nq,B,Ld,d] (view)
        gathered = torch.gather(D_exp, dim=2, index=idx)           # [Nq,B,Lq,d]
    else:
        # D_flat shape [Nq*B, Ld, d] -> reshape to [Nq, B, Ld, d].
        D_nb = D_f.view(Nq, B, Ld, d)
        idx = am.unsqueeze(-1).expand(Nq, B, Lq, d)
        gathered = torch.gather(D_nb, dim=2, index=idx)            # [Nq,B,Lq,d]

    # Weight by grad_scores and sum over B.
    grad_Q = (gs[:, :, None, None] * gathered).sum(dim=1)          # [Nq, Lq, d]

    # -----------------------------------------------------------------
    # grad_D[b, am[q,b,l], :] += gs[q, b] * Q[q, l, :]
    # -----------------------------------------------------------------
    # contribs[q, b, l, :] = gs[q, b] * Q[q, l, :]    -> [Nq, B, Lq, d]
    contribs = gs[:, :, None, None] * Q_f[:, None, :, :]           # [Nq, B, Lq, d]

    if shared_docs:
        # Sum over q first (routes into a single [B, Ld, d] accumulator).
        # We still need per-q scatter since argmax depends on q. Use a
        # [B, Ld, d] accumulator and loop-free scatter by flattening (q, l):
        #   index layout = am.permute(1,0,2).reshape(B, Nq*Lq)
        #   src          = contribs.permute(1,0,2,3).reshape(B, Nq*Lq, d)
        idx_bd = am.permute(1, 0, 2).reshape(B, Nq * Lq)           # [B, Nq*Lq]
        src_bd = contribs.permute(1, 0, 2, 3).reshape(B, Nq * Lq, d)
        grad_D_f = torch.zeros(B, Ld, d, device=Q.device, dtype=torch.float32)
        grad_D_f.scatter_add_(
            dim=1,
            index=idx_bd.unsqueeze(-1).expand(B, Nq * Lq, d),
            src=src_bd,
        )
    else:
        # Each (q, b) owns its own D slice in [Nq, B, Ld, d]. Scatter along
        # the Ld axis independently per (q, b).
        grad_D_nb = torch.zeros(Nq, B, Ld, d, device=Q.device, dtype=torch.float32)
        grad_D_nb.scatter_add_(
            dim=2,
            index=am.unsqueeze(-1).expand(Nq, B, Lq, d),
            src=contribs,
        )
        grad_D_f = grad_D_nb.reshape(Nq * B, Ld, d)

    # Cast back to D's dtype and reshape to user-facing shape.
    grad_D = grad_D_f.to(D.dtype)
    if shared_docs:
        grad_D = grad_D.view(orig_D_shape)
    else:
        if len(orig_D_shape) == 4:
            grad_D = grad_D.view(Nq, B, Ld, d)
        else:
            grad_D = grad_D.view(orig_D_shape)

    grad_Q = grad_Q.to(Q.dtype)
    return grad_Q, grad_D


def torch_backward_tiled(
    Q: torch.Tensor,
    D: torch.Tensor,
    argmax: torch.Tensor,
    grad_scores: torch.Tensor,
    shared_docs: bool = True,
    b_tile: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Memory-friendly variant: tile over the `B` (doc-batch) axis.

    Useful when `contribs = gs * Q` of shape [Nq, B, Lq, d] would exceed a
    GB or two. For ColBERT at B >= 2048 this is the recommended path.
    """
    Nq, Lq, d = Q.shape
    if shared_docs:
        B, Ld, _ = D.shape
        D_flat = D
    else:
        if D.dim() == 4:
            _, B, Ld, _ = D.shape
            D_flat = D.reshape(Nq * B, Ld, d)
        else:
            NB, Ld, _ = D.shape
            B = NB // Nq
            D_flat = D

    am = _resolve_argmax_layout(argmax, Nq, B, Lq, shared_docs)  # [Nq, B, Lq]
    gs = grad_scores.float()

    grad_Q_acc = torch.zeros(Nq, Lq, d, device=Q.device, dtype=torch.float32)
    if shared_docs:
        grad_D_f = torch.zeros(B, Ld, d, device=Q.device, dtype=torch.float32)
    else:
        grad_D_f = torch.zeros(Nq, B, Ld, d, device=Q.device, dtype=torch.float32)

    Q_f = Q.float()
    for b0 in range(0, B, b_tile):
        b1 = min(b0 + b_tile, B)
        am_t = am[:, b0:b1, :]                         # [Nq, bt, Lq]
        gs_t = gs[:, b0:b1]                            # [Nq, bt]
        bt = b1 - b0

        if shared_docs:
            D_slice = D_flat[b0:b1].float()            # [bt, Ld, d]
            idx = am_t.unsqueeze(-1).expand(Nq, bt, Lq, d)
            D_exp = D_slice.unsqueeze(0).expand(Nq, bt, Ld, d)
            gathered = torch.gather(D_exp, dim=2, index=idx)
        else:
            D_slice = D_flat.view(Nq, B, Ld, d)[:, b0:b1].float()  # [Nq, bt, Ld, d]
            idx = am_t.unsqueeze(-1).expand(Nq, bt, Lq, d)
            gathered = torch.gather(D_slice, dim=2, index=idx)

        grad_Q_acc += (gs_t[:, :, None, None] * gathered).sum(dim=1)

        contribs = gs_t[:, :, None, None] * Q_f[:, None, :, :]  # [Nq, bt, Lq, d]
        if shared_docs:
            idx_bd = am_t.permute(1, 0, 2).reshape(bt, Nq * Lq)
            src_bd = contribs.permute(1, 0, 2, 3).reshape(bt, Nq * Lq, d)
            grad_D_f[b0:b1].scatter_add_(
                dim=1,
                index=idx_bd.unsqueeze(-1).expand(bt, Nq * Lq, d),
                src=src_bd,
            )
        else:
            grad_D_f[:, b0:b1].scatter_add_(
                dim=2,
                index=am_t.unsqueeze(-1).expand(Nq, bt, Lq, d),
                src=contribs,
            )

    grad_Q = grad_Q_acc.to(Q.dtype)
    grad_D = grad_D_f.to(D.dtype)
    if not shared_docs and D.dim() == 3:
        grad_D = grad_D.view(Nq * B, Ld, d)
    return grad_Q, grad_D


# ---------------------------------------------------------------------------
# torch.compile variants
# ---------------------------------------------------------------------------
# `torch.compile` can fuse the (gs * Q) broadcast with the scatter_add_, and
# on recent CUDA+Inductor it can pick efficient segment-sum lowerings. We
# expose a couple of modes so the caller can benchmark.

try:
    torch_backward_compiled_default = torch.compile(torch_backward)
    torch_backward_compiled_reduce_overhead = torch.compile(
        torch_backward, mode="reduce-overhead"
    )
    torch_backward_compiled_max_autotune = torch.compile(
        torch_backward, mode="max-autotune"
    )
except Exception:  # pragma: no cover - older torch versions
    torch_backward_compiled_default = torch_backward
    torch_backward_compiled_reduce_overhead = torch_backward
    torch_backward_compiled_max_autotune = torch_backward


# ---------------------------------------------------------------------------
# Brute-force reference + correctness test
# ---------------------------------------------------------------------------


def _reference_backward(
    Q: torch.Tensor,
    D: torch.Tensor,
    argmax: torch.Tensor,
    grad_scores: torch.Tensor,
    shared_docs: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Simple nested-loop reference, matching the math in the docstring."""
    Nq, Lq, d = Q.shape
    if shared_docs:
        B, Ld, _ = D.shape
        D_flat = D
    else:
        if D.dim() == 4:
            _, B, Ld, _ = D.shape
            D_flat = D.reshape(Nq * B, Ld, d)
        else:
            NB, Ld, _ = D.shape
            B = NB // Nq
            D_flat = D

    am_2d = argmax.view(-1, Lq).to(torch.long)
    grad_Q = torch.zeros(Nq, Lq, d, device=Q.device, dtype=torch.float32)
    grad_D_f = torch.zeros(
        (B, Ld, d) if shared_docs else (Nq * B, Ld, d),
        device=Q.device, dtype=torch.float32,
    )
    Q_f = Q.float()
    D_f = D_flat.float()
    gs = grad_scores.float()

    for q in range(Nq):
        for b in range(B):
            if shared_docs:
                pid = b * Nq + q
                d_batch = b
            else:
                pid = q * B + b
                d_batch = q * B + b
            gsv = gs[q, b].item()
            for l in range(Lq):
                j = int(am_2d[pid, l].item())
                grad_D_f[d_batch, j, :] += gsv * Q_f[q, l, :]
                grad_Q[q, l, :] += gsv * D_f[d_batch, j, :]

    if shared_docs:
        grad_D_out = grad_D_f.to(D.dtype)
    else:
        grad_D_out = grad_D_f.to(D.dtype)
        if D.dim() == 4:
            grad_D_out = grad_D_out.view(Nq, B, Ld, d)
    return grad_Q.to(Q.dtype), grad_D_out


def _make_inputs(Nq, B, Lq, Ld, d, shared_docs, device, dtype=torch.float32, seed=0):
    """Build random (Q, D, argmax, grad_scores) triples with valid argmax."""
    g = torch.Generator(device=device).manual_seed(seed)
    Q = torch.randn(Nq, Lq, d, device=device, dtype=dtype, generator=g)
    if shared_docs:
        D = torch.randn(B, Ld, d, device=device, dtype=dtype, generator=g)
    else:
        D = torch.randn(Nq, B, Ld, d, device=device, dtype=dtype, generator=g)
    argmax = torch.randint(
        0, Ld, (Nq * B, Lq), device=device, dtype=torch.int32, generator=g,
    )
    grad_scores = torch.randn(Nq, B, device=device, dtype=torch.float32, generator=g)
    return Q, D, argmax, grad_scores


def _check(name, Nq, B, Lq, Ld, d, shared_docs, device):
    Q, D, argmax, gs = _make_inputs(
        Nq, B, Lq, Ld, d, shared_docs, device, dtype=torch.float32,
    )
    gq_ref, gd_ref = _reference_backward(Q, D, argmax, gs, shared_docs)
    gq, gd = torch_backward(Q, D, argmax, gs, shared_docs)

    dq = (gq - gq_ref).abs().max().item()
    dd = (gd - gd_ref).abs().max().item()
    tol = 1e-4 if Q.dtype == torch.float32 else 5e-2
    ok = dq < tol and dd < tol
    print(
        f"  [{name}] shared={shared_docs} Nq={Nq} B={B} Lq={Lq} Ld={Ld} d={d}: "
        f"max|dQ|={dq:.2e}  max|dD|={dd:.2e}  -> {'OK' if ok else 'FAIL'}"
    )
    assert ok, f"{name} mismatch"


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running correctness tests on {device} ...")

    # Tiny shapes: brute-force is only viable here.
    _check("tiny-shared",     Nq=2, B=3, Lq=4, Ld=5, d=8,  shared_docs=True,  device=device)
    _check("tiny-nonshared",  Nq=2, B=3, Lq=4, Ld=5, d=8,  shared_docs=False, device=device)
    _check("rect-shared",     Nq=3, B=5, Lq=7, Ld=11, d=16, shared_docs=True,  device=device)
    _check("rect-nonshared",  Nq=3, B=5, Lq=7, Ld=11, d=16, shared_docs=False, device=device)

    # Verify the tiled variant agrees with the monolithic one.
    Q, D, argmax, gs = _make_inputs(3, 17, 9, 23, 32, True, device)
    gq_a, gd_a = torch_backward(Q, D, argmax, gs, True)
    gq_b, gd_b = torch_backward_tiled(Q, D, argmax, gs, True, b_tile=4)
    assert torch.allclose(gq_a, gq_b, atol=1e-4)
    assert torch.allclose(gd_a, gd_b, atol=1e-4)
    print("  [tiled parity] OK")

    print("All correctness tests passed.")
