"""CSR invgrid backward for Chamfer distance — generalisation of
`flash_maxsim_invgrid_proposal` to Chamfer's argmin backward.

This is the implementation behind Embodiment A4 of the Inverse-Grid CSR
Backward IDR: same destination-owned reduction pattern, applied to a
different forward operator (`min` over `||P_i - Q_j||²` instead of `max`
over `Q_i · D_j`).

Math.
  Chamfer:  CD(P, Q) = (1/N) Σ_i min_j ||P_i − Q_j||²  +  (1/M) Σ_j min_i ||Q_j − P_i||²

  P→Q direction. Forward saves argmin_pq[i] = argmin_j ||P_i − Q_j||².
    grad_P[i] = (2g/N) · (P[i] − Q[argmin_pq[i]])         — direct, no contention
    grad_Q[j] = (2g/N) · (count_j · Q[j] − Σ_{i: pq[i]=j} P_i)  — destination-owned

  Q→P direction is symmetric: grad_Q direct, grad_P destination-owned.

Implementation.
  Per direction:
    1. build_chamfer_inverse_csr(argmin, M_or_N) → (row_ptr, col_idx).
       nnz = number of source points; n_dest = number of destination points.
    2. Launch one program per destination over the dD-equivalent kernel.
       The program reads its source list from CSR, accumulates source vectors
       in an FP32 register, and writes once. No atomics.
"""
import torch
import triton
import triton.language as tl

from .flash_chamfer import flash_chamfer_one_direction, _next_pow2


# ---------------------------------------------------------------------------
# CSR builder — generic (works for any argmin/argmax over a 1-D source axis).
# Same int64-cumsum-then-downcast safety pattern as build_inverse_csr.
# ---------------------------------------------------------------------------

def build_chamfer_inverse_csr(argmin: torch.Tensor, n_dest: int) -> tuple[torch.Tensor, torch.Tensor]:
    """argmin: [N] int32/int64 with values in [0, n_dest).
    Returns (row_ptr [n_dest+1], col_idx [N]).
    """
    device = argmin.device
    arg_i64 = argmin.to(torch.int64).contiguous()
    N = arg_i64.numel()

    # Sort sources by destination → contiguous segments per destination.
    order = torch.argsort(arg_i64, stable=True)
    col_idx = order.to(torch.int32).contiguous()       # [N] int32

    # row_ptr via bincount + cumsum (int64-safe per the IDR fix).
    counts = torch.bincount(arg_i64, minlength=n_dest)  # [n_dest] int64
    row_ptr_i64 = torch.zeros(n_dest + 1, dtype=torch.int64, device=device)
    row_ptr_i64[1:] = counts.cumsum(dim=0)
    if row_ptr_i64[-1].item() > torch.iinfo(torch.int32).max:
        row_ptr = row_ptr_i64
    else:
        row_ptr = row_ptr_i64.to(torch.int32)
    return row_ptr.contiguous(), col_idx


# ---------------------------------------------------------------------------
# Triton kernel: one program per destination point.
# Computes:  grad_dst[j] = scale * (count_j · DST[j] − Σ_{i in segment} SRC[i])
#
# Used for BOTH directions:
#   P→Q direction: SRC=P, DST=Q, scale = 2g/N. Output goes to grad_Q.
#   Q→P direction: SRC=Q, DST=P, scale = 2g/M. Output goes to grad_P.
# ---------------------------------------------------------------------------

@triton.jit
def _chamfer_invgrid_dDST_kernel(
    SRC_ptr, DST_ptr, grad_DST_ptr,
    row_ptr_ptr, col_idx_ptr,
    scale,                              # FP32 scalar: 2g / N or 2g / M
    n_dest: tl.constexpr,
    d: tl.constexpr, d_pad: tl.constexpr,
    stride_src_n, stride_src_d,
    stride_dst_m, stride_dst_d,
    stride_gd_m, stride_gd_d,
):
    j = tl.program_id(0)
    if j >= n_dest:
        return

    j_i64 = tl.cast(j, tl.int64)
    seg_start = tl.load(row_ptr_ptr + j).to(tl.int64)
    seg_end   = tl.load(row_ptr_ptr + j + 1).to(tl.int64)
    count = (seg_end - seg_start).to(tl.float32)

    k = tl.arange(0, d_pad)
    km = k < d

    # FP32 accumulator for Σ SRC[i] over the segment.
    acc = tl.zeros([d_pad], dtype=tl.float32)

    cursor = seg_start
    while cursor < seg_end:
        i = tl.load(col_idx_ptr + cursor).to(tl.int64)
        v = tl.load(
            SRC_ptr + i * stride_src_n + k * stride_src_d,
            mask=km, other=0.0,
        ).to(tl.float32)
        acc += v
        cursor += 1

    # Load DST[j] for the count·DST[j] − Σ SRC term.
    dst_j = tl.load(
        DST_ptr + j_i64 * stride_dst_m + k * stride_dst_d,
        mask=km, other=0.0,
    ).to(tl.float32)

    out = scale * (count * dst_j - acc)

    # Single coalesced write (no atomics — j owns this destination).
    tl.store(
        grad_DST_ptr + j_i64 * stride_gd_m + k * stride_gd_d,
        out, mask=km,
    )


# ---------------------------------------------------------------------------
# Autograd Function — full Chamfer with CSR invgrid backward in BOTH directions.
# ---------------------------------------------------------------------------

class _FlashChamferInvGridFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, P, Q):
        # P: [N, d], Q: [M, d]
        out_pq, argmin_pq = flash_chamfer_one_direction(P, Q, metric="l2_sq", save_argmin=True)
        out_qp, argmin_qp = flash_chamfer_one_direction(Q, P, metric="l2_sq", save_argmin=True)
        ctx.save_for_backward(P, Q, argmin_pq, argmin_qp)
        N, M = P.shape[0], Q.shape[0]
        ctx.N, ctx.M = N, M
        cd = out_pq.sum() / N + out_qp.sum() / M
        return cd

    @staticmethod
    def backward(ctx, grad_cd):
        P, Q, argmin_pq, argmin_qp = ctx.saved_tensors
        N, M = ctx.N, ctx.M
        d = P.shape[1]
        d_pad = _next_pow2(d)
        g = grad_cd.float().item() if grad_cd.dim() == 0 else grad_cd.float().mean().item()
        # `grad_cd` is a scalar (cd is a scalar reduction).

        Pf = P.float()
        Qf = Q.float()

        # ---- direct gradient parts (no contention; each source has unique destination) ----
        # grad_P[i] from P→Q: (2g/N) · (P[i] − Q[argmin_pq[i]])
        Q_at_pq = Qf[argmin_pq.long()]                       # [N, d]
        grad_P_direct = (2.0 * g / N) * (Pf - Q_at_pq)       # [N, d]
        # grad_Q[j] from Q→P: (2g/M) · (Q[j] − P[argmin_qp[j]])
        P_at_qp = Pf[argmin_qp.long()]                       # [M, d]
        grad_Q_direct = (2.0 * g / M) * (Qf - P_at_qp)       # [M, d]

        # ---- CSR-invgrid gradient parts (destination-owned, atomic-free) ----
        # grad_Q[j] from P→Q: (2g/N) · (count_j · Q[j] − Σ_{i: pq[i]=j} P[i])
        row_ptr_pq, col_idx_pq = build_chamfer_inverse_csr(argmin_pq, M)
        grad_Q_from_pq = torch.zeros((M, d), device=P.device, dtype=torch.float32)
        _chamfer_invgrid_dDST_kernel[(M,)](
            Pf.contiguous(), Qf.contiguous(), grad_Q_from_pq,
            row_ptr_pq, col_idx_pq,
            float(2.0 * g / N),
            M,
            d, d_pad,
            Pf.stride(0), Pf.stride(1),
            Qf.stride(0), Qf.stride(1),
            grad_Q_from_pq.stride(0), grad_Q_from_pq.stride(1),
        )

        # grad_P[i] from Q→P: (2g/M) · (count_i · P[i] − Σ_{j: qp[j]=i} Q[j])
        row_ptr_qp, col_idx_qp = build_chamfer_inverse_csr(argmin_qp, N)
        grad_P_from_qp = torch.zeros((N, d), device=P.device, dtype=torch.float32)
        _chamfer_invgrid_dDST_kernel[(N,)](
            Qf.contiguous(), Pf.contiguous(), grad_P_from_qp,
            row_ptr_qp, col_idx_qp,
            float(2.0 * g / M),
            N,
            d, d_pad,
            Qf.stride(0), Qf.stride(1),
            Pf.stride(0), Pf.stride(1),
            grad_P_from_qp.stride(0), grad_P_from_qp.stride(1),
        )

        # ---- combine direct + invgrid into the final per-tensor gradients ----
        grad_P = (grad_P_direct + grad_P_from_qp).to(P.dtype)
        grad_Q = (grad_Q_direct + grad_Q_from_pq).to(Q.dtype)
        return grad_P, grad_Q


def flash_chamfer_invgrid_train(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Chamfer distance with autograd — atomic-free CSR-invgrid backward.

    Same API as `flash_chamfer_train` but the destination-owned gradient
    accumulations (grad_Q from the P→Q direction, grad_P from the Q→P
    direction) use the inverse-grid CSR pattern from the parent IDR
    instead of `scatter_add_`.
    """
    return _FlashChamferInvGridFn.apply(P, Q)
