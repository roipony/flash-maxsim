"""Fused Triton kernel for Chamfer distance between 3D point clouds.

Chamfer distance between point sets P [N, d] and Q [M, d]:
    CD(P, Q) = (1/N) Σ_i min_j ||P_i - Q_j||² + (1/M) Σ_j min_i ||Q_j - P_i||²

Same tiling pattern as Flash-MaxSim, but with min instead of max.
Uses the dot product trick: ||a-b||² = ||a||² + ||b||² - 2(a·b)

Never materializes the [N, M] distance matrix → O(N+M) memory instead of O(N*M).
"""

import torch
import triton
import triton.language as tl


def _next_pow2(x):
    return 1 << (x - 1).bit_length()


# ---------------------------------------------------------------------------
# Triton kernel: blocks of BLOCK_I source points, tiles over target in BLOCK_J
# Same structure as Flash-MaxSim but min instead of max
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_I": 16, "BLOCK_J": 16}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_I": 16, "BLOCK_J": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_I": 16, "BLOCK_J": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_I": 32, "BLOCK_J": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_I": 32, "BLOCK_J": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_I": 32, "BLOCK_J": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_I": 64, "BLOCK_J": 64}, num_warps=4, num_stages=2),
    ],
    key=["M", "d_pad"],
)
@triton.jit
def _chamfer_l2_kernel(
    src_ptr, tgt_ptr, out_ptr, argmin_ptr,
    src_norms_ptr, tgt_norms_ptr,
    N: tl.constexpr, M: tl.constexpr,
    d: tl.constexpr, d_pad: tl.constexpr,
    stride_src_n, stride_src_d,
    stride_tgt_m, stride_tgt_d,
    save_argmin: tl.constexpr,
    BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr,
):
    """For a block of source points, find min_j ||src_i - tgt_j||² for each i."""
    pid = tl.program_id(0)
    i_start = pid * BLOCK_I
    i_off = i_start + tl.arange(0, BLOCK_I)
    i_valid = i_off < N

    k_off = tl.arange(0, d_pad)
    k_mask = k_off < d

    # Load source block [BLOCK_I, d_pad]
    src_block = tl.load(
        src_ptr + i_off[:, None] * stride_src_n + k_off[None, :] * stride_src_d,
        mask=i_valid[:, None] & k_mask[None, :], other=0.0,
    ).to(tl.float16)

    # Source norms [BLOCK_I]
    src_norms = tl.load(src_norms_ptr + i_off, mask=i_valid, other=0.0)

    # Running min distance per source point
    min_dist = tl.full([BLOCK_I], float("inf"), dtype=tl.float32)
    if save_argmin:
        min_idx = tl.full([BLOCK_I], 0, dtype=tl.int32)

    # Tile over target points
    for j_start in range(0, M, BLOCK_J):
        j_off = j_start + tl.arange(0, BLOCK_J)
        j_valid = j_off < M

        # Load target block [BLOCK_J, d_pad]
        tgt_block = tl.load(
            tgt_ptr + j_off[:, None] * stride_tgt_m + k_off[None, :] * stride_tgt_d,
            mask=j_valid[:, None] & k_mask[None, :], other=0.0,
        ).to(tl.float16)

        # Target norms [BLOCK_J]
        tgt_norms = tl.load(tgt_norms_ptr + j_off, mask=j_valid, other=0.0)

        # Dot products: [BLOCK_I, d_pad] @ [d_pad, BLOCK_J] -> [BLOCK_I, BLOCK_J]
        dots = tl.dot(src_block, tl.trans(tgt_block)).to(tl.float32)

        # Squared L2: ||a-b||² = ||a||² + ||b||² - 2(a·b)
        dists = src_norms[:, None] + tgt_norms[None, :] - 2.0 * dots
        dists = tl.maximum(dists, 0.0)  # clamp numerical noise
        dists = tl.where(j_valid[None, :], dists, float("inf"))

        # Update running min per source point
        tile_min = tl.min(dists, axis=1)  # [BLOCK_I]
        if save_argmin:
            tile_argmin = tl.argmin(dists, axis=1).to(tl.int32) + j_start
            update = tile_min < min_dist
            min_idx = tl.where(update, tile_argmin, min_idx)
        min_dist = tl.minimum(min_dist, tile_min)

    # Store results
    tl.store(out_ptr + i_off, min_dist, mask=i_valid)
    if save_argmin:
        tl.store(argmin_ptr + i_off, min_idx, mask=i_valid)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_I": 16, "BLOCK_J": 16}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_I": 16, "BLOCK_J": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_I": 16, "BLOCK_J": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_I": 32, "BLOCK_J": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_I": 32, "BLOCK_J": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_I": 64, "BLOCK_J": 64}, num_warps=4, num_stages=2),
    ],
    key=["M", "d_pad"],
)
@triton.jit
def _chamfer_l1_kernel(
    src_ptr, tgt_ptr, out_ptr,
    N: tl.constexpr, M: tl.constexpr,
    d: tl.constexpr, d_pad: tl.constexpr,
    stride_src_n, stride_src_d,
    stride_tgt_m, stride_tgt_d,
    BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr,
):
    """L1 Chamfer: for a block of source points, find min_j Σ|src_i - tgt_j|."""
    pid = tl.program_id(0)
    i_start = pid * BLOCK_I
    i_off = i_start + tl.arange(0, BLOCK_I)
    i_valid = i_off < N

    k_off = tl.arange(0, d_pad)
    k_mask = k_off < d

    min_dist = tl.full([BLOCK_I], float("inf"), dtype=tl.float32)

    for j_start in range(0, M, BLOCK_J):
        j_off = j_start + tl.arange(0, BLOCK_J)
        j_valid = j_off < M

        # For each (i, j) pair, compute L1 distance
        for ji in range(BLOCK_J):
            j_idx = j_start + ji
            if j_idx < M:
                tgt_vec = tl.load(
                    tgt_ptr + j_idx * stride_tgt_m + k_off * stride_tgt_d,
                    mask=k_mask, other=0.0,
                ).to(tl.float32)

                src_block = tl.load(
                    src_ptr + i_off[:, None] * stride_src_n + k_off[None, :] * stride_src_d,
                    mask=i_valid[:, None] & k_mask[None, :], other=0.0,
                ).to(tl.float32)

                diff = tl.abs(src_block - tgt_vec[None, :])
                dists = tl.sum(diff, axis=1)  # [BLOCK_I]
                min_dist = tl.where(i_valid, tl.minimum(min_dist, dists), min_dist)

    tl.store(out_ptr + i_off, min_dist, mask=i_valid)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def flash_chamfer_one_direction(P: torch.Tensor, Q: torch.Tensor,
                                 metric: str = "l2_sq",
                                 save_argmin: bool = False):
    """Per-point min distances from P to Q.

    Args:
        P: [N, d] source points
        Q: [M, d] target points
        metric: "l2_sq" (squared L2) or "l1" (Manhattan)
        save_argmin: if True, also return [N] int32 argmin indices

    Returns:
        out: [N] per-point minimum distances
        argmin: [N] int32 argmin indices (only if save_argmin=True)
    """
    assert P.dim() == 2 and Q.dim() == 2 and P.shape[1] == Q.shape[1]
    N, d = P.shape
    M = Q.shape[0]
    d_pad = _next_pow2(max(d, 16))  # ensure >= 16 for tl.dot

    P_f = P.contiguous().float()
    Q_f = Q.contiguous().float()
    out = torch.empty(N, device=P.device, dtype=torch.float32)
    argmin = torch.empty(N, device=P.device, dtype=torch.int32) if save_argmin else out  # dummy

    if metric == "l1":
        assert not save_argmin, "save_argmin not supported for L1 metric"
        grid = lambda meta: ((N + meta["BLOCK_I"] - 1) // meta["BLOCK_I"],)
        _chamfer_l1_kernel[grid](
            P_f, Q_f, out,
            N, M, d, d_pad,
            P_f.stride(0), P_f.stride(1),
            Q_f.stride(0), Q_f.stride(1),
        )
    else:
        P_norms = (P_f * P_f).sum(dim=1)
        Q_norms = (Q_f * Q_f).sum(dim=1)

        # Pad P and Q to d_pad if needed
        if d < d_pad:
            P_padded = torch.zeros(N, d_pad, device=P.device, dtype=torch.float32)
            P_padded[:, :d] = P_f
            Q_padded = torch.zeros(M, d_pad, device=Q.device, dtype=torch.float32)
            Q_padded[:, :d] = Q_f
        else:
            P_padded = P_f
            Q_padded = Q_f

        # Grid must match the autotuner's choice of BLOCK_I (else under-launched
        # configs leave the back half of `argmin` uninitialized — see the
        # `meta`-driven grid required by Triton autotune).
        grid = lambda meta: ((N + meta["BLOCK_I"] - 1) // meta["BLOCK_I"],)
        _chamfer_l2_kernel[grid](
            P_padded, Q_padded, out, argmin,
            P_norms, Q_norms,
            N, M, d, d_pad,
            P_padded.stride(0), P_padded.stride(1),
            Q_padded.stride(0), Q_padded.stride(1),
            1 if save_argmin else 0,
        )

    if save_argmin:
        return out, argmin
    return out


def flash_chamfer(P: torch.Tensor, Q: torch.Tensor,
                  metric: str = "l2_sq") -> torch.Tensor:
    """Chamfer distance between point sets P and Q.

    CD(P, Q) = (1/N) Σ min_j d(P_i, Q_j) + (1/M) Σ min_i d(Q_j, P_i)

    Args:
        P: [N, d] first point cloud
        Q: [M, d] second point cloud
        metric: "l2_sq" or "l1"

    Returns:
        scalar Chamfer distance
    """
    d1 = flash_chamfer_one_direction(P, Q, metric)
    d2 = flash_chamfer_one_direction(Q, P, metric)
    return d1.mean() + d2.mean()


# ---------------------------------------------------------------------------
# Autograd wrapper for training
# ---------------------------------------------------------------------------

class _FlashChamferFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, P, Q):
        # P: [N, d], Q: [M, d]
        # Run both directions with argmin saving
        out_pq, argmin_pq = flash_chamfer_one_direction(P, Q, metric="l2_sq", save_argmin=True)
        out_qp, argmin_qp = flash_chamfer_one_direction(Q, P, metric="l2_sq", save_argmin=True)
        N = P.shape[0]
        M = Q.shape[0]
        ctx.save_for_backward(P, Q, argmin_pq, argmin_qp)
        ctx.N = N
        ctx.M = M
        # CD = (1/N) * sum(out_pq) + (1/M) * sum(out_qp)
        cd = out_pq.sum() / N + out_qp.sum() / M
        return cd

    @staticmethod
    def backward(ctx, grad_cd):
        P, Q, argmin_pq, argmin_qp = ctx.saved_tensors
        N = ctx.N
        M = ctx.M
        P_f = P.float()
        Q_f = Q.float()
        g = grad_cd.float()

        # Gradient from the P->Q term: (1/N) * sum_i ||P_i - Q[argmin_pq[i]]||²
        # d/dP_i = (2/N) * (P_i - Q[argmin_pq[i]])
        # d/dQ_j += -(2/N) * (P_i - Q_j) for each i where argmin_pq[i] == j
        Q_nearest_for_P = Q_f[argmin_pq.long()]  # [N, d]
        grad_P_from_pq = (2.0 / N) * (P_f - Q_nearest_for_P)  # [N, d]

        # Gradient from the Q->P term: (1/M) * sum_j ||Q_j - P[argmin_qp[j]]||²
        # d/dQ_j = (2/M) * (Q_j - P[argmin_qp[j]])
        # d/dP_i += -(2/M) * (Q_j - P_i) for each j where argmin_qp[j] == i
        P_nearest_for_Q = P_f[argmin_qp.long()]  # [M, d]
        grad_Q_from_qp = (2.0 / M) * (Q_f - P_nearest_for_Q)  # [M, d]

        # Cross-term gradients via scatter_add
        # P->Q term contribution to grad_Q: -(2/N) * (P_i - Q[argmin_pq[i]]) for each i
        grad_Q_from_pq = torch.zeros_like(Q_f)
        idx_pq = argmin_pq.long().unsqueeze(1).expand_as(P_f)  # [N, d]
        grad_Q_from_pq.scatter_add_(0, idx_pq, -grad_P_from_pq)

        # Q->P term contribution to grad_P: -(2/M) * (Q_j - P[argmin_qp[j]]) for each j
        grad_P_from_qp = torch.zeros_like(P_f)
        idx_qp = argmin_qp.long().unsqueeze(1).expand_as(Q_f)  # [M, d]
        grad_P_from_qp.scatter_add_(0, idx_qp, -grad_Q_from_qp)

        grad_P = g * (grad_P_from_pq + grad_P_from_qp)
        grad_Q = g * (grad_Q_from_qp + grad_Q_from_pq)

        return grad_P.to(P.dtype), grad_Q.to(Q.dtype)


def flash_chamfer_train(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Chamfer distance with autograd backward support.

    CD(P, Q) = (1/N) Σ_i min_j ||P_i - Q_j||² + (1/M) Σ_j min_i ||Q_j - P_i||²

    Args:
        P: [N, d] first point cloud (requires_grad supported)
        Q: [M, d] second point cloud (requires_grad supported)

    Returns:
        scalar Chamfer distance (differentiable)
    """
    return _FlashChamferFn.apply(P, Q)


def naive_chamfer_one_direction(P: torch.Tensor, Q: torch.Tensor,
                                 metric: str = "l2_sq") -> torch.Tensor:
    """Reference: per-point min distances (materializes full N×M matrix)."""
    if metric == "l1":
        dists = torch.cdist(P.float(), Q.float(), p=1)
    else:
        dists = torch.cdist(P.float(), Q.float(), p=2) ** 2
    return dists.min(dim=1).values


def naive_chamfer(P: torch.Tensor, Q: torch.Tensor,
                  metric: str = "l2_sq") -> torch.Tensor:
    """Reference: full Chamfer distance."""
    d1 = naive_chamfer_one_direction(P, Q, metric)
    d2 = naive_chamfer_one_direction(Q, P, metric)
    return d1.mean() + d2.mean()
