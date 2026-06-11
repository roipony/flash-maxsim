"""Naive MaxSim implementations — all variants in one place.

Every naive MaxSim in the codebase should import from here.
No Q.float() or D.float() — inputs stay FP16, matched precision
via .float() on the sim matrix where needed.

Four compute methods:
  einsum:  torch.einsum('qd,bld->bql', Q, D)     — batched GEMM, simple
  at_op:   (Q @ D_flat.T).view(...)               — one big GEMM, fastest
  bmm:     torch.bmm(Q.expand(...), D.transpose(1,2))  — explicit batched
  matmul:  torch.matmul(Q, D[b].T) per doc       — loop, slowest

Two precision modes:
  fp16:    S stays FP16 → max → sum (fast, less precise)
  fp32:    S.float() → max → sum   (slower, matched precision with flash)
"""

import torch


# ---------------------------------------------------------------------------
# Core: einsum (most readable, standard baseline)
# ---------------------------------------------------------------------------

def maxsim_naive(Q: torch.Tensor, D: torch.Tensor, doc_lengths=None) -> torch.Tensor:
    """Reference MaxSim via einsum. Pure FP16, no .float() on inputs.

    Q: [Lq, d], D: [B, Ld, d] -> [B].
    """
    S = torch.einsum("qd,bld->bql", Q, D)
    if doc_lengths is not None:
        Ld = D.shape[1]
        pos = torch.arange(Ld, device=D.device)[None, None, :]
        S = S.masked_fill(pos >= doc_lengths[:, None, None], float("-inf"))
    return S.max(dim=2).values.sum(dim=1)


def maxsim_naive_fp32(Q: torch.Tensor, D: torch.Tensor, doc_lengths=None) -> torch.Tensor:
    """Matched-precision MaxSim: FP16 einsum, FP32 reduction.

    Same precision as flash_maxsim (FP16 tensor cores, FP32 accumulation).
    The .float() is on the sim matrix S, NOT on Q or D.
    """
    S = torch.einsum("qd,bld->bql", Q, D)
    if doc_lengths is not None:
        Ld = D.shape[1]
        pos = torch.arange(Ld, device=D.device)[None, None, :]
        S = S.masked_fill(pos >= doc_lengths[:, None, None], float("-inf"))
    return S.float().max(dim=2).values.sum(dim=1)


# ---------------------------------------------------------------------------
# @ operator: one big GEMM (fastest naive for large B)
# ---------------------------------------------------------------------------

def maxsim_at_operator(Q: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """Flat matmul: Q @ D_flat.T, one cuBLAS call. Fastest naive.

    Q: [Lq, d], D: [B, Ld, d] -> [B].
    """
    B, Ld, d = D.shape
    D_flat = D.reshape(B * Ld, d)           # free view
    S = (Q @ D_flat.T).view(Q.shape[0], B, Ld)  # [Lq, B, Ld]
    return S.max(dim=2).values.sum(dim=0)   # max over Ld, sum over Lq


# ---------------------------------------------------------------------------
# bmm: explicit batched matrix multiply
# ---------------------------------------------------------------------------

def maxsim_bmm(Q: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """Batched matmul via torch.bmm.

    Q: [Lq, d], D: [B, Ld, d] -> [B].
    """
    B = D.shape[0]
    Lq = Q.shape[0]
    S = torch.bmm(
        Q.unsqueeze(0).expand(B, -1, -1),  # [B, Lq, d]
        D.transpose(1, 2),                  # [B, d, Ld]
    )  # [B, Lq, Ld]
    return S.max(dim=2).values.sum(dim=1)


# ---------------------------------------------------------------------------
# Chunked: for large B to avoid OOM on the sim matrix
# ---------------------------------------------------------------------------

def maxsim_chunked(Q: torch.Tensor, D: torch.Tensor, chunk_size: int = 1000,
                   fp32_reduce: bool = False) -> torch.Tensor:
    """Chunked MaxSim — processes docs in mini-batches to avoid OOM.

    This is what production systems (vLLM, etc.) do.
    Q: [Lq, d], D: [B, Ld, d] -> [B].
    """
    B = D.shape[0]
    scores = []
    for i in range(0, B, chunk_size):
        S = torch.einsum("qd,bld->bql", Q, D[i:i+chunk_size])
        if fp32_reduce:
            S = S.float()
        scores.append(S.max(dim=2).values.sum(dim=1))
    return torch.cat(scores)


# ---------------------------------------------------------------------------
# Batched naive (multiple queries)
# ---------------------------------------------------------------------------

def maxsim_naive_batched(Q: torch.Tensor, D: torch.Tensor,
                         doc_lengths=None, shared_docs: bool = True) -> torch.Tensor:
    """Batched MaxSim via serial loop. Q: [Nq, Lq, d], D: [B, Ld, d] -> [Nq, B]."""
    Nq = Q.shape[0]
    scores = []
    for i in range(Nq):
        if shared_docs:
            s = maxsim_naive(Q[i], D, doc_lengths)
        else:
            s = maxsim_naive(Q[i], D[i], doc_lengths)
        scores.append(s)
    return torch.stack(scores)


# ---------------------------------------------------------------------------
# INT8 naive: dequant in HBM, then einsum
# ---------------------------------------------------------------------------

def maxsim_naive_int8(Q: torch.Tensor, D_int8: torch.Tensor,
                      d_scales: torch.Tensor) -> torch.Tensor:
    """Naive INT8: dequant D to FP16 in HBM, then standard einsum.

    This is what you'd do without flash-maxsim for an INT8 index.
    Q: [Lq, d], D_int8: [B, Ld, d] int8, d_scales: [B, Ld, 1] or [B, Ld] -> [B].
    """
    if d_scales.dim() == 2:
        d_scales = d_scales.unsqueeze(-1)
    D_fp16 = (D_int8.float() * d_scales.float()).half()
    S = torch.einsum("qd,bld->bql", Q, D_fp16)
    return S.max(dim=2).values.sum(dim=1)
