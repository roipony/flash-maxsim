"""Fused Triton kernels for ColBERT/ColPali MaxSim scoring."""

import torch
import triton
import triton.language as tl


def _next_pow2(x):
    return 1 << (x - 1).bit_length()

# ---------------------------------------------------------------------------
# Hardware-aware autotune configs
# ---------------------------------------------------------------------------

def _detect_gpu():
    if not torch.cuda.is_available():
        return "generic"
    name = torch.cuda.get_device_name().lower()
    if "h100" in name or "h200" in name:
        return "hopper"
    if "a100" in name:
        return "a100"
    return "generic"


def _get_configs(gpu=None):
    gpu = gpu or _detect_gpu()
    # Small blocks for large embedding dims (d=512, 1024, 2048)
    large_d_configs = [
        triton.Config({"BLOCK_Q": 16, "BLOCK_D": 16}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_Q": 16, "BLOCK_D": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 16}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 32}, num_warps=4, num_stages=2),
    ]
    if gpu == "hopper":
        return large_d_configs + [
            triton.Config({"BLOCK_Q": 32, "BLOCK_D": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_Q": 32, "BLOCK_D": 128}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_Q": 64, "BLOCK_D": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_Q": 64, "BLOCK_D": 128}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_Q": 128, "BLOCK_D": 64}, num_warps=8, num_stages=2),
        ]
    if gpu == "a100":
        return large_d_configs + [
            triton.Config({"BLOCK_Q": 32, "BLOCK_D": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_Q": 32, "BLOCK_D": 128}, num_warps=8, num_stages=1),
            triton.Config({"BLOCK_Q": 64, "BLOCK_D": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_Q": 64, "BLOCK_D": 128}, num_warps=8, num_stages=1),
            triton.Config({"BLOCK_Q": 128, "BLOCK_D": 64}, num_warps=8, num_stages=1),
        ]
    return large_d_configs + [
        triton.Config({"BLOCK_Q": 16, "BLOCK_D": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_Q": 64, "BLOCK_D": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 64, "BLOCK_D": 128}, num_warps=8, num_stages=2),
    ]


def _prune_configs(configs, named_args, **kwargs):
    Lq = named_args.get("Lq", 32)
    d = named_args.get("d", 128)
    pruned = []
    for cfg in configs:
        bq, bd = cfg.kwargs["BLOCK_Q"], cfg.kwargs["BLOCK_D"]
        if bq > Lq * 2:
            continue
        if (bq * d + bd * d) * 2 + bq * bd * 4 > 200_000:
            continue
        pruned.append(cfg)
    return pruned or configs[:4]


_CONFIGS = _get_configs()

# ---------------------------------------------------------------------------
# Unified forward kernel (single-query & batched)
# ---------------------------------------------------------------------------

@triton.autotune(configs=_CONFIGS, key=["Lq", "Ld", "d_pad"],
                 prune_configs_by={"early_config_prune": _prune_configs})
@triton.jit
def _maxsim_fwd_kernel(
    Q_ptr, D_ptr, lengths_ptr, scores_ptr, argmax_ptr,
    Nq: tl.constexpr, B: tl.constexpr,
    Lq: tl.constexpr, Ld: tl.constexpr, d: tl.constexpr, d_pad: tl.constexpr,
    stride_q_n, stride_q_l, stride_q_d,
    stride_d_b, stride_d_l, stride_d_d,
    stride_s_n, stride_s_b,
    shared_docs: tl.constexpr,
    save_argmax: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    q_idx = pid // B
    doc_idx = pid % B
    if q_idx >= Nq:
        return

    d_batch = doc_idx if shared_docs else q_idx * B + doc_idx
    doc_len = tl.load(lengths_ptr + d_batch).to(tl.int32)

    k_off = tl.arange(0, d_pad)
    k_mask = k_off < d
    score_acc = tl.zeros([], dtype=tl.float32)

    for q_start in tl.static_range(0, Lq, BLOCK_Q):
        q_off = q_start + tl.arange(0, BLOCK_Q)
        q_valid = q_off < Lq

        Q_block = tl.load(
            Q_ptr + q_idx * stride_q_n + q_off[:, None] * stride_q_l + k_off[None, :] * stride_q_d,
            mask=q_valid[:, None] & k_mask[None, :], other=0.0,
        ).to(tl.float16)

        m = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
        m_idx = tl.full([BLOCK_Q], 0, dtype=tl.int32)

        for d_start in range(0, Ld, BLOCK_D):
            d_off = d_start + tl.arange(0, BLOCK_D)
            d_valid = d_off < doc_len

            D_block = tl.load(
                D_ptr + d_batch * stride_d_b + d_off[:, None] * stride_d_l + k_off[None, :] * stride_d_d,
                mask=d_valid[:, None] & k_mask[None, :], other=0.0,
            ).to(tl.float16)

            S = tl.dot(Q_block, tl.trans(D_block))
            S = tl.where(d_valid[None, :], S, float("-inf"))

            tile_max = tl.max(S, axis=1)
            if save_argmax:
                tile_argmax = tl.argmax(S, axis=1).to(tl.int32) + d_start
                update = tile_max > m
                m_idx = tl.where(update, tile_argmax, m_idx)
            m = tl.maximum(m, tile_max)

        m = tl.where(q_valid, m, 0.0)
        score_acc += tl.sum(m)

        if save_argmax:
            tl.store(
                argmax_ptr + pid * Lq + q_off, m_idx, mask=q_valid,
            )

    tl.store(scores_ptr + q_idx * stride_s_n + doc_idx * stride_s_b, score_acc)


# ---------------------------------------------------------------------------
# Backward kernels for training
# ---------------------------------------------------------------------------

@triton.jit
def _maxsim_bwd_dQ_kernel(
    D_ptr, argmax_ptr, grad_s_ptr, grad_Q_ptr,
    B: tl.constexpr, Lq: tl.constexpr, d: tl.constexpr, d_pad: tl.constexpr,
    stride_d_b, stride_d_l, stride_d_d,
):
    q_idx = tl.program_id(0)
    k = tl.arange(0, d_pad)
    km = k < d
    acc = tl.zeros([d_pad], dtype=tl.float32)
    for b in range(B):
        gs = tl.load(grad_s_ptr + b).to(tl.float32)
        j = tl.load(argmax_ptr + b * Lq + q_idx)
        v = tl.load(D_ptr + b * stride_d_b + j * stride_d_l + k * stride_d_d, mask=km, other=0.0).to(tl.float32)
        acc += gs * v
    tl.store(grad_Q_ptr + q_idx * d + k, acc.to(tl.float16), mask=km)


@triton.jit
def _maxsim_bwd_dD_kernel(
    Q_ptr, argmax_ptr, grad_s_ptr, grad_D_ptr,
    Lq: tl.constexpr, Ld: tl.constexpr, d: tl.constexpr, d_pad: tl.constexpr,
    stride_d_b, stride_d_l, stride_d_d,
    stride_q_l, stride_q_d,
):
    doc_id = tl.program_id(0)
    k = tl.arange(0, d_pad)
    km = k < d
    gs = tl.load(grad_s_ptr + doc_id).to(tl.float32)
    for q_idx in range(Lq):
        j = tl.load(argmax_ptr + doc_id * Lq + q_idx)
        qv = tl.load(Q_ptr + q_idx * stride_q_l + k * stride_q_d, mask=km, other=0.0).to(tl.float32)
        tl.atomic_add(
            grad_D_ptr + doc_id * stride_d_b + j * stride_d_l + k * stride_d_d,
            (gs * qv).to(tl.float16), mask=km,
        )


# ---------------------------------------------------------------------------
# Helper: launch the unified kernel
# ---------------------------------------------------------------------------

def _launch_fwd(Q, D, lengths, Nq, B, Lq, Ld, d, shared_docs, save_argmax):
    d_pad = _next_pow2(d)
    scores = torch.empty(Nq, B, device=Q.device, dtype=torch.float32)
    argmax = torch.empty(Nq * B, Lq, device=Q.device, dtype=torch.int32) if save_argmax else Q  # dummy
    _maxsim_fwd_kernel[(Nq * B,)](
        Q, D, lengths, scores, argmax,
        Nq, B, Lq, Ld, d, d_pad,
        Q.stride(-3) if Q.dim() == 3 else 0, Q.stride(-2), Q.stride(-1),
        D.stride(0), D.stride(1), D.stride(2),
        scores.stride(0), scores.stride(1),
        1 if shared_docs else 0,
        1 if save_argmax else 0,
    )
    return scores, argmax


def _default_lengths(B, Ld, device, doc_lengths=None):
    if doc_lengths is not None:
        return doc_lengths.to(torch.int32).contiguous()
    return torch.full((B,), Ld, device=device, dtype=torch.int32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def flash_maxsim(Q: torch.Tensor, D: torch.Tensor, doc_lengths=None) -> torch.Tensor:
    """Compute MaxSim scores. Q: [Lq, d], D: [B, Ld, d] -> [B]."""
    assert Q.dim() == 2 and D.dim() == 3 and Q.shape[1] == D.shape[2]
    Lq, d = Q.shape
    B, Ld, _ = D.shape
    Q2 = Q.unsqueeze(0).contiguous().half()
    D2 = D.contiguous().half()
    lengths = _default_lengths(B, Ld, D.device, doc_lengths)
    scores, _ = _launch_fwd(Q2, D2, lengths, 1, B, Lq, Ld, d, True, False)
    return scores.squeeze(0)


def flash_maxsim_batched(Q: torch.Tensor, D: torch.Tensor, doc_lengths=None,
                         shared_docs: bool = True) -> torch.Tensor:
    """Batched MaxSim. Q: [Nq, Lq, d], D: [B, Ld, d] -> [Nq, B]."""
    assert Q.dim() == 3 and Q.shape[2] == D.shape[-1]
    Nq, Lq, d = Q.shape
    if shared_docs:
        assert D.dim() == 3
        B, Ld, _ = D.shape
    else:
        assert D.dim() == 4 and D.shape[0] == Nq
        _, B, Ld, _ = D.shape
        D = D.reshape(Nq * B, Ld, d)
    Q2 = Q.contiguous().half()
    D2 = D.contiguous().half()
    total = B if shared_docs else Nq * B
    lengths = _default_lengths(total, Ld, D.device, doc_lengths)
    scores, _ = _launch_fwd(Q2, D2, lengths, Nq, B, Lq, Ld, d, shared_docs, False)
    return scores


class _FlashMaxSimFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, D):
        Lq, d = Q.shape
        B, Ld, _ = D.shape
        Q2 = Q.unsqueeze(0).contiguous().half()
        D2 = D.contiguous().half()
        lengths = torch.full((B,), Ld, device=D.device, dtype=torch.int32)
        scores, argmax = _launch_fwd(Q2, D2, lengths, 1, B, Lq, Ld, d, True, True)
        ctx.save_for_backward(Q2.squeeze(0), D2, argmax)
        return scores.squeeze(0)

    @staticmethod
    def backward(ctx, grad_scores):
        Q, D, argmax = ctx.saved_tensors
        Lq, d = Q.shape
        B, Ld, _ = D.shape
        grad_scores = grad_scores.contiguous().float()

        grad_Q = torch.zeros_like(Q)
        d_pad = _next_pow2(d)
        _maxsim_bwd_dQ_kernel[(Lq,)](
            D, argmax, grad_scores, grad_Q,
            B, Lq, d, d_pad,
            D.stride(0), D.stride(1), D.stride(2),
        )
        grad_D = torch.zeros_like(D)
        _maxsim_bwd_dD_kernel[(B,)](
            Q, argmax, grad_scores, grad_D,
            Lq, Ld, d, d_pad,
            D.stride(0), D.stride(1), D.stride(2),
            Q.stride(0), Q.stride(1),
        )
        return grad_Q, grad_D


def flash_maxsim_train(Q: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """MaxSim with autograd backward. Q: [Lq, d], D: [B, Ld, d] -> [B]."""
    return _FlashMaxSimFn.apply(Q, D)


def maxsim_naive(Q: torch.Tensor, D: torch.Tensor, doc_lengths=None) -> torch.Tensor:
    """Reference PyTorch MaxSim. Q: [Lq, d], D: [B, Ld, d] -> [B]."""
    S = torch.einsum("qd,bld->bql", Q.float(), D.float())
    if doc_lengths is not None:
        Ld = D.shape[1]
        pos = torch.arange(Ld, device=D.device)[None, None, :]
        S = S.masked_fill(pos >= doc_lengths[:, None, None], float("-inf"))
    return S.max(dim=2).values.sum(dim=1)
