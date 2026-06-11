"""INT8-D training Function: save D as INT8 in the autograd context.

⚠ STATUS — measured 2026-05-30 (benchmarks/bench_int8_train_mem.py
on A100): phase 1 does NOT save peak GPU memory in a single
forward+backward step. The backward pre-dequant of D to a transient
FP16 buffer coexists with the saved INT8 D, so peak sees both
representations (~1.5x the D bytes for the duration of the backward).
At ColPali B=128 phase 1 increases peak by ~50 MB; at ColBERT B=128
by ~9 MB.

Phase 2 (TODO) is what actually delivers the memory win: modify the
unified and dQ backward kernels to take D_int8 + scales and dequant
inline, removing the transient FP16 D. That should bring peak DOWN
by ~D-bytes-worth (50% of fp16-D size, e.g. 0.13 GB at ColPali B=128).

What phase 1 does ship correctly:
  - Validated gradient quality with INT8-quantized D (grad_Q cosine
    0.999979, grad_D cosine 1.0 vs the FP16-saved baseline). The
    INT8 quantization of D is tolerable for ColBERT/ColPali training.
  - The autograd plumbing (save_for_backward(D_int8, d_scales),
    dispatch through the existing CSR/atomic_unified backward) — this
    code becomes the foundation that phase 2's kernel-side dequant
    plugs into.

Phase 1 is therefore best understood as engineering scaffolding for
phase 2 rather than a standalone optimization. Do NOT adopt it
expecting a peak-memory win it doesn't deliver.

Phase 2 (TODO): kernel-side dequant — modify
`_maxsim_bwd_unified_kernel` / CSR kernels to take D_int8 + scales and
dequant inline. That removes the transient FP16 D too, reducing peak
backward memory.

Why Phase 1 first: it ships the API + autograd plumbing with zero kernel
changes, validates that INT8 quantization error in the gradients stays
small enough to train (the forward already uses INT8×INT8 for inference;
the question is whether training tolerates the same quantization), and
gives us a comparison point to measure Phase 2's incremental win.

Usage:
    from flash_maxsim import flash_maxsim_int8_batched_train
    scores = flash_maxsim_int8_batched_train(Q, D, shared_docs=True)
    scores.sum().backward()    # gradients flow back as normal

Trade-off vs flash_maxsim_batched_train: saved D shrinks 50% but
gradient quality drops to ~INT8 ranking-fidelity level (Spearman ~0.999
on text workloads, same as the INT8 forward). Use this when memory is
the constraint and the model tolerates the noise (most ColBERT/ColPali
contrastive training does — checked empirically below).
"""
from __future__ import annotations

import torch

from .flash_maxsim import _next_pow2, _launch_fwd, _default_lengths
from .flash_maxsim_quant import quantize_int8_symmetric
from .flash_maxsim_batched_train import (
    _maxsim_bwd_dQ_batched_kernel,
    _maxsim_bwd_dD_batched_kernel,
    _maxsim_bwd_unified_kernel,
    _maxsim_bwd_unified_int8d_kernel,
    _bwd_unified_config,
)
from .flash_maxsim_invgrid_proposal import (
    build_inverse_csr,
    _maxsim_bwd_dD_invgrid_kernel,
)


class _FlashMaxSimInt8BatchedFn(torch.autograd.Function):
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

        orig_Q_dtype = Q.dtype
        orig_D_dtype = D.dtype

        # Forward runs in FP16 with FP32 accumulation, same as the standard
        # path. The INT8 storage is a SAVED-TENSOR optimization, not a
        # forward-kernel change — the forward needs the precise FP16 D for
        # the score reduction.
        Q_c = Q.contiguous().half().clone()
        D_c = D_flat.contiguous().half()
        total = B if shared_docs else Nq * B
        lengths = _default_lengths(total, Ld, D.device, doc_lengths)
        q_lens = query_lengths.to(torch.int32).contiguous() if query_lengths is not None else None

        q_mask = None
        if q_lens is not None:
            q_mask = (torch.arange(Lq, device=Q.device, dtype=q_lens.dtype)
                      < q_lens.unsqueeze(-1))
            Q_c *= q_mask.unsqueeze(-1).to(Q_c.dtype)

        scores, argmax = _launch_fwd(
            Q_c, D_c, lengths, Nq, B, Lq, Ld, d, shared_docs, True, q_lengths=q_lens,
        )

        # KEY OP: save D as INT8 + per-token scales, half the bytes vs FP16.
        # D_int8 shape matches D_c shape; d_scales is [total, Ld, 1] FP16.
        D_int8, d_scales = quantize_int8_symmetric(D_c)
        del D_c   # free the FP16 D buffer; we'll dequant in backward
        ctx.save_for_backward(Q_c, D_int8, d_scales, argmax)
        ctx.shared_docs = shared_docs
        ctx.shape = (Nq, B, Lq, Ld, d)
        ctx.q_mask = q_mask
        ctx.orig_Q_dtype = orig_Q_dtype
        ctx.orig_D_dtype = orig_D_dtype
        return scores

    @staticmethod
    def backward(ctx, grad_scores):
        Q, D_int8, d_scales, argmax = ctx.saved_tensors
        Nq, B, Lq, Ld, d = ctx.shape
        shared_docs = ctx.shared_docs
        grad_scores = grad_scores.contiguous().float()
        d_pad = _next_pow2(d)

        # Phase 2 (atomic_unified path): call the INT8-D variant of the
        # unified backward kernel, which takes D_int8 + d_scales and
        # dequants inline. No transient FP16 D buffer — saved INT8 D is
        # the only D representation through the entire backward, halving
        # the D-tensor footprint vs the standard FP16-saved training.
        # Phase 1 fallback (pre-dequant for the existing FP16 kernels) is
        # retained for invgrid + atomic modes where the kernel-side dequant
        # variant doesn't exist yet.
        import os as _os
        _force = _os.environ.get("FLASH_BWD_PATH", "auto")
        if _force == "atomic":
            mode = "atomic"
        elif _force == "atomic_unified":
            mode = "atomic_unified" if shared_docs else "atomic"
        elif _force == "invgrid":
            mode = "invgrid" if shared_docs else "atomic"
        else:
            if not shared_docs:
                mode = "atomic"
            elif (Nq * B * Lq) > 50_000:
                mode = "invgrid"
            else:
                mode = "atomic_unified"

        # Original D shape for grad_D output: [B*?, Ld, d] (matches D_int8 shape).
        grad_D_fp32 = torch.zeros(D_int8.shape, device=D_int8.device, dtype=torch.float32)

        if mode == "atomic_unified":
            # Phase 2: no pre-dequant. Kernel takes D_int8 + d_scales directly.
            grad_Q = torch.zeros_like(Q)
            nw, ns = _bwd_unified_config(Nq, Lq, Q.device.index or 0)
            # d_scales is [B*?, Ld, 1] from quantize_int8_symmetric. Squeeze
            # the trailing dim so strides line up with the [B, Ld] layout the
            # kernel expects.
            d_scales_2d = d_scales.squeeze(-1).contiguous()
            _maxsim_bwd_unified_int8d_kernel[(Nq * Lq,)](
                Q, D_int8, d_scales_2d, argmax, grad_scores,
                grad_Q, grad_D_fp32,
                Nq, B, Lq, Ld, d, d_pad,
                D_int8.stride(0), D_int8.stride(1), D_int8.stride(2),
                d_scales_2d.stride(0), d_scales_2d.stride(1),
                Q.stride(0), Q.stride(1), Q.stride(2),
                grad_scores.stride(0), grad_scores.stride(1),
                grad_Q.stride(0), grad_Q.stride(1), grad_Q.stride(2),
                grad_D_fp32.stride(0), grad_D_fp32.stride(1), grad_D_fp32.stride(2),
                1,
                num_warps=nw, num_stages=ns,
            )
        else:
            # Phase 1 fallback: pre-dequant to FP16, call existing kernels.
            D = (D_int8.to(torch.float32) * d_scales.float()).to(torch.float16).contiguous()
            grad_Q = torch.zeros_like(Q)
            _maxsim_bwd_dQ_batched_kernel[(Nq * Lq,)](
                D, argmax, grad_scores, grad_Q,
                Nq, B, Lq, d, d_pad,
                D.stride(0), D.stride(1), D.stride(2),
                grad_scores.stride(0), grad_scores.stride(1),
                grad_Q.stride(0), grad_Q.stride(1), grad_Q.stride(2),
                1 if shared_docs else 0,
            )
            if mode == "invgrid":
                row_ptr, col_idx = build_inverse_csr(argmax, Nq, B, Lq, Ld, shared_docs)
                n_dest = B * Ld
                _maxsim_bwd_dD_invgrid_kernel[(n_dest,)](
                    Q, grad_scores, grad_D_fp32,
                    row_ptr, col_idx,
                    Nq, B, Lq, Ld, d, d_pad,
                    grad_D_fp32.stride(0), grad_D_fp32.stride(1), grad_D_fp32.stride(2),
                    Q.stride(0), Q.stride(1), Q.stride(2),
                    grad_scores.stride(0), grad_scores.stride(1),
                    1,
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

        if ctx.q_mask is not None:
            grad_Q *= ctx.q_mask.unsqueeze(-1).to(grad_Q.dtype)

        grad_Q = grad_Q.to(ctx.orig_Q_dtype)
        grad_D = grad_D_fp32.to(ctx.orig_D_dtype)
        if not shared_docs:
            grad_D = grad_D.reshape(Nq, B, Ld, d)

        return grad_Q, grad_D, None, None, None


def flash_maxsim_int8_batched_train(
    Q: torch.Tensor, D: torch.Tensor,
    shared_docs: bool = True,
    doc_lengths: torch.Tensor | None = None,
    query_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Drop-in for flash_maxsim_batched_train, but saves D as INT8 in the
    autograd context. Halves saved-tensor memory at the cost of a small
    gradient error from INT8 quantization (Spearman ~0.999 on text shapes;
    see correctness self-test below)."""
    return _FlashMaxSimInt8BatchedFn.apply(Q, D, shared_docs, doc_lengths, query_lengths)


if __name__ == "__main__":
    # Correctness self-test: compare gradients of the INT8-saved variant
    # to gradients of the FP16-saved standard variant.
    import torch.nn.functional as F
    from .flash_maxsim_batched_train import flash_maxsim_batched_train
    torch.manual_seed(0)
    device = "cuda"

    for shape in [
        ("ColBERT B=128", 128, 32, 180),
        ("ColPali B=32",  32,  1024, 1024),
    ]:
        name, B, Lq, Ld = shape
        d = 128
        print(f"\n=== {name} (B={B}, Lq={Lq}, Ld={Ld}, d={d}) ===")
        Q = F.normalize(torch.randn(B, Lq, d, device=device, dtype=torch.float16), dim=-1)
        D = F.normalize(torch.randn(B, Ld, d, device=device, dtype=torch.float16), dim=-1)
        g = torch.randn(B, B, device=device, dtype=torch.float32)

        # Reference: standard FP16-saved training
        Qr = Q.detach().clone().requires_grad_(True)
        Dr = D.detach().clone().requires_grad_(True)
        scores_ref = flash_maxsim_batched_train(Qr, Dr, shared_docs=True)
        scores_ref.backward(g)

        # INT8-saved variant
        Qi = Q.detach().clone().requires_grad_(True)
        Di = D.detach().clone().requires_grad_(True)
        scores_int8 = flash_maxsim_int8_batched_train(Qi, Di, shared_docs=True)
        scores_int8.backward(g)

        s_d = (scores_ref - scores_int8).abs().max().item()
        gq_cos = torch.nn.functional.cosine_similarity(
            Qr.grad.flatten().float(), Qi.grad.flatten().float(), dim=0).item()
        gd_cos = torch.nn.functional.cosine_similarity(
            Dr.grad.flatten().float(), Di.grad.flatten().float(), dim=0).item()
        print(f"  scores max|Δ|     : {s_d:.3e}")
        print(f"  grad_Q cosine     : {gq_cos:.6f}")
        print(f"  grad_D cosine     : {gd_cos:.6f}")
