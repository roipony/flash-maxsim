"""Pre-compile flash-maxsim kernels for common shapes at startup.

For production deployments where the JIT compile cost on first call is
unacceptable (latency-sensitive serving), call `warmup()` once at
server startup. It runs each kernel with a synthetic small input across
the standard ColBERT / ColPali / fat-embedding shapes and the per-arch
launch configs that the dispatcher will pick at runtime. Triton's
persistent disk cache (3.4+) means this only matters on first install
per machine — but for serving systems where cold-start latency
matters, warmup makes the first request indistinguishable from the
hundredth.

Typical cost on A100/H100: 5-15s of one-time compilation; each
subsequent call to a flash-maxsim API is JIT-free.

Usage:
    import flash_maxsim
    flash_maxsim.warmup()        # default: standard shapes, current arch

    # Custom shape list:
    flash_maxsim.warmup(shapes=[(B, Lq, Ld, d) for ...])
"""
from __future__ import annotations

import torch

# Default shape coverage: spans ColBERT-textual, ColBERT-long, ColPali, and
# fat-d variants (Jina v2, Voyage, NV-Embed). Covers what the
# dispatcher will see in production for the published encoder models.
DEFAULT_SHAPES = [
    # (B, Lq, Ld, d)  for forward
    (1024, 32,   180,  128),   # ColBERT textual
    (1024, 32,   1024, 128),   # ColBERT long-doc
    (512,  128,  1024, 128),   # ColBERT-long Lq=128
    (256,  512,  1024, 128),   # ColPali half-res
    (128,  1024, 1024, 128),   # ColPali full-res
    (128,  1024, 1024, 512),   # Jina v2 ColBERT d=512
    (128,  1024, 1024, 768),   # d=768 encoder
    (128,  1024, 1024, 1024),  # Voyage v2 d=1024
]

# Training shapes: small subset of the above (contrastive in-batch negatives
# only). Skips fat-d at large B since memory budget is tight.
DEFAULT_TRAIN_SHAPES = [
    (128, 32,   180,  128),   # ColBERT B=128 contrastive
    (64,  1024, 1024, 128),   # ColPali B=64 contrastive
]


def _make_q_d(B, Lq, Ld, d, device):
    Q = torch.randn(1, Lq, d, device=device, dtype=torch.float16)
    D = torch.randn(B, Ld, d, device=device, dtype=torch.float16)
    return Q, D


def warmup(shapes=None, train_shapes=None, device=None, verbose=False) -> dict:
    """Pre-compile all flash-maxsim kernels for the given shapes.

    Args:
        shapes: list of (B, Lq, Ld, d) for forward warmup. Default covers
            standard ColBERT/ColPali/fat-d encoders.
        train_shapes: list of (B, Lq, Ld, d) for training warmup
            (forward + backward). Default is a small subset.
        device: CUDA device. Defaults to current device.
        verbose: print per-shape progress.

    Returns:
        dict {"fwd_shapes": N, "bwd_shapes": M, "wall_seconds": float}.
    """
    import time
    from .flash_maxsim import flash_maxsim_batched
    from .flash_maxsim_batched_train import flash_maxsim_batched_train

    shapes = shapes if shapes is not None else DEFAULT_SHAPES
    train_shapes = train_shapes if train_shapes is not None else DEFAULT_TRAIN_SHAPES
    device = device or torch.device("cuda")

    t0 = time.time()
    fwd_count = 0
    for B, Lq, Ld, d in shapes:
        if verbose:
            print(f"  warmup fwd: B={B} Lq={Lq} Ld={Ld} d={d}", flush=True)
        Q, D = _make_q_d(B, Lq, Ld, d, device)
        try:
            _ = flash_maxsim_batched(Q, D)
            torch.cuda.synchronize()
            fwd_count += 1
        except Exception as e:
            if verbose:
                print(f"    skipped ({type(e).__name__}: {str(e)[:60]})", flush=True)
        del Q, D
    torch.cuda.empty_cache()

    bwd_count = 0
    for B, Lq, Ld, d in train_shapes:
        if verbose:
            print(f"  warmup bwd: B={B} Lq={Lq} Ld={Ld} d={d}", flush=True)
        Q = torch.randn(B, Lq, d, device=device, dtype=torch.float16, requires_grad=True)
        D = torch.randn(B, Ld, d, device=device, dtype=torch.float16, requires_grad=True)
        try:
            scores = flash_maxsim_batched_train(Q, D, shared_docs=True)
            scores.sum().backward()
            torch.cuda.synchronize()
            bwd_count += 1
        except Exception as e:
            if verbose:
                print(f"    skipped ({type(e).__name__}: {str(e)[:60]})", flush=True)
        del Q, D
    torch.cuda.empty_cache()

    wall = time.time() - t0
    return {"fwd_shapes": fwd_count, "bwd_shapes": bwd_count, "wall_seconds": round(wall, 2)}


if __name__ == "__main__":
    # CLI: python -m flash_maxsim.warmup
    print(f"GPU: {torch.cuda.get_device_name()}", flush=True)
    import os as _os
    cache_dir = _os.environ.get("TRITON_CACHE_DIR", "~/.triton/cache (Triton 3.4 default)")
    print(f"Triton cache dir: {cache_dir}", flush=True)
    result = warmup(verbose=True)
    print(f"\nDone. {result}", flush=True)
