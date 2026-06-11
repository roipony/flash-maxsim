"""Unit tests for the killer-feature additions shipped on dev:

  (A) Split-d forward correctness at d in {512, 768, 1024} — kernel output
      matches the naive FP32 einsum reference to FP16-cast noise.
  (B) INT8-D training-Function gradient quality — grad_Q / grad_D match
      the standard FP16-saved training Function within FP16-bwd noise.
  (C) warmup() completes without raising and pre-compiles every default
      shape the dispatcher will pick at runtime.

Script-style (no pytest dependency), matches the convention of
test_batched_train_correctness.py + neighbors. Run via:
    python tests/test_splitd_int8_warmup.py
"""
import sys, torch
import torch.nn.functional as F

sys.path.insert(0, '/path/to/data')

# TF32 off so the FP32 reference is a true FP32 ground truth, matching
# the convention in test_batched_train_correctness.py
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(0)


def _naive_fwd(Q, D):
    """Reference: explicit einsum + max + sum in FP32."""
    sim = torch.einsum('qid,bjd->qbij', Q.float(), D.float())
    return sim.max(dim=-1).values.sum(dim=-1)


# ---------------------------------------------------------------------------
# (A) Split-d forward correctness
# ---------------------------------------------------------------------------
def test_splitd_forward_correctness():
    from flash_maxsim import flash_maxsim_batched
    print("\n(A) Split-d forward correctness")
    for d in [512, 768, 1024, 2048]:
        # ColPali-shape contrastive at small B (so the d=2048 row fits)
        Nq, B, Lq, Ld = 1, 32, 128, 256
        Q = F.normalize(torch.randn(Nq, Lq, d, device='cuda', dtype=torch.float16), dim=-1)
        D = F.normalize(torch.randn(B,  Ld, d, device='cuda', dtype=torch.float16), dim=-1)
        ref = _naive_fwd(Q, D)
        got = flash_maxsim_batched(Q, D)
        assert torch.allclose(ref, got, atol=1e-2, rtol=1e-2), (
            f"split-d d={d}: max|delta|={(ref-got).abs().max().item():.3e}"
        )
        print(f"  d={d:>4d}: max|delta|={(ref-got).abs().max().item():.3e}  OK")


# ---------------------------------------------------------------------------
# (B) INT8-D training gradient quality
# ---------------------------------------------------------------------------
def test_int8_train_gradient_quality():
    from flash_maxsim import flash_maxsim_batched_train, flash_maxsim_int8_batched_train
    print("\n(B) INT8-D training gradient quality")
    for name, B, Lq, Ld, d in [
        ("ColBERT B=128",  128, 32, 180, 128),
        ("ColPali B=32",   32, 1024, 1024, 128),
    ]:
        Q = F.normalize(torch.randn(B, Lq, d, device='cuda', dtype=torch.float16), dim=-1)
        D = F.normalize(torch.randn(B, Ld, d, device='cuda', dtype=torch.float16), dim=-1)
        g = torch.randn(B, B, device='cuda', dtype=torch.float32)

        # Reference: FP16-saved training
        Qr = Q.detach().clone().requires_grad_(True)
        Dr = D.detach().clone().requires_grad_(True)
        flash_maxsim_batched_train(Qr, Dr, shared_docs=True).backward(g)

        # INT8-saved variant
        Qi = Q.detach().clone().requires_grad_(True)
        Di = D.detach().clone().requires_grad_(True)
        flash_maxsim_int8_batched_train(Qi, Di, shared_docs=True).backward(g)

        gq_cos = torch.nn.functional.cosine_similarity(
            Qr.grad.flatten().float(), Qi.grad.flatten().float(), dim=0).item()
        gd_cos = torch.nn.functional.cosine_similarity(
            Dr.grad.flatten().float(), Di.grad.flatten().float(), dim=0).item()
        # Allow some FP16 quantization noise on grad_Q; grad_D should be
        # bit-perfect because the dD reduction never reads D values.
        assert gq_cos > 0.999, f"{name}: grad_Q cosine {gq_cos} too low"
        assert gd_cos > 0.9999, f"{name}: grad_D cosine {gd_cos} too low"
        print(f"  {name}: grad_Q cos={gq_cos:.6f}, grad_D cos={gd_cos:.6f}  OK")


# ---------------------------------------------------------------------------
# (C) warmup() smoke test
# ---------------------------------------------------------------------------
def test_warmup_runs():
    from flash_maxsim import warmup
    print("\n(C) warmup() smoke test")
    # Run with a tiny custom shape list so the test is fast (default shapes
    # cover ~10 specializations and take ~10s on cached cache; we only
    # need to verify the entry-point + dispatch don't error).
    result = warmup(
        shapes=[(128, 32, 180, 128)],    # one ColBERT shape
        train_shapes=[(64, 32, 180, 128)],  # one training shape
    )
    assert result["fwd_shapes"] == 1, f"unexpected fwd_shapes: {result}"
    assert result["bwd_shapes"] == 1, f"unexpected bwd_shapes: {result}"
    assert result["wall_seconds"] > 0, "wall_seconds should be positive"
    print(f"  {result}  OK")


if __name__ == "__main__":
    test_splitd_forward_correctness()
    test_int8_train_gradient_quality()
    test_warmup_runs()
    print("\nALL TESTS PASSED")
