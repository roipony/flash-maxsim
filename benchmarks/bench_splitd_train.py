"""Verify split-d works through the full training path (autograd
Function + forward + backward) at fat embedding dim. The split-d
forward is already dispatched at d > 512; this test confirms the
backward kernels also handle fat d correctly via the standard
atomic_unified / CSR paths (they load one d-vector per program,
which stays register-friendly to ~d=4096).

Compares to naive autograd through PyTorch einsum+max+sum at d in
{512, 768, 1024, 2048}. Reports max |delta| of grad_Q and grad_D.
"""
import os, sys, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from flash_maxsim import flash_maxsim_batched_train

torch.set_float32_matmul_precision('high')
torch.manual_seed(0)


def naive_step(Q, D):
    """Reference: explicit einsum + max + sum through PyTorch autograd."""
    Qf, Df = Q.float(), D.float()
    s = torch.einsum('nqd,bld->nbql', Qf, Df).max(dim=3).values.sum(dim=2)
    return s


print(f"GPU: {torch.cuda.get_device_name()}")

for d in [512, 768, 1024, 2048]:
    # Realistic-ish contrastive training shape: small B + Lq=128 so the
    # naive einsum doesn't OOM at d=2048 (Lq=1024 would explode).
    B, Lq, Ld = 16, 128, 128
    print(f"\n=== d={d} (B={B}, Lq={Lq}, Ld={Ld}, FP16) ===", flush=True)
    Q = F.normalize(torch.randn(B, Lq, d, device='cuda', dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(B, Ld, d, device='cuda', dtype=torch.float16), dim=-1)

    # Reference path
    Qr = Q.detach().clone().requires_grad_()
    Dr = D.detach().clone().requires_grad_()
    g = torch.randn(B, B, device='cuda', dtype=torch.float32)
    scores_ref = naive_step(Qr, Dr)
    scores_ref.backward(g)

    # Flash path (split-d forward kicks in at d > 512; backward is
    # atomic_unified or CSR per the heuristic dispatcher)
    Qf = Q.detach().clone().requires_grad_()
    Df = D.detach().clone().requires_grad_()
    scores_flash = flash_maxsim_batched_train(Qf, Df, shared_docs=True)
    scores_flash.backward(g)

    s_match = torch.allclose(scores_ref, scores_flash, atol=1e-2, rtol=1e-2)
    gq_match = torch.allclose(Qr.grad, Qf.grad, atol=5e-2, rtol=5e-2)
    gd_match = torch.allclose(Dr.grad, Df.grad, atol=5e-2, rtol=5e-2)
    s_d = (scores_ref - scores_flash).abs().max().item()
    gq_d = (Qr.grad - Qf.grad).abs().max().item()
    gd_d = (Dr.grad - Df.grad).abs().max().item()
    print(f"  scores match : {s_match}  max|Δ|={s_d:.3e}", flush=True)
    print(f"  grad_Q match : {gq_match}  max|Δ|={gq_d:.3e}", flush=True)
    print(f"  grad_D match : {gd_match}  max|Δ|={gd_d:.3e}", flush=True)
    del Q, D, Qr, Dr, Qf, Df; torch.cuda.empty_cache()
