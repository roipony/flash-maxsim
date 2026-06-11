import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from flash_maxsim.flash_maxsim_batched_train import flash_maxsim_batched_train
from flash_maxsim.flash_maxsim_contrastive import colbert_contrastive_step


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def _make_inputs(Nq, B, Lq, Ld, d, dtype, seed):
    torch.manual_seed(seed)
    Q = F.normalize(torch.randn(Nq, Lq, d, device="cuda"), dim=-1).to(dtype)
    D = F.normalize(torch.randn(B, Ld, d, device="cuda"), dim=-1).to(dtype)
    return Q, D


def _reference(Q, D, labels):
    Qr = Q.detach().clone().requires_grad_(True)
    Dr = D.detach().clone().requires_grad_(True)
    scores = flash_maxsim_batched_train(Qr, Dr, shared_docs=True)
    loss = F.cross_entropy(scores, labels)
    loss.backward()
    return loss.detach(), Qr.grad.detach(), Dr.grad.detach()


def _fused(Q, D, labels):
    Qf = Q.detach().clone().requires_grad_(True)
    Df = D.detach().clone().requires_grad_(True)
    loss = colbert_contrastive_step(Qf, Df, labels)
    torch.cuda.synchronize()
    return loss.detach(), Qf.grad.detach(), Df.grad.detach()


def _check_shape(name, Nq, B, Lq, Ld, d, dtype, seed=0):
    labels = torch.arange(Nq, device="cuda", dtype=torch.int64)
    Q, D = _make_inputs(Nq, B, Lq, Ld, d, dtype, seed)

    ref_loss, ref_gQ, ref_gD = _reference(Q, D, labels)
    got_loss, got_gQ, got_gD = _fused(Q, D, labels)

    atol = 1e-2 if dtype is torch.bfloat16 else 8e-3
    rtol = 1e-2 if dtype is torch.bfloat16 else 8e-3
    torch.testing.assert_close(got_loss, ref_loss, atol=atol, rtol=rtol, msg=f"{name} loss")
    torch.testing.assert_close(got_gQ.float(), ref_gQ.float(), atol=atol, rtol=rtol, msg=f"{name} Q.grad")
    torch.testing.assert_close(got_gD.float(), ref_gD.float(), atol=atol, rtol=rtol, msg=f"{name} D.grad")
    print(f"{name}: OK loss={got_loss.item():.6f} dtype={str(dtype).replace('torch.', '')}")


def _check_accumulates(dtype):
    labels = torch.arange(4, device="cuda", dtype=torch.int64)
    Q, D = _make_inputs(4, 4, 8, 20, 16, dtype, 123)
    loss1, gQ1, gD1 = _fused(Q, D, labels)

    Q2 = Q.detach().clone().requires_grad_(True)
    D2 = D.detach().clone().requires_grad_(True)
    Q2.grad = torch.ones_like(Q2) * 0.25
    D2.grad = torch.ones_like(D2) * -0.5
    loss2 = colbert_contrastive_step(Q2, D2, labels)
    torch.cuda.synchronize()

    torch.testing.assert_close(loss2, loss1, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(Q2.grad.float(), gQ1.float() + 0.25, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(D2.grad.float(), gD1.float() - 0.5, atol=1e-2, rtol=1e-2)
    print(f"accumulate {str(dtype).replace('torch.', '')}: OK")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("test_contrastive_step.py requires CUDA")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    for dtype in (torch.float16, torch.bfloat16):
        _check_shape("tiny", 4, 4, 8, 20, 16, dtype, seed=1)
        _check_shape("medium", 32, 32, 32, 300, 128, dtype, seed=2)
        _check_accumulates(dtype)
    print("Done.")


if __name__ == "__main__":
    main()

