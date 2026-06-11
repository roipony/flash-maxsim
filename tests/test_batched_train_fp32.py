"""Correctness: flash_maxsim_batched_train vs FP32 ground truth.

Eliminates FP16 argmax-tie noise by upcasting Q, D to FP32 for the reference
einsum path and disabling TF32 on the matmul. The reference represents the
'true' gradients; we measure how far FP16-input flash deviates from it.
"""
import sys, torch
import torch.nn.functional as F

sys.path.insert(0, '/path/to/data')
from flash_maxsim.flash_maxsim_batched_train import flash_maxsim_batched_train

# Force true FP32 matmul (no TF32) for ground truth
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

device = 'cuda'
DTYPE = torch.float16

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"TF32 allowed: {torch.backends.cuda.matmul.allow_tf32}\n")


def reference_fp32(Q16, D16):
    """True-FP32 ColBERT score (upcast, pure FP32 matmul). Nq == B."""
    Q32 = Q16.float()
    D32 = D16.float()
    # reproduce colbert_scores shape: Q[B, Lq, d], D[B, Ld, d] → scores[B, B]
    S = torch.einsum('ash,bth->abst', Q32, D32)     # [B, B, Lq, Ld]  FP32
    return S.max(dim=-1).values.sum(dim=-1)         # [B, B]


def check(B, Lq, Ld, d, seed):
    torch.manual_seed(seed)
    Q0 = F.normalize(torch.randn(B, Lq, d, dtype=DTYPE, device=device), dim=-1)
    D0 = F.normalize(torch.randn(B, Ld, d, dtype=DTYPE, device=device), dim=-1)

    # --- FP32 reference ---
    Qr = Q0.clone().requires_grad_(True); Dr = D0.clone().requires_grad_(True)
    sr = reference_fp32(Qr, Dr)
    sr.diagonal().sum().backward()
    gq_ref = Qr.grad.float().clone()
    gd_ref = Dr.grad.float().clone()

    # --- flash ---
    Qf = Q0.clone().requires_grad_(True); Df = D0.clone().requires_grad_(True)
    sf = flash_maxsim_batched_train(Qf, Df, shared_docs=True)
    sf.diagonal().sum().backward()
    gq_flash = Qf.grad.float()
    gd_flash = Df.grad.float()

    score_err = (sr - sf).abs().max().item() / max(sr.abs().max().item(), 1e-9)
    gq_err = (gq_ref - gq_flash).abs().max().item() / max(gq_ref.abs().max().item(), 1e-9)
    gd_err = (gd_ref - gd_flash).abs().max().item() / max(gd_ref.abs().max().item(), 1e-9)
    # Cosine sim of gradient vectors
    gq_cos = F.cosine_similarity(gq_ref.flatten().unsqueeze(0), gq_flash.flatten().unsqueeze(0)).item()
    gd_cos = F.cosine_similarity(gd_ref.flatten().unsqueeze(0), gd_flash.flatten().unsqueeze(0)).item()
    return score_err, gq_err, gd_err, gq_cos, gd_cos


print(f"  {'B':>4}  {'Lq':>4}  {'Ld':>5}  {'d':>4}  {'score_rel':>10}  {'gQ_rel':>9}  {'gD_rel':>9}  "
      f"{'gQ_cos':>8}  {'gD_cos':>8}")
print("  " + "-" * 82)
CASES = [
    (8,   32,  180, 128),
    (4,   64,  256, 128),
    (16,  32,  300, 128),
    (32,  32,  180, 128),
    (8,  128, 1024, 128),
    (4, 1024, 1024, 128),
    (16, 512,  512, 128),
]
for (B, Lq, Ld, d) in CASES:
    for seed in (0, 42):
        s, gq, gd, gqc, gdc = check(B, Lq, Ld, d, seed)
        print(f"  {B:>4}  {Lq:>4}  {Ld:>5}  {d:>4}  {s:>10.2e}  {gq:>9.2e}  {gd:>9.2e}  "
              f"{gqc:>8.5f}  {gdc:>8.5f}   seed={seed}")

print("""
Interpretation:
  score_rel / gQ_rel / gD_rel = max |ref - flash| / max |ref|  (relative element-wise)
  gQ_cos / gD_cos            = cosine similarity of gradient vectors (1.0 = perfect)
For FP16 inputs, element-wise relative error is bounded by FP16 precision
(~1e-3). Cosine similarity near 1.0 means the gradient *direction* matches.
""")
