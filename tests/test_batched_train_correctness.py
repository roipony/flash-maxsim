"""Full correctness sweep for flash_maxsim_batched_train after reviewer fixes.

Verifies (against FP32 ground truth, TF32 off):
  (A) Invgrid path        — shared_docs, large shapes
  (B) Atomic path         — shared_docs, small shapes
  (C) Non-shared (KD)     — atomic path (invgrid disabled for non-shared)
  (D) Varlen correctness  — padded Q with random values + query_lengths must not leak
  (E) Dtype preservation  — pass FP32 inputs, verify FP32 grads out
  (F) Dtype preservation  — pass BF16 inputs, verify BF16 grads out
"""
import sys, torch
import torch.nn.functional as F

sys.path.insert(0, '/path/to/data')
from flash_maxsim.flash_maxsim_batched_train import flash_maxsim_batched_train

# TF32 off for true FP32 ground truth
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

device = 'cuda'


def fp32_reference(Q, D, shared_docs, q_lengths=None, d_lengths=None):
    """True FP32 MaxSim with explicit -inf masking for varlen."""
    Q32 = Q.float(); D32 = D.float()
    if shared_docs:
        S = torch.einsum('nqd,bld->nbql', Q32, D32)  # [Nq, B, Lq, Ld]
        Nq, B, Lq, Ld = S.shape
    else:
        S = torch.einsum('nqd,nbld->nbql', Q32, D32)
        Nq, B, Lq, Ld = S.shape

    if d_lengths is not None:
        dl = d_lengths.view(-1)
        valid_d = torch.arange(Ld, device=device)[None, :] < dl[:, None]
        if shared_docs:
            S.masked_fill_(~valid_d.view(1, B, 1, Ld), float('-inf'))
        else:
            S.masked_fill_(~valid_d.view(Nq, B, 1, Ld), float('-inf'))

    m = S.max(dim=-1).values   # [Nq, B, Lq]
    if q_lengths is not None:
        valid_q = torch.arange(Lq, device=device)[None, :] < q_lengths[:, None]  # [Nq, Lq]
        m.masked_fill_(~valid_q.unsqueeze(1), 0.0)
    return m.sum(dim=-1)       # [Nq, B]


def check(tag, Nq, B, Lq, Ld, d, *, shared_docs=True, pad_Q_random=False,
          q_lens=None, d_lens=None, dtype=torch.float16, seed=0):
    torch.manual_seed(seed)
    Qf = F.normalize(torch.randn(Nq, Lq, d, device=device), dim=-1)
    if shared_docs:
        Df = F.normalize(torch.randn(B, Ld, d, device=device), dim=-1)
    else:
        Df = F.normalize(torch.randn(Nq, B, Ld, d, device=device), dim=-1)

    # Simulate pylate-style padding: random values at padded Q positions
    if pad_Q_random and q_lens is not None:
        pad_mask = torch.arange(Lq, device=device)[None, :] >= q_lens[:, None]  # [Nq, Lq]
        Qf = torch.where(pad_mask.unsqueeze(-1), torch.randn_like(Qf), Qf)

    # Reference — mask BEFORE reference computation so reference is "what the user wants"
    Qr = Qf.clone().requires_grad_(True)
    Dr = Df.clone().requires_grad_(True)
    Sr = fp32_reference(Qr, Dr, shared_docs, q_lens, d_lens)
    Sr.diagonal().sum().backward() if shared_docs else Sr.sum().backward()
    gQ_ref = Qr.grad.detach()
    gD_ref = Dr.grad.detach()

    # Flash — in user's chosen dtype
    Qt = Qf.to(dtype).clone().requires_grad_(True)
    Dt = Df.to(dtype).clone().requires_grad_(True)
    S = flash_maxsim_batched_train(Qt, Dt, shared_docs=shared_docs,
                                    doc_lengths=d_lens, query_lengths=q_lens)
    (S.diagonal().sum() if shared_docs else S.sum()).backward()
    gQ = Qt.grad.detach()
    gD = Dt.grad.detach()

    # Dtype preservation
    assert gQ.dtype == dtype, f"{tag}: gQ dtype {gQ.dtype} != input dtype {dtype}"
    assert gD.dtype == dtype, f"{tag}: gD dtype {gD.dtype} != input dtype {dtype}"

    # Mask reference grads at padded positions where flash is expected to write 0
    if q_lens is not None:
        pm = torch.arange(Lq, device=device)[None, :] >= q_lens[:, None]
        gQ_ref = gQ_ref.clone()
        gQ_ref[pm] = 0

    gQ_cos = F.cosine_similarity(gQ_ref.float().flatten().unsqueeze(0),
                                  gQ.float().flatten().unsqueeze(0)).item()
    gD_cos = F.cosine_similarity(gD_ref.float().flatten().unsqueeze(0),
                                  gD.float().flatten().unsqueeze(0)).item()
    score_err = (Sr.float() - S.float()).abs().max().item()
    # Threshold accounts for FP-precision floor: FP16 noise grows with B,
    # BF16 has fewer mantissa bits than FP16 (we downcast internally).
    if dtype == torch.bfloat16:
        thresh = 0.99
    elif Nq * B >= 50_000:
        thresh = 0.998      # large-B FP16: more accumulation noise
    else:
        thresh = 0.999
    verdict = "OK" if (gQ_cos > thresh and gD_cos > thresh) else "FAIL"
    print(f"  {tag:<40s}  gQ_cos={gQ_cos:.5f}  gD_cos={gD_cos:.5f}  "
          f"score_err={score_err:.2e}  dtype={str(dtype).replace('torch.','')}  [{verdict}]")


print(f"GPU: {torch.cuda.get_device_name(0)}\n")

print("=" * 100)
print(" (A) Invgrid path — shared_docs, large (Nq*B*Lq > 50K)")
print("=" * 100)
check("ColBERT contrastive B=128", 128, 128, 32, 180, 128)
check("ColBERT contrastive B=256", 256, 256, 32, 180, 128)
check("ColPali contrastive B=32",   32,  32, 1024, 1024, 128)

print()
print("=" * 100)
print(" (B) Atomic fallback — shared_docs, tiny (Nq*B*Lq <= 50K)")
print("=" * 100)
check("tiny B=8",                    8,   8,  16,   24, 64)
check("ColBERT B=32",               32,  32,  32,  180, 128)

print()
print("=" * 100)
print(" (C) Non-shared (KD) — always atomic")
print("=" * 100)
check("KD B=16 K=8",                 8,  16,  32,  180, 128, shared_docs=False)
check("KD B=16 K=32",               32,  16,  32,  180, 128, shared_docs=False)

print()
print("=" * 100)
print(" (D) Varlen — padded Q with RANDOM garbage; mask must hide it")
print("=" * 100)
q_lens = torch.tensor([20, 28, 24, 30, 18, 32, 16, 22], dtype=torch.int32, device=device)
d_lens = torch.tensor([150, 100, 175, 80, 170, 120, 60, 175], dtype=torch.int32, device=device)
check("varlen small (random pad)",   8,   8,  32,  175, 128,
      pad_Q_random=True, q_lens=q_lens, d_lens=d_lens)

q2 = torch.randint(16, 33, (64,), dtype=torch.int32, device=device)
d2 = torch.randint(50, 181, (64,), dtype=torch.int32, device=device)
check("varlen medium (random pad)", 64,  64,  32,  180, 128,
      pad_Q_random=True, q_lens=q2, d_lens=d2)

print()
print("=" * 100)
print(" (E) Dtype preservation — FP32 inputs")
print("=" * 100)
check("FP32 input (contrastive)",  128, 128,  32,  180, 128, dtype=torch.float32)

print()
print("=" * 100)
print(" (F) Dtype preservation — BF16 inputs")
print("=" * 100)
check("BF16 input (contrastive)",  128, 128,  32,  180, 128, dtype=torch.bfloat16)

print("\nDone.")
