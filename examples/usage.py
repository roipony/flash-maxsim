"""Flash-MaxSim usage examples."""

import torch
import torch.nn.functional as F
from flash_maxsim import (
    flash_maxsim, flash_maxsim_batched, flash_maxsim_train,
    flash_maxsim_int8, quantize_int8,
)

device = "cuda"

# --- Basic: single query vs document batch ---
Q = F.normalize(torch.randn(32, 128, device=device, dtype=torch.float16), dim=-1)
D = F.normalize(torch.randn(1000, 256, 128, device=device, dtype=torch.float16), dim=-1)
scores = flash_maxsim(Q, D)  # [1000]
print(f"Basic:   top-5 scores = {scores.topk(5).values.tolist()}")

# --- Variable-length documents ---
lengths = torch.randint(64, 256, (1000,), device=device, dtype=torch.int32)
scores_vl = flash_maxsim(Q, D, doc_lengths=lengths)  # [1000]
print(f"VarLen:  top-5 scores = {scores_vl.topk(5).values.tolist()}")

# --- Batched: multiple queries ---
Q_batch = F.normalize(torch.randn(10, 32, 128, device=device, dtype=torch.float16), dim=-1)
scores_b = flash_maxsim_batched(Q_batch, D, shared_docs=True)  # [10, 1000]
print(f"Batched: shape = {scores_b.shape}")

# --- INT8 quantized ---
D_q, scales, mins = quantize_int8(D)
scores_q = flash_maxsim_int8(Q, D_q, scales, mins)  # [1000]
print(f"INT8:    max |diff| vs FP16 = {(scores - scores_q).abs().max():.2f}")

# --- Training with autograd ---
Q_train = F.normalize(torch.randn(32, 128, device=device, dtype=torch.float16), dim=-1).requires_grad_(True)
D_train = F.normalize(torch.randn(50, 128, 128, device=device, dtype=torch.float16), dim=-1)
loss = flash_maxsim_train(Q_train, D_train).sum()
loss.backward()
print(f"Train:   Q.grad norm = {Q_train.grad.norm():.4f}")
