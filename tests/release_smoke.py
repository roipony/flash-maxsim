"""v0.2.0 release smoke test — verify every advertised kernel imports and runs."""
import sys, torch
import torch.nn.functional as F
sys.path.insert(0, '.')

device = 'cuda'
torch.manual_seed(0)
dtype = torch.float16

print(f"GPU: {torch.cuda.get_device_name(0)}")

# ---- 1. Every public symbol imports ----
from flash_maxsim import (
    flash_maxsim, flash_maxsim_batched, flash_maxsim_train,
    flash_maxsim_varlen, pack_pairs, flash_maxsim_packed, pack_docs,
    flash_maxsim_rerank, flash_maxsim_rerank_direct,
    flash_maxsim_int8, quantize_int8, dequantize_int8,
    flash_maxsim_int8x8, quantize_int8_symmetric, quantize_query_int8,
    flash_maxsim_int8_symmetric,
    flash_maxsim_topk, flash_maxsim_persistent,
    flash_maxsim_qreuse, flash_maxsim_splitk,
    maxsim_naive, maxsim_naive_fp32, maxsim_at_operator, maxsim_bmm,
    maxsim_chunked, maxsim_naive_batched, maxsim_naive_int8,
)
from flash_maxsim import __version__ as fm_version
print(f"flash_maxsim version: {fm_version}")
print("[OK] all imports")

# ---- 2. Core paths ----
Q = F.normalize(torch.randn(32, 128, dtype=dtype, device=device), dim=-1)
D = F.normalize(torch.randn(100, 300, 128, dtype=dtype, device=device), dim=-1)

s_flash = flash_maxsim(Q, D)
s_naive = maxsim_naive_fp32(Q, D)
err = (s_flash - s_naive).abs().max().item()
assert err < 0.1, f"flash_maxsim mismatch: {err}"
print(f"[OK] flash_maxsim  (max err vs matched-naive: {err:.2e})")

# ---- 3. Batched ----
Qb = F.normalize(torch.randn(5, 32, 128, dtype=dtype, device=device), dim=-1)
s_b = flash_maxsim_batched(Qb, D)
assert s_b.shape == (5, 100)
print(f"[OK] flash_maxsim_batched  {tuple(s_b.shape)}")

# ---- 4. Varlen ----
q_list = [F.normalize(torch.randn(l, 128, dtype=dtype, device=device), dim=-1)
          for l in (24, 32, 28)]
d_list = [F.normalize(torch.randn(l, 128, dtype=dtype, device=device), dim=-1)
          for l in (180, 250, 320)]
Q_pk, D_pk, cu_q, cu_d, mq, md = pack_pairs(q_list, d_list)
s_vl = flash_maxsim_varlen(Q_pk, D_pk, cu_q, cu_d, mq, md)
assert s_vl.shape == (3,)
print(f"[OK] flash_maxsim_varlen  {tuple(s_vl.shape)}")

# ---- 5. INT8 affine (legacy) ----
D_q, scales_aff, mins = quantize_int8(D)
s_i8 = flash_maxsim_int8(Q, D_q, scales_aff, mins)
err_i8 = (s_i8 - s_naive).abs().max().item() / max(s_naive.abs().max().item(), 1e-6)
print(f"[OK] flash_maxsim_int8  (rel err vs fp16-naive: {err_i8:.3e})")

# ---- 6. INT8 × INT8 (headline v0.2.0 feature) ----
D_i8, scales_sym = quantize_int8_symmetric(D)
s_i8x8 = flash_maxsim_int8x8(Q, D_i8, scales_sym)
err_i8x8 = (s_i8x8 - s_naive).abs().max().item() / max(s_naive.abs().max().item(), 1e-6)
print(f"[OK] flash_maxsim_int8x8  (rel err vs fp16-naive: {err_i8x8:.3e})  ← INT8 tensor cores")

# ---- 7. Training backward ----
Q_t = F.normalize(torch.randn(32, 128, dtype=dtype, device=device), dim=-1).requires_grad_(True)
D_t = F.normalize(torch.randn(50, 200, 128, dtype=dtype, device=device), dim=-1).requires_grad_(True)
scores = flash_maxsim_train(Q_t, D_t)
scores.sum().backward()
assert Q_t.grad is not None and D_t.grad is not None
print(f"[OK] flash_maxsim_train   (grad shapes Q:{tuple(Q_t.grad.shape)}, D:{tuple(D_t.grad.shape)})")

print("\n*** v0.2.0 smoke test: PASS ***")
