"""Benchmark: 4-bit fused kernel + TurboQuant rotation.

Tests:
1. Quantization quality: 4-bit uniform vs 4-bit+rotation vs INT8
2. Speed: fused 4-bit kernel vs flash FP16 vs flash INT8×INT8
3. Memory: 4x compression story

Run: python tests/bench_4bit.py
"""

import torch
import torch.nn.functional as F
import sys, time
sys.path.insert(0, '.')

from flash_maxsim import flash_maxsim, maxsim_naive
from flash_maxsim.flash_maxsim_quant import (
    flash_maxsim_int8x8, quantize_int8_symmetric,
    flash_maxsim_4bit, quantize_4bit_symmetric,
    make_rotation_matrix, rotate_embeddings,
)

device = 'cuda'
D_DIM, DTYPE = 128, torch.float16
Lq, Ld = 1024, 1024

print(f'GPU: {torch.cuda.get_device_name(0)}')
print()


def bench(fn, warmup=10, runs=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def fp32_ref(Q, D):
    return torch.einsum('qd,bld->bql', Q.float(), D.float()).max(dim=2).values.sum(dim=1)


def metrics(out, ref, B):
    err = (ref - out).abs()
    K = min(20, B)
    ref_topk = set(ref.argsort(descending=True)[:K].tolist())
    out_topk = set(out.argsort(descending=True)[:K].tolist())
    ovlp = len(ref_topk & out_topk) / K
    ref_rank = ref.argsort(descending=True).argsort().float()
    out_rank = out.argsort(descending=True).argsort().float()
    sp = (1 - 6 * ((ref_rank - out_rank)**2).sum() / (B * (B**2 - 1))).item()
    return err.mean().item(), err.max().item(), ovlp, sp


# =========================================================================
# 1. Precision: rotation effect on 4-bit
# =========================================================================

print('=' * 80)
print(' Precision: does rotation help 4-bit quantization?')
print('=' * 80)
print()

torch.manual_seed(42)
B = 1000
Q = F.normalize(torch.randn(Lq, D_DIM, dtype=DTYPE, device=device), dim=-1)
D = F.normalize(torch.randn(B, Ld, D_DIM, dtype=DTYPE, device=device), dim=-1)
R = make_rotation_matrix(D_DIM, device=device)

ref = fp32_ref(Q, D)

# All methods
results = []

# FP16 naive
out = maxsim_naive(Q, D)
m_err, mx_err, ovlp, sp = metrics(out, ref, B)
results.append(('FP16 naive', m_err, ovlp, sp))

# Flash FP16
out = flash_maxsim(Q, D)
m_err, mx_err, ovlp, sp = metrics(out, ref, B)
results.append(('Flash FP16', m_err, ovlp, sp))

# INT8×INT8
D_int8, d_sc8 = quantize_int8_symmetric(D)
out = flash_maxsim_int8x8(Q, D_int8, d_sc8)
m_err, mx_err, ovlp, sp = metrics(out, ref, B)
results.append(('Flash INT8×INT8', m_err, ovlp, sp))

# 4-bit NO rotation
D_4bit, d_sc4 = quantize_4bit_symmetric(D)
out = flash_maxsim_4bit(Q, D_4bit, d_sc4)
m_err, mx_err, ovlp, sp = metrics(out, ref, B)
results.append(('Flash 4-bit (no rot)', m_err, ovlp, sp))

# 4-bit WITH rotation
D_rot = rotate_embeddings(D, R)
D_4bit_r, d_sc4_r = quantize_4bit_symmetric(D_rot)
out = flash_maxsim_4bit(Q, D_4bit_r, d_sc4_r, R=R)
m_err, mx_err, ovlp, sp = metrics(out, ref, B)
results.append(('Flash 4-bit + rotation', m_err, ovlp, sp))

# INT8 WITH rotation (for comparison)
D_int8_r, d_sc8_r = quantize_int8_symmetric(D_rot)
d_sc8_r_2d = d_sc8_r.squeeze(-1) if d_sc8_r.dim() == 3 else d_sc8_r
Q_rot = rotate_embeddings(Q, R)
out = flash_maxsim_int8x8(Q_rot, D_int8_r, d_sc8_r_2d)
m_err, mx_err, ovlp, sp = metrics(out, ref, B)
results.append(('Flash INT8 + rotation', m_err, ovlp, sp))

print(f'{"Method":<25}  {"mean_err":>10}  {"top-20":>7}  {"spearman":>9}')
print('-' * 60)
for name, m_err, ovlp, sp in results:
    print(f'{name:<25}  {m_err:>10.6f}  {ovlp:>6.0%}  {sp:>9.6f}')

del D_int8, D_4bit, D_4bit_r, D_int8_r, D_rot
torch.cuda.empty_cache()


# =========================================================================
# 2. Speed comparison
# =========================================================================

print()
print('=' * 80)
print(' Speed: fused 4-bit kernel vs FP16 vs INT8')
print('=' * 80)
print()

print(f'{"B":>7}  {"naive":>8}  {"flash":>8}  {"int8x8":>8}  {"4bit":>8}  {"4bit+R":>8}  | {"f_sp":>6}  {"i8_sp":>6}  {"4b_sp":>6}  {"4bR_sp":>7}')
print('-' * 90)

for B in [100, 500, 1000, 5000]:
    Q = F.normalize(torch.randn(Lq, D_DIM, dtype=DTYPE, device=device), dim=-1)
    D = F.normalize(torch.randn(B, Ld, D_DIM, dtype=DTYPE, device=device), dim=-1)
    R = make_rotation_matrix(D_DIM, device=device)

    # Pre-quantize all (index time, not timed)
    D_int8, d_sc8 = quantize_int8_symmetric(D)
    D_4bit, d_sc4 = quantize_4bit_symmetric(D)
    D_rot = rotate_embeddings(D, R)
    D_4bit_r, d_sc4_r = quantize_4bit_symmetric(D_rot)

    naive_ms = bench(lambda: maxsim_naive(Q, D))
    flash_ms = bench(lambda: flash_maxsim(Q, D))
    int8_ms  = bench(lambda: flash_maxsim_int8x8(Q, D_int8, d_sc8))
    bit4_ms  = bench(lambda: flash_maxsim_4bit(Q, D_4bit, d_sc4))
    bit4r_ms = bench(lambda: flash_maxsim_4bit(Q, D_4bit_r, d_sc4_r, R=R))

    print(f'{B:>7,}  {naive_ms:>7.2f}  {flash_ms:>7.2f}  {int8_ms:>7.2f}  {bit4_ms:>7.2f}  {bit4r_ms:>7.2f}  | '
          f'{naive_ms/flash_ms:>5.1f}x  {naive_ms/int8_ms:>5.1f}x  {naive_ms/bit4_ms:>5.1f}x  {naive_ms/bit4r_ms:>6.1f}x')

    del D, D_int8, D_4bit, D_rot, D_4bit_r
    torch.cuda.empty_cache()


# =========================================================================
# 3. Memory
# =========================================================================

print()
print('=' * 80)
print(' Memory: docs per GPU budget')
print('=' * 80)
print()
d = D_DIM
print(f'Per doc (Ld={Ld}, d={d}):  FP16={Ld*d*2//1024}KB  INT8={Ld*d//1024}KB  4-bit={Ld*d//2//1024}KB')
print()
for gb in [20, 40]:
    n_fp16 = int(gb * 1e9 / (Ld * d * 2))
    n_int8 = int(gb * 1e9 / (Ld * d * 1))
    n_4bit = int(gb * 1e9 / (Ld * d // 2))
    print(f'  {gb}GB budget:  FP16={n_fp16:>9,}  INT8={n_int8:>9,} (2x)  4-bit={n_4bit:>9,} (4x)')

print()
print('Done.')
