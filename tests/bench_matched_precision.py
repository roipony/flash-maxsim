"""Benchmark: Flash vs naive at MATCHED precision.

Fair comparison: both use FP16 tensor cores for matmul, FP32 for reduction.
The ONLY difference is whether the sim matrix materializes in HBM.

- Naive: einsum(FP16) → sim matrix in HBM → .float() → max → sum
- Flash: FP16 dot in SRAM → FP32 max in registers → sum. No sim matrix.

Run: /u/pony/envs/flash_maxsim/bin/python tests/bench_matched_precision.py
"""
import torch
import torch.nn.functional as F
import sys, time
sys.path.insert(0, '.')
from flash_maxsim import flash_maxsim

device = 'cuda'
D_DIM, DTYPE = 128, torch.float16

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


from flash_maxsim.naive import maxsim_naive as naive_fp16_reduce
from flash_maxsim.naive import maxsim_naive_fp32 as naive_matched
from flash_maxsim.naive import maxsim_chunked


def naive_matched_chunked(Q, D, chunk_size=500):
    return maxsim_chunked(Q, D, chunk_size=chunk_size, fp32_reduce=True)


# Precision check
torch.manual_seed(42)
Q = F.normalize(torch.randn(1024, D_DIM, dtype=DTYPE, device=device), dim=-1)
D = F.normalize(torch.randn(1000, 1024, D_DIM, dtype=DTYPE, device=device), dim=-1)

ref = torch.einsum('qd,bld->bql', Q.float(), D.float()).max(dim=2).values.sum(dim=1)
out_fp16 = naive_fp16_reduce(Q, D)
out_matched = naive_matched(Q, D)
out_flash = flash_maxsim(Q, D)

print('Precision vs FP32 ground truth (ColPali B=1000):')
print(f'  Naive FP16 reduce:    mean_err = {(ref - out_fp16).abs().mean().item():.6f}')
print(f'  Naive matched (fp32): mean_err = {(ref - out_matched).abs().mean().item():.6f}')
print(f'  Flash:                mean_err = {(ref - out_flash).abs().mean().item():.6f}')
print()
print('Both "matched" and "flash" use FP16 matmul + FP32 reduction → same precision.')
print('The only difference: naive materializes the sim matrix, flash does not.')

del D
torch.cuda.empty_cache()

# Speed
print()
print('=' * 90)
print(' Speed: Flash vs Naive at MATCHED precision (FP16 matmul + FP32 reduction)')
print('=' * 90)
print()
print(f'{"Config":<30}  {"naive_fp16":>10}  {"naive_matched":>14}  {"flash":>10}  {"vs_fp16":>8}  {"vs_matched":>11}')
print('-' * 90)

configs = [
    ('ColBERT B=1K',     32,  300,  1000),
    ('ColBERT B=10K',    32,  300,  10000),
    ('ColBERT B=100K',   32,  300,  100000),
    ('ColPali B=100',    1024, 1024, 100),
    ('ColPali B=500',    1024, 1024, 500),
    ('ColPali B=1K',     1024, 1024, 1000),
    ('ColPali B=5K',     1024, 1024, 5000),
    ('ColPali B=10K',    1024, 1024, 10000),
]

for name, Lq, Ld, B in configs:
    Q = F.normalize(torch.randn(Lq, D_DIM, dtype=DTYPE, device=device), dim=-1)
    try:
        D = F.normalize(torch.randn(B, Ld, D_DIM, dtype=DTYPE, device=device), dim=-1)
    except:
        print(f'{name:<30}  OOM on D alloc')
        torch.cuda.empty_cache()
        continue

    flash_ms = bench(lambda: flash_maxsim(Q, D))
    fp16_ms = bench(lambda: naive_fp16_reduce(Q, D))

    # Matched precision — may need chunking for large sim matrix
    sim_gb = B * Lq * Ld * 4 / 1e9  # FP32 sim matrix after .float()
    try:
        if sim_gb > 30:
            matched_ms = bench(lambda: naive_matched_chunked(Q, D, 200))
        else:
            matched_ms = bench(lambda: naive_matched(Q, D))
    except:
        matched_ms = float('inf')
        torch.cuda.empty_cache()

    vs_fp16 = fp16_ms / flash_ms
    vs_matched = matched_ms / flash_ms if matched_ms < float('inf') else float('inf')

    matched_s = f'{matched_ms:>13.2f}' if matched_ms < float('inf') else '          OOM'
    print(f'{name:<30}  {fp16_ms:>9.2f}  {matched_s}  {flash_ms:>9.2f}  {vs_fp16:>7.1f}x  {vs_matched:>10.1f}x')

    del D
    torch.cuda.empty_cache()

print()
print('vs_fp16    = speedup vs FP16 naive (flash is more precise — unfair to naive)')
print('vs_matched = speedup vs matched-precision naive (same precision — FAIR)')
print()
print('Done.')
