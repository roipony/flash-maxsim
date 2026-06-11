"""Find the crossover point where Flash-MaxSim beats naive einsum.

Usage: python benchmarks/bench_threshold.py
"""
import torch, torch.nn.functional as F

assert torch.cuda.is_available(), "CUDA required"
print(f"GPU: {torch.cuda.get_device_name()}\n")

from flash_maxsim.flash_maxsim import _launch_fwd, _default_lengths, maxsim_naive
from utils import bench_interleaved, compile_high_precision
compiled_maxsim = compile_high_precision(maxsim_naive)


def bench_flash_raw(Q, D):
    """Call the Triton kernel directly (bypass auto-dispatch)."""
    Lq, d = Q.shape
    B, Ld, _ = D.shape
    Q2 = Q.unsqueeze(0).contiguous().half()
    D2 = D.contiguous().half()
    lengths = _default_lengths(B, Ld, D.device, None)
    scores, _ = _launch_fwd(Q2, D2, lengths, 1, B, Lq, Ld, d, True, False)
    return scores.squeeze(0)


print("=" * 70)
print("CROSSOVER FINDER: naive einsum vs Flash-MaxSim kernel")
print("=" * 70)
print(f"  {'Config':38s} {'B*Lq*Ld':<13s} {'naive':<10s} {'compiled':<10s} {'flash':<10s} {'naive_sp':<10s} {'compiled_sp'}")
print("-" * 110)

for B, Lq, Ld, d in [
    (10,   32,  300,  128),
    (20,   32,  300,  128),
    (50,   32,  300,  128),
    (100,  32,  300,  128),
    (200,  32,  300,  128),
    (500,  32,  300,  128),
    (1000, 32,  300,  128),
    (10,   32,  1024, 128),
    (50,   32,  1024, 128),
    (100,  32,  1024, 128),
    (500,  32,  1024, 128),
    (10,   128, 300,  128),
    (50,   128, 300,  128),
    (100,  128, 300,  128),
    (500,  128, 300,  128),
    (10,   1024, 1024, 128),
    (50,   1024, 1024, 128),
    (100,  1024, 1024, 128),
    (500,  1024, 1024, 128),
]:
    Q = F.normalize(torch.randn(Lq, d, device='cuda', dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(B, Ld, d, device='cuda', dtype=torch.float16), dim=-1)

    score_size = B * Lq * Ld

    t_naive, t_compiled, t_flash = bench_interleaved([maxsim_naive, compiled_maxsim, bench_flash_raw], [[Q, D]])

    label = f"B={B}, Lq={Lq}, Ld={Ld}"
    print(f"  {label:38s} {f'{score_size:,}':<13s} {f'{t_naive:.3f}ms':<10s} {f'{t_compiled:.3f}ms':<10s} {f'{t_flash:.3f}ms':<10s} {f'{t_naive/t_flash:.1f}x':<10s} {f'{t_compiled/t_flash:.1f}x'}")

    del Q, D; torch.cuda.empty_cache()

print("\nSuggested threshold: set _SMALL_THRESHOLD to the B*Lq*Ld value")
print("where Flash consistently wins.")
