"""Benchmark experimental features from flash_maxsim_exp.

(1) Cache-aware heuristic vs autotune — cold-start + steady-state latency
(2) Chunked streaming vs whole-corpus-on-GPU — correctness + throughput
"""
import sys, time, os, subprocess, json
import torch
import torch.nn.functional as F
sys.path.insert(0, '.')

device = 'cuda'
D_DIM = 128
DTYPE = torch.float16


def bench_steady(fn, warmup=10, runs=30):
    for _ in range(warmup): fn()
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


gpu = torch.cuda.get_device_name(0)
print(f"GPU: {gpu}\n")


# ============================================================================
# Part 1 — Heuristic vs autotune: cold start (first call) + steady state
# ============================================================================

print("="*90)
print(" Part 1: Heuristic vs autotune — cold start and steady state")
print("="*90)

# Use DISTINCT Ld values per shape so Triton's autotune cache (keyed on Lq/Ld/d)
# cold-compiles every test. This captures JIT + autotune overhead in-process.
SHAPES = [
    ("ColBERT",   32,   181, 1000),
    ("ColBERT",   32,   182, 10000),
    ("ColPali",   1024, 1025, 100),
    ("ColPali",   1024, 1026, 1000),
    ("Long-doc",  32,   1027, 1000),
    ("Medium",    128,  1028, 500),
]

from flash_maxsim import flash_maxsim
from flash_maxsim.flash_maxsim_exp import flash_maxsim_heuristic

print(f"  {'regime':>10}  {'Lq':>5}  {'Ld':>5}  {'B':>6}  "
      f"{'cold_auto':>9}  {'cold_heur':>9}  {'speed_up':>8}  "
      f"{'steady_auto':>11}  {'steady_heur':>11}  {'ratio':>6}")
print("  " + "-" * 100)

for tag, Lq, Ld, B in SHAPES:
    torch.manual_seed(42 + Ld)
    Q = F.normalize(torch.randn(Lq, D_DIM, dtype=DTYPE, device=device), dim=-1)
    D = F.normalize(torch.randn(B, Ld, D_DIM, dtype=DTYPE, device=device), dim=-1)

    # --- Cold-start: first call for this shape ---
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out_auto = flash_maxsim(Q, D)
    torch.cuda.synchronize()
    cold_auto = (time.perf_counter() - t0) * 1000

    # Use Ld+1 to force a different key for the heuristic's separate kernel too.
    # (Heuristic kernel is a separate @triton.jit function from autotune kernel,
    #  so they don't share compile cache — first call triggers its own compile.)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out_heur = flash_maxsim_heuristic(Q, D)
    torch.cuda.synchronize()
    cold_heur = (time.perf_counter() - t0) * 1000

    # Correctness
    err = (out_auto - out_heur).abs().max().item()
    ok = "OK" if err < 0.1 else f"MISMATCH({err:.3f})"

    # --- Steady state ---
    steady_auto = bench_steady(lambda: flash_maxsim(Q, D))
    steady_heur = bench_steady(lambda: flash_maxsim_heuristic(Q, D))

    ratio = steady_heur / steady_auto if steady_auto > 0 else float('nan')
    speedup = cold_auto / cold_heur if cold_heur > 0 else float('nan')

    print(f"  {tag:>10}  {Lq:>5}  {Ld:>5}  {B:>6}  "
          f"{cold_auto:>7.0f}ms  {cold_heur:>7.0f}ms  {speedup:>6.1f}x  "
          f"{steady_auto:>9.2f}ms  {steady_heur:>9.2f}ms  {ratio:>5.2f}x  {ok}")

    del Q, D, out_auto, out_heur
    torch.cuda.empty_cache()


# ============================================================================
# Part 2 — Streamed (CPU corpus) vs whole-GPU corpus
# ============================================================================

print()
print("="*90)
print(" Part 2: Chunked streaming vs whole-GPU")
print("="*90)

from flash_maxsim import flash_maxsim
from flash_maxsim.flash_maxsim_exp import flash_maxsim_streamed

print("  Comparison: 'copy_then_compute' (one-shot H→D + kernel) vs")
print("              'streamed' (overlapped chunked H→D + kernel)")
print()
print(f"  {'regime':>10}  {'Lq':>4}  {'Ld':>4}  {'B':>7}  {'D_GB':>5}  {'chunk':>6}  "
      f"{'copy+k':>9}  {'streamed':>9}  {'ratio':>6}  correct")
print("  " + "-" * 88)

for (tag, Lq, Ld, B, chunk) in [
    ("ColBERT",  32,   180,  50_000, 10_000),
    ("ColBERT",  32,   180,  100_000, 20_000),
    ("ColPali",  1024, 1024, 2_000, 500),
    ("ColPali",  1024, 1024, 5_000, 1_000),
]:
    torch.manual_seed(42)
    Q = F.normalize(torch.randn(Lq, D_DIM, dtype=DTYPE, device=device), dim=-1)
    D_cpu = F.normalize(torch.randn(B, Ld, D_DIM, dtype=DTYPE), dim=-1).pin_memory()
    D_gpu_gb = B * Ld * D_DIM * 2 / (1024**3)

    # --- Method A: one-shot H→D copy, then kernel ---
    def copy_then_compute():
        D_gpu = D_cpu.to(device, non_blocking=True)
        s = flash_maxsim(Q, D_gpu)
        del D_gpu
        return s

    # --- Method B: streamed (overlapped) ---
    def streamed():
        return flash_maxsim_streamed(Q, D_cpu, chunk_size=chunk)

    # Correctness
    torch.cuda.synchronize()
    ref = copy_then_compute()
    torch.cuda.synchronize()
    test = streamed()
    err = (ref - test).abs().max().item()
    corr = "OK" if err < 0.1 else f"MISMATCH({err:.3f})"
    del ref, test
    torch.cuda.empty_cache()

    # Benchmark (few runs — these allocate a lot each iter)
    copy_ms = bench_steady(copy_then_compute, warmup=1, runs=3)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    stream_ms = bench_steady(streamed, warmup=1, runs=3)
    torch.cuda.synchronize()
    ratio = stream_ms / copy_ms

    print(f"  {tag:>10}  {Lq:>4}  {Ld:>4}  {B:>7,}  {D_gpu_gb:>4.1f}G  {chunk:>6,}  "
          f"{copy_ms:>7.1f}ms  {stream_ms:>7.1f}ms  {ratio:>5.2f}x  {corr}")

    del Q, D_cpu
    torch.cuda.empty_cache()

print("\nDone.")
