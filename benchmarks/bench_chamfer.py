"""Flash-Chamfer benchmark. Run: python benchmarks/bench_chamfer.py

Tests correctness against naive implementation and benchmarks at various scales.
"""
import torch
from utils import bench_interleaved, compile_high_precision

assert torch.cuda.is_available(), "CUDA required"
print(f"GPU: {torch.cuda.get_device_name()}\n")

from flash_maxsim.flash_chamfer import (
    flash_chamfer,
    flash_chamfer_one_direction,
    naive_chamfer,
    naive_chamfer_one_direction,
)


def make_clouds(N, M, d, device="cuda"):
    """Generate random point clouds."""
    P = torch.randn(N, d, device=device, dtype=torch.float32)
    Q = torch.randn(M, d, device=device, dtype=torch.float32)
    return P, Q


# ══════════════════════════════════════════════════════════════════════
# CORRECTNESS TESTS
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("CORRECTNESS: flash_chamfer vs naive_chamfer")
print("=" * 70)

test_cases = [
    (100, 100, 3, "l2_sq"),
    (256, 512, 3, "l2_sq"),
    (1000, 1000, 3, "l2_sq"),
    (500, 300, 64, "l2_sq"),
    (100, 200, 128, "l2_sq"),
    (256, 512, 3, "l1"),
    (500, 300, 64, "l1"),
]

all_pass = True
for N, M, d, metric in test_cases:
    P, Q = make_clouds(N, M, d)

    # Test one-direction
    flash_1d = flash_chamfer_one_direction(P, Q, metric=metric)
    naive_1d = naive_chamfer_one_direction(P, Q, metric=metric)
    err_1d = (flash_1d - naive_1d).abs().max().item()
    rel_err_1d = err_1d / (naive_1d.abs().max().item() + 1e-8)

    # Test full chamfer
    flash_cd = flash_chamfer(P, Q, metric=metric)
    naive_cd = naive_chamfer(P, Q, metric=metric)
    err_cd = (flash_cd - naive_cd).abs().item()
    rel_err_cd = err_cd / (naive_cd.abs().item() + 1e-8)

    ok = rel_err_cd < 0.01  # 1% relative tolerance (fp16 intermediate)
    if not ok:
        all_pass = False
    status = "OK" if ok else "FAIL"

    print(f"  N={N}, M={M}, d={d}, {metric:5s}: "
          f"one_dir_err={rel_err_1d:.2e}  full_err={rel_err_cd:.2e}  [{status}]")

print(f"\n  Overall: {'ALL PASSED' if all_pass else 'SOME FAILURES'}\n")


# ══════════════════════════════════════════════════════════════════════
# SPEED BENCHMARKS
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SPEED: flash_chamfer vs naive (PyTorch cdist)")
print("=" * 70)

speed_cases = [
    (1_000, 1_000, 3, "1K x 1K, d=3"),
    (10_000, 10_000, 3, "10K x 10K, d=3"),
    (1_000, 1_000, 128, "1K x 1K, d=128"),
    (10_000, 10_000, 128, "10K x 10K, d=128"),
]

# Check if 100K is feasible (needs ~100K*100K*4 = 40GB for naive)
# We'll try but catch OOM
large_cases = [
    (100_000, 100_000, 3, "100K x 100K, d=3"),
]

compiled_chamfer = compile_high_precision(naive_chamfer)

for N, M, d, label in speed_cases:
    P, Q = make_clouds(N, M, d)

    t_naive, t_compiled, t_flash = bench_interleaved([naive_chamfer, compiled_chamfer, flash_chamfer], [[P, Q]])
    naive_speedup = t_naive / t_flash
    compiled_speedup = t_compiled / t_flash

    print(f"  {label:25s}: naive={t_naive:.2f}ms compiled={t_compiled:.2f}ms flash={t_flash:.2f}ms "
          f"naive_speedup={naive_speedup:.1f}x compiled_speedup={compiled_speedup:.1f}x")

# Large scale: naive may OOM
for N, M, d, label in large_cases:
    P, Q = make_clouds(N, M, d)

    try:
        t_naive, t_compiled, t_flash = bench_interleaved([naive_chamfer, compiled_chamfer, flash_chamfer], [[P, Q]]).item()
        naive_speedup_str = f"{t_naive / t_flash:5.1f}x"
        naive_str = f"{t_naive:.2f}ms"
        compiled_speedup_str = f"{t_compiled / t_flash:.1f}x"
        compiled_str = f"{t_compiled:.2f}ms"
    except RuntimeError:
        naive_str = "OOM"
        naive_speedup_str = "N/A"
        t_compiled, t_flash = bench_interleaved([compiled_chamfer, flash_chamfer], [[P, Q]]).item()
        compiled_speedup_str = f"{t_compiled / t_flash:.1f}x"
        compiled_str = f"{t_compiled:.2f}ms"


    print(f"  {label:25s}: naive={naive_str} compiled={compiled_str} flash={t_flash:.2f}ms "
          f"naive_speedup={naive_speedup_str} compiled_speedup={compiled_speedup_str}")


# ══════════════════════════════════════════════════════════════════════
# MEMORY SAVINGS
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("MEMORY: flash vs naive (peak GPU memory above baseline)")
print("=" * 70)

memory_cases = [
    (1_000, 1_000, 3, "1K x 1K, d=3"),
    (10_000, 10_000, 3, "10K x 10K, d=3"),
    (10_000, 10_000, 128, "10K x 10K, d=128"),
    (50_000, 50_000, 3, "50K x 50K, d=3"),
    (100_000, 100_000, 3, "100K x 100K, d=3"),
]

for N, M, d, label in memory_cases:
    P, Q = make_clouds(N, M, d)
    matrix_gb = N * M * 4 / 1e9  # theoretical [N,M] float32 matrix size

    # Measure naive memory
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    try:
        _ = naive_chamfer(P, Q)
        torch.cuda.synchronize()
        naive_gb = (torch.cuda.max_memory_allocated() - base) / 1e9
        naive_str = f"{naive_gb:.3f}GB"
        del _
    except RuntimeError:
        naive_gb = matrix_gb
        naive_str = f"OOM (~{matrix_gb:.1f}GB)"

    # Measure flash memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    _ = flash_chamfer(P, Q)
    torch.cuda.synchronize()
    flash_gb = (torch.cuda.max_memory_allocated() - base) / 1e9
    del _
    torch.cuda.empty_cache()

    ratio = max(1, naive_gb / max(flash_gb, 1e-6))
    print(f"  {label:25s}: matrix={matrix_gb:.3f}GB  naive={naive_str:15s}  "
          f"flash={flash_gb:.3f}GB  ({ratio:.0f}x less)")


# ══════════════════════════════════════════════════════════════════════
# L1 METRIC BENCHMARK
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("L1 METRIC (Manhattan distance)")
print("=" * 70)

l1_cases = [
    (1_000, 1_000, 3, "1K x 1K, d=3"),
    (10_000, 10_000, 3, "10K x 10K, d=3"),
    (1_000, 1_000, 128, "1K x 1K, d=128"),
]

for N, M, d, label in l1_cases:
    P, Q = make_clouds(N, M, d)

    t_naive, t_compiled, t_flash = bench_interleaved([naive_chamfer, compiled_chamfer, flash_chamfer], [[P, Q, "l1"]])
    naive_speedup = t_naive / t_flash
    compiled_speedup = t_compiled / t_flash

    print(f"  {label:25s}: naive={t_naive:.2f}ms compiled={t_compiled:.2f}ms flash={t_flash:.2f}ms "
          f"naive_speedup={naive_speedup:.1f}x compiled_speedup={compiled_speedup:.1f}x")


# ══════════════════════════════════════════════════════════════════════
# 4-TIER BENCHMARK (standard 3D point cloud community tiers)
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("4-TIER BENCHMARK (standard 3D point cloud community)")
print("=" * 70)

tier_configs = [
    (1, "Standard Research",    2_048,   2_048,   3),
    (2, "High-Res Generation",  32_768,  32_768,  3),
    (3, "Industrial/LiDAR",     131_072, 131_072, 3),
    (4, "Latent Space",         10_000,  10_000,  128),
]

for tier, desc, N, M, d in tier_configs:
    print(f"\n  --- Tier {tier}: {desc} (N={N}, M={M}, d={d}) ---")
    P, Q = make_clouds(N, M, d)

    # Flash chamfer (always expected to work)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_flash = torch.cuda.memory_allocated()
    t_flash = bench_interleaved([flash_chamfer], [[P, Q]]).item()
    _ = flash_chamfer(P, Q)
    torch.cuda.synchronize()
    flash_mem_gb = (torch.cuda.max_memory_allocated() - base_flash) / 1e9
    del _
    torch.cuda.empty_cache()

    # Naive chamfer (may OOM on large tiers)
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base_naive = torch.cuda.memory_allocated()
        t_naive = bench_interleaved([naive_chamfer], [[P, Q]]).item()
        _ = naive_chamfer(P, Q)
        torch.cuda.synchronize()
        naive_mem_gb = (torch.cuda.max_memory_allocated() - base_naive) / 1e9
        del _
        torch.cuda.empty_cache()

        speedup = t_naive / t_flash
        print(f"    flash : {t_flash:10.2f} ms | mem {flash_mem_gb:.4f} GB")
        print(f"    naive : {t_naive:10.2f} ms | mem {naive_mem_gb:.4f} GB")
        print(f"    speedup: {speedup:.1f}x  |  mem savings: {naive_mem_gb / max(flash_mem_gb, 1e-9):.0f}x")
    except RuntimeError:
        matrix_gb = N * M * 4 / 1e9
        print(f"    flash : {t_flash:10.2f} ms | mem {flash_mem_gb:.4f} GB")
        print(f"    naive :        OOM       | would need ~{matrix_gb:.1f} GB for distance matrix")
        print(f"    speedup: N/A (naive OOM)")

    torch.cuda.empty_cache()

    # compiled chamfer (may OOM on large tiers)
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base_compiled = torch.cuda.memory_allocated()
        t_compiled = bench_interleaved([compiled_chamfer], [[P, Q]]).item()
        _ = compiled_chamfer(P, Q)
        torch.cuda.synchronize()
        compiled_mem_gb = (torch.cuda.max_memory_allocated() - base_compiled) / 1e9
        del _
        torch.cuda.empty_cache()

        speedup = t_compiled / t_flash
        print(f"    flash : {t_flash:10.2f} ms | mem {flash_mem_gb:.4f} GB")
        print(f"    naive : {t_compiled:10.2f} ms | mem {compiled_mem_gb:.4f} GB")
        print(f"    speedup: {speedup:.1f}x  |  mem savings: {compiled_mem_gb / max(flash_mem_gb, 1e-9):.0f}x")
    except RuntimeError:
        matrix_gb = N * M * 4 / 1e9
        print(f"    flash : {t_flash:10.2f} ms | mem {flash_mem_gb:.4f} GB")
        print(f"    naive :        OOM       | would need ~{matrix_gb:.1f} GB for distance matrix")
        print(f"    speedup: N/A (naive OOM)")

    del P, Q
    torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════
# PYTORCH3D COMPARISON
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PYTORCH3D COMPARISON")
print("=" * 70)

try:
    from pytorch3d.loss import chamfer_distance as pt3d_chamfer

    pt3d_cases = [
        (1, "Standard Research",    2_048,   2_048,   3),
        (2, "High-Res Generation",  32_768,  32_768,  3),
        (3, "Industrial/LiDAR",     131_072, 131_072, 3),
        (4, "Latent Space",         10_000,  10_000,  128),
    ]

    for tier, desc, N, M, d in pt3d_cases:
        print(f"\n  --- Tier {tier}: {desc} (N={N}, M={M}, d={d}) ---")
        P, Q = make_clouds(N, M, d)

        # PyTorch3D expects (B, N, 3) batched input
        P_batch = P.unsqueeze(0)
        Q_batch = Q.unsqueeze(0)

        try:
            t_pt3d, t_flash = bench_interleaved([lambda: pt3d_chamfer(P_batch, Q_batch), flash_chamfer], [[], [P, Q]])
            speedup = t_pt3d / t_flash
            print(f"    flash     : {t_flash:10.2f} ms")
            print(f"    pytorch3d : {t_pt3d:10.2f} ms")
            print(f"    speedup vs pt3d: {speedup:.1f}x")
        except RuntimeError:
            t_flash = bench_interleaved([flash_chamfer], [[]])
            print(f"    flash     : {t_flash:10.2f} ms")
            print(f"    pytorch3d :        OOM")
            print(f"    speedup vs pt3d: N/A (pt3d OOM)")

        del P, Q, P_batch, Q_batch
        torch.cuda.empty_cache()

except ImportError:
    print("  PyTorch3D not installed, skipping comparison")


print(f"\n{'=' * 70}")
print("DONE")
print("=" * 70)
