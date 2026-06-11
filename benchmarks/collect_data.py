"""Collect continuous sweep benchmark data for visualization.

Usage:
    python benchmarks/collect_data.py
    # produces benchmarks/results_<GPU_NAME>.json
"""
import json, torch, torch.nn.functional as F

import triton
assert torch.cuda.is_available(), "CUDA required"
gpu_name = torch.cuda.get_device_name()
print(f"GPU: {gpu_name}\n")

from flash_maxsim import flash_maxsim, flash_maxsim_batched
from flash_maxsim import flash_maxsim_int8, quantize_int8
from utils import bench_interleaved, compile_high_precision

def make(B, Lq, Ld, d=128):
    Q = F.normalize(torch.randn(Lq, d, device='cuda', dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(B, Ld, d, device='cuda', dtype=torch.float16), dim=-1)
    return Q, D

def naive_fp32(Q, D):
    return torch.einsum('qd,bld->bql', Q.float(), D.float()).max(2).values.sum(1)

torch.set_float32_matmul_precision('high')
compiled_fp32 = compile_high_precision(naive_fp32)

results = {"gpu": gpu_name, "torch": torch.__version__, "triton": triton.__version__}

# ── Sweep 1: Vary Ld at different Lq values (fixed B=1000) ──
# Shows the textual → visual transition
print("=" * 60)
print("SWEEP 1: Sequence length landscape (B=1000)")
print("=" * 60)
B = 1000
sweep_seq = []
for Lq in [32, 128, 512, 1024]:
    for Ld in [128, 256, 512, 768, 1024]:
        # skip combos that would OOM on naive
        if Lq * Ld * B * 4 > 60e9:
            print(f"  Lq={Lq}, Ld={Ld}: skip (would OOM naive)")
            continue
        Q, D = make(B, Lq, Ld)
        try:
            n, c, f = bench_interleaved([naive_fp32, compiled_fp32, flash_maxsim], [[Q, D]])
        except RuntimeError:
            n = float('nan')
            c, f = bench_interleaved([compiled_fp32, flash_maxsim], [[Q, D]])
        c_sp = c/f
        n_sp = n / f if n == n else float('nan')
        print(f"  Lq={Lq:4d}, Ld={Ld:4d}: naive={n:7.2f}ms compiled={c:7.2f}ms flash={f:7.2f}ms "
              f"naive_speedup={n_sp:.1f}x compiled_speedup={c_sp:.1f}x")
        sweep_seq.append({"Lq": Lq, "Ld": Ld, "B": B, "naive_ms": round(n, 3), "flash_ms": round(f, 3),
                          "naive_speedup": round(n_sp, 1), "compiled_speedup": round(c_sp, 1)})
        del Q, D; torch.cuda.empty_cache()
results["sweep_seq"] = sweep_seq

# ── Sweep 2: Vary B (corpus size) at key configs ──
print(f"\n{'=' * 60}")
print("SWEEP 2: Corpus scaling")
print("=" * 60)
sweep_corpus = []
configs = [
    (32, 300, "textual"),
    (32, 1024, "long_doc"),
    (1024, 1024, "visual"),
]
for Lq, Ld, tag in configs:
    B_values = [100, 500, 1000, 2000, 5000]
    if Lq == 1024:
        B_values = [100, 500, 1000, 2000, 5000]
    for B in B_values:
        if Lq * Ld * B * 4 > 60e9:
            print(f"  {tag} B={B}: skip (would OOM naive)")
            # still benchmark flash
            Q, D = make(B, Lq, Ld)
            f = bench_interleaved([flash_maxsim], [[Q, D]]).item()
            sweep_corpus.append({"Lq": Lq, "Ld": Ld, "B": B, "tag": tag, "naive_ms": float('nan'), "flash_ms": round(f, 3), "speedup": float('nan')})
            del Q, D; torch.cuda.empty_cache()
            continue
        Q, D = make(B, Lq, Ld)
        try:
            n, c, f = bench_interleaved([naive_fp32, compiled_fp32, flash_maxsim], [[Q, D]]).item()
        except RuntimeError:
            n = float('nan')
            c, f = bench_interleaved([compiled_fp32, flash_maxsim], [[Q, D]])
        n_sp = n / f if n == n else float('nan')
        c_sp = c / f
        print(f"  {tag:10s} B={B:5d}: naive={n:8.2f}ms compiled={c:8.2f}ms flash={f:7.2f}ms  "
              f"naive_speedup={n_sp:.1f}x compiled_speedup={c_sp:.1f}x")
        sweep_corpus.append({"Lq": Lq, "Ld": Ld, "B": B, "tag": tag, "naive_ms": round(n, 3), "compiled_ms": round(c, 3),
                             "flash_ms": round(f, 3), "naive_speedup": round(n_sp, 1), "compiled_speedup": round(c_sp, 1)})
        del Q, D; torch.cuda.empty_cache()
results["sweep_corpus"] = sweep_corpus

# ── Sweep 3: Memory ──
print(f"\n{'=' * 60}")
print("SWEEP 3: Peak memory")
print("=" * 60)
sweep_mem = []
gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
mem_configs = [
    (32, 300, "textual"),
    (32, 1024, "long_doc"),
    (1024, 1024, "visual"),
]
for Lq, Ld, tag in mem_configs:
    B_values = [500, 1000, 2000, 5000, 10000] if tag != "visual" else [100, 500, 1000, 2000]
    for B in B_values:
        # Check if D alone would exceed GPU memory
        d_gb = B * Ld * 128 * 2 / 1e9  # FP16 storage
        if d_gb > gpu_mem_gb * 0.7:
            print(f"  {tag:10s} B={B:5d}: skip (D={d_gb:.1f}GB exceeds GPU)")
            continue

        # Check total memory needed: D_fp16 + D_fp32 + Q_fp32 + sim_matrix
        sim_gb = B * Lq * Ld * 4 / 1e9
        total_needed = d_gb + (B * Ld * 128 * 4 / 1e9) + sim_gb  # D_fp32 + sim
        if total_needed > gpu_mem_gb * 0.5:
            # Skip naive, use theoretical sim size
            Q = F.normalize(torch.randn(1, Lq, 128, device='cuda', dtype=torch.float16), dim=-1)
            D = F.normalize(torch.randn(B, Ld, 128, device='cuda', dtype=torch.float16), dim=-1)
            naive_gb = sim_gb

            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            try:
                _ = flash_maxsim_batched(Q, D, shared_docs=True); torch.cuda.synchronize()
                flash_gb = (torch.cuda.max_memory_allocated() - base) / 1e9
                del _
            except (RuntimeError, Exception):
                flash_gb = 0.0001
            torch.cuda.empty_cache()

            ratio = max(1, naive_gb / max(flash_gb, 0.001))
            print(f"  {tag:10s} B={B:5d}: naive={naive_gb:6.2f}GB (theoretical)  flash={flash_gb:.4f}GB  {ratio:.0f}x")
            sweep_mem.append({"Lq": Lq, "Ld": Ld, "B": B, "tag": tag, "naive_gb": round(naive_gb, 4), "flash_gb": round(flash_gb, 4), "ratio": round(ratio, 0)})
            del Q, D; torch.cuda.empty_cache()
            continue

        Q = F.normalize(torch.randn(1, Lq, 128, device='cuda', dtype=torch.float16), dim=-1)
        D = F.normalize(torch.randn(B, Ld, 128, device='cuda', dtype=torch.float16), dim=-1)

        # Naive memory
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        try:
            _ = torch.einsum('nqd,bld->nbql', Q.float(), D.float()).max(3).values.sum(2)
            torch.cuda.synchronize()
            naive_gb = (torch.cuda.max_memory_allocated() - base) / 1e9
            del _
        except Exception:
            naive_gb = sim_gb
            # Reset CUDA after error
            torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # Flash memory
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        try:
            _ = flash_maxsim_batched(Q, D, shared_docs=True); torch.cuda.synchronize()
            flash_gb = (torch.cuda.max_memory_allocated() - base) / 1e9
            del _
        except (RuntimeError, Exception):
            flash_gb = 0.0001
        torch.cuda.empty_cache()

        ratio = max(1, naive_gb / max(flash_gb, 0.001))
        print(f"  {tag:10s} B={B:5d}: naive={naive_gb:6.2f}GB  flash={flash_gb:.4f}GB  {ratio:.0f}x")
        sweep_mem.append({"Lq": Lq, "Ld": Ld, "B": B, "tag": tag, "naive_gb": round(naive_gb, 4), "flash_gb": round(flash_gb, 4), "ratio": round(ratio, 0)})
        del Q, D; torch.cuda.empty_cache()
results["sweep_mem"] = sweep_mem

# ── Sweep 4: INT8 vs FP16 ──
print(f"\n{'=' * 60}")
print("SWEEP 4: INT8 fused dequant")
print("=" * 60)
sweep_int8 = []
for B in [500, 1000, 2000, 5000]:
    for Lq, Ld, tag in [(32, 300, "textual"), (32, 1024, "long_doc")]:
        Q, D = make(B, Lq, Ld)
        Dq, s, m = quantize_int8(D)
        nf, cf, ff, fi = bench_interleaved([naive_fp32, compiled_fp32, flash_maxsim, flash_maxsim_int8],
                               [[Q, D], [Q, D], [Q, D], [Q, Dq, s, m]])
        print(f"  {tag:10s} B={B:5d}: naive_fp32={nf:.2f}ms compiled_fp32={cf:.2f}ms flash_fp16={ff:.2f}ms  "
              f"flash_q8={fi:.2f}ms  naive_q8_speedup={nf/fi:.1f}x compiled_q8_speedup={cf/fi:.1f}x")
        sweep_int8.append({"Lq": Lq, "Ld": Ld, "B": B, "tag": tag,
                           "naive_fp32_ms": round(nf, 3), "flash_fp16_ms": round(ff, 3),
                           "flash_q8_ms": round(fi, 3), "speedup_vs_naive": round(nf/fi, 1)})
        del Q, D, Dq, s, m; torch.cuda.empty_cache()
results["sweep_int8"] = sweep_int8

# ── Save ──
gpu_tag = gpu_name.replace(" ", "_").replace("/", "-")
out = f"benchmarks/results_{gpu_tag}.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out}")
