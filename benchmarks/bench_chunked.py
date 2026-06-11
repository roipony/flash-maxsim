"""Benchmark: Chunked naive MaxSim vs Flash-MaxSim.

Shows the U-shaped curve: small chunks = slow (many launches),
large chunks = OOM. Flash-MaxSim is a flat line below all of it.

Usage: python benchmarks/bench_chunked.py
Produces: benchmarks/results_chunked_<GPU>.json
"""
import json, torch, torch.nn.functional as F

assert torch.cuda.is_available(), "CUDA required"
gpu_name = torch.cuda.get_device_name()
gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {gpu_name} ({gpu_mem_gb:.0f}GB)\n")

from flash_maxsim import flash_maxsim
from utils import bench_interleaved, compile_high_precision

def chunked_maxsim(Q, D, chunk_size):
    """Naive MaxSim with chunking over documents (B axis)."""
    B = D.shape[0]
    scores = torch.empty(B, device=Q.device, dtype=torch.float32)
    for i in range(0, B, chunk_size):
        D_chunk = D[i:i+chunk_size]
        sim = torch.einsum('qd,bld->bql', Q.float(), D_chunk.float())
        scores[i:i+chunk_size] = sim.max(dim=2).values.sum(dim=1)
    return scores

compiled_chunked_maxsim = compile_high_precision(chunked_maxsim)

results = {"gpu": gpu_name, "gpu_mem_gb": round(gpu_mem_gb, 1), "configs": []}

print("=" * 65)
print("CHUNKED NAIVE vs FLASH-MAXSIM")
print("=" * 65)
print("Sweeping chunk sizes to find the U-shaped curve.\n")

for B, Lq, Ld, d, label in [
    (1000,  32,  300,  128, "Textual (Lq=32, Ld=300, B=1K)"),
    (5000,  32,  300,  128, "Textual (Lq=32, Ld=300, B=5K)"),
    (1000,  128, 1024, 128, "Medium (Lq=128, Ld=1024, B=1K)"),
    (1000,  1024, 1024, 128, "Visual (Lq=1024, Ld=1024, B=1K)"),
    (5000,  1024, 1024, 128, "Visual (Lq=1024, Ld=1024, B=5K)"),
    (10000, 1024, 1024, 128, "Visual (Lq=1024, Ld=1024, B=10K)"),
]:
    # Check if D fits in GPU memory
    d_gb = B * Ld * d * 2 / 1e9
    if d_gb > gpu_mem_gb * 0.6:
        print(f"\n  {label}")
        print(f"  SKIP: D alone = {d_gb:.1f}GB > 60% of GPU ({gpu_mem_gb:.0f}GB)")
        continue

    Q = F.normalize(torch.randn(Lq, d, device='cuda', dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(B, Ld, d, device='cuda', dtype=torch.float16), dim=-1)

    # Sim matrix size per doc
    sim_per_doc_mb = Lq * Ld * 4 / 1e6  # FP32

    # Flash-MaxSim (the flat line) — run FIRST before chunked can corrupt CUDA
    wup = 5 if Lq >= 512 else 10
    iters = 20 if Lq >= 512 else 40
    t_flash = bench_interleaved([flash_maxsim], [[Q, D]], warmup=wup, n=iters).item()

    print(f"\n  {label}")
    print(f"  sim_matrix/doc: {sim_per_doc_mb:.1f}MB  full_sim: {sim_per_doc_mb*B:.0f}MB")
    print(f"  Flash-MaxSim:   {t_flash:.3f}ms (parameter-free)")

    config_results = {
        "label": label, "B": B, "Lq": Lq, "Ld": Ld, "d": d,
        "sim_per_doc_mb": round(sim_per_doc_mb, 2),
        "flash_ms": round(t_flash, 3),
        "chunks": [],
    }

    # Sweep chunk sizes: 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, ..., B
    chunk_sizes = sorted(set([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, B]))
    chunk_sizes = [c for c in chunk_sizes if c <= B]

    print(f"  {'chunk':>6s}  {'sim_MB':>8s}  {'latency':>10s}  {'eager_vs_flash':>10s} {'compiled_vs_flash':>10s}  {'status':>8s}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")

    for chunk in chunk_sizes:
        sim_mb = chunk * sim_per_doc_mb
        free_mem_mb = (torch.cuda.mem_get_info()[0]) / 1e6

        # Skip if chunk sim matrix would clearly OOM (conservative: 90% of free)
        if sim_mb > free_mem_mb * 0.9:
            print(f"  {chunk:6d}  {sim_mb:7.0f}MB  {'---':>10s}  {'---':>10s}  {'OOM':>8s}")
            config_results["chunks"].append({
                "chunk_size": chunk, "sim_mb": round(sim_mb, 1),
                "latency_ms": None, "status": "OOM",
            })
            continue

        # Try it — catch OOM and CUDA errors gracefully
        try:
            torch.cuda.empty_cache()
            # Single warmup run to test if it fits
            _ = chunked_maxsim(Q, D, chunk)
            torch.cuda.synchronize()
            del _
            torch.cuda.empty_cache()

            t_chunked, t_compiled = bench_interleaved([chunked_maxsim, compiled_chunked_maxsim], [[Q, D, chunk]])
            eager_ratio = t_chunked / t_flash
            compiled_ratio = t_compiled / t_flash
            print(f"  {chunk:6d}  {sim_mb:7.0f}MB  {t_chunked:8.3f}ms {t_compiled:8.3f} "
                  f"{eager_ratio:8.1f}x {compiled_ratio:8.1f}x  {'ok':>8s}")
            config_results["chunks"].append({
                "chunk_size": chunk, "sim_mb": round(sim_mb, 1),
                "eager_ms": round(t_chunked, 3), "compiled_ms": round(t_chunked, 3),
                "eager_vs_flash": round(eager_ratio, 2), "compiled_vs_flash": round(compiled_ratio, 2), "status": "ok",
            })
        except (torch.cuda.OutOfMemoryError, RuntimeError, Exception) as e:
            print(f"  {chunk:6d}  {sim_mb:7.0f}MB  {'---':>10s}  {'---':>10s}  {'OOM':>8s}")
            config_results["chunks"].append({
                "chunk_size": chunk, "sim_mb": round(sim_mb, 1),
                "latency_ms": None, "status": "OOM",
            })
            # Reset CUDA state after OOM
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
            # Skip larger chunk sizes for this config — they'll also OOM
            for remaining_chunk in [c for c in chunk_sizes if c > chunk]:
                remaining_sim = remaining_chunk * sim_per_doc_mb
                print(f"  {remaining_chunk:6d}  {remaining_sim:7.0f}MB  {'---':>10s}  {'---':>10s}  {'OOM':>8s}")
                config_results["chunks"].append({
                    "chunk_size": remaining_chunk, "sim_mb": round(remaining_sim, 1),
                    "latency_ms": None, "status": "OOM",
                })
            break

    results["configs"].append(config_results)
    del Q, D; torch.cuda.empty_cache()

# ── Save ──
gpu_tag = gpu_name.replace(" ", "_").replace("/", "-")
out = f"benchmarks/results_chunked_{gpu_tag}.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out}")
