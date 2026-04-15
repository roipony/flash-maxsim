"""Collect benchmark data for A100 — generates JSON for make_figure.py.

Run: /u/pony/envs/flash_maxsim/bin/python benchmarks/collect_a100_data.py
"""
import json, sys, time, torch
import torch.nn.functional as F
sys.path.insert(0, '.')
from flash_maxsim import flash_maxsim
from flash_maxsim.naive import maxsim_naive, maxsim_naive_fp32

device = 'cuda'
D_DIM = 128
DTYPE = torch.float16

def bench(fn, warmup=10, runs=20):
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
print(f"GPU: {gpu}")

results = {
    "gpu": gpu,
    "triton": "3.6.0",
    "torch": torch.__version__,
    "d": D_DIM,
    "dtype": "float16",
}

# Speedup sweep
sweep_data = []
configs = [
    ("textual",  32,  300),
    ("long_doc", 32,  1024),
    ("medium",   128, 1024),
    ("visual",   1024, 1024),
    ("short_d",  1024, 512),
]

for tag, Lq, Ld in configs:
    Q = F.normalize(torch.randn(Lq, D_DIM, dtype=DTYPE, device=device), dim=-1)
    row = {"tag": tag, "Lq": Lq, "Ld": Ld, "points": []}
    for B in [10, 50, 100, 500, 1000, 2000, 5000, 10000]:
        try:
            D = F.normalize(torch.randn(B, Ld, D_DIM, dtype=DTYPE, device=device), dim=-1)
        except:
            torch.cuda.empty_cache()
            continue
        naive_ms = bench(lambda: maxsim_naive(Q, D))
        matched_ms = bench(lambda: maxsim_naive_fp32(Q, D))
        flash_ms = bench(lambda: flash_maxsim(Q, D))
        row["points"].append({
            "B": B,
            "naive_ms": round(naive_ms, 3),
            "matched_ms": round(matched_ms, 3),
            "flash_ms": round(flash_ms, 3),
            "speedup_fp16": round(naive_ms / flash_ms, 2),
            "speedup_matched": round(matched_ms / flash_ms, 2),
        })
        print(f"  {tag} B={B}: naive={naive_ms:.2f} matched={matched_ms:.2f} flash={flash_ms:.2f} "
              f"sp_fp16={naive_ms/flash_ms:.1f}x sp_matched={matched_ms/flash_ms:.1f}x")
        del D; torch.cuda.empty_cache()
    sweep_data.append(row)

# Memory data
memory_data = []
for tag, Lq, Ld, B in [("textual", 32, 300, 10000), ("long_doc", 32, 1024, 10000),
                         ("visual", 1024, 1024, 1000), ("visual", 1024, 1024, 5000)]:
    sim_mb = B * Lq * Ld * 2 / 1e6
    memory_data.append({"tag": tag, "Lq": Lq, "Ld": Ld, "B": B, "sim_matrix_mb": round(sim_mb, 1)})

results["sweep"] = sweep_data
results["memory"] = memory_data

outfile = f"benchmarks/results_v020_{gpu.replace(' ', '_')}.json"
with open(outfile, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {outfile}")
print("Done.")
