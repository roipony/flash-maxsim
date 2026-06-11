"""Probe: at which embedding dim `d` does the current Flash-MaxSim kernel
start to fail or spill?

Motivation: split-d killer feature for fat embeddings (Jina d=128/512,
d=384–768, Voyage d=1024, NV-Embed d=4096). The current kernel
uses `d_pad = next_pow2(d)` as a constexpr tile, which creates per-program
register tiles of size [BLOCK_Q, d_pad] and [BLOCK_D, d_pad] fp16 in
shared memory. At d=1024 that's 64KB per tile — already at the SRAM
budget on A100 (164 KB) once we account for `num_stages=2` double-
buffering.

For each d we measure: (a) does it compile + run at all?  (b) is the
forward correctness OK against a naive einsum?  (c) what's the latency
+ peak memory profile?  Output drives the design of the split-d
kernel.
"""
import json, os, sys, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from flash_maxsim import flash_maxsim_batched

torch.set_float32_matmul_precision('high')
WARMUP, RUNS = 3, 10

# d values that cover real production encoders.
D_VALUES = [128, 256, 384, 512, 768, 1024, 2048]
SHAPES = [
    # (name, B, Lq, Ld)
    ("textual",  1024, 32,   180),
    ("ColPali",  128,  1024, 1024),
]


def _bench(fn, runs=RUNS, warmup=WARMUP):
    for _ in range(warmup):
        fn(); torch.cuda.synchronize()
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(runs):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); e.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2], torch.cuda.max_memory_allocated() / 1e9


def naive_fp32(Q, D):
    sim = torch.einsum('qid,bjd->qbij', Q.float(), D.float())
    return sim.max(dim=-1).values.sum(dim=-1)


gpu = torch.cuda.get_device_name()
print(f"GPU: {gpu}", flush=True)
out = {"gpu": gpu, "shapes": []}
for name, B, Lq, Ld in SHAPES:
    print(f"\n=== {name} (B={B}, Lq={Lq}, Ld={Ld}) ===", flush=True)
    for d in D_VALUES:
        cell = {"shape": name, "B": B, "Lq": Lq, "Ld": Ld, "d": d}
        d_tile_bytes = d * 2 * 2 * 64  # rough SRAM lower bound: 2 tiles of [64, d] fp16
        cell["d_tile_estimate_kb"] = round(d_tile_bytes / 1024, 1)
        try:
            Q = F.normalize(torch.randn(1, Lq, d, device='cuda', dtype=torch.float16), dim=-1)
            D = F.normalize(torch.randn(B, Ld, d, device='cuda', dtype=torch.float16), dim=-1)
            # forward only — that's where the d-tile SRAM constraint hits hardest
            flash_ms, flash_peak = _bench(lambda: flash_maxsim_batched(Q, D))
            # correctness vs naive
            ref = naive_fp32(Q, D)
            got = flash_maxsim_batched(Q, D)
            max_abs = float((ref - got).abs().max())
            cell["flash_ms"] = round(flash_ms, 3)
            cell["flash_peak_gb"] = round(flash_peak, 3)
            cell["max_abs_delta_vs_fp32"] = round(max_abs, 4)
            cell["status"] = "ok"
            print(f"  d={d:>4d} (est tile {cell['d_tile_estimate_kb']:>6.1f} KB): "
                  f"flash={flash_ms:.3f}ms  peak={flash_peak:.2f}GB  |Δ|={max_abs:.2e}",
                  flush=True)
            del Q, D, ref, got; torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as e:
            cell["status"] = "OOM"; cell["error"] = str(e)[:120]
            print(f"  d={d:>4d}: OOM", flush=True); torch.cuda.empty_cache()
        except RuntimeError as e:
            cell["status"] = "RuntimeError"; cell["error"] = str(e)[:200]
            print(f"  d={d:>4d}: RuntimeError: {str(e)[:120]}", flush=True); torch.cuda.empty_cache()
        except Exception as e:
            cell["status"] = type(e).__name__; cell["error"] = str(e)[:200]
            print(f"  d={d:>4d}: {type(e).__name__}: {str(e)[:120]}", flush=True); torch.cuda.empty_cache()
        out["shapes"].append(cell)

p = os.path.join(HERE, f"bench_fat_embeddings_{gpu.replace(' ', '_').replace('/', '_')}.json")
json.dump(out, open(p, "w"), indent=2)
print(f"\nwrote {p}", flush=True)
