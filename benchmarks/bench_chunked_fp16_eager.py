"""Chunked FP16 eager — the production-default inference baseline.

Reviewer asks for the "actual default in deployed ColBERT/ColPali stacks":
plain FP16 eager einsum with corpus chunking, no torch.compile, no FP32
promotion. This is what removes the materialised-tensor OOM without
graph compilation.

For each canonical (shape, B) cell, sweeps chunk sizes and reports the
BEST (lowest-latency, non-OOM) chunked-FP16-eager latency, plus the
Flash latency for direct comparison. Output is one row per cell, with
the best chunk size identified so the table can cite it.

Shape grid mirrors:
  - tab:fwd (textual, long-doc, medium, visual, ColPali) at B=1K
  - App. A OOM frontier (Visual, ColPali) at B={5K, 10K, 20K}

Output: bench_chunked_fp16_eager_<GPU>.json
"""
import json, os, sys, time
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from flash_maxsim import flash_maxsim

# (label, Lq, Ld, B)
CELLS = [
    # tab:fwd row expansion (B=1K, all 5 shapes)
    ("textual",  32,   300,  1000),
    ("long-doc", 32,   1024, 1000),
    ("medium",   128,  1024, 1000),
    ("visual",   512,  1024, 1000),
    ("ColPali",  1024, 1024, 1000),
    # OOM-frontier expansion (ColPali / visual at growing B)
    ("ColPali",  1024, 1024, 5000),
    ("ColPali",  1024, 1024, 10000),
    ("ColPali",  1024, 1024, 20000),
    ("visual",   512,  1024, 5000),
    ("visual",   512,  1024, 10000),
    ("visual",   512,  1024, 20000),
]
d = 128
CHUNK_GRID = [1, 4, 16, 64, 256, 1024, 4096, 16384]


def make(Lq, Ld, B):
    Q = F.normalize(torch.randn(Lq, d, device='cuda', dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(B, Ld, d, device='cuda', dtype=torch.float16), dim=-1)
    return Q, D


def chunked_fp16_eager(Q, D, chunk):
    """Production-default: FP16 eager einsum with corpus chunking,
    NO torch.compile, NO FP32 promotion. FP16 accumulation throughout."""
    B = D.shape[0]
    out = torch.empty(B, device=Q.device, dtype=torch.float16)
    for i in range(0, B, chunk):
        De = D[i:i+chunk]
        S = torch.einsum("qd,bld->bql", Q, De)  # FP16 in/out
        out[i:i+chunk] = S.max(2).values.sum(1)
    return out


def vanilla_fp16_eager(Q, D):
    """Same FP16 einsum but no chunking — materialises full [B,Lq,Ld]."""
    S = torch.einsum("qd,bld->bql", Q, D)
    return S.max(2).values.sum(1)


def measure(fn, args, warmup=3, n=10):
    try:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        for _ in range(warmup):
            _ = fn(*args); torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(n):
            s.record(); _ = fn(*args); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        ts.sort()
        return {"ms": round(float(ts[len(ts)//2]), 3),
                "peak_gb": round(torch.cuda.max_memory_allocated()/1e9, 3),
                "ok": True}
    except (torch.cuda.OutOfMemoryError, RuntimeError, AssertionError) as ex:
        torch.cuda.empty_cache()
        return {"ms": None, "peak_gb": None, "ok": False,
                "error": f"{type(ex).__name__}: {str(ex)[:80]}"}


def main():
    gpu = torch.cuda.get_device_name()
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu} ({vram:.1f} GB)")
    print(f"torch={torch.__version__}\n")

    out = {"gpu": gpu, "vram_gb": round(vram, 1), "torch": torch.__version__,
           "note": "Production-default inference baseline: FP16 eager + corpus "
                   "chunking, no torch.compile, no FP32 promotion. For each cell "
                   "the BEST chunk size (lowest non-OOM median latency) is reported.",
           "rows": []}

    for label, Lq, Ld, B in CELLS:
        d_gb = B * Ld * d * 2 / 1e9
        sim_gb_full = B * Lq * Ld * 2 / 1e9
        print(f"=== {label}  Lq={Lq}  Ld={Ld}  B={B}  (D={d_gb:.2f}GB  "
              f"S_full={sim_gb_full:.2f}GB) ===")
        if d_gb > vram * 0.7:
            print(f"  SKIP: D alone {d_gb:.1f}GB > 70% VRAM")
            out["rows"].append({"label": label, "Lq": Lq, "Ld": Ld, "B": B,
                                "skip": "D > 70% VRAM"})
            continue

        Q, D = make(Lq, Ld, B)

        # Flash
        fl = measure(flash_maxsim, [Q, D])
        print(f"  flash:                 {fl}")

        # Vanilla FP16 eager (no chunking) — usually OOM at large B
        van = measure(vanilla_fp16_eager, [Q, D])
        print(f"  vanilla_fp16_eager:    {van}")

        # Chunked sweep
        chunk_results = []
        best = None
        for chunk in CHUNK_GRID:
            if chunk > B: continue
            sim_chunk_gb = chunk * Lq * Ld * 2 / 1e9
            r = measure(chunked_fp16_eager, [Q, D, chunk])
            r["chunk"] = chunk
            r["sim_chunk_gb"] = round(sim_chunk_gb, 3)
            chunk_results.append(r)
            tag = f"chunk={chunk:>5d}  S_chunk={sim_chunk_gb:.2f}GB"
            if r["ok"]:
                print(f"  chunked_fp16  {tag}  ms={r['ms']:.2f}  peak={r['peak_gb']:.2f}GB")
                if best is None or r["ms"] < best["ms"]:
                    best = r
            else:
                print(f"  chunked_fp16  {tag}  OOM")

        row = {"label": label, "Lq": Lq, "Ld": Ld, "B": B,
               "d_gb": round(d_gb, 3), "s_full_gb": round(sim_gb_full, 3),
               "flash": fl,
               "vanilla_fp16_eager": van,
               "chunked_fp16_eager_sweep": chunk_results,
               "chunked_fp16_eager_best": best}
        if fl["ok"] and best and best["ok"]:
            spd = best["ms"] / fl["ms"]
            row["best_chunk_vs_flash"] = round(spd, 2)
            print(f"  → BEST chunked: chunk={best['chunk']}  {best['ms']:.2f}ms  "
                  f"({spd:.2f}× of Flash {fl['ms']:.2f}ms)")
        out["rows"].append(row)
        del Q, D
        torch.cuda.empty_cache()
        print()

    op = os.path.join(HERE, f"bench_chunked_fp16_eager_{gpu.replace(' ', '_').replace('/', '_')}.json")
    with open(op, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {op}")


if __name__ == '__main__':
    main()
