"""Backward-path 3-way ablation: CSR (invgrid) / atomic (plain) / atomic_unified.

Refreshes the tab:bwd_abl row data, adding the atomic_unified column
that became the production default after the dispatcher pin.

Toggles via FLASH_BWD_PATH={invgrid, atomic, atomic_unified} env var on
the same 6 training shapes as the existing tab:bwd_abl, on H100 to match
the prior measurement context.

Output: bench_bwd_paths_3way_<GPU>.json next to this script.
"""
from __future__ import annotations
import gc, json, os, time
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))

SHAPES = [
    ("ColBERT B=128",  128,  32,   180),
    ("ColBERT B=512",  512,  32,   180),
    ("ColBERT B=1024", 1024, 32,   180),
    ("ColPali B=32",   32,   1024, 1024),
    ("ColPali B=64",   64,   1024, 1024),
    ("ColPali B=128",  128,  1024, 1024),
]

# The three backward paths we want to compare (the dispatcher exposes
# FLASH_BWD_PATH for this exact purpose).
PATHS = ["invgrid", "atomic", "atomic_unified"]


def _reset():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def _time_step(make_step, iters=5, warmup=2):
    try:
        for _ in range(warmup): make_step()
        torch.cuda.synchronize(); _reset()
        ts = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(); make_step(); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        ts.sort()
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        return {"ms": round(ts[len(ts)//2], 3), "peak_gb": round(peak, 3)}
    except torch.cuda.OutOfMemoryError:
        _reset()
        return {"ms": None, "peak_gb": None, "error": "OOM"}
    except Exception as e:
        _reset()
        return {"ms": None, "peak_gb": None, "error": f"{type(e).__name__}: {str(e)[:120]}"}


def main():
    # Import after we touch the env var to make sure the dispatcher reads it.
    from flash_maxsim import flash_maxsim_batched_train
    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}  torch={torch.__version__}", flush=True)

    rows = []
    for name, B, Lq, Ld in SHAPES:
        print(f"\n=== {name}  (Lq={Lq}, Ld={Ld}) ===", flush=True)
        row = {"name": name, "B": B, "Lq": Lq, "Ld": Ld}
        for path in PATHS:
            os.environ["FLASH_BWD_PATH"] = path
            def step():
                torch.manual_seed(0)
                Q = F.normalize(torch.randn(B, Lq, 128, device='cuda', dtype=torch.float16), dim=-1).requires_grad_(True)
                D = F.normalize(torch.randn(B, Ld, 128, device='cuda', dtype=torch.float16), dim=-1).requires_grad_(True)
                scores = flash_maxsim_batched_train(Q, D, shared_docs=True)
                scores.sum().backward()
            r = _time_step(step)
            row[path] = r
            print(f"  {path:>16s}: {r}", flush=True)
        rows.append(row)

    out = {"gpu": gpu, "torch_version": torch.__version__, "paths": PATHS, "rows": rows}
    out_path = os.path.join(HERE, f"bench_bwd_paths_3way_{gpu.replace(' ', '_').replace('/', '_')}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
