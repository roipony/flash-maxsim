"""Per-shape sweep of (num_warps, num_stages) configs for the unified
backward kernel — used to build the deterministic per-arch heuristic
table that replaces @triton.autotune.

Inspired by Flash-KMeans's `_heuristic_euclid_config` pattern (paper §4.3,
github.com/svg-project/flash-kmeans/.../assign_euclid_triton.py:568-598).
The goal: pick the optimal (num_warps, num_stages) for every shape at
package-build time, ship the table, eliminate Triton's autotune-trial
overhead on first call.

For each shape we time every config and emit the best for the per-arch
table. Bypasses our @triton.autotune by monkey-patching it to a single
fixed config per run.
"""
import json, os, sys, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import triton, importlib
from flash_maxsim import flash_maxsim_batched_train
# The MODULE is shadowed by the same-named function in `flash_maxsim/__init__.py`;
# importlib bypasses the attribute lookup and grabs the real module.
fm_train = importlib.import_module("flash_maxsim.flash_maxsim_batched_train")
assert hasattr(fm_train, "_maxsim_bwd_unified_kernel"), "module not loaded"

torch.set_float32_matmul_precision('high')
d = 128

SHAPES = [
    ("ColBERT B=128",   128, 32,  180),
    ("ColBERT B=512",   512, 32,  180),
    ("ColBERT B=1024",  1024, 32,  180),
    ("ColPali B=32",    32,  1024, 1024),
    ("ColPali B=64",    64,  1024, 1024),
    ("ColPali B=128",   128, 1024, 1024),
]

CONFIGS = [
    (nw, ns) for nw in [1, 2, 4, 8] for ns in [2, 3]
]


def make(B, Lq, Ld):
    Q = F.normalize(torch.randn(B, Lq, d, device='cuda', dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(B, Ld, d, device='cuda', dtype=torch.float16), dim=-1)
    return Q, D


def _bench(Q, D, runs=30, warmup=5):
    for _ in range(warmup):
        Q.grad = None; D.grad = None
        flash_maxsim_batched_train(Q, D, shared_docs=True).sum().backward()
        torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(runs):
        Q.grad = None; D.grad = None
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        flash_maxsim_batched_train(Q, D, shared_docs=True).sum().backward()
        e.record(); e.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2], torch.cuda.max_memory_allocated() / 1e9


_ORIG_KERNEL = None

def patch_unified_kernel(nw, ns):
    """Monkey-patch the autotune-decorated kernel with a single-config one
    so we can measure each (num_warps, num_stages) in isolation."""
    global _ORIG_KERNEL
    if _ORIG_KERNEL is None:
        _ORIG_KERNEL = fm_train._maxsim_bwd_unified_kernel
    # The autotune wrapper's .fn attribute is the underlying JITFunction.
    inner = _ORIG_KERNEL.fn
    single = triton.autotune(
        configs=[triton.Config({}, num_warps=nw, num_stages=ns)],
        key=["Nq", "B", "Lq", "d_pad"],
    )(inner)
    fm_train._maxsim_bwd_unified_kernel = single


gpu = torch.cuda.get_device_name()
os.environ["FLASH_BWD_PATH"] = "atomic_unified"
print(f"GPU: {gpu}", flush=True)
out = {"gpu": gpu, "shapes": []}
for name, B, Lq, Ld in SHAPES:
    print(f"\n=== {name} (B={B}, Lq={Lq}, Ld={Ld}) ===", flush=True)
    shape_rec = {"name": name, "B": B, "Lq": Lq, "Ld": Ld, "results": {}}
    best_ms = 1e18; best_cfg = None
    for nw, ns in CONFIGS:
        try:
            patch_unified_kernel(nw, ns)
            Q, D = make(B, Lq, Ld); Q.requires_grad_(True); D.requires_grad_(True)
            t, mem = _bench(Q, D)
            shape_rec["results"][f"w{nw}_s{ns}"] = {"ms": round(t, 3), "peak_gb": round(mem, 3)}
            mark = ""
            if t < best_ms:
                best_ms = t; best_cfg = (nw, ns); mark = " ★"
            print(f"  w={nw}, s={ns}: {t:.3f}ms / {mem:.3f}GB{mark}", flush=True)
            del Q, D; torch.cuda.empty_cache()
        except Exception as e:
            shape_rec["results"][f"w{nw}_s{ns}"] = {"error": type(e).__name__}
            print(f"  w={nw}, s={ns}: FAILED ({type(e).__name__})", flush=True)
    shape_rec["best_config"] = {"num_warps": best_cfg[0], "num_stages": best_cfg[1], "ms": round(best_ms, 3)}
    print(f"  → BEST: num_warps={best_cfg[0]}, num_stages={best_cfg[1]} ({best_ms:.3f}ms)", flush=True)
    out["shapes"].append(shape_rec)

p = os.path.join(HERE, f"bench_config_sweep_{gpu.replace(' ', '_').replace('/', '_')}.json")
json.dump(out, open(p, "w"), indent=2)
print(f"\nwrote {p}", flush=True)
print("\n=== Per-shape best configs (paste into heuristic table) ===")
for s in out["shapes"]:
    bc = s["best_config"]
    print(f'  ({s["B"]}, {s["Lq"]}, {s["Ld"]}): num_warps={bc["num_warps"]}, num_stages={bc["num_stages"]},  # {s["name"]} {bc["ms"]}ms')
