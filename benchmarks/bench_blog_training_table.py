"""Refresh the blog's training table: all 6 shapes, fwd+bwd, matched precision.

Same shapes the blog uses (lines 217-223 of blog/draft.md):
  ColBERT B=128, 1024, 2048   (Lq=32,  Ld=180,  d=128)
  ColPali B=32, 64, 128       (Lq=1024, Ld=1024, d=128)

Output: bench_blog_training_table_<GPU>.json next to this script.
"""
from __future__ import annotations
import gc, json, os, time
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))

SHAPES = [
    ("ColBERT B=128",  128, 32, 180),
    ("ColBERT B=1024", 1024, 32, 180),
    ("ColBERT B=2048", 2048, 32, 180),
    ("ColPali B=32",   32, 1024, 1024),
    ("ColPali B=64",   64, 1024, 1024),
    ("ColPali B=128",  128, 1024, 1024),
]


def _reset():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def naive_step(Q, D):
    sim = torch.einsum('nqd,bld->nbql', Q.float(), D.float())
    return sim.max(dim=-1).values.sum(dim=-1)


def time_step(make_step, iters=5, warmup=2):
    try:
        for _ in range(warmup):
            make_step()
        torch.cuda.synchronize(); _reset()
        ts = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(); make_step(); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        ts.sort()
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        return {"ms": round(ts[len(ts)//2], 2), "peak_gb": round(peak, 3)}
    except torch.cuda.OutOfMemoryError:
        _reset()
        return {"ms": None, "peak_gb": None, "error": "OutOfMemoryError"}


def main():
    from flash_maxsim import flash_maxsim_batched_train

    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}  torch={torch.__version__}", flush=True)
    rows = []
    for name, B, Lq, Ld in SHAPES:
        print(f"\n=== {name}  (Lq={Lq}, Ld={Ld}) ===", flush=True)

        # naive step (fwd+bwd via PyTorch autograd, matched FP32 precision)
        def make_naive():
            torch.manual_seed(0)
            Q = F.normalize(torch.randn(B, Lq, 128, device='cuda', dtype=torch.float16), dim=-1).requires_grad_(True)
            D = F.normalize(torch.randn(B, Ld, 128, device='cuda', dtype=torch.float16), dim=-1).requires_grad_(True)
            scores = naive_step(Q, D)
            scores.sum().backward()
        naive = time_step(make_naive)
        print(f"  naive: {naive}", flush=True)

        # flash step (fwd+bwd via autograd Function)
        def make_flash():
            torch.manual_seed(0)
            Q = F.normalize(torch.randn(B, Lq, 128, device='cuda', dtype=torch.float16), dim=-1).requires_grad_(True)
            D = F.normalize(torch.randn(B, Ld, 128, device='cuda', dtype=torch.float16), dim=-1).requires_grad_(True)
            scores = flash_maxsim_batched_train(Q, D, shared_docs=True)
            scores.sum().backward()
        flash = time_step(make_flash)
        print(f"  flash: {flash}", flush=True)

        row = {"name": name, "B": B, "Lq": Lq, "Ld": Ld, "naive": naive, "flash": flash}
        if naive.get("ms") and flash.get("ms") and naive.get("peak_gb") and flash.get("peak_gb"):
            row["speedup"] = round(naive["ms"] / flash["ms"], 2)
            row["mem_ratio"] = round(naive["peak_gb"] / flash["peak_gb"], 1)
        rows.append(row)

    out = {"gpu": gpu, "torch_version": torch.__version__, "rows": rows}
    out_path = os.path.join(HERE, f"bench_blog_training_table_{gpu.replace(' ', '_').replace('/', '_')}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
