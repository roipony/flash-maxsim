"""Chunked-recompute training baseline (reviewer Q2 / re-review item 3).

The memory-matched alternative to the fused backward: gradient
checkpointing at the operator level. The corpus is split into doc
chunks; each chunk's einsum+max+sum runs under
torch.utils.checkpoint, so the [Nq, B_chunk, Lq, Ld] tensor is
re-materialised (twice: forward + recompute-in-backward) but never
held for more than one chunk at a time.

Measures, at ColPali contrastive shape (shared docs, Nq = B):
  naive     : full autograd einsum (the tab:train baseline)
  recompute : chunked checkpoint, chunk size swept, best reported
  flash     : flash_maxsim_batched_train (atomic-unified default)

step = forward + loss.backward() at the MaxSim-operator level
(loss = scores.mean(); matched precision: FP16 inputs / FP32 scores).

Output: bench_recompute_train_<GPU>.json
"""
import json, os, sys
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from flash_maxsim import flash_maxsim_batched_train

torch.set_float32_matmul_precision('high')
d = 128
Lq = Ld = 1024
RUNS, WARMUP = 10, 3
CHUNKS = [4, 8, 16, 32]


def make(B):
    Q = F.normalize(torch.randn(B, Lq, d, device='cuda', dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(B, Ld, d, device='cuda', dtype=torch.float16), dim=-1)
    return Q, D


def naive_step(Q, D):
    S = torch.einsum('aqd,bld->abql', Q.float(), D.float())
    scores = S.max(dim=3).values.sum(dim=2)
    scores.mean().backward()


def _chunk_scores(Q, D_chunk):
    S = torch.einsum('aqd,bld->abql', Q.float(), D_chunk.float())
    return S.max(dim=3).values.sum(dim=2)


def recompute_step(Q, D, chunk):
    parts = []
    for i in range(0, D.shape[0], chunk):
        parts.append(checkpoint(_chunk_scores, Q, D[i:i+chunk],
                                use_reentrant=False))
    torch.cat(parts, dim=1).mean().backward()


def flash_step(Q, D):
    scores = flash_maxsim_batched_train(Q, D, shared_docs=True)
    scores.mean().backward()


def measure(fn, B, *extra):
    try:
        Q, D = make(B)
        Q.requires_grad_(True); D.requires_grad_(True)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        for _ in range(WARMUP):
            fn(Q, D, *extra); torch.cuda.synchronize()
            Q.grad = None; D.grad = None
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(RUNS):
            s.record(); fn(Q, D, *extra); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
            Q.grad = None; D.grad = None
        ts.sort()
        del Q, D
        torch.cuda.empty_cache()
        return {"ms": round(ts[len(ts)//2], 2),
                "peak_gb": round(torch.cuda.max_memory_allocated()/1e9, 3), "ok": True}
    except torch.cuda.OutOfMemoryError as ex:
        torch.cuda.empty_cache()
        return {"ok": False, "error": str(ex)[:100]}


def main():
    gpu = torch.cuda.get_device_name()
    print(f"GPU: {gpu}", flush=True)
    out = {"gpu": gpu, "Lq": Lq, "Ld": Ld, "d": d, "rows": []}
    for B in (64, 128):
        print(f"\n=== ColPali contrastive, Nq=B={B} ===", flush=True)
        row = {"B": B}
        row["naive"] = measure(naive_step, B)
        print(f"  naive:     {row['naive']}", flush=True)
        best = None
        for c in CHUNKS:
            r = measure(recompute_step, B, c); r["chunk"] = c
            print(f"  recompute chunk={c:3d}: {r}", flush=True)
            if r.get("ok") and (best is None or r["ms"] < best["ms"]):
                best = r
        row["recompute_best"] = best
        row["flash"] = measure(flash_step, B)
        print(f"  flash:     {row['flash']}", flush=True)
        if best and best.get("ok") and row["flash"].get("ok"):
            row["flash_vs_recompute"] = round(best["ms"] / row["flash"]["ms"], 2)
            row["mem_ratio"] = round(best["peak_gb"] / max(row["flash"]["peak_gb"], 1e-9), 1)
            print(f"  -> flash {row['flash_vs_recompute']}x faster than best recompute, "
                  f"{row['mem_ratio']}x less peak", flush=True)
        out["rows"].append(row)
    op = os.path.join(HERE, f"bench_recompute_train_{gpu.replace(' ', '_').replace('/', '_')}.json")
    json.dump(out, open(op, "w"), indent=2)
    print("\nwrote", op, flush=True)


if __name__ == '__main__':
    main()
