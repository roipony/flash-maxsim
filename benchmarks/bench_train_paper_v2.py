"""tab:train re-measurement with the backward path labeled per row.

Re-review P2: the original bench_train_paper.py ran under the
then-default invgrid-CSR backward (0.40 / 1.43 GB peaks); the
dispatcher default is now atomic-unified. This script measures, in
ONE campaign, at ColPali Nq=B in-batch negatives:

  naive          : full autograd FP32 einsum (OOMs at B=128)
  flash-unified  : FLASH_BWD_PATH=atomic_unified (dispatcher default)
  flash-invgrid  : FLASH_BWD_PATH=invgrid (opt-in deterministic CSR)

Protocol identical to bench_train_paper.py: sum().backward(), 5
warmup, 20 CUDA-event medians, per-method peak via
reset_peak_memory_stats.

Output: bench_train_paper_v2_<GPU>.json
"""
import json, os, sys
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from flash_maxsim import flash_maxsim_batched_train

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
Lq = Ld = 1024
d = 128
RUNS, WARMUP = 20, 5


def make(B):
    Q = F.normalize(torch.randn(B, Lq, d, device='cuda', dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(B, Ld, d, device='cuda', dtype=torch.float16), dim=-1)
    Q.requires_grad_(True); D.requires_grad_(True)
    return Q, D


def step_naive(Q, D):
    Q.grad = None; D.grad = None
    sim = torch.einsum("nqd,bld->nbql", Q.float(), D.float())
    sim.max(dim=3).values.sum(dim=2).sum().backward()


def step_flash(Q, D):
    Q.grad = None; D.grad = None
    flash_maxsim_batched_train(Q, D, shared_docs=True).sum().backward()


def measure(fn, B, env_path=None):
    if env_path is None:
        os.environ.pop("FLASH_BWD_PATH", None)
    else:
        os.environ["FLASH_BWD_PATH"] = env_path
    try:
        Q, D = make(B)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        for _ in range(WARMUP):
            fn(Q, D); torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(RUNS):
            s.record(); fn(Q, D); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        ts.sort()
        del Q, D
        torch.cuda.empty_cache()
        return {"ms": round(ts[len(ts)//2], 2),
                "peak_gb": round(torch.cuda.max_memory_allocated()/1e9, 3), "ok": True}
    except torch.cuda.OutOfMemoryError as ex:
        torch.cuda.empty_cache()
        return {"ok": False, "error": str(ex)[:100]}
    finally:
        os.environ.pop("FLASH_BWD_PATH", None)


def main():
    gpu = torch.cuda.get_device_name()
    print(f"GPU: {gpu}", flush=True)
    out = {"gpu": gpu, "Lq": Lq, "Ld": Ld, "protocol":
           "sum().backward(), 5 warmup + 20 CUDA-event medians, fresh peak stats per method",
           "rows": []}
    for B in (64, 128):
        print(f"\n=== Nq=B={B} ===", flush=True)
        row = {"B": B}
        row["naive"] = measure(step_naive, B)
        print(f"  naive:         {row['naive']}", flush=True)
        row["flash_unified"] = measure(step_flash, B, "atomic_unified")
        print(f"  flash-unified: {row['flash_unified']}", flush=True)
        row["flash_invgrid"] = measure(step_flash, B, "invgrid")
        print(f"  flash-invgrid: {row['flash_invgrid']}", flush=True)
        if row["naive"].get("ok") and row["flash_unified"].get("ok"):
            row["speedup_unified"] = round(row["naive"]["ms"] / row["flash_unified"]["ms"], 2)
            row["mem_ratio_unified"] = round(row["naive"]["peak_gb"] / row["flash_unified"]["peak_gb"], 1)
            print(f"  -> unified: {row['speedup_unified']}x speed, {row['mem_ratio_unified']}x mem", flush=True)
        out["rows"].append(row)
    op = os.path.join(HERE, f"bench_train_paper_v2_{gpu.replace(' ', '_').replace('/', '_')}.json")
    json.dump(out, open(op, "w"), indent=2)
    print("\nwrote", op, flush=True)


if __name__ == '__main__':
    main()
