"""E4 — itemized reconciliation of analytic vs measured training memory.

EUF4 W2/Q2: (a) analytic ~1000x vs measured 217x; (b) FM kernel measured
6-12x above its analytic transient; (c) CSR measured below its analytic
"lower bound". This script reproduces the tab:train protocol (ColPali scale,
Lq=Ld=1024, d=128, fp16 embeddings, Nq=B in-batch) and decomposes the
measured peak into named parts so every gap has an owner:

  baseline   : Q + D embeddings resident (inputs; excluded from analytic)
  grads      : Q.grad + D.grad allocated by autograd (excluded from analytic)
  argmax     : saved int32/int64 winner map (the analytic FM budget)
  transient  : peak - (baseline + grads + argmax)  [kernel workspace, CSR
               build, allocator rounding]

Runs vanilla / flash-unified / flash-invgrid at B=64 and B=128.
Output: analysis/e1_results/e4_mem_audit.json
"""
from __future__ import annotations
import gc, json, os, sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

Lq = Ld = 1024
D_MODEL = 128
GB = 1e9


def make(B):
    Q = F.normalize(torch.randn(B, Lq, D_MODEL, device="cuda",
                                dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(B, Ld, D_MODEL, device="cuda",
                                dtype=torch.float16), dim=-1)
    Q.requires_grad_(True); D.requires_grad_(True)
    return Q, D


def run_method(B: int, method: str) -> dict:
    from flash_maxsim import flash_maxsim_batched_train

    gc.collect(); torch.cuda.empty_cache()
    base0 = torch.cuda.memory_allocated()
    Q, D = make(B)
    baseline = torch.cuda.memory_allocated() - base0    # Q + D bytes

    def step():
        Q.grad = None; D.grad = None
        if method == "vanilla":
            sim = torch.einsum("nqd,bld->nbql", Q.float(), D.float())
            sim.max(dim=3).values.sum(dim=2).sum().backward()
        else:
            os.environ["FLASH_BWD_PATH"] = (
                "atomic_unified" if method == "flash-unified" else "invgrid")
            s = flash_maxsim_batched_train(Q, D, shared_docs=True)
            s.sum().backward()

    try:
        step()                                          # warmup / autotune
        Q.grad = None; D.grad = None
        gc.collect(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        step()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        grads = Q.grad.numel() * Q.grad.element_size() \
              + D.grad.numel() * D.grad.element_size()
        # analytic sizes at this shape
        s_fp32 = B * B * Lq * Ld * 4                    # S (vanilla, fp32)
        argmax_i32 = B * B * Lq * 4                     # winner map
        res = {
            "ok": True,
            "peak_gb": round(peak / GB, 4),
            "baseline_QD_gb": round(baseline / GB, 4),
            "grads_gb": round(grads / GB, 4),
            "analytic_S_gb": round(s_fp32 / GB, 4),
            "analytic_argmax_gb": round(argmax_i32 / GB, 4),
            "residual_transient_gb": round(
                (peak - baseline - grads
                 - (argmax_i32 if method != "vanilla" else 0)) / GB, 4),
        }
    except torch.cuda.OutOfMemoryError:
        res = {"ok": False, "oom": True,
               "baseline_QD_gb": round(baseline / GB, 4)}
    finally:
        Q = D = None
        gc.collect(); torch.cuda.empty_cache()
    return res


def main():
    out = {"gpu": torch.cuda.get_device_name(0),
           "shape": {"Lq": Lq, "Ld": Ld, "d": D_MODEL, "Nq": "=B"},
           "rows": []}
    for B in (64, 128):
        for method in ("flash-unified", "flash-invgrid", "vanilla"):
            r = run_method(B, method)
            r.update({"B": B, "method": method})
            out["rows"].append(r)
            print(json.dumps(r), flush=True)
    path = HERE / "e1_results" / "e4_mem_audit.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
