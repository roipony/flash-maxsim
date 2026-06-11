"""Dispersion (median + IQR) for the paper's headline latency cells.

Reviewer ask: medians alone hide tail behaviour. Reports p25/p50/p75
and IQR/median for: flash, naive FP16 eager, naive FP32 (TF32), and
chunked FP16 eager (best chunk from bench_chunked_fp16_eager) at the
two headline shapes: ColPali B=1K and ColPali B=10K.

Output: bench_dispersion_<GPU>.json
"""
import json, os, sys
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from flash_maxsim import flash_maxsim

RUNS, WARMUP = 50, 5
d = 128


def make(Lq, Ld, B):
    Q = F.normalize(torch.randn(Lq, d, device='cuda', dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(B, Ld, d, device='cuda', dtype=torch.float16), dim=-1)
    return Q, D


def naive_fp16(Q, D):
    return torch.einsum("qd,bld->bql", Q, D).max(2).values.sum(1)


def naive_fp32(Qf, Df):
    return torch.einsum("qd,bld->bql", Qf, Df).max(2).values.sum(1)


def chunked_fp16(Q, D, chunk):
    B = D.shape[0]
    out = torch.empty(B, device=Q.device, dtype=torch.float16)
    for i in range(0, B, chunk):
        S = torch.einsum("qd,bld->bql", Q, D[i:i+chunk])
        out[i:i+chunk] = S.max(2).values.sum(1)
    return out


def samples(fn, args):
    for _ in range(WARMUP):
        fn(*args); torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(RUNS):
        s.record(); fn(*args); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    def pct(p):
        i = p * (len(ts) - 1)
        lo, hi = int(i), min(int(i) + 1, len(ts) - 1)
        return ts[lo] + (ts[hi] - ts[lo]) * (i - lo)
    p25, p50, p75 = pct(.25), pct(.50), pct(.75)
    return {"p25": round(p25, 4), "p50": round(p50, 4), "p75": round(p75, 4),
            "iqr": round(p75 - p25, 4),
            "iqr_over_median_pct": round(100 * (p75 - p25) / p50, 2),
            "min": round(ts[0], 4), "max": round(ts[-1], 4)}


def main():
    gpu = torch.cuda.get_device_name()
    print(f"GPU: {gpu}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    out = {"gpu": gpu, "runs": RUNS, "warmup": WARMUP, "cells": []}

    for B, chunk in [(1000, 256), (10000, 1024)]:
        Q, D = make(1024, 1024, B)
        Qf, Df = Q.float(), D.float()
        cell = {"shape": "ColPali", "B": B}
        cell["flash"] = samples(flash_maxsim, [Q, D])
        cell["naive_fp16"] = samples(naive_fp16, [Q, D]) if B <= 1000 else None
        if B <= 1000:
            cell["naive_fp32_tf32"] = samples(naive_fp32, [Qf, Df])
        cell["chunked_fp16"] = samples(chunked_fp16, [Q, D, chunk])
        cell["chunk"] = chunk
        out["cells"].append(cell)
        for k, v in cell.items():
            if isinstance(v, dict):
                print(f"  B={B} {k:16s} p50={v['p50']:8.3f}  IQR={v['iqr']:7.3f}  "
                      f"({v['iqr_over_median_pct']:.2f}% of median)")
        del Q, D, Qf, Df
        torch.cuda.empty_cache()

    op = os.path.join(HERE, f"bench_dispersion_{gpu.replace(' ', '_').replace('/', '_')}.json")
    json.dump(out, open(op, "w"), indent=2)
    print("wrote", op)


if __name__ == '__main__':
    main()
