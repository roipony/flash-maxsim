"""Fair forward-latency re-measurement for the paper's tab:fwd.

Fixes the cast-inside-the-timed-region artifact (old `naive_fp32` cast
`Q.float()/D.float()` INSIDE the timed call, charging a full-tensor HBM cast
to every iteration). All casts here are hoisted OUT of the timed region.

Measured per shape (B=1000):
  flash            : FP16 in, FP32 accumulation, fused (Triton)
  naive_fp16       : FP16 einsum + reduce (FP16 accumulation, lower accuracy)
  naive_fp32_tf32  : cast once, FP32 einsum with TF32 tensor cores (precision 'high')
  naive_fp32_true  : cast once, true FP32 matmul, TF32 OFF (precision 'highest')
                     -- this is the accuracy-matched baseline (matches flash's
                        FP32-reference quality), so it is the honest "matched
                        precision" comparison.
"""
import json, os, sys, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from flash_maxsim import flash_maxsim
from utils import bench_interleaved

B = 1000
SHAPES = [("textual", 32, 300), ("long-doc", 32, 1024), ("medium", 128, 1024),
          ("visual", 512, 1024), ("ColPali", 1024, 1024)]


def make(B, Lq, Ld, d=128):
    Q = F.normalize(torch.randn(Lq, d, device="cuda", dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(B, Ld, d, device="cuda", dtype=torch.float16), dim=-1)
    return Q, D


def n_fp16(Q, D):
    return torch.einsum("qd,bld->bql", Q, D).max(2).values.sum(1)

def n_fp32(Qf, Df):  # already FP32; TF32 behaviour set by global flag
    return torch.einsum("qd,bld->bql", Qf, Df).max(2).values.sum(1)


def tf32(on):
    torch.backends.cuda.matmul.allow_tf32 = on
    torch.backends.cudnn.allow_tf32 = on
    torch.set_float32_matmul_precision('high' if on else 'highest')


gpu = torch.cuda.get_device_name()
print("GPU:", gpu)
rows = []
for nm, Lq, Ld in SHAPES:
    Q, D = make(B, Lq, Ld)
    Qf, Df = Q.float(), D.float()
    tf32(True)
    fp16_ms, flash_ms = bench_interleaved([n_fp16, flash_maxsim], [[Q, D]])
    (fp32_tf32_ms,) = bench_interleaved([n_fp32], [[Qf, Df]])
    tf32(False)
    (fp32_true_ms,) = bench_interleaved([n_fp32], [[Qf, Df]])
    r = {"shape": nm, "Lq": Lq, "Ld": Ld, "B": B,
         "flash_ms": round(float(flash_ms), 3),
         "naive_fp16_ms": round(float(fp16_ms), 3),
         "naive_fp32_tf32_ms": round(float(fp32_tf32_ms), 3),
         "naive_fp32_true_ms": round(float(fp32_true_ms), 3),
         "spd_fp16": round(float(fp16_ms / flash_ms), 2),
         "spd_fp32_tf32": round(float(fp32_tf32_ms / flash_ms), 2),
         "spd_fp32_true": round(float(fp32_true_ms / flash_ms), 2)}
    rows.append(r)
    print(f"  {nm:9s} | flash={r['flash_ms']:.2f}  fp16={r['naive_fp16_ms']:.2f}({r['spd_fp16']}x)  "
          f"fp32_tf32={r['naive_fp32_tf32_ms']:.2f}({r['spd_fp32_tf32']}x)  "
          f"fp32_true={r['naive_fp32_true_ms']:.2f}({r['spd_fp32_true']}x)")

out = os.path.join(HERE, f"bench_fwd_fair_{gpu.replace(' ', '_').replace('/', '_')}.json")
json.dump({"gpu": gpu, "rows": rows}, open(out, "w"), indent=2)
print("wrote", out)
