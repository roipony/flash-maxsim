"""Lock down the four UNVERIFIED correctness claims:
  - Flash max relative error vs FP32 reference (paper says 2e-6)
  - Top-20 ranking overlap (paper says 100%)
  - INT8xINT8 Spearman rho (paper says 0.999)
  - INT8 top-20 overlap (paper says 100%)

Runs at ColPali shape (Lq=Ld=1024, d=128, B=1000) which is the
representative reranking workload from tab:fwd.
"""
import json, os, sys, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from flash_maxsim import (flash_maxsim, flash_maxsim_int8x8,
                          quantize_int8_symmetric, quantize_query_int8)

torch.set_float32_matmul_precision('highest')  # FP32 reference, no TF32 rounding
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

Lq = Ld = 1024
d = 128
B = 1000


def make(B, seed=0):
    g = torch.Generator(device='cuda').manual_seed(seed)
    Q = F.normalize(torch.randn(Lq, d, device='cuda', dtype=torch.float16, generator=g), dim=-1)
    D = F.normalize(torch.randn(B, Ld, d, device='cuda', dtype=torch.float16, generator=g), dim=-1)
    return Q, D


def spearman(a, b):
    """Spearman rank correlation, single batch."""
    ar = a.argsort().argsort().float()
    br = b.argsort().argsort().float()
    return float(torch.corrcoef(torch.stack([ar, br]))[0, 1])


def topk_overlap(a, b, K):
    ta = set(torch.topk(a, K).indices.cpu().tolist())
    tb = set(torch.topk(b, K).indices.cpu().tolist())
    return len(ta & tb) / K


gpu = torch.cuda.get_device_name()
print(f"GPU: {gpu}")
out = {"gpu": gpu, "Lq": Lq, "Ld": Ld, "d": d, "B": B}

# === FP32 reference ===
Q, D = make(B, seed=0)
Qf, Df = Q.float(), D.float()
print("\n=== FP32 reference (dense, true FP32 matmul) ===")
ref = torch.einsum("qd,bld->bql", Qf, Df).max(dim=2).values.sum(dim=1)  # [B] FP32
print(f"  ref range: [{ref.min():.4f}, {ref.max():.4f}]")

# === Flash FP16 (TF32 'high' would be the paper's matched-precision; here we test the kernel itself) ===
print("\n=== Flash FP16 (vs FP32 reference) ===")
flash_score = flash_maxsim(Q, D).float()
abs_err = (flash_score - ref).abs()
rel_err = abs_err / ref.abs().clamp(min=1e-9)
max_abs = float(abs_err.max())
max_rel = float(rel_err.max())
mean_rel = float(rel_err.mean())
top20 = topk_overlap(flash_score, ref, 20)
top50 = topk_overlap(flash_score, ref, 50)
spear = spearman(flash_score, ref)
print(f"  max_abs_err={max_abs:.2e}  max_rel_err={max_rel:.2e}  mean_rel_err={mean_rel:.2e}")
print(f"  top-20 overlap={top20*100:.1f}%  top-50 overlap={top50*100:.1f}%  spearman={spear:.6f}")
out["flash_fp16"] = {"max_abs_err": max_abs, "max_rel_err": max_rel,
                    "mean_rel_err": mean_rel,
                    "top20_overlap": top20, "top50_overlap": top50,
                    "spearman_rho": spear}

# === INT8 x INT8 (with per-token scales) ===
print("\n=== Flash INT8xINT8 (vs FP32 reference) ===")
try:
    D_int8, d_scales = quantize_int8_symmetric(D)
    Q_int8, q_scales = quantize_query_int8(Q)
    int8_score = flash_maxsim_int8x8(Q, D_int8, d_scales,
                                     Q_int8=Q_int8, q_scales=q_scales).float()
    abs_err8 = (int8_score - ref).abs()
    rel_err8 = abs_err8 / ref.abs().clamp(min=1e-9)
    max_abs8 = float(abs_err8.max())
    max_rel8 = float(rel_err8.max())
    mean_rel8 = float(rel_err8.mean())
    top20_8 = topk_overlap(int8_score, ref, 20)
    top50_8 = topk_overlap(int8_score, ref, 50)
    spear8 = spearman(int8_score, ref)
    print(f"  max_abs_err={max_abs8:.2e}  max_rel_err={max_rel8:.2e}  mean_rel_err={mean_rel8:.2e}")
    print(f"  top-20 overlap={top20_8*100:.1f}%  top-50 overlap={top50_8*100:.1f}%  spearman={spear8:.6f}")
    out["flash_int8"] = {"max_abs_err": max_abs8, "max_rel_err": max_rel8,
                        "mean_rel_err": mean_rel8,
                        "top20_overlap": top20_8, "top50_overlap": top50_8,
                        "spearman_rho": spear8}
except Exception as e:
    print(f"  INT8 path FAILED: {type(e).__name__}: {e}")
    out["flash_int8"] = {"error": f"{type(e).__name__}: {str(e)[:150]}"}

p = os.path.join(HERE, f"bench_correctness_{gpu.replace(' ', '_').replace('/', '_')}.json")
json.dump(out, open(p, "w"), indent=2)
print(f"\nwrote {p}")
