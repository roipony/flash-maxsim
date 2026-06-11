"""Lock down tab:varlen: speedup of Flash cu_seqlens vs naive padded einsum
across the three named fill-ratio buckets the paper cites.
Buckets (Ld_max=512): uniform [256,512] (ρ≈0.75), HotpotQA-like (ρ≈0.30),
highly ragged (ρ≈0.16). All matched-precision (TF32 'high'), cast hoisted in naive.
"""
import json, os, sys, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from flash_maxsim import flash_maxsim_packed
from utils import bench_interleaved

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True

Lq, d, B, Ld_max = 32, 128, 1000, 512


def make_lengths(B, Ld_max, kind, seed=0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    if kind == "uniform":
        return torch.randint(256, Ld_max + 1, (B,), generator=g)
    if kind == "hotpotqa":
        ln = torch.distributions.LogNormal(torch.tensor(4.6), torch.tensor(0.7)).sample((B,))
        return torch.clamp(ln.long(), min=10, max=Ld_max)
    # ragged
    ln = torch.distributions.LogNormal(torch.tensor(3.9), torch.tensor(0.9)).sample((B,))
    return torch.clamp(ln.long(), min=10, max=Ld_max)


def naive_padded_fp32(Qf, Df, doc_lengths):
    """Matched-precision (FP32, TF32) naive: einsum on padded D, mask, reduce."""
    sim = torch.einsum("qd,bld->bql", Qf, Df)
    ar = torch.arange(Df.shape[1], device=Df.device)
    keep = ar[None, :] < doc_lengths[:, None]
    sim = sim.masked_fill(~keep[:, None, :], float("-inf"))
    return sim.max(dim=2).values.sum(dim=1)


gpu = torch.cuda.get_device_name()
print(f"GPU: {gpu}")
out = {"gpu": gpu, "Lq": Lq, "d": d, "Ld_max": Ld_max, "B": B, "buckets": []}

for kind, label in [("uniform", "uniform [256,512]"),
                    ("hotpotqa", "HotpotQA-like"),
                    ("ragged", "highly ragged")]:
    print(f"\n=== {kind} ===")
    lengths = make_lengths(B, Ld_max, kind)
    rho = float(lengths.sum().item()) / (B * Ld_max)
    print(f"  ρ={rho:.3f} mean={lengths.float().mean():.1f} max={lengths.max():d}")

    Q = F.normalize(torch.randn(Lq, d, device='cuda', dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(B, Ld_max, d, device='cuda', dtype=torch.float16), dim=-1)
    doc_lengths = lengths.to('cuda', dtype=torch.int32)
    Qf, Df = Q.float(), D.float()
    (naive_ms,) = bench_interleaved([naive_padded_fp32], [[Qf, Df, doc_lengths]])

    total = int(lengths.sum().item())
    D_pack = torch.empty(total, d, device='cuda', dtype=torch.float16)
    off = 0
    for i, L in enumerate(lengths.tolist()):
        D_pack[off:off + L] = D[i, :L]
        off += L
    cu = torch.cat([torch.zeros(1, dtype=torch.int32, device='cuda'),
                    lengths.cumsum(0).to(torch.int32).to('cuda')])
    msd = int(lengths.max().item())
    (flash_ms,) = bench_interleaved([flash_maxsim_packed], [[Q, D_pack, cu, msd]])

    spd = float(naive_ms) / float(flash_ms)
    rec = {"bucket": kind, "label": label, "B": B, "Ld_max": Ld_max,
           "avg_ld": round(lengths.float().mean().item(), 1),
           "max_ld": int(lengths.max().item()),
           "rho": round(rho, 3),
           "naive_ms": round(float(naive_ms), 3),
           "flash_ms": round(float(flash_ms), 3),
           "speedup": round(spd, 2)}
    out["buckets"].append(rec)
    print(f"  naive={naive_ms:.2f}ms flash={flash_ms:.2f}ms  speedup={spd:.2f}×")
    del Q, D, Qf, Df, D_pack, doc_lengths, cu; torch.cuda.empty_cache()

p = os.path.join(HERE, f"bench_varlen_buckets_{gpu.replace(' ', '_').replace('/', '_')}.json")
json.dump(out, open(p, "w"), indent=2)
print("wrote", p)
