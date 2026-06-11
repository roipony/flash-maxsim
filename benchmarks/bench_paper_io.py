"""Empirical IO-validation for the Flash-MaxSim paper (FA-style).

Produces:
  1. traffic   : analytical HBM bytes (naive materialized-S vs flash) + measured
                 latency + achieved bandwidth  -> the "traffic ~ runtime" table.
  2. micro     : ideal matmul-compute time vs measured naive latency
                 (the "compute is cheap, materialization is the cost" example).
  3. backward  : CSR backward (flash_maxsim_train) vs naive autograd backward,
                 latency + effective gradient bandwidth (GB/s).
  4. chunk_abl : query_chunk_size sweep (the exposed tile knob) -> sensitivity/plateau.
All casts hoisted out of timed regions; matched precision (FP32 accumulation, TF32).
"""
import json, os, sys, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from flash_maxsim import flash_maxsim, flash_maxsim_train, maxsim_naive_fp32
from utils import bench_interleaved

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')
d = 128
gpu = torch.cuda.get_device_name()
# peak dense throughput (TF32 tensor-core) for the compute floor, TFLOP/s
PEAK_TFLOPS = 312.0 if "H100" in gpu else 156.0  # A100 TF32 ~156, H100 TF32 ~312(approx for floor)
out = {"gpu": gpu, "traffic": [], "backward": [], "chunk_abl": []}


def make(B, Lq, Ld):
    Q = F.normalize(torch.randn(Lq, d, device='cuda', dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(B, Ld, d, device='cuda', dtype=torch.float16), dim=-1)
    return Q, D

def naive_precast(Qf, Df):
    return torch.einsum("qd,bld->bql", Qf, Df).max(2).values.sum(1)

# ---- 1+2. traffic table + compute-floor micro-example -------------------------
for nm, Lq, Ld, B in [("medium", 128, 1024, 1000), ("visual", 512, 1024, 1000),
                      ("ColPali", 1024, 1024, 1000)]:
    Q, D = make(B, Lq, Ld); Qf, Df = Q.float(), D.float()
    (f_ms,) = bench_interleaved([flash_maxsim], [[Q, D]])
    (n_ms,) = bench_interleaved([naive_precast], [[Qf, Df]])
    in_bytes = B * Ld * d * 2 + Lq * d * 2            # FP16 inputs
    S_bytes = B * Lq * Ld * 4                          # FP32 score tensor
    naive_bytes = 2 * S_bytes + in_bytes               # write S + read S + read inputs
    flash_bytes = in_bytes + B * 4                     # inputs + scalar scores
    flops = 2 * B * Lq * Ld * d
    compute_floor_ms = flops / (PEAK_TFLOPS * 1e12) * 1e3
    out["traffic"].append({
        "shape": nm, "Lq": Lq, "Ld": Ld, "B": B,
        "naive_ms": round(float(n_ms), 3), "flash_ms": round(float(f_ms), 3),
        "naive_GB": round(naive_bytes / 1e9, 2), "flash_GB": round(flash_bytes / 1e9, 3),
        "traffic_ratio": round(naive_bytes / flash_bytes, 1),
        "naive_achieved_GBs": round(naive_bytes / 1e9 / (float(n_ms) / 1e3)),
        "flash_achieved_GBs": round(flash_bytes / 1e9 / (float(f_ms) / 1e3)),
        "compute_floor_ms": round(compute_floor_ms, 3),
    })
    print(f"  {nm:8s} naive {n_ms:.2f}ms ({naive_bytes/1e9:.1f}GB) flash {f_ms:.2f}ms "
          f"({flash_bytes/1e9:.2f}GB)  traffic {naive_bytes/flash_bytes:.0f}x  "
          f"compute-floor {compute_floor_ms:.2f}ms")

# ---- 3. backward: CSR vs naive autograd, effective gradient bandwidth ----------
for nm, Lq, Ld, B in [("medium", 128, 1024, 128), ("ColPali", 1024, 1024, 64)]:
    try:
        grad_bytes = (Lq * d + B * Ld * d) * 4        # dQ + dD in FP32
        # CSR backward
        Qc, Dc = make(B, Lq, Ld); Qc.requires_grad_(True); Dc.requires_grad_(True)
        def csr_bwd(Q, D):
            Q.grad = None; D.grad = None
            flash_maxsim_train(Q, D).sum().backward()
        (csr_ms,) = bench_interleaved([csr_bwd], [[Qc, Dc]], warmup=5, n=20)
        # naive autograd backward
        Qn, Dn = make(B, Lq, Ld); Qn.requires_grad_(True); Dn.requires_grad_(True)
        def naive_bwd(Q, D):
            Q.grad = None; D.grad = None
            maxsim_naive_fp32(Q, D).sum().backward()
        try:
            (nai_ms,) = bench_interleaved([naive_bwd], [[Qn, Dn]], warmup=5, n=20)
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            nai_ms = None; torch.cuda.empty_cache()
        rec = {"shape": nm, "Lq": Lq, "Ld": Ld, "B": B,
               "csr_ms": round(float(csr_ms), 3),
               "naive_ms": round(float(nai_ms), 3) if nai_ms else None,
               "csr_GBs": round(grad_bytes / 1e9 / (float(csr_ms) / 1e3)),
               "naive_GBs": round(grad_bytes / 1e9 / (float(nai_ms) / 1e3)) if nai_ms else None,
               "speedup": round(float(nai_ms) / float(csr_ms), 2) if nai_ms else "OOM"}
        out["backward"].append(rec); print("  bwd", rec)
    except Exception as e:
        print("  bwd FAIL", nm, e)

# ---- 4. query_chunk_size ablation (the exposed tile knob) ----------------------
Q, D = make(1000, 1024, 1024)
for qcs in [64, 128, 256, 512, 1024]:
    try:
        (ms,) = bench_interleaved([lambda Q, D: flash_maxsim(Q, D, query_chunk_size=qcs)], [[Q, D]])
        out["chunk_abl"].append({"query_chunk_size": qcs, "ms": round(float(ms), 3)})
        print(f"  qcs={qcs}: {ms:.3f}ms")
    except Exception as e:
        print(f"  qcs={qcs} FAIL", e)

p = os.path.join(HERE, f"bench_paper_io_{gpu.replace(' ', '_').replace('/', '_')}.json")
json.dump(out, open(p, "w"), indent=2); print("wrote", p)
