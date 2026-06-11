"""INT8 cross-shape evaluation.

Reviewer R2 #11 ("strengthen or demote INT8"): the paper currently asserts
INT8 ranking-fidelity (Spearman 0.999) and a forward-only 4.7x at "tested
shapes" but does not report which shapes. This bench sweeps the same five
shapes as tab:fwd so the reader can see (a) where INT8 is a net latency
win vs FP16-flash and (b) whether the ranking-fidelity claim survives at
ColPali scale.

For each shape we report:
  - FP32 naive einsum latency + scores (reference for fidelity)
  - FP16 flash latency
  - INT8xINT8 flash latency + max |Δ|, Spearman ρ, top-K=20 overlap
  - Speedup of INT8 over FP16-flash and over naive FP32
  - Memory footprint of D in bytes (D-tensor is the dominant storage)

The fidelity numbers are computed once per shape on a single representative
query (the bench's Q[0]); for the headline "INT8 preserves rankings" claim
this is sufficient because INT8 is content-agnostic — the quantization
error budget only depends on the per-token distribution, not on which
query is asked.
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from flash_maxsim import (  # noqa: E402
    flash_maxsim_batched,
    flash_maxsim_int8x8,
    quantize_int8_symmetric,
    quantize_query_int8,
)

torch.set_float32_matmul_precision('high')
WARMUP, RUNS = 5, 30

SHAPES = [
    {"name": "textual",  "Lq": 32,   "Ld": 180,  "d": 128, "B": 1024},
    {"name": "long-doc", "Lq": 32,   "Ld": 1024, "d": 128, "B": 1024},
    {"name": "medium",   "Lq": 128,  "Ld": 1024, "d": 128, "B": 512},
    {"name": "visual",   "Lq": 512,  "Ld": 1024, "d": 128, "B": 256},
    {"name": "ColPali",  "Lq": 1024, "Ld": 1024, "d": 128, "B": 128},
]


def _bench(fn, runs=RUNS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); e.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def fp32_naive(Q, D):
    sim = torch.einsum('id,bjd->bij', Q.float(), D.float())
    return sim.max(dim=-1).values.sum(dim=-1)


def run_shape(s):
    Lq, Ld, d, B = s["Lq"], s["Ld"], s["d"], s["B"]
    # Embeddings ℓ2-normalised — matches what real encoders produce
    Q = torch.randn(Lq, d, device='cuda', dtype=torch.float16)
    Q = Q / Q.float().norm(dim=-1, keepdim=True).clamp(min=1e-12).to(torch.float16)
    D = torch.randn(B, Ld, d, device='cuda', dtype=torch.float16)
    D = D / D.float().norm(dim=-1, keepdim=True).clamp(min=1e-12).to(torch.float16)

    # 1) FP32 naive (reference scores + reference latency)
    ref_scores = fp32_naive(Q, D)
    t_naive = _bench(lambda: fp32_naive(Q, D))

    # 2) FP16 flash
    Qb = Q.unsqueeze(0)  # [1, Lq, d]
    t_fp16 = _bench(lambda: flash_maxsim_batched(Qb, D))
    fp16_scores = flash_maxsim_batched(Qb, D).squeeze(0)

    # 3) INT8 x INT8 flash — pre-quantize D and Q out of the timed region
    D_int8, d_scales = quantize_int8_symmetric(D)
    Q_int8, q_scales = quantize_query_int8(Q)
    t_int8 = _bench(lambda: flash_maxsim_int8x8(Q, D_int8, d_scales, Q_int8, q_scales))
    int8_scores = flash_maxsim_int8x8(Q, D_int8, d_scales, Q_int8, q_scales)

    # Ranking fidelity vs the FP32 reference (the actual published metric)
    ref_np = ref_scores.cpu().numpy()
    int8_np = int8_scores.cpu().numpy()
    fp16_np = fp16_scores.cpu().numpy()

    max_abs_delta_int8 = float(np.abs(ref_np - int8_np).max())
    rho_int8 = float(spearmanr(ref_np, int8_np).statistic)
    K = 20
    ref_top = set(np.argsort(-ref_np)[:K].tolist())
    int8_top = set(np.argsort(-int8_np)[:K].tolist())
    overlap_top20 = len(ref_top & int8_top) / K

    # Memory footprint of D
    fp16_bytes = B * Ld * d * 2
    int8_bytes = B * Ld * d * 1 + B * Ld * 2  # int8 + fp16 per-token scale

    speedup_int8_vs_fp16 = t_fp16 / t_int8
    speedup_int8_vs_naive = t_naive / t_int8

    row = {
        "shape": s["name"], "Lq": Lq, "Ld": Ld, "d": d, "B": B,
        "naive_ms": round(t_naive, 4),
        "fp16_flash_ms": round(t_fp16, 4),
        "int8_flash_ms": round(t_int8, 4),
        "int8_vs_fp16": round(speedup_int8_vs_fp16, 2),
        "int8_vs_naive": round(speedup_int8_vs_naive, 2),
        "int8_max_abs_delta": round(max_abs_delta_int8, 4),
        "int8_spearman_rho": round(rho_int8, 6),
        "int8_top20_overlap": round(overlap_top20, 4),
        "D_fp16_bytes": fp16_bytes,
        "D_int8_bytes": int8_bytes,
        "D_storage_ratio": round(fp16_bytes / int8_bytes, 2),
    }
    print(f"  {s['name']:>9s}: naive={t_naive:>6.2f}ms  fp16={t_fp16:>5.2f}ms  "
          f"int8={t_int8:>5.2f}ms  ({speedup_int8_vs_fp16:.2f}x vs fp16, "
          f"{speedup_int8_vs_naive:.2f}x vs naive)  "
          f"int8: ρ={rho_int8:.4f}, top-20 ovl={overlap_top20:.0%}, "
          f"max|Δ|={max_abs_delta_int8:.3f}", flush=True)
    return row


print(f"GPU: {torch.cuda.get_device_name()}", flush=True)
results = []
for s in SHAPES:
    print(f"\n=== {s['name']} (B={s['B']}, Lq={s['Lq']}, Ld={s['Ld']}, d={s['d']}) ===", flush=True)
    try:
        results.append(run_shape(s))
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
        results.append({"shape": s["name"], "error": str(e)[:200]})

gpu = torch.cuda.get_device_name()
out = os.path.join(HERE, f"bench_int8_shapes_{gpu.replace(' ', '_').replace('/', '_')}.json")
json.dump({"gpu": gpu, "warmup": WARMUP, "runs": RUNS, "results": results}, open(out, "w"), indent=2)
print(f"\nwrote {out}")
