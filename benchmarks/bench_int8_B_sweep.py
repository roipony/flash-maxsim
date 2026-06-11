"""INT8 B-sweep at ColPali shape.

Sanity check on the cross-shape INT8 result: at B=128 the D tensor (33 MB FP16,
16 MB INT8) fits in A100's L2 cache (~40 MB), so the kernel is compute-bound
and INT8 wins only by the compute-throughput delta. The storage advantage
should kick in when D no longer fits in L2 — i.e., at B that pushes D beyond
~40 MB.

We sweep B at ColPali shape (Lq=Ld=1024, d=128, FP16):
  B=128   D=  33 MB    (in L2)
  B=256   D=  67 MB    (out of L2)
  B=1024  D= 270 MB    (deep out of L2)
  B=10K   D= 2.6 GB    (memory-bound)

For each B we report FP16-flash, INT8-flash, and the ratio. If INT8's
storage advantage is real for memory-bound serving, the ratio should grow
toward 2x at large B.

Also two ref-baseline columns to keep the original 4.7x-vs-dequantize claim
in context: "naive FP32 einsum" (= naive column of tab:fwd) and
"dequant-then-FP32-einsum" (= the strawman baseline the original paper
compared INT8 against).
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import torch  # noqa: E402
from flash_maxsim import (  # noqa: E402
    flash_maxsim_batched,
    flash_maxsim_int8x8,
    quantize_int8_symmetric,
    quantize_query_int8,
)

torch.set_float32_matmul_precision('high')
WARMUP, RUNS = 5, 30

BS = [128, 256, 512, 1024, 2048, 4096, 10000]
Lq, Ld, d = 1024, 1024, 128


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


def dequant_then_naive(Q, D_int8, d_scales):
    """The strawman the original paper compared INT8 against:
    materialize a dequantized FP32 D tensor, then naive einsum + max + sum.

    d_scales arrives as [B, Ld, 1] from quantize_int8_symmetric (keepdim=True),
    so no extra unsqueeze is needed for broadcasting against [B, Ld, d].
    """
    D_fp32 = D_int8.float() * d_scales.float()
    return fp32_naive(Q, D_fp32)


print(f"GPU: {torch.cuda.get_device_name()}", flush=True)
print(f"ColPali shape: Lq={Lq}, Ld={Ld}, d={d}\n", flush=True)
results = []
Q = torch.randn(Lq, d, device='cuda', dtype=torch.float16)
Q = Q / Q.float().norm(dim=-1, keepdim=True).clamp(min=1e-12).to(torch.float16)
for B in BS:
    print(f"--- B={B} (D = {B*Ld*d*2 / 1e6:.0f} MB FP16, {B*Ld*d / 1e6:.0f} MB INT8) ---",
          flush=True)
    try:
        D = torch.randn(B, Ld, d, device='cuda', dtype=torch.float16)
        D = D / D.float().norm(dim=-1, keepdim=True).clamp(min=1e-12).to(torch.float16)
        D_int8, d_scales = quantize_int8_symmetric(D)
        Q_int8, q_scales = quantize_query_int8(Q)
        Qb = Q.unsqueeze(0)

        # FP32 paths can OOM at large B (sim tensor is B*Lq*Ld*4 = 4 GB at
        # B=1K ColPali and grows linearly). Catch OOM and report as None so
        # the rest of the row still lands.
        try:
            t_naive = _bench(lambda: fp32_naive(Q, D))
        except torch.cuda.OutOfMemoryError:
            t_naive = None; torch.cuda.empty_cache()
        try:
            t_dequant = _bench(lambda: dequant_then_naive(Q, D_int8, d_scales))
        except torch.cuda.OutOfMemoryError:
            t_dequant = None; torch.cuda.empty_cache()
        t_fp16 = _bench(lambda: flash_maxsim_batched(Qb, D))
        t_int8 = _bench(lambda: flash_maxsim_int8x8(Q, D_int8, d_scales, Q_int8, q_scales))

        row = {
            "B": B, "Lq": Lq, "Ld": Ld, "d": d,
            "D_fp16_mb": round(B * Ld * d * 2 / 1e6, 1),
            "D_int8_mb": round(B * Ld * d / 1e6, 1),
            "naive_ms": round(t_naive, 4) if t_naive else "OOM",
            "dequant_then_naive_ms": round(t_dequant, 4) if t_dequant else "OOM",
            "fp16_flash_ms": round(t_fp16, 4),
            "int8_flash_ms": round(t_int8, 4),
            "int8_vs_fp16_flash": round(t_fp16 / t_int8, 3),
            "int8_vs_naive": round(t_naive / t_int8, 3) if t_naive else None,
            "int8_vs_dequant_then_naive": round(t_dequant / t_int8, 3) if t_dequant else None,
        }
        naive_str = f"{t_naive:.2f}ms" if t_naive else "OOM"
        dequant_str = f"{t_dequant:.2f}ms" if t_dequant else "OOM"
        sp_naive = f"{row['int8_vs_naive']}x" if row['int8_vs_naive'] else "OOM"
        sp_dequant = f"{row['int8_vs_dequant_then_naive']}x" if row['int8_vs_dequant_then_naive'] else "OOM"
        print(f"  naive={naive_str}  dequant_then_naive={dequant_str}  "
              f"fp16={t_fp16:.2f}ms  int8={t_int8:.2f}ms", flush=True)
        print(f"  int8 vs fp16-flash: {row['int8_vs_fp16_flash']}x  "
              f"vs naive: {sp_naive}  "
              f"vs dequant-then-naive (paper baseline): {sp_dequant}",
              flush=True)
        results.append(row)
        del D, D_int8, d_scales, Q_int8, q_scales
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
        results.append({"B": B, "error": str(e)[:200]})

gpu = torch.cuda.get_device_name()
out = os.path.join(HERE, f"bench_int8_B_sweep_{gpu.replace(' ', '_').replace('/', '_')}.json")
json.dump({"gpu": gpu, "warmup": WARMUP, "runs": RUNS, "results": results}, open(out, "w"), indent=2)
print(f"\nwrote {out}")
