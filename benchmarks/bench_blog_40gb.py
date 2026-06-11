"""Lock down the blog's 40 GB-specific ColPali claims with a JSON of record.

Times four baselines at ColPali shape (Lq=Ld=1024, d=128) on A100-40GB:

  - naive_fp16        : einsum + max + sum, FP16 inputs / FP16 accum
  - naive_matched_fp32: einsum + max + sum, FP32 inputs / FP32 accum (TF32 'high')
  - compile_ma_fp32   : torch.compile(naive_fp32, mode='max-autotune-no-cudagraphs',
                                       dynamic=False), FP32 inputs
  - flash             : flash_maxsim, FP16 inputs

Fairness protocol (revised 2026-06 after reviewer feedback):

  - Each (B, method) runs in an ISOLATED allocator state. Inputs are generated
    in the method's NATIVE dtype, used only by that method, then freed before
    the next method runs. The previous version cast FP16 -> FP32 with the FP16
    copy still alive, unfairly penalising the FP32 peak by ~1-2 GB per row.
  - Peak memory reset BEFORE inputs are allocated, so reported peak includes
    BOTH the resident input tensors AND the per-call intermediates / workspace
    -- this is end-to-end resident peak under the user's allocator, not a
    naked workspace number.
  - OOM handling distinguishes torch.cuda.OutOfMemoryError (and RuntimeError
    whose message contains 'out of memory') from real RuntimeError bugs; the
    latter are re-raised so a buggy run does not silently appear as an OOM cliff.
  - A small correctness check runs at startup verifying that Flash's scores
    match the FP32-precast naive baseline within FP16-cast noise on B=4.

Output JSON shape (per row): {B, naive_fp16, naive_matched_fp32,
compile_ma_fp32, flash}. Each method is a {ms, peak_gb} dict or
{ms: None, peak_gb: None, error: <name>} on OOM.
"""
import json, os, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from flash_maxsim import flash_maxsim

torch.set_float32_matmul_precision('high')  # TF32 — paper's stated baseline
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

Lq = Ld = 1024
d = 128


def make_inputs(B, dtype):
    """Generate Q and D in the requested dtype, with in-place L2-norm so
    the allocation peak stays at one D-sized tensor (the previous
    F.normalize(...) returned a NEW tensor of the same shape, transiently
    doubling the peak to ~2x|D| during the divide and OOMing at large B
    even when D alone would have fit — see reviewer feedback 2026-06)."""
    Q = torch.randn(Lq, d, device='cuda', dtype=dtype)
    Q.div_(Q.norm(dim=-1, keepdim=True).clamp_min(1e-12))
    D = torch.randn(B, Ld, d, device='cuda', dtype=dtype)
    D.div_(D.norm(dim=-1, keepdim=True).clamp_min(1e-12))
    return Q, D


def naive(Q, D):
    return torch.einsum("qd,bld->bql", Q, D).max(2).values.sum(1)


# torch.compile(max-autotune-no-cudagraphs) over the same einsum-max-sum
# expression. We disable CUDA graph capture for two reasons: (i) paper
# protocol says no CUDA graphs; (ii) the cudagraph_trees backend has an
# assertion bug on our torch build that crashes the bench. CUDA-graphs-on
# numbers (the production max-autotune flavor) are audited separately in
# bench_compile_ma_audit.py. Two separate compile instances so FP32 and
# FP16 each get their own trace cache and config.
compile_ma_fp32 = torch.compile(naive, mode="max-autotune-no-cudagraphs",
                                 dynamic=False)
compile_ma_fp16 = torch.compile(naive, mode="max-autotune-no-cudagraphs",
                                 dynamic=False)


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def run_case(B, method):
    """Run a single (B, method) in an isolated allocator state.

    Returns {ms, peak_gb} on success or {ms: None, peak_gb: None, error: name}
    on OOM (during input allocation OR during the call). Non-OOM RuntimeErrors
    are re-raised so they don't silently masquerade as OOM cliffs.
    """
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    if method == "flash":
        dtype, fn = torch.float16, flash_maxsim
    elif method == "naive_fp16":
        dtype, fn = torch.float16, naive
    elif method == "naive_matched_fp32":
        dtype, fn = torch.float32, naive
    elif method == "compile_ma_fp32":
        dtype, fn = torch.float32, compile_ma_fp32
    elif method == "compile_ma_fp16":
        dtype, fn = torch.float16, compile_ma_fp16
    else:
        raise ValueError(method)

    Q = D = None
    try:
        Q, D = make_inputs(B, dtype)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if _is_oom(e):
            torch.cuda.empty_cache()
            return {"ms": None, "peak_gb": None, "error": "OOM_inputs"}
        raise

    warmup, n = 3, 5
    try:
        for _ in range(warmup):
            _ = fn(Q, D); torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(n):
            start.record(); _ = fn(Q, D); end.record(); torch.cuda.synchronize()
            ts.append(start.elapsed_time(end))
        ts.sort()
        result = {"ms": round(float(ts[len(ts) // 2]), 2),
                  "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if _is_oom(e):
            result = {"ms": None, "peak_gb": None, "error": "OOM_call"}
        else:
            raise
    finally:
        del Q, D
        torch.cuda.empty_cache()
    return result


def correctness_check():
    """Verify Flash matches the FP32-precast naive baseline within FP16-cast
    noise at small B. Fail loudly so the bench isn't silently broken."""
    B = 4
    Q16, D16 = make_inputs(B, torch.float16)
    Qf, Df   = Q16.float(), D16.float()
    ref = naive(Qf, Df)            # FP32 reference
    fl  = flash_maxsim(Q16, D16)   # FP16 inputs, FP32 accum inside
    err = (ref - fl).abs().max().item()
    print(f"correctness: max|naive_fp32 - flash_fp16| = {err:.3e}  (tol 1e-1)")
    assert err < 1e-1, f"Flash vs FP32 reference disagrees by {err:.3e} — abort"
    del Q16, D16, Qf, Df, ref, fl
    torch.cuda.empty_cache()


gpu = torch.cuda.get_device_name()
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {gpu} ({vram:.1f} GB VRAM)")
print(f"torch={torch.__version__}\n")

correctness_check()

Bs = [1000, 2000, 3000, 4000, 5000, 7000, 10000, 12000, 15000, 20000,
      50000, 100000, 150000, 175000]
out = {"gpu": gpu, "vram_gb": round(vram, 1), "Lq": Lq, "Ld": Ld, "d": d,
       "torch": torch.__version__,
       "protocol": ("Each (B, method) runs in an isolated allocator state with "
                    "inputs generated in the method's native dtype. Peak memory "
                    "includes resident inputs + per-call intermediates / "
                    "workspace under the user allocator (not a naked workspace "
                    "number); cudagraph private pool excluded (no cudagraphs). "
                    "OOM caught and recorded; non-OOM RuntimeErrors propagate."),
       "rows": []}

for B in Bs:
    print(f"=== B={B} ===")
    n16  = run_case(B, "naive_fp16");          print(f"  naive_fp16:        {n16}")
    n32  = run_case(B, "naive_matched_fp32");  print(f"  naive_fp32(TF32):  {n32}")
    c32  = run_case(B, "compile_ma_fp32");     print(f"  compile-MA(FP32):  {c32}")
    c16  = run_case(B, "compile_ma_fp16");     print(f"  compile-MA(FP16):  {c16}")
    fl   = run_case(B, "flash");               print(f"  flash:             {fl}")
    out["rows"].append({"B": B, "naive_fp16": n16, "naive_matched_fp32": n32,
                         "compile_ma_fp32": c32, "compile_ma_fp16": c16, "flash": fl})

p = os.path.join(HERE, f"bench_blog_40gb_{gpu.replace(' ', '_').replace('/', '_')}.json")
json.dump(out, open(p, "w"), indent=2)
print(f"\nwrote {p}")
