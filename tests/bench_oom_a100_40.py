"""Empirical OOM test on A100 40GB with a REAL ColPali model loaded.

Realistic scoring pipeline:
  1. Load ColPali (vidore/colpali-v1.2, PaliGemma-3B backbone) in FP16 on GPU
  2. Run one forward pass on a dummy image to warm caches / allocate activations
  3. Attempt to score against 10K / 20K fake D embeddings
  4. Record OOM or success for three paths:
       a. naive FP16 einsum
       b. naive matched precision (FP32 reduction)
       c. flash_maxsim
"""
import sys, time, torch
import torch.nn.functional as F
sys.path.insert(0, '.')

device = 'cuda'
DTYPE = torch.float16


def mem_str():
    a = torch.cuda.memory_allocated() / (1024**3)
    r = torch.cuda.memory_reserved() / (1024**3)
    p = torch.cuda.max_memory_allocated() / (1024**3)
    return f"alloc={a:5.1f}GB reserved={r:5.1f}GB peak={p:5.1f}GB"


print(f"GPU: {torch.cuda.get_device_name(0)}")
total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
print(f"Total VRAM: {total_gb:.1f} GB\n")

# ----------------------------------------------------------------------
# 1. Load the real ColPali model
# ----------------------------------------------------------------------
print("Loading ColPali v1.2 (PaliGemma-3B backbone)...")
t0 = time.perf_counter()
from colpali_engine.models import ColPali, ColPaliProcessor

model = ColPali.from_pretrained(
    "vidore/colpali-v1.2",
    torch_dtype=torch.float16,
).eval().to(device)
processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")

print(f"  Loaded in {time.perf_counter()-t0:.1f}s  ({mem_str()})")
model_resident_gb = torch.cuda.memory_allocated() / (1024**3)
print(f"  Model resident: {model_resident_gb:.2f} GB\n")

# ----------------------------------------------------------------------
# 2. Forward pass on a dummy query to allocate any activation buffers
# ----------------------------------------------------------------------
print("Running one forward pass (query text)...")
try:
    queries = ["What is the main finding of this document?"]
    batch_q = processor.process_queries(queries).to(device)
    with torch.no_grad():
        q_out = model(**batch_q)
        Q = q_out.float().squeeze(0).half()  # [Lq, 128]
    torch.cuda.synchronize()
    print(f"  Query embedded: shape={tuple(Q.shape)}  ({mem_str()})")
except Exception as e:
    print(f"  Query forward failed: {e!r}")
    # Fallback: random Q that looks like ColPali output
    Q = F.normalize(torch.randn(32, 128, dtype=DTYPE, device=device), dim=-1)
    print(f"  Using dummy Q: shape={tuple(Q.shape)}")
torch.cuda.empty_cache()
print()

# We'll benchmark at ColPali-typical Lq (query ~32 tokens). For stress-test
# with a wide Q, also test Lq=1024 (what an image query would produce).
# Pick Lq by zero-padding or repeating Q.
def _make_Q(target_Lq):
    if Q.shape[0] >= target_Lq:
        return Q[:target_Lq].contiguous()
    rep = (target_Lq + Q.shape[0] - 1) // Q.shape[0]
    return Q.repeat(rep, 1)[:target_Lq].contiguous()


# ----------------------------------------------------------------------
# 3. Score against fake D embeddings at various B
# ----------------------------------------------------------------------
def run_case(label, fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"  [{label}]  OK  {ms:.1f} ms  peak={peak:.1f} GB  ({mem_str()})")
        return out
    except torch.cuda.OutOfMemoryError as e:
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        msg = str(e).splitlines()[0][:140]
        print(f"  [{label}]  OOM  peak={peak:.1f} GB  ({msg})")
        return None
    except Exception as e:
        print(f"  [{label}]  ERR  {type(e).__name__}: {str(e)[:120]}")
        return None


from flash_maxsim import flash_maxsim

Lq_test = 1024  # ColPali-image query (worst case; ~1024 patches)
Ld = 1024
d = 128

Q_img = _make_Q(Lq_test)

for B in [10_000, 15_000, 20_000]:
    sim_fp16_gb = B * Lq_test * Ld * 2 / (1024**3)
    d_gb = B * Ld * d * 2 / (1024**3)
    total_est = model_resident_gb + d_gb + sim_fp16_gb
    print("=" * 100)
    print(f" ColPali image-query scoring:  Lq={Lq_test}, Ld={Ld}, d={d}, B={B:,}")
    print(f"   model={model_resident_gb:.1f}GB  D={d_gb:.1f}GB  sim(FP16)={sim_fp16_gb:.1f}GB  "
          f"expected_peak≥{total_est:.1f}GB")
    print("=" * 100)

    torch.cuda.empty_cache()
    try:
        D = F.normalize(torch.randn(B, Ld, d, dtype=DTYPE, device=device), dim=-1)
        print(f"  D allocated ({mem_str()})")
    except torch.cuda.OutOfMemoryError as e:
        print(f"  Can't allocate D ({d_gb:.1f} GB) alongside model — SKIP")
        torch.cuda.empty_cache()
        continue

    run_case(f"naive FP16      ", lambda: torch.einsum('qd,bld->bql', Q_img, D).max(dim=2).values.sum(dim=1))
    run_case(f"naive matched   ", lambda: torch.einsum('qd,bld->bql', Q_img, D).float().max(dim=2).values.sum(dim=1))
    run_case(f"flash_maxsim    ", lambda: flash_maxsim(Q_img, D))

    del D
    torch.cuda.empty_cache()
    print()

print("Done.")
