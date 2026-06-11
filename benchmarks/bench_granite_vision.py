"""End-to-end benchmark with Granite Vision Embedding model.

Uses ibm-granite/granite-vision-3.3-2b-embedding (multi-vector, MaxSim-based).
Benchmarks Flash-MaxSim as drop-in replacement for the model's scoring.

Usage:
    pip install flash-maxsim transformers pillow requests
    python benchmarks/bench_granite_vision.py

Produces: benchmarks/results_granite_vision_<GPU>.json
"""
import json, torch

assert torch.cuda.is_available(), "CUDA required"
gpu_name = torch.cuda.get_device_name()
print(f"GPU: {gpu_name}\n")

from flash_maxsim import flash_maxsim, flash_maxsim_int8, quantize_int8
from utils import bench_interleaved, compile_high_precision


def naive_maxsim(Q, D):
    return torch.einsum('qd,bld->bql', Q.float(), D.float()).max(2).values.sum(1)

compiled_naive_maxsim = compile_high_precision(naive_maxsim)

# ── Load model ──
print("Loading Granite Vision Embedding model...")
from transformers import AutoProcessor, AutoModel

model_name = "ibm-granite/granite-vision-3.3-2b-embedding"
model = AutoModel.from_pretrained(
    model_name, trust_remote_code=True,
    torch_dtype=torch.float16, device_map="cuda",
).eval()
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
print(f"  Loaded: {model_name}\n")

# ── Encode sample data ──
print("Encoding sample queries and images...")

queries = [
    "What is the revenue breakdown by segment?",
    "Show me the organizational chart",
    "What are the key financial metrics?",
    "Describe the product architecture diagram",
    "What is the market share distribution?",
]

from PIL import Image
images = []
for i in range(20):
    img = Image.new("RGB", (224, 224), color=(i * 12, i * 8, 255 - i * 10))
    images.append(img)

print("  Encoding queries...")
text_inputs = processor.process_queries(queries)
text_inputs = {k: v.to("cuda") for k, v in text_inputs.items()}
with torch.no_grad():
    Q_all = model(**text_inputs).to(torch.float16)
print(f"  Q: {Q_all.shape}")

print("  Encoding images...")
img_inputs = processor.process_images(images)
img_inputs = {k: v.to("cuda") for k, v in img_inputs.items()}
with torch.no_grad():
    D = model(**img_inputs).to(torch.float16)
print(f"  D: {D.shape}")

Nq, Lq, d = Q_all.shape
B, Ld, _ = D.shape
print(f"  Lq={Lq}, Ld={Ld}, d={d}\n")

results = {
    "gpu": gpu_name,
    "model": model_name,
    "Lq": Lq, "Ld": Ld, "d": d,
    "Nq": Nq, "B": B,
}

# ── Ranking verification ──
print("=" * 60)
print("Ranking Verification")
print("=" * 60)
all_match = True
for qi in range(Nq):
    Q = Q_all[qi]
    s_naive = naive_maxsim(Q, D)
    s_flash = flash_maxsim(Q, D)
    r_naive = s_naive.argsort(descending=True).tolist()
    r_flash = s_flash.argsort(descending=True).tolist()
    match = r_naive == r_flash
    if not match:
        all_match = False
    print(f"  Q{qi}: top-3={r_flash[:3]}  match={match}")
print(f"  All match: {all_match}\n")
results["all_rankings_match"] = all_match

# ── Scoring speed ──
print("=" * 60)
print("Scoring Speed (single query)")
print("=" * 60)

Q = Q_all[0]
scoring_results = []

t_naive, t_compiled, t_flash = bench_interleaved([naive_maxsim, compiled_naive_maxsim, flash_maxsim], [[Q, D]])
print(f"  B={B}: naive={t_naive:.3f}ms compiled={t_compiled:.3f}ms flash={t_flash:.3f}ms  naive_speedup={t_naive/t_flash:.1f}x compiled_speedup={t_compiled/t_flash:.1f}x")
scoring_results.append({"B": B, "naive_ms": round(t_naive, 3), "flash_ms": round(t_flash, 3)})

# Scale up D
for scale in [5, 10, 25, 50, 100]:
    B_scaled = B * scale
    if B_scaled * Ld * d * 2 > 50e9:
        break
    D_scaled = D.repeat(scale, 1, 1)

    wup = 5 if B_scaled > 500 else 10
    iters = 20 if B_scaled > 500 else 50

    t_naive, t_compiled, t_flash = bench_interleaved([naive_maxsim, t_compiled, flash_maxsim], [[Q, D_scaled]])

    sim_mb = Lq * B_scaled * Ld * 4 / 1e6
    print(f"  B={B_scaled}: naive={t_naive:.3f}ms  flash={t_flash:.3f}ms compiled={t_compiled:.3f}ms  flash={t_flash:.3f}ms "
          f"naive_speedup={t_naive/t_flash:.1f}x compiled_speedup={t_compiled/t_flash:.1f}x sim_matrix={sim_mb:.0f}MB")

    scoring_results.append({"B": B_scaled, "naive_ms": round(t_naive, 3), "compiled_ms": round(t_compiled, 3), "flash_ms": round(t_flash, 3)})
    del D_scaled; torch.cuda.empty_cache()

results["scoring"] = scoring_results

# ── INT8 ──
print(f"\n{'=' * 60}")
print("INT8 Quantized Scoring")
print("=" * 60)
D_q, scales, mins = quantize_int8(D)
t_fp16, t_q8 = bench_interleaved([flash_maxsim, flash_maxsim_int8], [[Q, D], [Q, D_q, scales, mins]])
s_fp16 = flash_maxsim(Q, D)
s_q8 = flash_maxsim_int8(Q, D_q, scales, mins)
rank_match = (s_fp16.argsort(descending=True)[:5] == s_q8.argsort(descending=True)[:5]).all().item()
err = (s_fp16 - s_q8).abs().max().item()
print(f"  FP16={t_fp16:.3f}ms  INT8={t_q8:.3f}ms  compression={D.nbytes/D_q.nbytes:.0f}x")
print(f"  rank_match={rank_match}  max_err={err:.4f}")
results["int8"] = {
    "fp16_ms": round(t_fp16, 3), "q8_ms": round(t_q8, 3),
    "compression": round(D.nbytes / D_q.nbytes, 1),
    "rank_match": rank_match, "max_err": round(err, 4),
}

# ── Memory ──
print(f"\n{'=' * 60}")
print("Peak Memory")
print("=" * 60)
mem_results = []
for scale in [1, 5, 25, 50]:
    B_test = B * scale
    if B_test * Ld * d * 2 > 50e9:
        break
    D_test = D.repeat(scale, 1, 1) if scale > 1 else D
    sim_mb = Lq * B_test * Ld * 4 / 1e6

    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    try:
        _ = naive_maxsim(Q, D_test); torch.cuda.synchronize()
        naive_mb = (torch.cuda.max_memory_allocated() - base) / 1e6
        del _
    except Exception:
        naive_mb = sim_mb
    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    _ = flash_maxsim(Q, D_test); torch.cuda.synchronize()
    flash_mb = (torch.cuda.max_memory_allocated() - base) / 1e6
    del _; torch.cuda.empty_cache()

    ratio = naive_mb / max(flash_mb, 0.001)
    print(f"  B={B_test}: naive={naive_mb:.1f}MB  flash={flash_mb:.2f}MB  {ratio:.0f}x less")
    mem_results.append({"B": B_test, "naive_mb": round(naive_mb, 1), "flash_mb": round(flash_mb, 2), "ratio": round(ratio, 0)})

    if scale > 1:
        del D_test; torch.cuda.empty_cache()

results["memory"] = mem_results

# ── Save ──
gpu_tag = gpu_name.replace(" ", "_").replace("/", "-")
out = f"benchmarks/results_granite_vision_{gpu_tag}.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out}")
print("Done!")
