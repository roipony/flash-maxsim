"""Collect ALL benchmark data for the Flash-MaxSim landing page.

Runs six benchmark suites (A-F) and saves everything to a single JSON file.

Usage:
    python benchmarks/collect_all_benchmarks.py
    # produces benchmarks/results_landing_<GPU>.json
"""
import json, math, torch, torch.nn.functional as F
import numpy as np

assert torch.cuda.is_available(), "CUDA required"
gpu_name = torch.cuda.get_device_name()
gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {gpu_name} ({gpu_mem_gb:.0f} GB)")

import triton
from flash_maxsim import flash_maxsim, flash_maxsim_batched, flash_maxsim_int8, quantize_int8
from utils import bench_interleaved, compile_high_precision

# ── Helpers ──────────────────────────────────────────────────────────────────

def make(B, Lq, Ld, d=128):
    Q = F.normalize(torch.randn(Lq, d, device="cuda", dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(B, Ld, d, device="cuda", dtype=torch.float16), dim=-1)
    return Q, D


def naive_fp32(Q, D):
    return torch.einsum("qd,bld->bql", Q.float(), D.float()).max(2).values.sum(1)

torch.set_float32_matmul_precision('high')
compiled_fp32 = compile_high_precision(naive_fp32)

def chunked_maxsim(Q, D, chunk_size):
    B = D.shape[0]
    scores = torch.empty(B, device=Q.device, dtype=torch.float32)
    for i in range(0, B, chunk_size):
        D_chunk = D[i : i + chunk_size]
        sim = torch.einsum("qd,bld->bql", Q.float(), D_chunk.float())
        scores[i : i + chunk_size] = sim.max(dim=2).values.sum(dim=1)
    return scores


def safe_round(v, digits=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return round(v, digits)


def clear():
    torch.cuda.empty_cache()


# ── Results container ────────────────────────────────────────────────────────

results = {
    "gpu": gpu_name,
    "gpu_mem_gb": round(gpu_mem_gb, 1),
    "torch": torch.__version__,
    "triton": triton.__version__,
}


# ═══════════════════════════════════════════════════════════════════════════════
# A. KERNEL SPEEDUP SWEEP  (single query, B=1000)
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 65}")
print("A. KERNEL SPEEDUP SWEEP (B=1000)")
print("=" * 65)

B = 1000
sweep_a = []
for Lq in [32, 128, 512, 1024]:
    for Ld in [128, 256, 512, 768, 1024]:
        label = f"Lq={Lq}, Ld={Ld}"
        try:
            Q, D = make(B, Lq, Ld)
            wup = 5 if Lq >= 512 else 10
            iters = 20 if Lq >= 512 else 50

            # Naive
            try:
                n_ms, c_ms, f_ms = bench_interleaved([naive_fp32, compiled_fp32, flash_maxsim], [[Q, D]])
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                n_ms = None
                torch.cuda.synchronize()
                clear()
                c_ms, f_ms = bench_interleaved([compiled_fp32, flash_maxsim], [[Q, D]])

            n_sp = round(n_ms / f_ms, 1) if n_ms is not None else None
            c_sp = round(c_ms / f_ms, 1)

            print(f"  {label:20s}: naive={str(safe_round(n_ms)):>8s}ms compiled={str(safe_round(c_ms)):>8s}ms  "
                  f"flash={f_ms:.2f}ms  naive_speedup={n_sp}x compiled_speedup={c_sp}x")
            sweep_a.append({
                "Lq": Lq, "Ld": Ld, "B": B,
                "naive_ms": safe_round(n_ms),
                "compiled_ms": safe_round(c_ms),
                "flash_ms": safe_round(f_ms),
                "naive_speedup": n_sp,
                "compiled_speedup": c_sp,
            })
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            print(f"  {label:20s}: SKIPPED ({e.__class__.__name__})")
            sweep_a.append({
                "Lq": Lq, "Ld": Ld, "B": B,
                "naive_ms": None, "flash_ms": None, "speedup": None,
            })
            torch.cuda.synchronize()
        finally:
            clear()

results["kernel_speedup_sweep"] = sweep_a


# ═══════════════════════════════════════════════════════════════════════════════
# B. CORPUS SCALING  (single query, varying B)
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 65}")
print("B. CORPUS SCALING")
print("=" * 65)

corpus_configs = [
    (32, 300, "textual"),
    (32, 1024, "long_doc"),
    (1024, 1024, "visual"),
]
B_values_corpus = [100, 500, 1000, 2000, 5000]
sweep_b = []

for Lq, Ld, tag in corpus_configs:
    for B in B_values_corpus:
        label = f"{tag} B={B}"
        try:
            Q, D = make(B, Lq, Ld)

            # Naive
            try:
                n_ms, c_ms, f_ms = bench_interleaved([naive_fp32, compiled_fp32, flash_maxsim], [[Q, D]])
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                n_ms = None
                torch.cuda.synchronize()
                clear()
                c_ms, f_ms = bench_interleaved([compiled_fp32, flash_maxsim], [[Q, D]])

            n_sp = round(n_ms / f_ms, 1) if n_ms is not None else None
            c_sp = round(c_ms / f_ms, 1)

            print(f"  {label:25s}: naive={str(safe_round(n_ms)):>8s}ms compiled={str(safe_round(c_ms)):>8s}ms "
                  f"flash={f_ms:.2f}ms  naive_speedup={n_sp}x compiled_speedup={c_sp}x")
            sweep_b.append({
                "Lq": Lq, "Ld": Ld, "B": B, "tag": tag,
                "naive_ms": safe_round(n_ms),
                "compiled_ms": safe_round(c_ms),
                "flash_ms": safe_round(f_ms),
                "naive_speedup": n_sp,
                "compiled_speedup": c_sp,
            })
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            print(f"  {label:25s}: SKIPPED ({e.__class__.__name__})")
            sweep_b.append({
                "Lq": Lq, "Ld": Ld, "B": B, "tag": tag,
                "naive_ms": None, "flash_ms": None, "speedup": None,
            })
            torch.cuda.synchronize()
        finally:
            clear()

results["corpus_scaling"] = sweep_b


# ═══════════════════════════════════════════════════════════════════════════════
# C. PEAK MEMORY
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 65}")
print("C. PEAK MEMORY")
print("=" * 65)

mem_configs = [
    (32, 300, "textual"),
    (32, 1024, "long_doc"),
    (1024, 1024, "visual"),
]
B_values_mem = [500, 1000, 2000, 5000, 10000]
sweep_c = []

for Lq, Ld, tag in mem_configs:
    for B in B_values_mem:
        label = f"{tag} B={B}"

        # Check if D alone exceeds GPU
        d_gb = B * Ld * 128 * 2 / 1e9
        if d_gb > gpu_mem_gb * 0.7:
            print(f"  {label:25s}: SKIP (D={d_gb:.1f}GB > GPU budget)")
            continue

        sim_gb = B * Lq * Ld * 4 / 1e9  # theoretical sim matrix in FP32

        try:
            Q = F.normalize(torch.randn(1, Lq, 128, device="cuda", dtype=torch.float16), dim=-1)
            D = F.normalize(torch.randn(B, Ld, 128, device="cuda", dtype=torch.float16), dim=-1)

            # Naive memory
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            try:
                _ = torch.einsum("nqd,bld->nbql", Q.float(), D.float()).max(3).values.sum(2)
                torch.cuda.synchronize()
                naive_gb = (torch.cuda.max_memory_allocated() - base) / 1e9
                del _
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                naive_gb = sim_gb  # theoretical
                torch.cuda.synchronize()
            clear()

            # Flash memory
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            _ = flash_maxsim_batched(Q, D, shared_docs=True)
            torch.cuda.synchronize()
            flash_gb = (torch.cuda.max_memory_allocated() - base) / 1e9
            del _
            clear()

            ratio = max(1.0, naive_gb / max(flash_gb, 1e-6))
            print(f"  {label:25s}: naive={naive_gb:.3f}GB  flash={flash_gb:.4f}GB  ratio={ratio:.0f}x")
            sweep_c.append({
                "Lq": Lq, "Ld": Ld, "B": B, "tag": tag,
                "naive_gb": safe_round(naive_gb, 4),
                "flash_gb": safe_round(flash_gb, 4),
                "ratio": round(ratio, 1),
            })
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            print(f"  {label:25s}: SKIPPED ({e.__class__.__name__})")
            torch.cuda.synchronize()
        finally:
            clear()

results["peak_memory"] = sweep_c


# ═══════════════════════════════════════════════════════════════════════════════
# D. INT8 FUSED
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 65}")
print("D. INT8 FUSED DEQUANT")
print("=" * 65)

int8_configs = [
    (32, 300, "textual"),
    (32, 1024, "long_doc"),
]
B_values_int8 = [500, 1000, 2000, 5000]
sweep_d = []

for Lq, Ld, tag in int8_configs:
    for B in B_values_int8:
        label = f"{tag} B={B}"
        try:
            Q, D = make(B, Lq, Ld)
            Dq, s, m = quantize_int8(D)

            nf_ms, cf_ms, ff_ms, fi_ms = bench_interleaved([naive_fp32, compiled_fp32, flash_maxsim, flash_maxsim_int8],
                                      [[Q, D], [Q, D], [Q, D], [Q, Dq, s, m]])

            print(f"  {label:25s}: naive_fp32={nf_ms:.2f}ms compiled_fp32={cf_ms:.2f}ms flash_fp16={ff_ms:.2f}ms  flash_q8={fi_ms:.2f}ms")
            sweep_d.append({
                "Lq": Lq, "Ld": Ld, "B": B, "tag": tag,
                "naive_fp32_ms": safe_round(nf_ms),
                "compiled_fp32_ms": safe_round(cf_ms),
                "flash_fp16_ms": safe_round(ff_ms),
                "flash_q8_ms": safe_round(fi_ms),
            })
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            print(f"  {label:25s}: SKIPPED ({e.__class__.__name__})")
            torch.cuda.synchronize()
        finally:
            clear()

results["int8_fused"] = sweep_d


# ═══════════════════════════════════════════════════════════════════════════════
# E. CHUNKED BASELINE  (parameter-free argument)
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 65}")
print("E. CHUNKED BASELINE (Visual Lq=1024, Ld=1024, B=5000)")
print("=" * 65)

Lq_e, Ld_e, B_e = 1024, 1024, 5000
chunk_sizes = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
sweep_e = {"Lq": Lq_e, "Ld": Ld_e, "B": B_e, "flash_ms": None, "chunks": []}

try:
    Q_e, D_e = make(B_e, Lq_e, Ld_e)

    # Flash-MaxSim baseline
    t_flash = bench_interleaved([flash_maxsim], [[Q_e, D_e]]).item()
    sweep_e["flash_ms"] = safe_round(t_flash)
    print(f"  Flash-MaxSim: {t_flash:.3f}ms (parameter-free baseline)\n")

    print(f"  {'chunk':>6s}  {'latency':>10s}  {'vs_flash':>10s}  {'status':>8s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}")

    oom_seen = False
    for chunk in chunk_sizes:
        if oom_seen:
            print(f"  {chunk:6d}  {'---':>10s}  {'---':>10s}  {'OOM':>8s}")
            sweep_e["chunks"].append({
                "chunk_size": chunk, "latency_ms": None, "status": "OOM",
            })
            continue

        try:
            clear()
            # Test run
            _ = chunked_maxsim(Q_e, D_e, chunk)
            torch.cuda.synchronize()
            del _
            clear()

            t = bench_interleaved([chunked_maxsim], [[Q_e, D_e, chunk]]).item()
            ratio = t / t_flash
            print(f"  {chunk:6d}  {t:8.3f}ms  {ratio:8.1f}x  {'ok':>8s}")
            sweep_e["chunks"].append({
                "chunk_size": chunk,
                "latency_ms": safe_round(t),
                "vs_flash": round(ratio, 2),
                "status": "ok",
            })
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            oom_seen = True
            print(f"  {chunk:6d}  {'---':>10s}  {'---':>10s}  {'OOM':>8s}")
            sweep_e["chunks"].append({
                "chunk_size": chunk, "latency_ms": None, "status": "OOM",
            })
            torch.cuda.synchronize()
            clear()

    del Q_e, D_e
except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
    print(f"  SKIPPED entirely ({e.__class__.__name__})")
    torch.cuda.synchronize()
finally:
    clear()

results["chunked_baseline"] = sweep_e


# ═══════════════════════════════════════════════════════════════════════════════
# F. REAL MODEL  (if pylate installed)
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 65}")
print("F. REAL MODEL (answerai-colbert-small-v1)")
print("=" * 65)

sweep_f = None

try:
    from pylate import models

    model = models.ColBERT("answerdotai/answerai-colbert-small-v1")
    print("  Model loaded.")

    # Encode
    queries = [
        "What is information retrieval?",
        "How does ColBERT work?",
        "What are transformer models used for?",
        "Explain neural search engines",
        "What is late interaction in retrieval?",
        "How does quantization reduce model size?",
        "What is dense passage retrieval?",
        "How do inverted indices work?",
        "What is BM25 scoring?",
        "Explain semantic search with embeddings",
    ]
    passages = [
        "Information retrieval is the process of finding relevant documents from a large collection based on a user query.",
        "ColBERT uses late interaction between query and document token embeddings to score relevance efficiently.",
        "Transformer models are deep learning architectures used for NLP tasks like translation, summarization, and search.",
        "Neural search engines use learned representations to match queries with documents based on semantic meaning.",
        "Late interaction computes token-level similarities between queries and documents, then aggregates via MaxSim.",
        "Traditional search engines rely on keyword matching using inverted indices like BM25.",
        "Dense retrieval encodes queries and documents as single vectors and uses approximate nearest neighbor search.",
        "ColPali extends ColBERT to visual document retrieval by treating document pages as image patches.",
        "Flash Attention avoids materializing the attention matrix by tiling computation in SRAM.",
        "Quantization reduces model size by representing weights with fewer bits, such as INT8 or INT4.",
        "BM25 is a bag-of-words retrieval function that ranks documents based on term frequency and inverse document frequency.",
        "Semantic search uses dense vector representations to find documents that are semantically similar to a query.",
        "Knowledge distillation transfers knowledge from a large teacher model to a smaller student model.",
        "Re-ranking uses a more expensive model to re-order the top candidates from a first-stage retriever.",
        "Approximate nearest neighbor search uses techniques like HNSW or IVF to find similar vectors efficiently.",
        "Cross-encoders process query and document together but are too slow for first-stage retrieval.",
        "Contrastive learning trains models by bringing positive pairs closer and pushing negative pairs apart.",
        "Document expansion adds predicted terms to documents to improve keyword-based retrieval.",
        "Hybrid search combines dense and sparse retrieval methods for better coverage.",
        "Multi-vector representations like ColBERT preserve token-level information for fine-grained matching.",
    ]

    def encode_with_pylate(mdl, texts, is_query=False):
        embs = mdl.encode(texts, is_query=is_query)
        max_len = max(e.shape[0] for e in embs)
        d = embs[0].shape[1]
        T = torch.zeros(len(texts), max_len, d, device="cuda", dtype=torch.float16)
        for i, e in enumerate(embs):
            t = torch.tensor(np.array(e), device="cuda", dtype=torch.float16)
            T[i, : t.shape[0]] = t
        return T

    print("  Encoding 10 queries, 20 passages...")
    Q_all = encode_with_pylate(model, queries, is_query=True)
    D_base = encode_with_pylate(model, passages)
    Nq, Lq, d = Q_all.shape
    B_base, Ld, _ = D_base.shape
    print(f"  Q: {Q_all.shape}, D: {D_base.shape}")

    # Use first query for scoring benchmarks
    Q0 = Q_all[0]

    sweep_f = {
        "model": "answerai-colbert-small-v1",
        "Nq": Nq, "Lq": int(Lq), "d": int(d),
        "B_base": int(B_base), "Ld": int(Ld),
        "scoring": [],
    }

    # Score at various B values by repeating D
    target_Bs = [20, 200, 1000, 2000, 10000]
    for target_B in target_Bs:
        if target_B < B_base:
            D_test = D_base[:target_B]
            actual_B = target_B
        else:
            repeats = max(1, target_B // B_base)
            D_test = D_base.repeat(repeats, 1, 1)
            actual_B = D_test.shape[0]

        # Check memory
        if actual_B * Ld * d * 2 > gpu_mem_gb * 0.6 * 1e9:
            print(f"  B={actual_B}: SKIP (would exceed GPU memory)")
            continue

        try:
            t_naive, t_compiled, t_flash = bench_interleaved([naive_fp32, compiled_fp32, flash_maxsim], [[Q0, D_test]])
            n_sp = round(t_naive / t_flash, 1)
            c_sp = round(t_compiled / t_flash, 1)
            print(f"  B={actual_B:6d}: naive={t_naive:.3f}ms compiled={t_compiled:.3f}ms flash={t_flash:.3f}ms "
                  f"naive_speedup={n_sp}x compiled_speedup={c_sp}x")
            sweep_f["scoring"].append({
                "B": actual_B,
                "naive_ms": safe_round(t_naive),
                "compiled_ms": safe_round(t_compiled),
                "flash_ms": safe_round(t_flash),
                "naive_speedup": n_sp,
                "compiled_speedup": c_sp,
            })
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            print(f"  B={actual_B:6d}: SKIPPED ({e.__class__.__name__})")
            torch.cuda.synchronize()
        finally:
            if target_B > B_base:
                del D_test
            clear()

    del model, Q_all, D_base, Q0
    clear()

except ImportError:
    print("  pylate not installed -- skipping. Install: pip install pylate sentence-transformers")
except Exception as e:
    print(f"  FAILED: {e}")
    torch.cuda.synchronize()
    clear()

results["real_model"] = sweep_f


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════════════

gpu_tag = gpu_name.replace(" ", "_").replace("/", "-")
out = f"benchmarks/results_landing_{gpu_tag}.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'=' * 65}")
print(f"ALL DONE. Saved: {out}")
print("=" * 65)
