"""End-to-end benchmark with real ColBERT/ColPali models.

Encodes real queries and documents, then benchmarks scoring.
Verifies that Flash-MaxSim produces identical rankings to naive.

Usage:
    pip install flash-maxsim pylate sentence-transformers datasets
    python benchmarks/bench_real_models.py

Produces: benchmarks/results_real_models_<GPU>.json
"""
import json, torch
import numpy as np
from utils import bench_interleaved, compile_high_precision

assert torch.cuda.is_available(), "CUDA required"
gpu_name = torch.cuda.get_device_name()
print(f"GPU: {gpu_name}\n")

from flash_maxsim import flash_maxsim, flash_maxsim_int8, quantize_int8


def naive_maxsim(Q, D):
    return torch.einsum('qd,bld->bql', Q.float(), D.float()).max(2).values.sum(1)

compiled_maxsim = compile_high_precision(naive_maxsim)

def encode_with_pylate(model, texts, is_query=False):
    """Encode texts with pylate model, return padded tensor on GPU."""
    embs = model.encode(texts, is_query=is_query)
    max_len = max(e.shape[0] for e in embs)
    d = embs[0].shape[1]
    T = torch.zeros(len(texts), max_len, d, device='cuda', dtype=torch.float16)
    for i, e in enumerate(embs):
        t = torch.tensor(np.array(e), device='cuda', dtype=torch.float16)
        T[i, :t.shape[0]] = t
    return T


results = {"gpu": gpu_name, "models": []}

# ── Model configs ──
model_configs = [
    {
        "name": "answerai-colbert-small-v1",
        "hf_id": "answerdotai/answerai-colbert-small-v1",
        "type": "text",
        "d": 128,
    },
]

# ── Queries and passages ──
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

# ── Run experiments per model ──
for cfg in model_configs:
    print(f"{'=' * 60}")
    print(f"Model: {cfg['name']} ({cfg['type']})")
    print(f"{'=' * 60}")

    try:
        from pylate import models
        model = models.ColBERT(cfg["hf_id"])
        print(f"  Loaded: {cfg['hf_id']}")
    except ImportError:
        print("  pylate not installed, skipping real model. Install: pip install pylate sentence-transformers")
        continue

    # Encode
    print("  Encoding queries and passages...")
    Q_all = encode_with_pylate(model, queries, is_query=True)
    D = encode_with_pylate(model, passages)
    Nq, Lq, d = Q_all.shape
    B, Ld, _ = D.shape
    print(f"  Q: {Q_all.shape}, D: {D.shape}")

    model_results = {
        "name": cfg["name"],
        "type": cfg["type"],
        "d": d,
        "Nq": Nq,
        "Lq": Lq,
        "B": B,
        "Ld": Ld,
        "scoring": [],
    }

    # Score each query
    print("\n  Scoring and ranking verification:")
    all_match = True
    for qi in range(Nq):
        Q = Q_all[qi]  # [Lq, d]

        score_naive = naive_maxsim(Q, D)
        score_flash = flash_maxsim(Q, D)

        rank_naive = score_naive.argsort(descending=True).tolist()
        rank_flash = score_flash.argsort(descending=True).tolist()

        match = rank_naive == rank_flash
        if not match:
            all_match = False
        print(f"    Q{qi}: top-3={rank_flash[:3]}  match={match}")

    print(f"  All rankings match: {all_match}")

    # Benchmark scoring speed at various scales
    print("\n  Scoring speed (single query):")

    # Use first query for benchmarking
    Q = Q_all[0]

    # Benchmark with original D
    t_naive, t_compiled, t_flash = bench_interleaved([naive_maxsim, compiled_maxsim, flash_maxsim], [[Q, D]])
    print(f"    B={B}: naive={t_naive:.3f}ms compiled={t_compiled:.3f}ms flash={t_flash:.3f}ms  "
          f"naive_speedup={t_naive/t_flash:.1f}x compiled_speedup={t_compiled/t_flash:.1f}x")

    model_results["scoring"].append({
        "B": B, "naive_ms": round(t_naive, 3), "flash_ms": round(t_flash, 3),
        "speedup": round(t_naive / t_flash, 1),
    })

    # Scale up D by repeating
    for scale in [10, 50, 100, 500]:
        B_scaled = B * scale
        if B_scaled * Ld * d * 2 > 40e9:  # skip if D would exceed GPU
            break
        D_scaled = D.repeat(scale, 1, 1)
        t_naive, t_compiled, t_flash = bench_interleaved([naive_maxsim, compiled_maxsim, flash_maxsim], [[Q, D_scaled]])
        print(f"    B={B_scaled}: naive={t_naive:.3f}ms compiled={t_compiled:.3f}ms flash={t_flash:.3f}ms  "
              f"naive_speedup={t_naive/t_flash:.1f}x compiled_speedup={t_compiled/t_flash:.1f}x")

        model_results["scoring"].append({
            "B": B_scaled, "naive_ms": round(t_naive, 3), "flash_ms": round(t_flash, 3),
            "speedup": round(t_naive / t_flash, 1),
        })
        del D_scaled; torch.cuda.empty_cache()

    # INT8
    print("\n  INT8 quantized scoring:")
    D_q, scales, mins = quantize_int8(D)
    t_fp16, t_q8 = bench_interleaved([flash_maxsim, flash_maxsim_int8], [[Q, D], [Q, D_q, scales, mins]])
    score_fp16 = flash_maxsim(Q, D)
    score_q8 = flash_maxsim_int8(Q, D_q, scales, mins)
    rank_match = (score_fp16.argsort(descending=True)[:5] == score_q8.argsort(descending=True)[:5]).all().item()
    err = (score_fp16 - score_q8).abs().max().item()
    print(f"    FP16={t_fp16:.3f}ms  INT8={t_q8:.3f}ms  compression={D.nbytes/D_q.nbytes:.0f}x  rank_match={rank_match}  max_err={err:.4f}")

    model_results["int8"] = {
        "fp16_ms": round(t_fp16, 3), "q8_ms": round(t_q8, 3),
        "compression": round(D.nbytes / D_q.nbytes, 1),
        "rank_match": rank_match, "max_err": round(err, 4),
    }

    # Memory comparison
    print("\n  Peak memory (single query):")
    for scale in [1, 50, 500]:
        B_test = B * scale
        if B_test * Ld * d * 2 > 40e9:
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
        print(f"    B={B_test}: naive={naive_mb:.1f}MB  flash={flash_mb:.2f}MB  {ratio:.0f}x less")

        if scale > 1:
            del D_test; torch.cuda.empty_cache()

    results["models"].append(model_results)
    del model; torch.cuda.empty_cache()

# ── Save ──
gpu_tag = gpu_name.replace(" ", "_").replace("/", "-")
out = f"benchmarks/results_real_models_{gpu_tag}.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out}")
