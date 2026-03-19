"""
Flash-MaxSim Demo: Real ColBERT model on real queries.

Shows Flash-MaxSim as a drop-in replacement for naive MaxSim scoring
with a real pretrained model and real text queries.

    pip install flash-maxsim pylate sentence-transformers
    python examples/demo.py
"""
import time
import torch
import torch.nn.functional as F

assert torch.cuda.is_available(), "This demo requires a CUDA GPU"
print(f"GPU: {torch.cuda.get_device_name()}\n")

# ── 1. Load a real ColBERT model ─────────────────────────────────
print("Loading ColBERT model...")
try:
    from pylate import models
    model = models.ColBERT("answerdotai/answerai-colbert-small-v1")
    print("  Model: answerai-colbert-small-v1\n")
except ImportError:
    print("  PyLate not installed. Run: pip install pylate sentence-transformers")
    print("  Falling back to simulated embeddings.\n")
    model = None

# ── 2. Encode real queries and passages ──────────────────────────
queries = [
    "What is information retrieval?",
    "How does ColBERT work?",
    "What are transformer models used for?",
    "Explain neural search engines",
    "What is late interaction in retrieval?",
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
]

if model is not None:
    print(f"Encoding {len(queries)} queries and {len(passages)} passages...")
    q_embs = model.encode(queries, is_query=True)
    p_embs = model.encode(passages)

    # Convert to padded tensors
    import numpy as np
    max_lq = max(e.shape[0] for e in q_embs)
    max_lp = max(e.shape[0] for e in p_embs)
    d = q_embs[0].shape[1]

    Q = torch.zeros(len(queries), max_lq, d, device='cuda', dtype=torch.float16)
    for i, e in enumerate(q_embs):
        t = torch.tensor(np.array(e), device='cuda', dtype=torch.float16)
        Q[i, :t.shape[0]] = t

    D = torch.zeros(len(passages), max_lp, d, device='cuda', dtype=torch.float16)
    for i, e in enumerate(p_embs):
        t = torch.tensor(np.array(e), device='cuda', dtype=torch.float16)
        D[i, :t.shape[0]] = t
else:
    # Simulated
    d = 128
    Q = F.normalize(torch.randn(len(queries), 32, d, device='cuda', dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(len(passages), 300, d, device='cuda', dtype=torch.float16), dim=-1)

Nq, Lq, d = Q.shape
Np, Ld, _ = D.shape
print(f"  Q: {Q.shape}  D: {D.shape}\n")

# ── 3. Score with naive PyTorch ──────────────────────────────────
print("=" * 55)
print("Scoring: Naive PyTorch vs Flash-MaxSim")
print("=" * 55)

# Flash-MaxSim
from flash_maxsim import flash_maxsim_batched

# Warmup BOTH (includes Triton compilation for Flash)
print("  Warming up...", end=" ", flush=True)
for _ in range(5):
    _ = torch.einsum('nqd,bld->nbql', Q.float(), D.float()).max(3).values.sum(2)
    _ = flash_maxsim_batched(Q, D, shared_docs=True)
torch.cuda.synchronize()
print("done")

# Naive
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(20):
    S_naive = torch.einsum('nqd,pld->npql', Q.float(), D.float())
    scores_naive = S_naive.max(dim=3).values.sum(dim=2)
torch.cuda.synchronize()
t_naive = (time.perf_counter() - t0) * 1000 / 20

# Flash
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(20):
    scores_flash = flash_maxsim_batched(Q, D, shared_docs=True)
torch.cuda.synchronize()
t_flash = (time.perf_counter() - t0) * 1000 / 20

print(f"  Naive:  {t_naive:.3f} ms")
print(f"  Flash:  {t_flash:.3f} ms")
if t_naive > t_flash:
    print(f"  Speedup: {t_naive/t_flash:.1f}x")
else:
    print(f"  (Small input — speedup shows at larger scale)")
print()

# ── 4. Show rankings ─────────────────────────────────────────────
print("=" * 55)
print("Rankings (top-3 passages per query)")
print("=" * 55)

for i, q in enumerate(queries):
    naive_rank = scores_naive[i].argsort(descending=True)[:3].tolist()
    flash_rank = scores_flash[i].argsort(descending=True)[:3].tolist()
    match = naive_rank == flash_rank

    print(f"\n  Query: \"{q}\"")
    print(f"  Naive top-3: {naive_rank}")
    print(f"  Flash top-3: {flash_rank}  {'MATCH' if match else 'DIFFER'}")
    print(f"  Best passage: \"{passages[flash_rank[0]]}\"")

# ── 5. INT8 quantized scoring ────────────────────────────────────
print(f"\n{'=' * 55}")
print("INT8 Quantized Scoring (2x compression)")
print("=" * 55)

from flash_maxsim import flash_maxsim_int8, quantize_int8

D_q, scales, mins = quantize_int8(D)
print(f"  D original: {D.nbytes/1e3:.0f} KB")
print(f"  D quantized: {D_q.nbytes/1e3:.0f} KB ({D.nbytes/D_q.nbytes:.1f}x smaller)")

# Score each query with INT8
for i, q in enumerate(queries[:3]):
    score_fp = scores_flash[i]
    score_q8 = flash_maxsim_int8(Q[i], D_q, scales, mins)
    fp_rank = score_fp.argsort(descending=True)[:3].tolist()
    q8_rank = score_q8.argsort(descending=True)[:3].tolist()
    err = (score_fp - score_q8).abs().max().item()
    print(f"\n  \"{q}\"")
    print(f"    FP16 top-3: {fp_rank}")
    print(f"    INT8 top-3: {q8_rank}  err={err:.4f}  {'MATCH' if fp_rank==q8_rank else 'DIFFER'}")

# ── 6. Memory comparison ─────────────────────────────────────────
print(f"\n{'=' * 55}")
print("Memory: Similarity matrix eliminated")
print("=" * 55)

sim_bytes = Nq * Np * Lq * Ld * 4
print(f"  Sim matrix [Nq={Nq}, Np={Np}, Lq={Lq}, Ld={Ld}]: {sim_bytes/1e6:.1f} MB")
print(f"  Naive allocates this in GPU memory")
print(f"  Flash-MaxSim: 0 bytes (computed in SRAM tiles)")

print(f"\nDone!")
