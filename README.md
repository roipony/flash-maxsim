# Flash-MaxSim

Fused Triton GPU kernel for ColBERT/ColPali MaxSim scoring. **2.5–2.9x faster** than FP16 eager PyTorch, **3.9x / 4.7x** (A100 / H100) at matched FP32-accumulation precision, **up to 4.6x on variable-length corpora**, and **2.6–5.1x faster than `torch.compile(max-autotune)`** — the strongest PyTorch configuration, which most public kernels never benchmark against. The `B × Lq × Ld` similarity matrix is never materialized. Drop-in replacement — same API, no configuration.

```bash
pip install flash-maxsim
```

```python
from flash_maxsim import flash_maxsim

scores = flash_maxsim(Q, D)  # that's it
```

## Why Flash-MaxSim?

Every existing MaxSim implementation computes and stores the full similarity matrix in GPU memory. Flash-MaxSim eliminates it — the matrix never exists outside the chip.

### 1. Memory: the similarity matrix is gone

The standard `einsum` / `bmm` approach allocates `B × Lq × Ld × 2 bytes` for the similarity matrix. For ColPali at 10K docs, that's **21 GB** — instant OOM.

| ColPali (Lq=1024, Ld=1024) | Naive sim matrix | Flash-MaxSim |
|---|---|---|
| B=1,000 | 2,097 MB | **0 MB** |
| B=5,000 | 10,486 MB | **0 MB** |
| B=10,000 | 20,972 MB | **0 MB** |

That's 21 GB of temporary memory just for scoring — on top of model weights, KV cache, and the embeddings themselves. On a 40 GB GPU, this OOMs. On 80 GB, it eats a quarter of your memory for a tensor that gets immediately reduced and thrown away.

Flash-MaxSim uses zero extra HBM. The similarity is computed tile-by-tile in SRAM and reduced on the fly.

### 2. Speed: 2.5–2.9x vs FP16 eager, 3.9–4.7x at matched precision

All timings cast-hoisted and CUDA-event-measured (medians); raw JSON in `benchmarks/bench_fwd_fair_*.json` and `benchmarks/bench_chunked_fp16_eager_*.json`.

Per-shape at B=1K, vs naive einsum at **matched precision** (FP32 accumulation, TF32 tensor cores, cast hoisted out of the timed region):

| Shape (Lq, Ld) | A100 | H100 |
|---|---|---|
| textual (32, 300) | 1.4x | 1.2x |
| long-doc (32, 1024) | 2.0x | 1.8x |
| medium (128, 1024) | 3.0x | 3.3x |
| visual (512, 1024) | 3.5x | 4.2x |
| ColPali (1024, 1024) | **3.9x** | **4.7x** |

Vs `torch.compile(mode="max-autotune")` of the same expression — the strongest PyTorch baseline, with CUDA graphs and Inductor autotuning — Flash is **2.6–5.1x** across the five shapes (peak 5.1x at medium Lq=128; 3.8x at ColPali; `benchmarks/bench_compile_ma_*.json`). Most published kernel comparisons stop at eager; we report both because `compile` narrows the latency gap but cannot remove the materialized `[B, Lq, Ld]` intermediate — the memory profile and OOM cliffs are compile-invariant.

Vs the fastest (but less precise) baseline — plain FP16 eager einsum, including its production chunked variant — Flash is 2.5–2.9x faster with 5–9x lower peak memory (A100, ColPali):

| B | FP16 eager | Chunked FP16 eager (best chunk) | Flash | Flash peak | Eager peak |
|---|---|---|---|---|---|
| 1,000 | 4.3 ms (2.1x) | 4.3 ms (2.1x) | **2.0 ms** | 0.3 GB | 2.4 GB |
| 10,000 | 45.9 ms (2.8x) | 43.0 ms (2.6x) | **16.4 ms** | 2.6 GB | 23.9 GB |
| 20,000 | 92.9 ms (2.8x) | 86.3 ms (2.6x) | **32.6 ms** | 5.3 GB | 47.7 GB |

### 3. Zero parameters — no chunk size to tune

Production systems (vLLM, etc.) chunk documents into mini-batches to avoid OOM. Too large → OOM. Too small → launch overhead. Flash-MaxSim has **zero configuration** — same code on a 16 GB GPU and an 80 GB GPU.

### 4. Variable-length documents — zero padding waste

Real collections have variable doc lengths, and this is where the gap is widest: padding wastes compute proportional to the fill ratio. At matched precision vs padded naive einsum (B=1K, Ld_max=512, A100): **3.2x** on uniform lengths, **4.3x** at a HotpotQA-like length distribution, **4.6x** on highly ragged collections (`benchmarks/bench_varlen_buckets_*.json`). Flash-MaxSim supports packed variable-length documents:

```python
from flash_maxsim import flash_maxsim_packed, pack_docs

D_packed, cu_seqlens, max_ld = pack_docs(doc_embeddings)
scores = flash_maxsim(Q, D_packed, doc_lengths=cu_seqlens)
```

| Regime | N | Speedup | Padding saved |
|---|---|---|---|
| ColBERT skewed (avg_Ld≈49) | 100K | **5.1x** | 39% |
| ColBERT uniform | 100K | **2.7x** | 42% |
| ColPali uniform | 500 | **4.2x** | 37% |
| ColPali skewed | 5K | **3.9x** | 19% |

### 5. INT8 index — half storage, faster, more precise

Store embeddings as INT8 (2x compression). The kernel uses **INT8 tensor cores** (624 TOPS on A100 — 2x FP16 throughput). No dequantization in HBM.

```python
from flash_maxsim import flash_maxsim_int8x8, quantize_int8_symmetric

# Index time: quantize once (50% storage savings)
D_int8, scales = quantize_int8_symmetric(D)

# Query time: drop-in
scores = flash_maxsim_int8x8(Q, D_int8, scales)
```

| Method (ColPali B=5K) | Latency | D Storage | Extra HBM | Precision |
|---|---|---|---|---|
| Naive dequant+einsum | 30.9 ms | 1 byte/dim | D_fp16 copy + sim matrix | 0.065 |
| Flash FP16 | 8.0 ms | 2 bytes/dim | ~0 | **0.00008** |
| **Flash INT8×INT8** | **6.6 ms** | **1 byte/dim** | **~0** | 0.023 |

Flash INT8×INT8 is **4.7x faster** than naive dequant, uses **half the storage**, and is **3x more precise** (FP32 accumulation vs FP16 einsum).

### 6. Training — autograd backward pass

Full gradient support via saved argmax indices. Sparse backward — no full matrix in either direction:

```python
# Single query (e.g. cross-encoder rerank training)
from flash_maxsim import flash_maxsim_train
scores = flash_maxsim_train(Q, D)              # Q: [Lq, d], D: [B, Ld, d]
scores.sum().backward()                         # gradients to both Q and D

# Batched (contrastive / in-batch negatives — new in v0.2.1)
from flash_maxsim import flash_maxsim_batched_train
scores = flash_maxsim_batched_train(            # Q: [Nq, Lq, d], D: [B, Ld, d]
    Q_batch, D, shared_docs=True,               #   shared_docs=True for contrastive
    doc_lengths=d_lens, query_lengths=q_lens,   #   varlen — masks padded tokens
)                                                # → scores [Nq, B]
scores.diagonal().sum().backward()              # gradients to Q_batch and D

# Knowledge distillation (each query has its own doc set)
scores = flash_maxsim_batched_train(            # Q: [Nq, Lq, d], D: [Nq, B, Ld, d]
    Q_batch, D_per_query, shared_docs=False,    # → scores [Nq, B]
)
```

The batched path uses an inverse-grid CSR backward (atomic-free, runs on
tensor cores) when work is non-trivial, falling back to FP32-atomic scatter
otherwise. Saved activations are O(Nq × B × Lq) argmax indices instead of
the full O(Nq × B × Lq × Ld) similarity matrix that vanilla autograd would
materialize — **95–205× less scoring memory** at typical contrastive shapes,
**1.4–3.8× faster** full training step than `colbert_scores`-style baselines
on A100, and **lifts the OOM ceiling 2×** (e.g. ColPali contrastive B=128
becomes feasible on a single 80 GB A100).

Verified bit-exact for grad_Q vs FP32 reference at fixed-length shapes;
cosine similarity > 0.999 across all tested batched shapes; correct under
variable-length inputs even when padded query positions hold non-zero values.

### 7. 800x more precise

Flash-MaxSim uses FP32 accumulation for the running max and score sum. The standard FP16 einsum has compounding rounding errors:

| Method | Mean error vs FP32 | Top-20 overlap | Spearman |
|---|---|---|---|
| FP16 naive (einsum) | 6.2×10⁻² | 95% | 0.993 |
| **Flash FP16** | **7.6×10⁻⁵** | **100%** | **1.000** |
| Flash INT8×INT8 | 2.3×10⁻² | 100% | 0.999 |

## Quick Start

```python
import torch
from flash_maxsim import flash_maxsim

# Score one query against 1000 documents
Q = torch.randn(32, 128, device="cuda", dtype=torch.float16)   # query: 32 tokens
D = torch.randn(1000, 300, 128, device="cuda", dtype=torch.float16)  # 1000 docs, 300 tokens each
scores = flash_maxsim(Q, D)  # [1000]

# ColPali (long query) — automatic chunking, no configuration needed
Q_colpali = torch.randn(1024, 128, device="cuda", dtype=torch.float16)
D_colpali = torch.randn(1000, 1024, 128, device="cuda", dtype=torch.float16)
scores = flash_maxsim(Q_colpali, D_colpali)  # [1000]

# Batched: 16 queries vs same corpus (up to 15x faster than serial loop)
Q_batch = torch.randn(16, 32, 128, device="cuda", dtype=torch.float16)
scores = flash_maxsim_batched(Q_batch, D, shared_docs=True)  # [16, 1000]
```

### Variable-Length Documents

```python
from flash_maxsim import flash_maxsim_varlen, pack_pairs

# Each pair has different lengths — zero padding waste
q_embs = [torch.randn(32, 128, ...), torch.randn(48, 128, ...)]
d_embs = [torch.randn(180, 128, ...), torch.randn(250, 128, ...)]

Q_packed, D_packed, cu_q, cu_d, max_lq, max_ld = pack_pairs(q_embs, d_embs)
scores = flash_maxsim_varlen(Q_packed, D_packed, cu_q, cu_d, max_lq, max_ld)
```

### INT8 Index

```python
from flash_maxsim import flash_maxsim_int8x8, quantize_int8_symmetric

# Index time (once): 50% smaller storage
D_int8, scales = quantize_int8_symmetric(D)

# Query time: INT8 tensor cores, zero overhead
scores = flash_maxsim_int8x8(Q, D_int8, scales)
```

### Training

```python
# Single query
from flash_maxsim import flash_maxsim_train

Q = torch.randn(32, 128, device="cuda", dtype=torch.float16, requires_grad=True)
D = torch.randn(100, 300, 128, device="cuda", dtype=torch.float16, requires_grad=True)
scores = flash_maxsim_train(Q, D)
scores.sum().backward()                   # Q.grad and D.grad

# Batched contrastive training (new in v0.2.1)
from flash_maxsim import flash_maxsim_batched_train

Q = torch.randn(64, 32, 128, device="cuda", dtype=torch.float16, requires_grad=True)
D = torch.randn(64, 300, 128, device="cuda", dtype=torch.float16, requires_grad=True)
scores = flash_maxsim_batched_train(Q, D, shared_docs=True)   # [64, 64] scores
scores.diagonal().sum().backward()        # contrastive loss → grads
```

### Zero-Copy Reranking

Score documents directly from a model's output tensor — zero additional memory:

```python
from flash_maxsim import flash_maxsim_rerank_direct

scores = flash_maxsim_rerank_direct(
    Q, batch_tensor, doc_offsets, doc_lengths, max_ld
)  # 0 bytes allocated for scoring
```

## How It Works

```
Q_block = load(Q)                      # SRAM (small — one query)
m = [-inf] * Lq                        # registers (running max per query token)

for tile in D.tiles(BLOCK_D):
    D_tile = load(tile)                # SRAM
    S = tl.dot(Q_block, D_tile.T)     # tensor cores — stays in SRAM
    m = max(m, S.max(axis=1))         # online max reduction
    # S dies here — never written to HBM

score = sum(m)                          # one scalar per doc → HBM
```

Same principle as Flash Attention, but simpler: `max` is trivially composable across tiles (no log-sum-exp rescaling needed).

## API Reference

### Core Scoring
| Function | Signature | Description |
|---|---|---|
| `flash_maxsim` | `[Lq,d] × [B,Ld,d] → [B]` | Single query, auto-chunking for long queries |
| `flash_maxsim_batched` | `[Nq,Lq,d] × [B,Ld,d] → [Nq,B]` | Multi-query (shared or per-query docs) |
| `flash_maxsim_varlen` | packed Q,D + cu_seqlens → `[N]` | Variable-length pairs, zero padding |
| `flash_maxsim_packed` | `[Lq,d] × packed [T,d] + cu_seqlens → [B]` | Shared Q + variable-length packed D |

### INT8 Quantization
| Function | Description |
|---|---|
| `flash_maxsim_int8x8` | True INT8×INT8 tensor core scoring (recommended) |
| `quantize_int8_symmetric` | Per-token symmetric INT8 quantization for D |
| `quantize_query_int8` | Per-token INT8 quantization for Q |
| `flash_maxsim_int8` | Legacy: fused affine INT8 dequant+scoring |

### Training & Utilities
| Function | Signature | Description |
|---|---|---|
| `flash_maxsim_train` | `[Lq,d] × [B,Ld,d] → [B]` | Single-query MaxSim with autograd backward (sparse argmax) |
| `flash_maxsim_batched_train` | `[Nq,Lq,d] × [B,Ld,d] → [Nq,B]` | **Batched** MaxSim with autograd — for contrastive in-batch negatives or KD; supports `shared_docs`, `doc_lengths`, `query_lengths` |
| `flash_maxsim_int8_batched_train` | same signature | Saves D as INT8 in the autograd context. Drop-in for the niche case where the caller releases the FP16 D between forward and backward; see `CHANGELOG.md` for the honest peak-memory measurement |
| `flash_maxsim_rerank_direct` | scattered batch tensor → `[B]` | Zero-copy scoring from a serving model's output |
| `pack_pairs` | list of (q, d) → packed | Variable-length (Q, D) pair packing into cu_seqlens format |
| `pack_docs` | list of D → packed | Variable-length doc packing for `flash_maxsim_packed` |
| `maxsim_naive` | `[Lq,d] × [B,Ld,d] → [B]` | Pure PyTorch reference (FP16 einsum) |

### Serving / production utilities
| Function | Description |
|---|---|
| `warmup()` | Pre-compile every kernel specialization the dispatcher can pick at runtime. Call once at server startup so the first request doesn't pay Triton JIT cost. CLI: `python -m flash_maxsim.warmup`. |

### What's new in v0.3.0
See `CHANGELOG.md` for the full list. Highlights:
- **Auto-routes to a split-d forward** at `d > 512` so fat-embedding
  encoders (Jina v2 d=512, Granite-Embedding d=384/768, Voyage v2
  d=1024, NV-Embed d=4096) run without the SRAM-spill latency cliff
  the standard kernel hits at those dims.
- **Backward unified kernel** (atomic-mode default for small shapes):
  fused dQ+dD in one launch with Q register hoisting — a single D
  load serves both gradients.
- **Per-arch heuristic launch-config table** replaces Triton's
  autotune — deterministic launch every call (CUDA-graph friendly),
  no first-call trial overhead.
- **int32 CSR build** for the invgrid backward at large shapes:
  25-35% lower CSR transient memory at ColPali B=128.

## Requirements

- NVIDIA GPU (Ampere or newer recommended)
- PyTorch >= 2.0
- Triton >= 3.4
- CUDA

Tested on: H100 80GB, A100 80GB/40GB, V100.

## Authors

**IBM Research Israel**

- Roi Pony
- Adi Raz Goldfarb
- Idan Friedman
- Udi Barzelay

## License

Apache 2.0
