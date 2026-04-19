# Flash-MaxSim

Fused Triton GPU kernel for ColBERT/ColPali MaxSim scoring. **Up to 6.5x faster** at matched precision, zero memory overhead. Drop-in replacement — same API, no configuration.

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

### 2. Speed: up to 6.5x faster at matched precision (A100)

Flash-MaxSim uses FP32 accumulation — more precise than FP16 naive. For a fair comparison, naive must also reduce in FP32 (`.float().max().sum()`), which adds the cost of casting the sim matrix. At matched precision:

| Config (A100) | Naive (matched) | Flash | Speedup |
|---|---|---|---|
| ColBERT B=10K | 1.61 ms | 0.51 ms | **3.2x** |
| ColBERT B=100K | 16.36 ms | 4.28 ms | **3.8x** |
| ColPali B=100 | 1.11 ms | 0.27 ms | **4.0x** |
| ColPali B=1K | 10.09 ms | 1.63 ms | **6.2x** |
| ColPali B=10K | 100.63 ms | 15.49 ms | **6.5x** |

Even vs the fastest FP16 naive (less precise), flash is 2.5–2.9x faster:

| B (ColPali) | vs FP16 naive | vs matched naive | Sim matrix eliminated |
|---|---|---|---|
| 1,000 | **2.6x** | **6.2x** | 2.1 GB |
| 5,000 | **2.9x** | **6.4x** | 10.5 GB |
| 10,000 | **2.9x** | **6.5x** | 21.0 GB |

### 3. Zero parameters — no chunk size to tune

Production systems (vLLM, etc.) chunk documents into mini-batches to avoid OOM. Too large → OOM. Too small → launch overhead. Flash-MaxSim has **zero configuration** — same code on a 16 GB GPU and an 80 GB GPU.

### 4. Variable-length documents — zero padding waste

Real collections have variable doc lengths. Padding wastes compute. Flash-MaxSim supports packed variable-length documents:

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
from flash_maxsim import flash_maxsim_train

scores = flash_maxsim_train(Q, D)  # Q: [Lq, d], D: [B, Ld, d]
scores.sum().backward()            # gradients to both Q and D
```

Verified correct against naive backward (max gradient error < 0.001).

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
from flash_maxsim import flash_maxsim_train

Q = torch.randn(32, 128, device="cuda", dtype=torch.float16, requires_grad=True)
D = torch.randn(100, 300, 128, device="cuda", dtype=torch.float16)
scores = flash_maxsim_train(Q, D)
loss = scores.sum()
loss.backward()  # Q.grad computed via sparse argmax backward
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
| Function | Description |
|---|---|
| `flash_maxsim_train` | MaxSim with autograd backward (sparse argmax) |
| `flash_maxsim_rerank_direct` | Zero-copy scoring from scattered batch tensor |
| `pack_pairs` | Pack variable-length (Q, D) pairs into cu_seqlens format |
| `pack_docs` | Pack variable-length docs for `flash_maxsim_packed` |
| `maxsim_naive` | Pure PyTorch reference (FP16 einsum) |

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
