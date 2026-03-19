# Flash-MaxSim

Fused Triton GPU kernel for ColBERT/ColPali MaxSim scoring. Up to **13x faster**, **143x less memory**. The similarity matrix never touches HBM.

## Get Started (copy-paste)

```bash
git clone -b release --single-branch git@github.ibm.com:Video-AI/flash_maxsim.git
cd flash_maxsim
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest tests/ -v
python benchmarks/bench.py

# For the notebook (includes pylate, jupyter, matplotlib):
uv pip install -e ".[dev,notebook]"
jupyter notebook examples/demo_notebook.ipynb
```

## Quick Start

```python
import torch
from flash_maxsim import flash_maxsim, flash_maxsim_batched

# Single query scoring
Q = torch.randn(32, 128, device="cuda", dtype=torch.float16)
D = torch.randn(1000, 300, 128, device="cuda", dtype=torch.float16)
scores = flash_maxsim(Q, D)  # [1000]

# Batched: all queries vs all docs
Q_batch = torch.randn(100, 32, 128, device="cuda", dtype=torch.float16)
scores = flash_maxsim_batched(Q_batch, D, shared_docs=True)  # [100, 1000]

# INT8 quantized (2x compression, same speed)
from flash_maxsim import flash_maxsim_int8, quantize_int8
D_q, scales, mins = quantize_int8(D)
scores = flash_maxsim_int8(Q, D_q, scales, mins)

# Training (autograd)
from flash_maxsim import flash_maxsim_train
Q = torch.nn.Parameter(Q)
scores = flash_maxsim_train(Q, D)
scores.sum().backward()  # gradients to Q and D
```

## Benchmarks (H100 80GB)

### Single Query (vs naive FP32 einsum)

| Config | Naive | Flash | Speedup |
|--------|-------|-------|---------|
| ColBERT (Lq=32, Ld=300, B=1000) | 0.27 ms | 0.07 ms | **3.9x** |
| ColPali text (Lq=32, Ld=1024, B=500) | 0.42 ms | 0.09 ms | **5.0x** |
| ColPali image (Lq=1024, Ld=1024, B=1000) | 9.19 ms | 0.83 ms | **11.1x** |
| ColPali image (Lq=1024, Ld=1024, B=5000) | 46.51 ms | 3.77 ms | **12.3x** |

### INT8 Fused Dequantization

| Config | Naive INT8 | Flash Q8 | Speedup |
|--------|-----------|----------|---------|
| ColBERT (B=1000) | 0.54 ms | 0.08 ms | **7.0x** |
| ColBERT (B=5000) | 2.40 ms | 0.19 ms | **12.4x** |

### Batched Multi-Query

| Config | Naive | Flash | Throughput |
|--------|-------|-------|-----------|
| 100q × 1000d | 26.1 ms | 2.43 ms | **41.2M pairs/s** |
| 100q × 100p (Lq=Ld=1024) | 97.3 ms | 5.96 ms | **16.3x** |

### Peak Memory

| Config | Naive | Flash | Reduction |
|--------|-------|-------|-----------|
| 1q × 1000p (Lq=Ld=1024) | 4.7 GB | 0.01 GB | **470x** |
| 10q × 1000p (Lq=Ld=1024) | 42.5 GB | 0.01 GB | **4247x** |

## How It Works

```
Q_block = load(Q)                      # SRAM
m = [-inf] * Lq                        # registers

for tile in D.tiles(BLOCK_D):
    D_tile = load(tile)                # SRAM
    S = tl.dot(Q_block, D_tile.T)     # tensor cores — SRAM only
    m = max(m, S.max(axis=1))         # online max
    # S dies here — never in HBM

score = sum(m)                          # → HBM
```

Same pattern as Flash Attention, but simpler: `max` is trivially composable (no rescaling needed unlike `softmax`).

## API

| Function | Input → Output | Description |
|----------|---------------|-------------|
| `flash_maxsim` | `[Lq,d] × [B,Ld,d] → [B]` | Single query |
| `flash_maxsim_batched` | `[Nq,Lq,d] × [B,Ld,d] → [Nq,B]` | Multi-query |
| `flash_maxsim_int8` | `[Lq,d] × [B,Ld,d] uint8 → [B]` | Fused INT8 |
| `flash_maxsim_train` | `[Lq,d] × [B,Ld,d] → [B]` | With autograd |
| `quantize_int8` | `[B,Ld,d] → uint8 + scales + mins` | Quantization |
| `maxsim_naive` | `[Lq,d] × [B,Ld,d] → [B]` | Reference |

## Files

```
flash_maxsim/
  flash_maxsim.py        # FP16 + batched + training kernels (290 lines)
  flash_maxsim_quant.py  # INT8 fused kernel (140 lines)
tests/
  test_flash_maxsim.py   # pytest suite
benchmarks/
  bench.py               # full benchmark
examples/
  demo.py                # real model demo
  demo_notebook.ipynb    # interactive notebook with plots
```

## License

Apache 2.0
