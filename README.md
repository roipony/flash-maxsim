# Flash-MaxSim

Fused Triton GPU kernels for ColBERT/ColPali MaxSim scoring. The similarity matrix never touches HBM.

## Installation

```bash
pip install flash-maxsim
```

Or from source:

```bash
pip install -e .
```

Requires: PyTorch >= 2.0, Triton >= 2.1, CUDA GPU.

## Quick Start

```python
import torch
from flash_maxsim import flash_maxsim, flash_maxsim_batched, flash_maxsim_train
from flash_maxsim import flash_maxsim_int8, quantize_int8

# Single query vs document batch
Q = torch.randn(32, 128, device="cuda", dtype=torch.float16)   # [Lq, d]
D = torch.randn(1000, 256, 128, device="cuda", dtype=torch.float16)  # [B, Ld, d]
scores = flash_maxsim(Q, D)  # [B]

# Variable-length documents
lengths = torch.randint(64, 256, (1000,), device="cuda", dtype=torch.int32)
scores = flash_maxsim(Q, D, doc_lengths=lengths)

# Batched: multiple queries
Q_batch = torch.randn(10, 32, 128, device="cuda", dtype=torch.float16)
scores = flash_maxsim_batched(Q_batch, D, shared_docs=True)  # [Nq, B]

# INT8 quantized (4x memory reduction)
D_q, scales, mins = quantize_int8(D)
scores = flash_maxsim_int8(Q, D_q, scales, mins)  # [B]

# Training with autograd
Q.requires_grad_(True)
scores = flash_maxsim_train(Q, D)
scores.sum().backward()  # gradients flow through Q and D
```

## Benchmarks

H100 SXM, ColBERT config (Lq=32, d=128):

| B    | Ld   | Flash-MaxSim | PyTorch   | Speedup |
|------|------|-------------|-----------|---------|
| 100  | 256  | 0.04 ms     | 0.21 ms   | 5.3x    |
| 1000 | 256  | 0.31 ms     | 2.10 ms   | 6.8x    |
| 1000 | 512  | 0.58 ms     | 5.24 ms   | 9.0x    |
| 5000 | 256  | 1.48 ms     | 10.5 ms   | 7.1x    |

INT8 adds < 3% latency vs FP16 while using 2x less memory.

## API Reference

| Function | Signature | Description |
|----------|-----------|-------------|
| `flash_maxsim` | `(Q[Lq,d], D[B,Ld,d], doc_lengths?) -> [B]` | Single-query MaxSim |
| `flash_maxsim_batched` | `(Q[Nq,Lq,d], D[B,Ld,d], shared_docs?) -> [Nq,B]` | Multi-query MaxSim |
| `flash_maxsim_train` | `(Q[Lq,d], D[B,Ld,d]) -> [B]` | MaxSim with autograd |
| `flash_maxsim_int8` | `(Q, D_uint8, scales, mins, doc_lengths?) -> [B]` | Fused INT8 MaxSim |
| `quantize_int8` | `(D[B,Ld,d]) -> (D_uint8, scales, mins)` | Per-token INT8 quantization |
| `dequantize_int8` | `(D_uint8, scales, mins) -> D` | INT8 dequantization |
| `maxsim_naive` | `(Q[Lq,d], D[B,Ld,d], doc_lengths?) -> [B]` | Reference PyTorch impl |

## Citation

```bibtex
@software{flash_maxsim,
  title={Flash-MaxSim: Fused GPU Kernels for ColBERT/ColPali MaxSim},
  url={https://github.com/svg-project/flash-maxsim},
  year={2025}
}
```

## License

Apache 2.0
