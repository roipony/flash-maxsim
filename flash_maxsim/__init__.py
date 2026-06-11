"""Fused GPU kernel for ColBERT/ColPali MaxSim scoring."""
__version__ = "0.3.0"

# Core scoring
from .flash_maxsim import flash_maxsim, flash_maxsim_batched, flash_maxsim_train, flash_maxsim_pairs
from .flash_maxsim_batched_train import flash_maxsim_batched_train
from .naive import maxsim_naive, maxsim_naive_fp32, maxsim_at_operator, maxsim_bmm
from .naive import maxsim_chunked, maxsim_naive_batched, maxsim_naive_int8

# Variable-length pairs
from .flash_maxsim_varlen import flash_maxsim_varlen, pack_pairs

# Dense (equal-length) pairs — fast path used by flash_maxsim_pairs when
# inputs are already batched tensors.
from .flash_maxsim_pairs_dense import flash_maxsim_pairs_dense

# Shared-Q + packed-D (variable-length docs, no padding)
from .flash_maxsim_varlen import flash_maxsim_packed, pack_docs

# Zero-copy reranking
from .flash_maxsim_rerank import flash_maxsim_rerank, flash_maxsim_rerank_direct
# Batched padded rerank (drop-in for erikkaum/maxsim's score_candidates_padded)
from .flash_maxsim_rerank import flash_maxsim_rerank_padded

# INT8 quantization
from .flash_maxsim_quant import flash_maxsim_int8, quantize_int8, dequantize_int8
from .flash_maxsim_quant import flash_maxsim_int8x8, quantize_int8_symmetric, quantize_query_int8
from .flash_maxsim_quant import flash_maxsim_int8_symmetric

# Persistent grid-strided kernel (small N + varlen)
from .flash_maxsim import flash_maxsim_persistent

# Two-stage top-K: INT8 coarse scan + FP16 re-score
from .flash_maxsim_topk import flash_maxsim_topk

# Advanced kernels: Q-reuse (opt 2) and split-K (opt 3)
from .flash_maxsim_advanced import flash_maxsim_qreuse, flash_maxsim_splitk

# INT8-D training: save D as INT8 + per-token scale (phase 1, saves 50%
# on the long-lived D in autograd context; backward pre-dequants).
from .flash_maxsim_int8_train import flash_maxsim_int8_batched_train

# Warmup: pre-compile common kernels at startup so the first request
# in a serving system isn't slowed by Triton JIT (~200-500ms per shape).
from .warmup import warmup

# Chamfer distance: forward kernel + autograd (scatter_add) + invgrid CSR backward
from .flash_chamfer import (
    flash_chamfer, flash_chamfer_train,
    naive_chamfer, naive_chamfer_one_direction,
)
from .flash_chamfer_invgrid import flash_chamfer_invgrid_train
