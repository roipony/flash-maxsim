"""Fused GPU kernel for ColBERT/ColPali MaxSim scoring."""
__version__ = "0.2.1"

# Core scoring
from .flash_maxsim import flash_maxsim, flash_maxsim_batched, flash_maxsim_train
from .flash_maxsim_batched_train import flash_maxsim_batched_train
from .naive import maxsim_naive, maxsim_naive_fp32, maxsim_at_operator, maxsim_bmm
from .naive import maxsim_chunked, maxsim_naive_batched, maxsim_naive_int8

# Variable-length pairs
from .flash_maxsim_varlen import flash_maxsim_varlen, pack_pairs

# Shared-Q + packed-D (variable-length docs, no padding)
from .flash_maxsim_varlen import flash_maxsim_packed, pack_docs

# Zero-copy reranking
from .flash_maxsim_rerank import flash_maxsim_rerank, flash_maxsim_rerank_direct

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
