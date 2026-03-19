"""Fused GPU kernel for ColBERT/ColPali MaxSim scoring."""
__version__ = "0.1.0"
from .flash_maxsim import flash_maxsim, flash_maxsim_batched, flash_maxsim_train, maxsim_naive
from .flash_maxsim_quant import flash_maxsim_int8, quantize_int8, dequantize_int8
