from typing import List, Callable, Sequence
import torch
import numpy as np
import time

def flush_l2_cache():
    tmp = torch.empty(100 * 1024 * 1024, device="cuda") # Usually 100MB is sufficient to flush L2 cache
    tmp.zero_()

"""
Benchmark the functions by interleaving invocations to reduce GPU noise
Inputs:
    - fns: list of functions
    - args_list: list of arguments to functions. If the list has length 1, the arguments will be used for all functions.
    - warmup: number of warmup iterations
    - n: number of test iterations
"""
def bench_interleaved(fns: List[Callable], args_list: List[Sequence], warmup=20, n=50):
    # Broadcast args
    if len(args_list) == 1:
        args_list = [args_list[0]] * len(fns)

    # Warmup all functions
    for fn, args in zip(fns, args_list):
        for _ in range(warmup):
            fn(*args)
            torch.cuda.synchronize()

    # bench interleaved
    ts = np.zeros((n, len(fns)))
    for i in range(n):
        for j, (fn, args) in enumerate(zip(fns, args_list)):
            flush_l2_cache()
            torch.cuda.synchronize()
            start = time.perf_counter()
            fn(*args)
            torch.cuda.synchronize()
            end = time.perf_counter()
            ts[i][j] = (end - start) * 1000
    return np.median(ts, axis=0)


def print_timing_results(wl_label, naive_time, compiled_time, flash_time):
    print(f"  {wl_label:35s}: naive={naive_time:.2f}ms compiled={compiled_time:.2f}ms flash={flash_time:.2f}ms "
          f"naive_speedup={naive_time / flash_time:.1f}x compiled_speedup={compiled_time / flash_time:.1f}x")

def compile_high_precision(fn: Callable):
    """Bug-fix: original missed the `return`, so all callers got None.
    This wrapper compiles `fn` with TF32-enabled FP32 matmul ('high' precision).
    """
    torch.set_float32_matmul_precision('high')
    return torch.compile(fn)