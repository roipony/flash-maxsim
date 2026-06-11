"""Quantify memory + scaling overhead of the invgrid (CSR) backward proposal.

Reports per shape:
  - Memory cost of CSR (col_idx, row_ptr) and grad_D_fp32, vs argmax baseline
    and vs the would-be sim matrix.
  - Median build_inverse_csr() wall time (5 trials, GPU sync'd).
  - dD kernel-only time, end-to-end backward time for both the new (invgrid)
    and the existing FP32-atomic path.
  - Worst-case behaviour: every (q, l) collides on dest j=0.

Run with the pylate env / 1xA100-80; sample command:
    submit_A100_80 -g 1 -inf ~/envs/pylate_env/bin/python \
        /path/to/data
"""
from __future__ import annotations

import gc
import math
import sys
import time
from statistics import median

import torch

sys.path.insert(0, "/path/to/data")

from flash_maxsim.flash_maxsim import _next_pow2, _launch_fwd, _default_lengths  # noqa: E402
from flash_maxsim.flash_maxsim_batched_train import flash_maxsim_batched_train as flash_atomic  # noqa: E402
from flash_maxsim.flash_maxsim_invgrid_proposal import (  # noqa: E402
    build_inverse_csr,
    flash_maxsim_invgrid_train,
    _maxsim_bwd_dD_invgrid_kernel,
)
from flash_maxsim.flash_maxsim_batched_train import _maxsim_bwd_dD_batched_kernel  # noqa: E402

device = torch.device("cuda")
DTYPE = torch.float16
D_DIM = 128

SHAPES = [
    # (Nq, B, Lq, Ld, label)
    (32, 32, 32, 180, "ColBERT-cont"),
    (128, 128, 32, 180, "ColBERT-B128"),
    (1024, 1024, 32, 180, "ColBERT-B1024"),
    (32, 32, 1024, 1024, "ColPali-B32"),
    (64, 64, 1024, 1024, "ColPali-B64"),
    (256, 256, 1024, 1024, "ColPali-B256"),
]


def reset():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def gpu_time_median(fn, n=5, warmup=2):
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()
    times = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)  # ms
    return median(times)


def make_inputs(Nq, B, Lq, Ld, d, shared_docs):
    Q = torch.randn(Nq, Lq, d, device=device, dtype=DTYPE, requires_grad=True)
    if shared_docs:
        D = torch.randn(B, Ld, d, device=device, dtype=DTYPE, requires_grad=True)
    else:
        D = torch.randn(Nq, B, Ld, d, device=device, dtype=DTYPE, requires_grad=True)
    return Q, D


def make_argmax(Nq, B, Lq, Ld, shared_docs):
    """Run the forward to get a saved argmax in the right layout."""
    Q, D = make_inputs(Nq, B, Lq, Ld, D_DIM, shared_docs)
    Q_c = Q.detach().contiguous().half()
    D_flat = D.detach() if shared_docs else D.detach().reshape(Nq * B, Ld, D_DIM)
    D_c = D_flat.contiguous().half()
    total = B if shared_docs else Nq * B
    lengths = _default_lengths(total, Ld, device, None)
    _scores, argmax = _launch_fwd(
        Q_c, D_c, lengths, Nq, B, Lq, Ld, D_DIM, shared_docs, True, q_lengths=None,
    )
    return Q_c, D_c, argmax


def memory_table(Nq, B, Lq, Ld, d, shared_docs):
    n_dest = (B if shared_docs else Nq * B) * Ld
    col_idx_mb = 4 * Nq * B * Lq / 1e6
    row_ptr_mb = 4 * (n_dest + 1) / 1e6
    if shared_docs:
        grad_D_fp32_mb = 4 * B * Ld * d / 1e6
    else:
        grad_D_fp32_mb = 4 * Nq * B * Ld * d / 1e6
    argmax_mb = 4 * Nq * B * Lq / 1e6        # int32 [total_pairs, Lq], total_pairs = Nq*B
    sim_mb = 2 * Nq * B * Lq * Ld / 1e6      # FP16 sim matrix
    return dict(
        col_idx=col_idx_mb,
        row_ptr=row_ptr_mb,
        grad_D_fp32=grad_D_fp32_mb,
        argmax=argmax_mb,
        sim=sim_mb,
    )


def bench_csr_build(Q_c, D_c, argmax, Nq, B, Lq, Ld, shared_docs):
    def fn():
        build_inverse_csr(argmax, Nq, B, Lq, Ld, shared_docs)
    return gpu_time_median(fn, n=5, warmup=2)


def bench_invgrid_kernel_only(Q_c, D_c, argmax, Nq, B, Lq, Ld, shared_docs):
    """Time just the dD kernel call (CSR built once outside)."""
    row_ptr, col_idx = build_inverse_csr(argmax, Nq, B, Lq, Ld, shared_docs)
    n_dest = (B if shared_docs else Nq * B) * Ld
    grad_D_fp32 = torch.zeros(D_c.shape, device=device, dtype=torch.float32)
    grad_scores = torch.randn(Nq, B, device=device, dtype=torch.float32).contiguous()
    d = D_DIM
    d_pad = _next_pow2(d)

    def fn():
        grad_D_fp32.zero_()
        _maxsim_bwd_dD_invgrid_kernel[(n_dest,)](
            Q_c, grad_scores, grad_D_fp32,
            row_ptr, col_idx,
            Nq, B, Lq, Ld, d, d_pad,
            grad_D_fp32.stride(0), grad_D_fp32.stride(1), grad_D_fp32.stride(2),
            Q_c.stride(0), Q_c.stride(1), Q_c.stride(2),
            grad_scores.stride(0), grad_scores.stride(1),
            1 if shared_docs else 0,
        )
    return gpu_time_median(fn, n=5, warmup=2)


def bench_atomic_kernel_only(Q_c, D_c, argmax, Nq, B, Lq, Ld, shared_docs):
    grad_D_fp32 = torch.zeros(D_c.shape, device=device, dtype=torch.float32)
    grad_scores = torch.randn(Nq, B, device=device, dtype=torch.float32).contiguous()
    d = D_DIM
    d_pad = _next_pow2(d)
    grid_size = B if shared_docs else Nq * B

    def fn():
        grad_D_fp32.zero_()
        _maxsim_bwd_dD_batched_kernel[(grid_size,)](
            Q_c, argmax, grad_scores, grad_D_fp32,
            Nq, B, Lq, Ld, d, d_pad,
            grad_D_fp32.stride(0), grad_D_fp32.stride(1), grad_D_fp32.stride(2),
            Q_c.stride(0), Q_c.stride(1), Q_c.stride(2),
            grad_scores.stride(0), grad_scores.stride(1),
            1 if shared_docs else 0,
        )
    return gpu_time_median(fn, n=5, warmup=2)


def bench_full_backward(impl, Nq, B, Lq, Ld, shared_docs):
    """Time scores.backward(g) for a given impl. Forward done outside."""
    Q, D = make_inputs(Nq, B, Lq, Ld, D_DIM, shared_docs)
    scores = impl(Q, D, shared_docs=shared_docs)
    g = torch.randn_like(scores)

    times = []
    # warmup
    for _ in range(2):
        Q.grad = None
        D.grad = None
        s = impl(Q, D, shared_docs=shared_docs)
        s.backward(g, retain_graph=False)
        torch.cuda.synchronize()

    for _ in range(5):
        Q.grad = None
        D.grad = None
        s = impl(Q, D, shared_docs=shared_docs)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        s.backward(g, retain_graph=False)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return median(times)


def run_shape(Nq, B, Lq, Ld, label, shared_docs):
    d = D_DIM
    tag = f"{label} sd={int(shared_docs)} (Nq={Nq},B={B},Lq={Lq},Ld={Ld})"
    print(f"\n=== {tag} ===")
    try:
        Q_c, D_c, argmax = make_argmax(Nq, B, Lq, Ld, shared_docs)
    except torch.cuda.OutOfMemoryError as e:
        print(f"  OOM building inputs: {e}")
        reset()
        return None

    mem = memory_table(Nq, B, Lq, Ld, d, shared_docs)
    print(
        f"  MEM (MB):  col_idx={mem['col_idx']:.2f}  "
        f"row_ptr={mem['row_ptr']:.2f}  "
        f"grad_D_fp32={mem['grad_D_fp32']:.2f}  "
        f"| argmax={mem['argmax']:.2f}  sim_matrix(would-be)={mem['sim']:.2f}"
    )
    csr_overhead_vs_argmax = (mem['col_idx'] + mem['row_ptr']) / mem['argmax']
    print(f"  CSR (col_idx+row_ptr) / argmax_baseline = {csr_overhead_vs_argmax:.2f}x")

    try:
        t_csr = bench_csr_build(Q_c, D_c, argmax, Nq, B, Lq, Ld, shared_docs)
        t_kern_inv = bench_invgrid_kernel_only(Q_c, D_c, argmax, Nq, B, Lq, Ld, shared_docs)
        t_kern_atom = bench_atomic_kernel_only(Q_c, D_c, argmax, Nq, B, Lq, Ld, shared_docs)
    except torch.cuda.OutOfMemoryError as e:
        print(f"  OOM during kernel timing: {e}")
        reset()
        return None
    print(f"  TIME: CSR-build={t_csr:.3f}ms  kern_invgrid={t_kern_inv:.3f}ms  kern_atomic={t_kern_atom:.3f}ms")
    print(f"        kernel-only invgrid/atomic = {t_kern_inv / max(t_kern_atom, 1e-9):.2f}x")

    # Free per-shape buffers before doing autograd path (which builds graphs).
    del Q_c, D_c, argmax
    reset()

    try:
        t_bwd_inv = bench_full_backward(flash_maxsim_invgrid_train, Nq, B, Lq, Ld, shared_docs)
    except torch.cuda.OutOfMemoryError as e:
        print(f"  OOM in invgrid full bwd: {e}")
        t_bwd_inv = float("nan")
    reset()
    try:
        t_bwd_atom = bench_full_backward(flash_atomic, Nq, B, Lq, Ld, shared_docs)
    except torch.cuda.OutOfMemoryError as e:
        print(f"  OOM in atomic full bwd: {e}")
        t_bwd_atom = float("nan")
    reset()

    csr_frac = t_csr / max(t_bwd_inv, 1e-9) * 100 if math.isfinite(t_bwd_inv) else float("nan")
    speedup = t_bwd_atom / max(t_bwd_inv, 1e-9) if math.isfinite(t_bwd_inv) and math.isfinite(t_bwd_atom) else float("nan")
    print(f"  TOTAL bwd: invgrid={t_bwd_inv:.3f}ms  atomic={t_bwd_atom:.3f}ms  e2e speedup={speedup:.2f}x")
    print(f"  >>> CSR-build is {csr_frac:.1f}% of total invgrid backward")
    return dict(
        label=label, shared=shared_docs, Nq=Nq, B=B, Lq=Lq, Ld=Ld,
        mem=mem, t_csr=t_csr, t_kern_inv=t_kern_inv, t_kern_atom=t_kern_atom,
        t_bwd_inv=t_bwd_inv, t_bwd_atom=t_bwd_atom, csr_frac=csr_frac, speedup=speedup,
    )


def scaling_check():
    """Build CSR at increasing Nq*B*Lq, check linearity."""
    print("\n=== SCALING: CSR build vs Nq*B*Lq (shared_docs=True, Ld=180, d=128) ===")
    rows = []
    for (Nq, B, Lq) in [(8, 8, 32), (32, 32, 32), (128, 128, 32), (512, 512, 32), (1024, 1024, 32)]:
        Ld = 180
        try:
            Q_c, D_c, argmax = make_argmax(Nq, B, Lq, Ld, shared_docs=True)
            t = bench_csr_build(Q_c, D_c, argmax, Nq, B, Lq, Ld, shared_docs=True)
            nnz = Nq * B * Lq
            print(f"  Nq*B*Lq = {nnz:>10,d}  CSR-build = {t:.3f} ms  ({t/nnz*1e6:.3f} ns/elem)")
            rows.append((nnz, t))
            del Q_c, D_c, argmax; reset()
        except torch.cuda.OutOfMemoryError:
            print(f"  Nq={Nq},B={B},Lq={Lq}: OOM"); reset()
    if len(rows) >= 2:
        nnz0, t0 = rows[0]; nnz1, t1 = rows[-1]
        scale_ratio = (t1 / t0) / (nnz1 / nnz0)
        print(f"  scale ratio (last/first, time-vs-size) = {scale_ratio:.2f}x  (1.0 == perfectly linear)")


def worst_case():
    """Adversarial argmax: every (q, l) → j=0. Measures hot-CTA serialization."""
    print("\n=== WORST-CASE: all argmax = 0 (max contention on j=0) ===")
    Nq, B, Lq, Ld, d = 256, 256, 1024, 1024, D_DIM
    shared_docs = True
    print(f"  shape: Nq={Nq} B={B} Lq={Lq} Ld={Ld} (shared_docs)")
    try:
        Q_c, D_c, argmax_real = make_argmax(Nq, B, Lq, Ld, shared_docs)
    except torch.cuda.OutOfMemoryError:
        print("  OOM building worst-case inputs"); reset(); return
    argmax_zero = torch.zeros_like(argmax_real)

    # CSR + kernel-only for both:
    t_csr_real = bench_csr_build(Q_c, D_c, argmax_real, Nq, B, Lq, Ld, shared_docs)
    t_csr_zero = bench_csr_build(Q_c, D_c, argmax_zero, Nq, B, Lq, Ld, shared_docs)

    t_kern_real = bench_invgrid_kernel_only(Q_c, D_c, argmax_real, Nq, B, Lq, Ld, shared_docs)
    t_kern_zero = bench_invgrid_kernel_only(Q_c, D_c, argmax_zero, Nq, B, Lq, Ld, shared_docs)

    print(f"  CSR-build:    real={t_csr_real:.3f}ms  worst(all-zero)={t_csr_zero:.3f}ms  ratio={t_csr_zero/max(t_csr_real,1e-9):.2f}x")
    print(f"  invgrid kern: real={t_kern_real:.3f}ms  worst(all-zero)={t_kern_zero:.3f}ms  ratio={t_kern_zero/max(t_kern_real,1e-9):.2f}x")

    # Inspect skew distribution.
    row_ptr, _ = build_inverse_csr(argmax_zero, Nq, B, Lq, Ld, shared_docs)
    counts = (row_ptr[1:] - row_ptr[:-1]).to(torch.int64)
    print(f"  worst-case row counts: max={counts.max().item()}  mean={counts.float().mean().item():.2f}  "
          f"#nonzero_dest={(counts>0).sum().item()}/{counts.numel()}")
    del Q_c, D_c, argmax_real, argmax_zero; reset()


def main():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Torch: {torch.__version__}  Triton: {__import__('triton').__version__}")
    print(f"d = {D_DIM} (FP16 acts, FP32 grad_D accum)")

    results = []
    for shared in (True, False):
        for (Nq, B, Lq, Ld, label) in SHAPES:
            r = run_shape(Nq, B, Lq, Ld, label, shared_docs=shared)
            if r is not None:
                results.append(r)
            reset()

    scaling_check()
    worst_case()

    # Summary table
    print("\n=== SUMMARY ===")
    print(f"{'shape':30s} {'sd':3s} {'CSR_MB':>8s} {'gradD_MB':>9s} {'argmax_MB':>10s} "
          f"{'csr_ms':>8s} {'kInv_ms':>8s} {'kAtom_ms':>9s} {'bwdInv':>8s} {'bwdAtom':>8s} "
          f"{'CSR%':>6s} {'e2e_x':>6s}")
    for r in results:
        nm = f"{r['label']}({r['Nq']},{r['B']},{r['Lq']},{r['Ld']})"
        m = r['mem']
        csr_mb = m['col_idx'] + m['row_ptr']
        print(
            f"{nm:30s} {int(r['shared']):>3d} {csr_mb:8.2f} {m['grad_D_fp32']:9.2f} {m['argmax']:10.2f} "
            f"{r['t_csr']:8.3f} {r['t_kern_inv']:8.3f} {r['t_kern_atom']:9.3f} "
            f"{r['t_bwd_inv']:8.3f} {r['t_bwd_atom']:8.3f} {r['csr_frac']:6.1f} {r['speedup']:6.2f}"
        )


if __name__ == "__main__":
    main()
