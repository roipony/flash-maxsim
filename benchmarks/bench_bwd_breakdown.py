"""Backward timing breakdown for the CSR (invgrid) training path.

Reviewer R2 #3: justify the dispatch heuristic (Nq*B*Lq > 50K → CSR) with
a per-component breakdown of the backward, so a reader can see where the
time goes and verify that CSR build is small relative to the ∇Q / ∇D
kernels at training-relevant shapes.

Components timed (CUDA events, post-warmup median over N runs):
  1. forward (returns scores + saved argmax)
  2. CSR build (build_inverse_csr: bincount → cumsum row_ptr + argsort col_idx)
  3. ∇Q kernel (_maxsim_bwd_dQ_batched_kernel)
  4. ∇D kernel (_maxsim_bwd_dD_invgrid_kernel)
  5. total backward (CSR build + ∇Q + ∇D), measured end-to-end for sanity
  6. autograd-roundtrip (loss.backward()) for the apples-to-apples total

Workloads match Tab.~\\ref{tab:bwd_abl} (the dispatch ablation):
ColBERT B ∈ {128, 512, 1024} (Lq=32, Ld=180) and ColPali B ∈ {32, 64, 128}
(Lq=Ld=1024, d=128, FP16 inputs).
"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import torch  # noqa: E402
from flash_maxsim.flash_maxsim_batched_train import (  # noqa: E402
    _FlashMaxSimBatchedFn,
    flash_maxsim_batched_train,
)
from flash_maxsim.flash_maxsim_invgrid_proposal import (  # noqa: E402
    build_inverse_csr, _maxsim_bwd_dD_invgrid_kernel,
)
from flash_maxsim.flash_maxsim_batched_train import (  # noqa: E402
    _maxsim_bwd_dQ_batched_kernel,
)

torch.set_float32_matmul_precision('high')
WARMUP, RUNS = 5, 30

WORKLOADS = [
    {"name": "ColBERT B=128",  "Nq": 1, "B": 128,  "Lq": 32,   "Ld": 180,  "d": 128},
    {"name": "ColBERT B=512",  "Nq": 1, "B": 512,  "Lq": 32,   "Ld": 180,  "d": 128},
    {"name": "ColBERT B=1024", "Nq": 1, "B": 1024, "Lq": 32,   "Ld": 180,  "d": 128},
    {"name": "ColPali B=32",   "Nq": 1, "B": 32,   "Lq": 1024, "Ld": 1024, "d": 128},
    {"name": "ColPali B=64",   "Nq": 1, "B": 64,   "Lq": 1024, "Ld": 1024, "d": 128},
    {"name": "ColPali B=128",  "Nq": 1, "B": 128,  "Lq": 1024, "Ld": 1024, "d": 128},
]


def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def _bench_event(fn, runs=RUNS, warmup=WARMUP):
    """Return median ms over `runs` measured with CUDA events."""
    times = []
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    for _ in range(runs):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def run_one(w):
    Nq, B, Lq, Ld, d = w["Nq"], w["B"], w["Lq"], w["Ld"], w["d"]
    print(f"\n=== {w['name']} (Nq={Nq}, B={B}, Lq={Lq}, Ld={Ld}, d={d}) ===")
    Q = torch.randn(Nq, Lq, d, device='cuda', dtype=torch.float16, requires_grad=True)
    D = torch.randn(B, Ld, d, device='cuda', dtype=torch.float16, requires_grad=True)

    # 1) Forward (returns scores + saves argmax)
    def fwd():
        return flash_maxsim_batched_train(Q, D, shared_docs=True)
    t_fwd = _bench_event(fwd)

    # Pre-compute one forward to capture argmax for the per-kernel passes
    s = fwd()
    # Re-run with grad enabled to capture context, but for component timing
    # we'll call the underlying functions directly.
    g = torch.ones_like(s)
    # Run forward once to obtain argmax via the Function's saved tensors.
    # Use the autograd.Function directly with a dummy ctx-like to extract argmax.
    # Easier: re-run the forward kernel via the Function and capture the
    # saved_tensors using torch's hooks. Simpler still: call the public
    # train forward and let autograd save argmax internally, then for
    # piecewise timing we extract via torch.autograd.grad with retain_graph.

    # 2) CSR build, isolated
    # We need an argmax tensor of shape [Nq, B, Lq] int64 in [0, Ld).
    # Re-derive: rerun the forward (with grad-disabled) and recover argmax.
    with torch.no_grad():
        # The forward kernel writes argmax into a buffer of shape [Nq, B, Lq].
        # We rebuild it by calling the underlying forward kernel directly;
        # but for simplicity we synthesize a representative argmax via a
        # vanilla einsum + argmax. This matches the actual run-time
        # distribution (uniform-ish), and CSR build is data-distribution-
        # agnostic in cost (it's O(NqBLq + BLd)).
        Q32 = Q.detach().float()
        D32 = D.detach().float()
        # naive sim per (q, b): [Nq, B, Lq, Ld]
        sim = torch.einsum('qid,bjd->qbij', Q32, D32)
        argmax = sim.argmax(dim=-1).contiguous()  # [Nq, B, Lq]
    # build_inverse_csr (shared_docs=True) expects [B*Nq, Lq] flattened
    # in the pair_idx = b*Nq + q_idx order. With Nq=1 a simple reshape
    # produces the same layout.
    argmax_i32 = argmax.permute(1, 0, 2).reshape(B * Nq, Lq).contiguous().to(torch.int32)

    def csr_build():
        return build_inverse_csr(argmax_i32, Nq, B, Lq, Ld, True)
    t_csr = _bench_event(csr_build)

    # 3) ∇Q kernel
    row_ptr, col_idx = build_inverse_csr(argmax_i32, Nq, B, Lq, Ld, True)
    grad_scores = torch.ones(Nq, B, Lq, device='cuda', dtype=torch.float32)
    grad_Q = torch.zeros_like(Q)
    d_pad = _next_pow2(d)

    def dq():
        grad_Q.zero_()
        _maxsim_bwd_dQ_batched_kernel[(Nq * Lq,)](
            D, argmax_i32, grad_scores, grad_Q,
            Nq, B, Lq, d, d_pad,
            D.stride(0), D.stride(1), D.stride(2),
            grad_scores.stride(0), grad_scores.stride(1),
            grad_Q.stride(0), grad_Q.stride(1), grad_Q.stride(2),
            1,
        )
    t_dq = _bench_event(dq)

    # 4) ∇D kernel (invgrid path)
    grad_D_fp32 = torch.zeros(D.shape, device=D.device, dtype=torch.float32)
    n_dest = B * Ld

    def dd():
        grad_D_fp32.zero_()
        _maxsim_bwd_dD_invgrid_kernel[(n_dest,)](
            Q, grad_scores, grad_D_fp32,
            row_ptr, col_idx,
            Nq, B, Lq, Ld, d, d_pad,
            grad_D_fp32.stride(0), grad_D_fp32.stride(1), grad_D_fp32.stride(2),
            Q.stride(0), Q.stride(1), Q.stride(2),
            grad_scores.stride(0), grad_scores.stride(1),
            1,
        )
    t_dd = _bench_event(dd)

    # 5) Total backward (CSR build + ∇Q + ∇D), measured end-to-end
    def total_bwd_pieces():
        rp, ci = build_inverse_csr(argmax_i32, Nq, B, Lq, Ld, True)
        grad_Q.zero_()
        _maxsim_bwd_dQ_batched_kernel[(Nq * Lq,)](
            D, argmax_i32, grad_scores, grad_Q,
            Nq, B, Lq, d, d_pad,
            D.stride(0), D.stride(1), D.stride(2),
            grad_scores.stride(0), grad_scores.stride(1),
            grad_Q.stride(0), grad_Q.stride(1), grad_Q.stride(2),
            1,
        )
        grad_D_fp32.zero_()
        _maxsim_bwd_dD_invgrid_kernel[(n_dest,)](
            Q, grad_scores, grad_D_fp32,
            rp, ci,
            Nq, B, Lq, Ld, d, d_pad,
            grad_D_fp32.stride(0), grad_D_fp32.stride(1), grad_D_fp32.stride(2),
            Q.stride(0), Q.stride(1), Q.stride(2),
            grad_scores.stride(0), grad_scores.stride(1),
            1,
        )
    t_total_pieces = _bench_event(total_bwd_pieces)

    # 6) Autograd-roundtrip (forward + loss.backward()) for the headline total
    Q2 = Q.detach().clone().requires_grad_(True)
    D2 = D.detach().clone().requires_grad_(True)
    def autograd_step():
        if Q2.grad is not None: Q2.grad = None
        if D2.grad is not None: D2.grad = None
        s = flash_maxsim_batched_train(Q2, D2, shared_docs=True)
        s.sum().backward()
    t_autograd = _bench_event(autograd_step)

    row = {
        "name": w["name"], "Nq": Nq, "B": B, "Lq": Lq, "Ld": Ld, "d": d,
        "forward_ms": round(t_fwd, 4),
        "csr_build_ms": round(t_csr, 4),
        "dQ_kernel_ms": round(t_dq, 4),
        "dD_kernel_ms": round(t_dd, 4),
        "backward_pieces_ms": round(t_total_pieces, 4),
        "autograd_step_ms": round(t_autograd, 4),
        "csr_pct_of_bwd": round(100 * t_csr / t_total_pieces, 1),
        "dQ_pct_of_bwd": round(100 * t_dq / t_total_pieces, 1),
        "dD_pct_of_bwd": round(100 * t_dd / t_total_pieces, 1),
    }
    print(f"  fwd={t_fwd:.3f}ms  csr_build={t_csr:.3f}ms ({row['csr_pct_of_bwd']}%)  "
          f"dQ={t_dq:.3f}ms ({row['dQ_pct_of_bwd']}%)  dD={t_dd:.3f}ms ({row['dD_pct_of_bwd']}%)  "
          f"bwd_total={t_total_pieces:.3f}ms  autograd={t_autograd:.3f}ms")
    return row


results = []
for w in WORKLOADS:
    try:
        results.append(run_one(w))
    except Exception as e:
        print(f"  FAILED on {w['name']}: {type(e).__name__}: {e}")
        results.append({"name": w["name"], "error": f"{type(e).__name__}: {str(e)[:200]}"})

gpu = torch.cuda.get_device_name()
out = os.path.join(HERE, f"bench_bwd_breakdown_{gpu.replace(' ', '_').replace('/', '_')}.json")
json.dump({"gpu": gpu, "warmup": WARMUP, "runs": RUNS, "results": results}, open(out, "w"), indent=2)
print(f"\nwrote {out}")
