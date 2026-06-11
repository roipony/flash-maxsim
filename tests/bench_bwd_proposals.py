"""Benchmark all 5 backward-optimization proposals at the small-B/small-Ld
bottleneck shapes that pylate currently wins on.

Tests correctness (vs FP32 reference) and full-step latency (fwd + bwd).
"""
import sys, time, gc, traceback
import torch
import torch.nn.functional as F

sys.path.insert(0, '/path/to/data')
sys.path.insert(0, '/path/to/data')

from flash_maxsim.flash_maxsim_batched_train import flash_maxsim_batched_train as flash_atomic
from pylate.scores import colbert_scores

# Proposal imports (each may fail to load — capture and continue).
PROPOSALS = {}

try:
    from flash_maxsim.flash_maxsim_sortbwd_proposal import flash_maxsim_batched_train_sortbwd
    PROPOSALS["sortbwd"] = flash_maxsim_batched_train_sortbwd
except Exception as e:
    print(f"[skip] sortbwd: {e}")

try:
    from flash_maxsim.flash_maxsim_persistbwd_proposal import flash_maxsim_batched_train_persistbwd
    PROPOSALS["persistbwd"] = flash_maxsim_batched_train_persistbwd
except Exception as e:
    print(f"[skip] persistbwd: {e}")

try:
    from flash_maxsim.flash_maxsim_fusedbwd_proposal import flash_maxsim_batched_train_fused
    PROPOSALS["fusedbwd"] = flash_maxsim_batched_train_fused
except Exception as e:
    print(f"[skip] fusedbwd: {e}")

try:
    from flash_maxsim.flash_maxsim_invgrid_proposal import flash_maxsim_invgrid_train
    PROPOSALS["invgrid"] = flash_maxsim_invgrid_train
except Exception as e:
    print(f"[skip] invgrid: {e}")

# Wrap the torchbwd path as an autograd-mimicking function (forward + manual bwd).
try:
    from flash_maxsim.flash_maxsim_torchbwd_proposal import torch_backward
    from flash_maxsim import flash_maxsim_batched

    class _TorchBwdFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, Q, D, shared_docs, dl, ql):
            # Use the existing forward (no autograd) to also save argmax.
            from flash_maxsim.flash_maxsim import _launch_fwd, _next_pow2, _default_lengths
            Nq, Lq, d = Q.shape
            B, Ld, _ = D.shape
            Q_c = Q.contiguous().half()
            D_c = D.contiguous().half()
            lengths = _default_lengths(B, Ld, D.device, dl)
            scores, argmax = _launch_fwd(Q_c, D_c, lengths, Nq, B, Lq, Ld, d,
                                         shared_docs, True, q_lengths=ql)
            ctx.save_for_backward(Q_c, D_c, argmax)
            ctx.shared_docs = shared_docs
            return scores
        @staticmethod
        def backward(ctx, grad_scores):
            Q, D, argmax = ctx.saved_tensors
            grad_Q, grad_D = torch_backward(Q, D, argmax, grad_scores.float(),
                                            shared_docs=ctx.shared_docs)
            return grad_Q.to(Q.dtype), grad_D.to(D.dtype), None, None, None

    PROPOSALS["torchbwd"] = lambda Q, D, sd=True, dl=None, ql=None: _TorchBwdFn.apply(Q, D, sd, dl, ql)
except Exception as e:
    print(f"[skip] torchbwd: {e}\n{traceback.format_exc()[:500]}")


print(f"\nProposals loaded: {list(PROPOSALS.keys())}\n")
device = 'cuda'
DTYPE = torch.float16


def reset():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()


def bench_step(fn, mk_args, warmup=3, runs=5):
    try:
        for _ in range(warmup):
            args = mk_args()
            scores = fn(*args)
            (scores.diagonal().sum() if scores.dim() == 2 else scores.sum()).backward()
            del args, scores
        reset()
        times = []
        for _ in range(runs):
            args = mk_args()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            scores = fn(*args)
            (scores.diagonal().sum() if scores.dim() == 2 else scores.sum()).backward()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
            del args, scores
        times.sort()
        return ("OK", times[len(times)//2], torch.cuda.max_memory_allocated() / (1024**3))
    except torch.cuda.OutOfMemoryError:
        reset()
        return ("OOM", float('nan'), float('nan'))
    except Exception as e:
        reset()
        return (f"ERR:{type(e).__name__}", float('nan'), float('nan'))


def correctness_vs_fp32(fn, B, Lq, Ld, d, seed=0):
    """Compare fn's grads against FP32 ground truth. fn signature: fn(Q, D, ...)."""
    torch.manual_seed(seed)
    Q0 = F.normalize(torch.randn(B, Lq, d, dtype=DTYPE, device=device), dim=-1)
    D0 = F.normalize(torch.randn(B, Ld, d, dtype=DTYPE, device=device), dim=-1)

    # FP32 reference (TF32 off)
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    Qr = Q0.clone().float().requires_grad_(True); Dr = D0.clone().float().requires_grad_(True)
    Sr = torch.einsum('ash,bth->abst', Qr, Dr).max(dim=-1).values.sum(dim=-1)
    Sr.diagonal().sum().backward()
    torch.backends.cuda.matmul.allow_tf32 = prev

    # candidate
    Qf = Q0.clone().requires_grad_(True); Df = D0.clone().requires_grad_(True)
    try:
        Sf = fn(Qf, Df)
        Sf.diagonal().sum().backward()
    except TypeError:
        Sf = fn(Qf, Df, True, None, None)
        Sf.diagonal().sum().backward()

    gQ_cos = F.cosine_similarity(Qr.grad.float().flatten().unsqueeze(0),
                                  Qf.grad.float().flatten().unsqueeze(0)).item()
    gD_cos = F.cosine_similarity(Dr.grad.float().flatten().unsqueeze(0),
                                  Df.grad.float().flatten().unsqueeze(0)).item()
    return gQ_cos, gD_cos


print(f"GPU: {torch.cuda.get_device_name(0)}\n")
print("=" * 110)
print(" Correctness — gradient cosine similarity vs FP32 ground truth (B=8 Lq=32 Ld=180)")
print("=" * 110)
print(f"  {'method':<14}  {'gQ_cos':>10}  {'gD_cos':>10}")
print("  " + "-" * 50)
for name, fn in PROPOSALS.items():
    try:
        gQ, gD = correctness_vs_fp32(fn, 8, 32, 180, 128, seed=0)
        verdict = "OK" if min(gQ, gD) > 0.999 else "MISMATCH"
        print(f"  {name:<14}  {gQ:>10.5f}  {gD:>10.5f}  [{verdict}]")
    except Exception as e:
        print(f"  {name:<14}  ERR  {type(e).__name__}: {str(e)[:80]}")


print()
print("=" * 110)
print(" Full training step (fwd + bwd) — bottleneck shapes (median ms)")
print("=" * 110)

SHAPES = [
    (128,  32,  180, 128, "ColBERT B=128"),
    (256,  32,  180, 128, "ColBERT B=256"),
    (512,  32,  180, 128, "ColBERT B=512"),
    (1024, 32,  180, 128, "ColBERT B=1K"),
    (32,  1024, 1024, 128, "ColPali B=32"),
    (64,  1024, 1024, 128, "ColPali B=64"),
]
methods = [("pylate", colbert_scores), ("flash_atomic", flash_atomic)]
methods += [(name, fn) for name, fn in PROPOSALS.items()]

header = f"  {'shape':<22} " + "".join(f"  {m[0]:>14}" for m in methods)
print(header)
print("  " + "-" * (22 + 16 * len(methods)))

for B, Lq, Ld, d, tag in SHAPES:
    row = f"  {tag:<22} "
    def mk():
        torch.manual_seed(B)
        Q = F.normalize(torch.randn(B, Lq, d, dtype=DTYPE, device=device), dim=-1).requires_grad_(True)
        D = F.normalize(torch.randn(B, Ld, d, dtype=DTYPE, device=device), dim=-1).requires_grad_(True)
        return (Q, D)
    def mk_flash():
        Q, D = mk()
        return (Q, D, True, None, None)

    for name, fn in methods:
        if name == "pylate":
            status, ms, _ = bench_step(fn, mk)
        else:
            status, ms, _ = bench_step(fn, mk_flash)
        cell = f"{ms:>11.1f} ms" if status == "OK" else f"{status:>14}"
        row += f"  {cell}"
    print(row)

print("\nDone.")
