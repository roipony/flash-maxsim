"""Test flash_maxsim_batched_train — correctness + speed + memory vs pylate naive."""
import sys, time, gc, torch
import torch.nn.functional as F

sys.path.insert(0, '.')
sys.path.insert(0, '/path/to/data')

from flash_maxsim.flash_maxsim_batched_train import flash_maxsim_batched_train
from pylate.scores import colbert_scores, colbert_kd_scores

device = 'cuda'
DTYPE = torch.float16


def reset():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def bench_training_step(fn_name, fn, *args):
    """One forward + backward; returns (status, ms, peak_gb, loss, grad_sum)."""
    try:
        reset()
        t0 = time.perf_counter()
        scores = fn(*args)
        loss = scores.diagonal().sum() if scores.dim() == 2 else scores.sum()
        loss.backward()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        grad_sum = sum(a.grad.abs().sum().item() for a in args if a.requires_grad and a.grad is not None)
        return ("OK", ms, peak, loss.item(), grad_sum)
    except torch.cuda.OutOfMemoryError:
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        reset()
        return ("OOM", float("nan"), peak, float("nan"), float("nan"))


print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB\n")


# =============================================================================
# 0) Correctness: gradients agree with pylate on small shapes
# =============================================================================
print("=" * 100)
print(" 0) Correctness — flash grads vs pylate grads (small shapes)")
print("=" * 100)

for (B, Lq, Ld, d) in [(8, 32, 180, 128), (4, 64, 256, 128), (16, 32, 300, 128)]:
    torch.manual_seed(0)
    Q0 = F.normalize(torch.randn(B, Lq, d, dtype=DTYPE, device=device), dim=-1)
    D0 = F.normalize(torch.randn(B, Ld, d, dtype=DTYPE, device=device), dim=-1)

    # pylate path
    Qp = Q0.clone().requires_grad_(True); Dp = D0.clone().requires_grad_(True)
    sp = colbert_scores(Qp, Dp)
    sp.diagonal().sum().backward()

    # flash path
    Qf = Q0.clone().requires_grad_(True); Df = D0.clone().requires_grad_(True)
    sf = flash_maxsim_batched_train(Qf, Df, shared_docs=True)
    sf.diagonal().sum().backward()

    score_err = (sp - sf).abs().max().item()
    gq_err = (Qp.grad.float() - Qf.grad.float()).abs().max().item()
    gd_err = (Dp.grad.float() - Df.grad.float()).abs().max().item()
    status = "OK" if (score_err < 1e-1 and gq_err < 1e-2 and gd_err < 1e-2) else "FAIL"
    print(f"  B={B:>3} Lq={Lq:>4} Ld={Ld:>4}   "
          f"score_err={score_err:.2e}  grad_Q_err={gq_err:.2e}  grad_D_err={gd_err:.2e}   [{status}]")


# =============================================================================
# 1) ColBERT training — fwd + bwd at increasing batch
# =============================================================================
print()
print("=" * 100)
print(" 1) ColBERT training (Lq=32, Ld=180)   fwd + bwd")
print("=" * 100)
print(f"  {'B':>5}   {'pylate':>25}   {'flash-batched':>30}   speedup  mem_ratio")
print("  " + "-" * 95)

for B in [128, 512, 1024, 2048, 4096]:
    torch.manual_seed(0)
    def mk(): return (
        F.normalize(torch.randn(B, 32, 128, dtype=DTYPE, device=device), dim=-1).requires_grad_(True),
        F.normalize(torch.randn(B, 180, 128, dtype=DTYPE, device=device), dim=-1).requires_grad_(True),
    )

    Qp, Dp = mk()
    s_p, ms_p, peak_p, _, _ = bench_training_step("pylate", colbert_scores, Qp, Dp)

    Qf, Df = mk()
    s_f, ms_f, peak_f, _, _ = bench_training_step("flash", flash_maxsim_batched_train, Qf, Df, True, None, None)

    p_lat = f"{ms_p:>8.1f} ms / {peak_p:>5.2f} GB" if s_p == "OK" else f"{'OOM':>8}          "
    f_lat = f"{ms_f:>8.1f} ms / {peak_f:>5.2f} GB" if s_f == "OK" else f"{'OOM':>8}          "
    sp = f"{ms_p/ms_f:>5.1f}×" if (s_p == "OK" and s_f == "OK") else "    —"
    mr = f"{peak_p/peak_f:>5.0f}×" if (s_p == "OK" and s_f == "OK") else "    —"
    print(f"  {B:>5}   {p_lat:>25}   {f_lat:>30}   {sp}   {mr}")


# =============================================================================
# 2) ColPali training — fwd + bwd at increasing batch
# =============================================================================
print()
print("=" * 100)
print(" 2) ColPali training (Lq=Ld=1024)   fwd + bwd")
print("=" * 100)
print(f"  {'B':>5}   {'pylate':>25}   {'flash-batched':>30}   speedup  mem_ratio")
print("  " + "-" * 95)

for B in [16, 32, 64, 128, 256]:
    torch.manual_seed(0)
    def mk():
        return (
            F.normalize(torch.randn(B, 1024, 128, dtype=DTYPE, device=device), dim=-1).requires_grad_(True),
            F.normalize(torch.randn(B, 1024, 128, dtype=DTYPE, device=device), dim=-1).requires_grad_(True),
        )

    Qp, Dp = mk()
    s_p, ms_p, peak_p, _, _ = bench_training_step("pylate", colbert_scores, Qp, Dp)

    Qf, Df = mk()
    s_f, ms_f, peak_f, _, _ = bench_training_step("flash", flash_maxsim_batched_train, Qf, Df, True, None, None)

    p_lat = f"{ms_p:>8.1f} ms / {peak_p:>5.2f} GB" if s_p == "OK" else f"{'OOM':>8}          "
    f_lat = f"{ms_f:>8.1f} ms / {peak_f:>5.2f} GB" if s_f == "OK" else f"{'OOM':>8}          "
    sp = f"{ms_p/ms_f:>5.1f}×" if (s_p == "OK" and s_f == "OK") else "    —"
    mr = f"{peak_p/peak_f:>5.0f}×" if (s_p == "OK" and s_f == "OK") else "    —"
    print(f"  {B:>5}   {p_lat:>25}   {f_lat:>30}   {sp}   {mr}")

print("\nDone.")
