"""Training-loop parity check: torch MaxSim vs flash-maxsim.

ColPali-scale contrastive training (B=64, Lq=Ld=1024, d=128). Trains two
"models" — identical FP32 init, identical data per step, identical Adam —
one scored through PyTorch's einsum reference, one through
flash_maxsim_batched_train. Plots:

  • overlapping loss curves
  • per-step drift |loss_torch − loss_flash|
  • peak GPU memory each path actually used during the step

If Flash is numerically faithful, curves overlap and final losses match;
the memory bar makes the structural memory advantage tangible (the
[B, B, Lq, Ld] similarity matrix vs nothing).

⚠️ Memory note: at B=64 Lq=Ld=1024 the torch path needs ~17 GB peak (the
all-pairs FP32 sim matrix). Run on a ≥40 GB GPU. Flash needs ~1 GB.

Run:   python tests/training_parity_demo.py
Out:   ~/training_parity.png + per-step diff + peak memory.
"""
from __future__ import annotations
import sys, time, gc
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/path/to/data")
from flash_maxsim import flash_maxsim_batched_train

DEV = "cuda"

# ── ColPali regime ──────────────────────────────────────────────────────
B, Lq, Ld, d = 128, 1024, 1024, 128
STEPS = 100
LR    = 1e-3
SEED  = 42
TEMP  = float(Lq) ** 0.5    # softmax temperature, scores grow with Lq


# ── scoring functions ───────────────────────────────────────────────────
# Both receive FP16 inputs (mixed-precision pattern). The FP16 cast is
# done ONCE per step OUTSIDE these functions — keeps timing apples-to-apples
# (the cast cost is shared, not double-counted against flash).

def torch_scores(Q_fp16, D_fp16):
    """Standard PyTorch path: FP16 einsum (the autocast default in real
    training), no internal upcast — materialise [B,B,Lq,Ld] FP16, max
    over Ld, sum over Lq. Apples-to-apples vs flash: same FP16 inputs,
    no extra cast inside. → [B, B]."""
    sim = torch.einsum("iqd,jld->ijql", Q_fp16, D_fp16)
    return sim.max(dim=-1).values.sum(dim=-1)

def flash_scores(Q_fp16, D_fp16):
    """flash_maxsim_batched_train: FP16 input, FP32 internal accumulation,
    autograd-clean (inverse-grid CSR backward). → [B, B]."""
    return flash_maxsim_batched_train(Q_fp16, D_fp16, shared_docs=True)


# ── one identically-seeded training run ─────────────────────────────────

def run(score_fn, label):
    torch.manual_seed(SEED)
    Q = nn.Parameter(F.normalize(torch.randn(B, Lq, d, device=DEV), dim=-1))
    D = nn.Parameter(F.normalize(torch.randn(B, Ld, d, device=DEV), dim=-1))
    opt = torch.optim.Adam([Q, D], lr=LR)
    target = torch.arange(B, device=DEV)

    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()

    losses = []
    t0 = time.time()
    try:
        for _ in range(STEPS):
            Q_h = Q.to(torch.float16)
            D_h = D.to(torch.float16)
            scores = score_fn(Q_h, D_h) / TEMP
            loss   = F.cross_entropy(scores, target)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        torch.cuda.synchronize()
        dt = time.time() - t0
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  {label:<6}: {STEPS} steps in {dt:6.1f}s "
              f"({dt*1000/STEPS:5.1f} ms/step)  peak {peak_gb:5.2f} GB  "
              f"final loss = {losses[-1]:.4f}")
        return np.array(losses), peak_gb, dt
    except torch.cuda.OutOfMemoryError as e:
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  {label:<6}: ❌ OOM after {len(losses)} steps  "
              f"(peak before OOM: {peak_gb:.2f} GB)")
        return None, peak_gb, float("inf")


# ── main ────────────────────────────────────────────────────────────────

print(f"GPU: {torch.cuda.get_device_name(0)}  "
      f"({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB)")
print(f"ColPali regime: B={B}  Lq={Lq}  Ld={Ld}  d={d}  | "
      f"{STEPS} steps, lr={LR}\n")

print("Running PyTorch reference path (materialises [B,B,Lq,Ld] sim matrix)…")
losses_torch, peak_torch, t_torch = run(torch_scores, "torch")

print("Running flash-maxsim path (no sim matrix)…")
losses_flash, peak_flash, t_flash = run(flash_scores, "flash")

torch_ok = losses_torch is not None
if torch_ok:
    diff     = np.abs(losses_torch - losses_flash)
    rel_diff = diff / np.maximum(np.abs(losses_torch), 1e-6)
    print(f"\nParity over {STEPS} steps:")
    print(f"  max |Δ loss|       = {diff.max():.5f}")
    print(f"  mean |Δ loss|      = {diff.mean():.5f}")
    print(f"  max relative drift = {rel_diff.max()*100:.3f} %")
    print(f"  final loss — torch = {losses_torch[-1]:.4f}")
    print(f"  final loss — flash = {losses_flash[-1]:.4f}")
else:
    print(f"\n⚠️  torch path OOMed — flash trained fine. Memory-unlock case.")
    print(f"  final loss — flash = {losses_flash[-1]:.4f}")

print(f"\nPeak GPU memory across the training step:")
print(f"  torch  : {peak_torch:6.2f} GB" + ("  (OOM)" if not torch_ok else ""))
print(f"  flash  : {peak_flash:6.2f} GB" +
      (f"  ({peak_torch/max(peak_flash,1e-6):.1f}× less)" if torch_ok else ""))
if torch_ok:
    print(f"\nStep latency (median over {STEPS}):")
    print(f"  torch  : {t_torch*1000/STEPS:6.1f} ms/step")
    print(f"  flash  : {t_flash*1000/STEPS:6.1f} ms/step  ({t_torch/t_flash:.1f}× faster)")


# ── 3-panel plot ─────────────────────────────────────────────────────────

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.2))

if torch_ok:
    ax1.plot(losses_torch, label="torch (FP16 einsum)", lw=2.0, color="#444")
ax1.plot(losses_flash, label="flash (FP16 input, FP32 accum)",
         lw=1.6, linestyle="--", color="#1E88E5")
ax1.set_xlabel("step"); ax1.set_ylabel("contrastive cross-entropy loss")
ax1.set_title("Loss curves — same init, data, optimizer"
              + ("" if torch_ok else "  (torch OOM)"))
ax1.legend(); ax1.grid(alpha=0.3)

if torch_ok:
    ax2.semilogy(np.maximum(diff, 1e-8), color="#D81B60", lw=1.2)
    ax2.set_xlabel("step"); ax2.set_ylabel("|loss_torch − loss_flash|  (log)")
    ax2.set_title("Per-step drift between paths")
else:
    ax2.text(0.5, 0.5, "torch OOMed — no drift to compare",
             ha="center", va="center", transform=ax2.transAxes,
             fontsize=11, color="#888")
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title("Per-step drift between paths")
ax2.grid(alpha=0.3, which="both")

bars = ax3.bar(["torch", "flash"], [peak_torch, peak_flash],
               color=["#666", "#1E88E5"])
ax3.set_ylabel("peak GPU memory (GB)")
ax3.set_title(f"Peak training-step memory "
              f"({peak_torch/max(peak_flash,1e-6):.0f}× less)")
ax3.grid(alpha=0.3, axis="y")
for bar, v in zip(bars, [peak_torch, peak_flash]):
    ax3.text(bar.get_x() + bar.get_width()/2, v, f"{v:.2f} GB",
             ha="center", va="bottom", fontsize=10, fontweight="bold")

fig.suptitle(f"flash-maxsim ↔ PyTorch training parity  —  ColPali regime "
             f"(B={B}, Lq={Lq}, Ld={Ld}, d={d})",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig("~/training_parity.png", dpi=140, bbox_inches="tight")
print("\nwrote ~/training_parity.png")
