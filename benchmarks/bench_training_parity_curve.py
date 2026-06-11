"""Training-curve parity for the paper's bit-faithfulness claim.

The paper claims (§5.x, line 828) that 500 steps of contrastive training
through flash_maxsim_batched_train reproduce the naive loss trajectory to
within 0.001% relative drift. The reviewer flagged this as unsubstantiated
without a figure. This bench produces the figure.

Differences from tests/training_parity_demo.py:
  - 500 steps (paper claim) instead of 100
  - writes loss arrays as JSON next to the script so the figure is
    reproducible without rerunning training
  - writes the figure into the paper repo so the LaTeX build sees it
"""
from __future__ import annotations
import gc, json, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from flash_maxsim import flash_maxsim_batched_train

DEV = "cuda"
# B=64 chosen so the torch reference fits in 80GB: the [B,B,Lq,Ld] FP16
# sim matrix at B=64 Lq=Ld=1024 is 8 GB, peak with grads/intermediates
# ~17 GB. B=128 OOMs (32 GB sim matrix → 64 GB+ peak); a parity curve
# needs both paths to actually train.
B, Lq, Ld, d = 64, 1024, 1024, 128
STEPS = 500
LR    = 1e-3
SEED  = 42
TEMP  = float(Lq) ** 0.5

PAPER_DIR = "/path/to/data"


def torch_scores(Q_fp16, D_fp16):
    sim = torch.einsum("iqd,jld->ijql", Q_fp16, D_fp16)
    return sim.max(dim=-1).values.sum(dim=-1)


def flash_scores(Q_fp16, D_fp16):
    return flash_maxsim_batched_train(Q_fp16, D_fp16, shared_docs=True)


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
        for step in range(STEPS):
            Q_h = Q.to(torch.float16)
            D_h = D.to(torch.float16)
            scores = score_fn(Q_h, D_h) / TEMP
            loss = F.cross_entropy(scores, target)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
            if step % 50 == 0:
                print(f"  {label:<6}: step {step:4d}/{STEPS}  loss={loss.item():.4f}",
                      flush=True)
        torch.cuda.synchronize()
        dt = time.time() - t0
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  {label:<6}: {STEPS} steps in {dt:6.1f}s "
              f"({dt*1000/STEPS:5.1f} ms/step)  peak {peak_gb:5.2f} GB  "
              f"final loss = {losses[-1]:.4f}", flush=True)
        return losses, peak_gb, dt
    except torch.cuda.OutOfMemoryError:
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  {label:<6}: OOM after {len(losses)} steps  "
              f"(peak before OOM: {peak_gb:.2f} GB)", flush=True)
        return None, peak_gb, float("inf")


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB)", flush=True)
    print(f"ColPali regime: B={B}  Lq={Lq}  Ld={Ld}  d={d}  | "
          f"{STEPS} steps, lr={LR}\n", flush=True)

    print("Running PyTorch reference path (materialises [B,B,Lq,Ld] sim matrix)...", flush=True)
    losses_torch, peak_torch, t_torch = run(torch_scores, "torch")

    print("\nRunning flash-maxsim path (no sim matrix)...", flush=True)
    losses_flash, peak_flash, t_flash = run(flash_scores, "flash")

    torch_ok = losses_torch is not None
    out = {
        "gpu": torch.cuda.get_device_name(0),
        "shape": {"B": B, "Lq": Lq, "Ld": Ld, "d": d},
        "steps": STEPS, "lr": LR, "seed": SEED, "temp": TEMP,
        "peak_gb": {"torch": peak_torch, "flash": peak_flash},
        "wall_s":  {"torch": t_torch,   "flash": t_flash},
        "loss":    {"flash": losses_flash, "torch": losses_torch if torch_ok else None},
    }
    if torch_ok:
        lt = np.array(losses_torch); lf = np.array(losses_flash)
        diff = np.abs(lt - lf); rel = diff / np.maximum(np.abs(lt), 1e-6)
        out["parity"] = {
            "max_abs_drift": float(diff.max()),
            "mean_abs_drift": float(diff.mean()),
            "max_rel_drift_pct": float(rel.max() * 100),
            "final_loss_torch": float(lt[-1]),
            "final_loss_flash": float(lf[-1]),
        }
        print(f"\nParity over {STEPS} steps:", flush=True)
        print(f"  max  |Δ loss|      = {diff.max():.5f}", flush=True)
        print(f"  mean |Δ loss|      = {diff.mean():.5f}", flush=True)
        print(f"  max relative drift = {rel.max()*100:.4f} %", flush=True)
        print(f"  final loss — torch = {lt[-1]:.4f}", flush=True)
        print(f"  final loss — flash = {lf[-1]:.4f}", flush=True)
    else:
        print("\ntorch OOMed — flash trained fine. Memory-unlock case.", flush=True)

    gpu_tag = torch.cuda.get_device_name(0).replace(" ", "_").replace("/", "_")
    json_path = os.path.join(HERE, f"bench_training_parity_curve_{gpu_tag}.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {json_path}", flush=True)

    fig_path  = os.path.join(PAPER_DIR, "fig_training_parity.png")
    fig_path2 = os.path.join(HERE, "fig_training_parity.png")
    plot(losses_torch, losses_flash, peak_torch, peak_flash, fig_path)
    plot(losses_torch, losses_flash, peak_torch, peak_flash, fig_path2)


def plot(losses_torch, losses_flash, peak_torch, peak_flash, path):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.2))
    torch_ok = losses_torch is not None
    if torch_ok:
        ax1.plot(losses_torch, label="naive eager (FP16 einsum)", lw=2.0, color="#444")
    ax1.plot(losses_flash, label="flash-maxsim (FP16 in, FP32 accum)",
             lw=1.4, linestyle="--", color="#1E88E5")
    ax1.set_xlabel("step"); ax1.set_ylabel("contrastive CE loss")
    ax1.set_title("Loss curves — identical init, data, optimizer")
    ax1.legend(); ax1.grid(alpha=0.3)

    if torch_ok:
        lt = np.array(losses_torch); lf = np.array(losses_flash)
        diff = np.abs(lt - lf)
        ax2.semilogy(np.maximum(diff, 1e-8), color="#D81B60", lw=1.2)
        ax2.set_xlabel("step"); ax2.set_ylabel("|loss_naive − loss_flash|  (log)")
        ax2.set_title("Per-step loss drift")
    else:
        ax2.text(0.5, 0.5, "naive OOMed — no drift to compare",
                 ha="center", va="center", transform=ax2.transAxes,
                 fontsize=11, color="#888")
        ax2.set_xticks([]); ax2.set_yticks([])
        ax2.set_title("Per-step loss drift")
    ax2.grid(alpha=0.3, which="both")

    bars = ax3.bar(["naive eager", "flash"], [peak_torch, peak_flash],
                   color=["#666", "#1E88E5"])
    ax3.set_ylabel("peak GPU memory (GB)")
    ratio = peak_torch / max(peak_flash, 1e-6) if torch_ok else 0
    ax3.set_title(f"Peak training-step memory ({ratio:.0f}× less)")
    ax3.grid(alpha=0.3, axis="y")
    for bar, v in zip(bars, [peak_torch, peak_flash]):
        ax3.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f} GB",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.suptitle(f"flash-maxsim ↔ naive eager training parity — ColPali "
                 f"(B={B}, Lq={Lq}, Ld={Ld}, d={d}, {STEPS} steps)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
