"""Memory comparison: FP16-saved training vs INT8-saved (Phase 1).

Goal: quantify the memory savings of saving D as INT8 in the autograd
context, vs the standard FP16-D save. Two scenarios:

  (1) Single forward + backward step (what most training loops do).
      Peak memory during backward should be similar — Phase 1 transient
      FP16 dequant offsets the smaller saved tensor.

  (2) "Saved-tensor-lifetime peak" — measures the memory held BETWEEN
      forward and backward (the regime where Phase 1 actually wins).
      Approximated by allocating a large auxiliary tensor between the
      forward and the backward, simulating activations of subsequent
      layers in a real training graph.

If (1) shows Phase 1 == FP16 within noise (expected) AND (2) shows
Phase 1 winning by ~D-bytes-worth (the actual saved-tensor delta), the
killer-feature framing is "useful for gradient-checkpointing or
multi-step graphs, not for single-step train", and Phase 2 (kernel-
side dequant) becomes the real peak-memory unlock.
"""
import os, sys, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from flash_maxsim import flash_maxsim_batched_train, flash_maxsim_int8_batched_train

torch.set_float32_matmul_precision('high')
torch.manual_seed(0)


def step_peak(fn, Q, D, hold_bytes=0):
    """Run fn(Q, D).sum().backward() and return the peak GPU memory in GB.
    If hold_bytes > 0, allocate a temporary FP16 buffer between forward
    and backward to simulate activations of subsequent layers."""
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    Q.grad = None; D.grad = None
    s = fn(Q, D)
    held = None
    if hold_bytes > 0:
        held = torch.empty(hold_bytes // 2, device='cuda', dtype=torch.float16)
    s.sum().backward()
    if held is not None: del held
    return torch.cuda.max_memory_allocated() / 1e9


print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

# Shapes: small (ColBERT) and large (ColPali) contrastive training
for name, B, Lq, Ld, d in [
    ("ColBERT B=128",  128, 32,   180,  128),
    ("ColPali B=128",  128, 1024, 1024, 128),
]:
    print(f"\n=== {name} (B={B}, Lq={Lq}, Ld={Ld}, d={d}) ===", flush=True)
    Q = F.normalize(torch.randn(B, Lq, d, device='cuda', dtype=torch.float16), dim=-1).requires_grad_(True)
    D = F.normalize(torch.randn(B, Ld, d, device='cuda', dtype=torch.float16), dim=-1).requires_grad_(True)

    # Estimate held tensor size as 4x the D tensor — simulates a few
    # subsequent layer activations of comparable size that stay live
    # alongside the saved D.
    D_bytes = B * Ld * d * 2
    hold = 0
    for hold_label, hold_b in [("hold=0", 0), (f"hold=4*D ({4*D_bytes/1e9:.2f}GB)", 4 * D_bytes)]:
        # Warmup (compile kernels, prime any cache)
        _ = step_peak(flash_maxsim_batched_train, Q, D, 0)
        _ = step_peak(flash_maxsim_int8_batched_train, Q, D, 0)
        fp16_peak = step_peak(flash_maxsim_batched_train, Q, D, hold_b)
        int8_peak = step_peak(flash_maxsim_int8_batched_train, Q, D, hold_b)
        delta = fp16_peak - int8_peak
        delta_kind = "saved" if delta > 0 else "lost"
        print(f"  {hold_label:<25s}: fp16={fp16_peak:.3f}GB  int8={int8_peak:.3f}GB  "
              f"int8 {delta_kind} {abs(delta):.3f}GB", flush=True)
    del Q, D; torch.cuda.empty_cache()
