"""Realistic training OOM test — includes model weights + grads + Adam + encoder activations.

For each (model_family, batch_size) we allocate:
  - model weights (FP16, size = params × 2)
  - model gradients (FP16, size = params × 2)
  - Adam state: m + v (FP32 × 2, size = params × 8)
  - encoder forward activations (proportional to B × seq_len × hidden)

THEN we run the scoring fwd+bwd with flash vs pylate.

This is the memory picture that determines whether pylate training fits.
"""
import sys, time, gc, torch
import torch.nn.functional as F

sys.path.insert(0, '/path/to/data')
sys.path.insert(0, '/path/to/data')

from flash_maxsim.flash_maxsim_batched_train import flash_maxsim_batched_train
from pylate.scores import colbert_scores

device = 'cuda'
DTYPE = torch.float16


def reset():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


# Simulate training memory: model weights + grads + 2× Adam moments + activations
class TrainingMemoryModel:
    """Reserves VRAM equivalent to a real encoder training state."""
    def __init__(self, params, act_per_token_bytes, B, seq_len):
        # weights (fp16), grads (fp16), adam m (fp32), adam v (fp32)
        total_fp16_bytes = 2 * params * 2              # weights + grads
        total_fp32_bytes = 2 * params * 4              # m + v
        act_bytes = B * seq_len * act_per_token_bytes
        self.weights = torch.empty(total_fp16_bytes // 2, dtype=DTYPE, device=device)
        self.adam    = torch.empty(total_fp32_bytes // 4, dtype=torch.float32, device=device)
        self.acts    = torch.empty(act_bytes // 2, dtype=DTYPE, device=device)

    def free(self):
        del self.weights, self.adam, self.acts
        reset()


def bench_training_step(fn, *args):
    try:
        reset()
        t0 = time.perf_counter()
        scores = fn(*args)
        if scores.dim() == 2:
            loss = scores.diagonal().sum() - scores.mean() * 0.01
        else:
            loss = scores.sum()
        loss.backward()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        return ("OK", ms, peak)
    except torch.cuda.OutOfMemoryError:
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        reset()
        return ("OOM", float('nan'), peak)


print(f"GPU: {torch.cuda.get_device_name(0)}")
total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
print(f"Total VRAM: {total_gb:.1f} GB\n")


# ==========================================================================
# Realistic encoder configurations
# ==========================================================================
# params, act_per_token (rough encoder activation memory per token in fwd),
# Lq, Ld, name
SCENARIOS = [
    # ColBERT text — small encoder (~110M params for BERT-base)
    (110e6,  8 * 1024, 32, 180, "ColBERT (BERT-base 110M, Lq=32, Ld=180)"),
    # ModernColBERT / GTE — larger
    (330e6, 24 * 1024, 32, 180, "ModernColBERT (330M params, Lq=32, Ld=180)"),
    # ColPali — PaliGemma-3B backbone
    (3e9,   48 * 1024, 1024, 1024, "ColPali (PaliGemma-3B, Lq=Ld=1024)"),
]

for params, apt, Lq, Ld, name in SCENARIOS:
    # training overhead: weights(fp16) + grads(fp16) + adam m,v (fp32 each) = 12 × params bytes
    model_gb = 12 * params / (1024**3)
    print(f"\n{'='*110}")
    print(f"  {name}")
    print(f"  Training overhead (weights + grads + Adam m,v) = {model_gb:.1f} GB")
    print(f"{'='*110}")
    print(f"  {'B':>5}   {'pyt_ms':>8}  {'pyt_peak':>9}  {'flash_ms':>9}  {'flash_peak':>11}   speedup  mem_save")

    if Lq <= 32:
        batches = [16, 32, 128, 256, 512, 1024, 2048]
    else:
        batches = [4, 8, 16, 32, 64, 128]

    for B in batches:
        torch.manual_seed(B)
        try:
            # Allocate full training state first — this is what fails in reality
            tm = TrainingMemoryModel(int(params), apt, B, max(Lq, Ld))
        except torch.cuda.OutOfMemoryError:
            print(f"  {B:>5}   {'OOM':>8}  (training state alone doesn't fit)")
            reset()
            continue

        # Now try scoring forward+backward on top
        def mk_inputs():
            Q = F.normalize(torch.randn(B, Lq, 128, dtype=DTYPE, device=device), dim=-1).requires_grad_(True)
            D = F.normalize(torch.randn(B, Ld, 128, dtype=DTYPE, device=device), dim=-1).requires_grad_(True)
            return Q, D

        # pylate path
        try:
            Qp, Dp = mk_inputs()
            sp, mp, pkp = bench_training_step(colbert_scores, Qp, Dp)
            del Qp, Dp
        except torch.cuda.OutOfMemoryError:
            sp, mp, pkp = "OOM", float('nan'), float('nan')
            reset()

        # flash path
        try:
            Qf, Df = mk_inputs()
            sf, mf, pkf = bench_training_step(flash_maxsim_batched_train, Qf, Df, True, None, None)
            del Qf, Df
        except torch.cuda.OutOfMemoryError:
            sf, mf, pkf = "OOM", float('nan'), float('nan')
            reset()

        tm.free()

        pyt_ms_s = f"{mp:>6.1f}" if sp == "OK" else f"{'OOM':>6}"
        pyt_pk_s = f"{pkp:>5.1f}GB" if sp == "OK" else f"{'OOM':>7}"
        fl_ms_s  = f"{mf:>7.1f}" if sf == "OK" else f"{'OOM':>7}"
        fl_pk_s  = f"{pkf:>5.1f}GB" if sf == "OK" else f"{'OOM':>7}"
        if sp == "OK" and sf == "OK":
            sp_str = f"{mp/mf:>5.1f}×"
            mr_str = f"{pkp/pkf:>5.0f}×"
        else:
            sp_str = "    —"
            mr_str = "    —"
        print(f"  {B:>5}   {pyt_ms_s}  {pyt_pk_s:>9}  {fl_ms_s}  {fl_pk_s:>11}   {sp_str}    {mr_str}")

print("\nDone.")
