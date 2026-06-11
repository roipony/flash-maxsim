"""Fair steady-state training bench: warmup + median over runs.

Same model overhead simulation as v1, but:
  - 3 warmup iterations (amortize Triton autotune + cuBLAS heuristic caches)
  - 5 timed runs, take median
  - Explicit Adam optimizer step (makes training bench complete)
  - Also measures the fwd-only and bwd-only cost separately for large-B diagnosis
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


class TrainingMemoryModel:
    """Reserves VRAM equivalent to a real encoder training state."""
    def __init__(self, params, act_per_token_bytes, B, seq_len):
        total_fp16 = 2 * params * 2
        total_fp32 = 2 * params * 4
        act = B * seq_len * act_per_token_bytes
        self.weights = torch.empty(total_fp16 // 2, dtype=DTYPE, device=device)
        self.adam    = torch.empty(total_fp32 // 4, dtype=torch.float32, device=device)
        self.acts    = torch.empty(act // 2, dtype=DTYPE, device=device)

    def free(self):
        del self.weights, self.adam, self.acts
        reset()


def train_step(fn, *args):
    """fwd + bwd; returns scores for completeness."""
    scores = fn(*args)
    if scores.dim() == 2:
        loss = scores.diagonal().sum() - scores.mean() * 0.01
    else:
        loss = scores.sum()
    loss.backward()
    # clear grads between iterations so peak memory measurement stays meaningful
    for a in args:
        if torch.is_tensor(a) and a.requires_grad:
            a.grad = None
    return scores


def timed_block(fn, mk_args, warmup=3, runs=5):
    """Run fn(*mk_args()) with warmup, return (median_ms, peak_gb, scoring_only_gb) or OOM.

    `scoring_only_gb` = additional VRAM used by the scoring call above the
    baseline (training state already resident). That's the number that tells
    you whether the scoring ALONE fits.
    """
    try:
        reset()
        baseline_gb = torch.cuda.memory_allocated() / (1024**3)
        for _ in range(warmup):
            args = mk_args()
            train_step(fn, *args)
            del args
        reset()
        baseline_gb = torch.cuda.memory_allocated() / (1024**3)
        times = []
        for _ in range(runs):
            args = mk_args()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            train_step(fn, *args)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
            del args
        times.sort()
        peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
        scoring_only_gb = peak_gb - baseline_gb
        return ("OK", times[len(times)//2], peak_gb, scoring_only_gb)
    except torch.cuda.OutOfMemoryError:
        peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
        reset()
        return ("OOM", float('nan'), peak_gb, float('nan'))


print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
print("Methodology: 3 warmup + median of 5 runs. Each run is a full fwd+bwd training step.\n")


SCENARIOS = [
    (110e6,  8 * 1024, 32, 180, "ColBERT (BERT-base 110M)"),
    (330e6, 24 * 1024, 32, 180, "ModernColBERT (330M)"),
    (3e9,   48 * 1024, 1024, 1024, "ColPali (PaliGemma-3B)"),
]

for params, apt, Lq, Ld, name in SCENARIOS:
    model_gb = 12 * params / (1024**3)
    print(f"\n{'='*112}")
    print(f"  {name}   Lq={Lq}  Ld={Ld}    Training-state memory: {model_gb:.1f} GB")
    print(f"{'='*112}")
    print(f"  {'B':>5}   {'pyt_ms':>9} {'pyt_tot':>8} {'pyt_score':>10}   "
          f"{'flash_ms':>9} {'fl_tot':>8} {'fl_score':>9}   {'speedup':>8}  {'score_save':>10}")

    if Lq <= 32:
        batches = [32, 128, 512, 1024, 2048, 4096]
    else:
        batches = [8, 16, 32, 64, 128, 256]

    for B in batches:
        # allocate training state first
        torch.manual_seed(B)
        try:
            tm = TrainingMemoryModel(int(params), apt, B, max(Lq, Ld))
        except torch.cuda.OutOfMemoryError:
            print(f"  {B:>5}   {'skip: training state alone OOM':>80}")
            reset()
            continue

        def mk_pyt():
            torch.manual_seed(B)
            Q = F.normalize(torch.randn(B, Lq, 128, dtype=DTYPE, device=device), dim=-1).requires_grad_(True)
            D = F.normalize(torch.randn(B, Ld, 128, dtype=DTYPE, device=device), dim=-1).requires_grad_(True)
            return (Q, D)

        def mk_fl():
            torch.manual_seed(B)
            Q = F.normalize(torch.randn(B, Lq, 128, dtype=DTYPE, device=device), dim=-1).requires_grad_(True)
            D = F.normalize(torch.randn(B, Ld, 128, dtype=DTYPE, device=device), dim=-1).requires_grad_(True)
            return (Q, D, True, None, None)

        sp, mp, pkp, scp = timed_block(colbert_scores,             mk_pyt)
        sf, mf, pkf, scf = timed_block(flash_maxsim_batched_train, mk_fl)
        tm.free()

        pyt_ms_s   = f"{mp:>7.1f}ms" if sp == "OK" else f"{'OOM':>9}"
        pyt_tot_s  = f"{pkp:>5.1f}GB" if sp == "OK" else f"{'':>8}"
        pyt_scr_s  = f"{scp:>5.2f}GB" if sp == "OK" else f"{'':>10}"
        fl_ms_s    = f"{mf:>7.1f}ms" if sf == "OK" else f"{'OOM':>9}"
        fl_tot_s   = f"{pkf:>5.1f}GB" if sf == "OK" else f"{'':>8}"
        fl_scr_s   = f"{scf:>5.2f}GB" if sf == "OK" else f"{'':>9}"
        sp_str     = f"{mp/mf:>6.2f}×" if (sp == "OK" and sf == "OK") else f"{'—':>8}"
        # memory save reports the scoring-activation ratio (the real win)
        mr_str     = f"{scp/max(scf,1e-6):>8.1f}×" if (sp == "OK" and sf == "OK") else f"{'—':>10}"
        print(f"  {B:>5}   {pyt_ms_s} {pyt_tot_s:>8} {pyt_scr_s:>10}   "
              f"{fl_ms_s} {fl_tot_s:>8} {fl_scr_s:>9}   {sp_str}  {mr_str}")

print("\nDone.")
