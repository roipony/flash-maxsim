"""Probe the argmax sparsity assumption for sparse-output dD.

For the KD-4D layout, our sparse-output kernel idea assumes that per (q, k)
pair, the saved argmax distributes its Lq winners across only a few unique
Ld destinations (so we can skip 97%+ of dD writes).

This script measures the actual distribution on a few realistic shapes:
- colpali-neg B256 n16 Ld1030 Lq32 d128
- colpali-neg B128 n8 Ld1030 Lq32 d128
- pylate-text B256 Lq32 Ld300 d128 (cross-product, for reference)
- longq B64 Lq1030 Ld1030 (where Lq is no longer small)

For each shape and dtype:
- Run forward to get argmax[Nq*Nd, Lq]
- For each (q, k) pair: count unique destinations
- Report histogram + summary stats (min, mean, median, p90, max)
"""
import statistics

import torch

from flash_maxsim import flash_maxsim_batched_train
from flash_maxsim.flash_maxsim import _launch_fwd, _default_lengths


def measure_sparsity(Nq, K, Lq, Ld, d, dtype, label):
    """Run forward, count unique argmax destinations per (q, k) slab."""
    Q = torch.randn(Nq, Lq, d, device="cuda", dtype=dtype)
    # KD-4D layout: D is [Nq, K, Ld, d] but the forward expects [Nq*K, Ld, d]
    D = torch.randn(Nq * K, Ld, d, device="cuda", dtype=dtype)

    Q_c = Q.contiguous().half().clone()
    D_c = D.contiguous().half()
    lengths = _default_lengths(Nq * K, Ld, D.device, None)

    # Run forward; the non-shared / KD layout: D_c is [Nq*K, Ld, d], shared_docs=False.
    scores, argmax = _launch_fwd(
        Q_c, D_c, lengths, Nq, K, Lq, Ld, d, shared_docs=False, save_argmax=True,
    )
    # argmax shape: [Nq*K, Lq] int32
    argmax = argmax.view(Nq, K, Lq)

    # For each (q, k), count unique destinations
    uniques = []
    for q in range(Nq):
        for k in range(K):
            row = argmax[q, k, :].cpu().tolist()
            uniques.append(len(set(row)))

    n = len(uniques)
    print(f"\n=== {label} ({dtype}) ===")
    print(f"  shape: Nq={Nq} K={K} Lq={Lq} Ld={Ld} d={d}")
    print(f"  pairs measured: {n}")
    print(f"  uniques: min={min(uniques)}  mean={sum(uniques)/n:.1f}  "
          f"median={statistics.median(uniques):.1f}  max={max(uniques)}")
    print(f"  uniques distribution (deciles 10/25/50/75/90/95):  "
          f"{sorted(uniques)[n*1//10]} / {sorted(uniques)[n*1//4]} / "
          f"{sorted(uniques)[n//2]} / {sorted(uniques)[n*3//4]} / "
          f"{sorted(uniques)[n*9//10]} / {sorted(uniques)[n*95//100]}")
    # Sparsity: fraction of Ld destinations that are zero per slab
    avg_uniques = sum(uniques) / n
    sparsity = 1.0 - (avg_uniques / Ld)
    print(f"  avg sparsity vs Ld: {sparsity*100:.1f}% of Ld rows are zero")
    print(f"  → write-bandwidth saving: {sparsity*100:.1f}% of grad_D writes can be skipped")
    return uniques


if __name__ == "__main__":
    print("# Argmax sparsity probe (KD-4D layout) — random unit-norm embeddings\n")
    print("Hypothesis: sparse-output dD wins when uniques << Ld\n")

    shapes = [
        # KD-4D shapes (where sparse-output should win big)
        ("colpali-neg B256 n16 Ld1030", 256, 16, 32, 1030, 128),
        ("colpali-neg B128 n8 Ld1030",  128,  8, 32, 1030, 128),
        # pylate-text cross-product for reference (not the target regime)
        ("pylate-text B256 Lq32 Ld300", 256,  1, 32,  300, 128),  # K=1 ≈ cross
        # long-Lq stress (where sparse-output is less helpful)
        ("longq B64 Lq1030 Ld1030",      64,  1, 1030, 1030, 128),  # K=1
    ]
    for dtype in (torch.bfloat16, torch.float16):
        for tag, Nq, K, Lq, Ld, d in shapes:
            try:
                measure_sparsity(Nq, K, Lq, Ld, d, dtype, tag)
            except Exception as e:
                print(f"\n!! {tag} ({dtype}): {type(e).__name__}: {e}")
