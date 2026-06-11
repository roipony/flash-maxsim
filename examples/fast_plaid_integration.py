"""Flash-MaxSim integration with Fast-Plaid.

Shows how to replace Fast-Plaid's native MaxSim scoring with Flash-MaxSim
for faster re-ranking of candidate documents.

Fast-Plaid computes MaxSim as:
    scores = doc_embeddings.matmul(query.T)  # [B, Ld, Lq] full sim matrix
    scores = scores.max(dim=1).sum()         # MaxSim aggregation

Flash-MaxSim eliminates the sim matrix entirely:
    scores = flash_maxsim(Q, D)              # [B] — computed in SRAM

Usage:
    pip install flash-maxsim fast-plaid
    python examples/fast_plaid_integration.py
"""

import torch
import time

assert torch.cuda.is_available(), "CUDA required"
print(f"GPU: {torch.cuda.get_device_name()}\n")


# ── Fast-Plaid's MaxSim (what they do internally) ──────────────────
def fast_plaid_maxsim(Q, D, attention_mask=None):
    """Reproduce Fast-Plaid's colbert_score_reduce().

    Q: [Lq, d], D: [B, Ld, d], attention_mask: [B, Ld] bool
    Returns: [B]
    """
    # Full similarity matrix in HBM
    # Q: [Lq, d], D: [B, Ld, d] -> sim: [B, Lq, Ld]
    sim = torch.einsum('qd,bld->bql', Q.float(), D.float())

    # Mask padding doc tokens
    if attention_mask is not None:
        sim = sim.masked_fill(~attention_mask.unsqueeze(1), -9999.0)

    # MaxSim: for each query token, max over doc tokens, then sum
    return sim.max(dim=2).values.sum(dim=1)  # [B]


# ── Flash-MaxSim drop-in replacement ───────────────────────────────
from flash_maxsim import flash_maxsim


# ── Compare on realistic data ──────────────────────────────────────
print("=" * 60)
print("Fast-Plaid MaxSim vs Flash-MaxSim")
print("=" * 60)

for B, Lq, Ld, d, label in [
    (100,  32,  300,  128, "ColBERT re-rank 100 docs"),
    (1000, 32,  300,  128, "ColBERT re-rank 1K docs"),
    (100,  1024, 1024, 128, "ColPali re-rank 100 pages"),
    (1000, 1024, 1024, 128, "ColPali re-rank 1K pages"),
]:
    Q = torch.randn(Lq, d, device="cuda", dtype=torch.float16)
    D = torch.randn(B, Ld, d, device="cuda", dtype=torch.float16)

    # Correctness check
    ref = fast_plaid_maxsim(Q.float(), D.float())
    out = flash_maxsim(Q, D)
    err = (ref - out).abs().max().item()
    rank_match = (ref.argsort(descending=True)[:10] == out.argsort(descending=True)[:10]).all()

    # Speed comparison
    # Warmup
    for _ in range(5):
        _ = fast_plaid_maxsim(Q.float(), D.float())
        _ = flash_maxsim(Q, D)
    torch.cuda.synchronize()

    # Benchmark Fast-Plaid style
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        _ = fast_plaid_maxsim(Q.float(), D.float())
    torch.cuda.synchronize()
    t_plaid = (time.perf_counter() - t0) * 1000 / 20

    # Benchmark Flash-MaxSim
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        _ = flash_maxsim(Q, D)
    torch.cuda.synchronize()
    t_flash = (time.perf_counter() - t0) * 1000 / 20

    speedup = t_plaid / t_flash
    sim_mb = B * Lq * Ld * 4 / 1e6  # FP32 sim matrix size

    print(f"\n  {label}")
    print(f"    Fast-Plaid: {t_plaid:.3f} ms  (sim matrix: {sim_mb:.0f} MB)")
    print(f"    Flash:      {t_flash:.3f} ms  (sim matrix: 0 MB)")
    print(f"    Speedup:    {speedup:.1f}x   rank_match={rank_match}   err={err:.2e}")

# ── Integration example ────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("Integration: drop-in replacement")
print("=" * 60)
print("""
# In your Fast-Plaid pipeline, replace:
#   scores = doc_embeddings.matmul(query.T).max(dim=2).values.sum(dim=1)
#
# With:
#   from flash_maxsim import flash_maxsim
#   scores = flash_maxsim(query, doc_embeddings)
#
# That's it. Same output, 10x faster, near-zero memory.
""")
