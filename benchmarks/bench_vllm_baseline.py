"""Benchmark Flash-MaxSim vs vLLM's MaxSim implementation.

vLLM uses padded mini-batches with torch.bmm and manual chunk size control
(max_score_matrix_elements). Flash-MaxSim eliminates this entirely.

Usage: python benchmarks/bench_vllm_baseline.py
"""
import torch, torch.nn.functional as F

assert torch.cuda.is_available(), "CUDA required"
print(f"GPU: {torch.cuda.get_device_name()}\n")

from flash_maxsim import flash_maxsim, flash_maxsim_pairs
from utils import bench_interleaved, compile_high_precision

# ── vLLM's implementation (cleaned up from their PR) ──

def vllm_maxsim(q_embs, d_embs, max_batch_size=16, max_score_matrix_elements=16_000_000):
    """vLLM's compute_maxsim_scores: padded mini-batches with torch.bmm."""
    num_pairs = len(q_embs)
    device = q_embs[0].device
    scores = []
    start = 0
    while start < num_pairs:
        end = min(start + max_batch_size, num_pairs)
        max_q = max(x.shape[0] for x in q_embs[start:end])
        max_d = max(x.shape[0] for x in d_embs[start:end])

        # Shrink batch if score matrix too large
        while (end - start > 1
               and (end - start) * max_q * max_d > max_score_matrix_elements):
            end -= 1
            max_q = max(x.shape[0] for x in q_embs[start:end])
            max_d = max(x.shape[0] for x in d_embs[start:end])

        batch_q = q_embs[start:end]
        batch_d = d_embs[start:end]
        batch_size = end - start
        dim = batch_q[0].shape[1]
        dtype = batch_q[0].dtype

        q_batch = torch.zeros(batch_size, max_q, dim, dtype=dtype, device=device)
        d_batch = torch.zeros(batch_size, max_d, dim, dtype=dtype, device=device)
        q_mask = torch.zeros(batch_size, max_q, dtype=torch.bool, device=device)
        d_mask = torch.zeros(batch_size, max_d, dtype=torch.bool, device=device)

        for i, (q, d) in enumerate(zip(batch_q, batch_d)):
            q_batch[i, :q.shape[0]] = q
            d_batch[i, :d.shape[0]] = d
            q_mask[i, :q.shape[0]] = True
            d_mask[i, :d.shape[0]] = True

        token_scores = torch.bmm(q_batch, d_batch.transpose(1, 2))
        token_scores.masked_fill_(~d_mask.unsqueeze(1), float("-inf"))
        max_per_query = token_scores.amax(dim=-1)
        max_per_query.masked_fill_(~q_mask, 0)
        batch_scores = max_per_query.sum(dim=-1)
        scores.extend(batch_scores.unbind(0))
        start = end

    return torch.stack(scores)

compiled_vllm_maxsim = compile_high_precision(vllm_maxsim)

# ── Flash-MaxSim wrapper for same interface ──

def flash_maxsim_pairs_wrapper(q_embs, d_embs):
    """Score pairs using flash_maxsim_pairs (batched, one kernel)."""
    return flash_maxsim_pairs(q_embs, d_embs)


# ── Benchmark ──

print("=" * 70)
print("vLLM MaxSim vs Flash-MaxSim")
print("=" * 70)
print("vLLM uses padded mini-batches + torch.bmm with manual chunk control.")
print("Flash-MaxSim: fused kernel, no padding, no chunking.\n")

for num_pairs, Lq, Ld, d, label in [
    (10,   32,  300,  128, "10 pairs, ColBERT"),
    (50,   32,  300,  128, "50 pairs, ColBERT"),
    (100,  32,  300,  128, "100 pairs, ColBERT"),
    (500,  32,  300,  128, "500 pairs, ColBERT"),
    (10,   38,  729,  128, "10 pairs, Granite Vision"),
    (50,   38,  729,  128, "50 pairs, Granite Vision"),
    (100,  38,  729,  128, "100 pairs, Granite Vision"),
    (10,   1024, 1024, 128, "10 pairs, Visual (Lq=Ld=1024)"),
    (50,   1024, 1024, 128, "50 pairs, Visual (Lq=Ld=1024)"),
    (100,  1024, 1024, 128, "100 pairs, Visual (Lq=Ld=1024)"),
]:
    q_embs = [F.normalize(torch.randn(Lq, d, device='cuda', dtype=torch.float16), dim=-1) for _ in range(num_pairs)]
    d_embs = [F.normalize(torch.randn(Ld, d, device='cuda', dtype=torch.float16), dim=-1) for _ in range(num_pairs)]

    # Correctness — compare both against FP32 reference
    ref_fp32 = torch.stack([torch.einsum('qd,ld->ql', q.float(), dd.float()).max(1).values.sum() for q, dd in zip(q_embs, d_embs)])
    vllm_scores = vllm_maxsim(q_embs, d_embs)
    flash_scores = flash_maxsim_pairs_wrapper(q_embs, d_embs)
    err_vllm = (ref_fp32.cuda() - vllm_scores.cuda()).abs().max().item()
    err_flash = (ref_fp32.cuda() - flash_scores).abs().max().item()

    # Speed
    t_eager, t_compiled, t_flash = bench_interleaved([vllm_maxsim, compiled_vllm_maxsim, flash_maxsim_pairs_wrapper], [[q_embs, d_embs]])
    eager_speedup = t_eager / t_flash
    compiled_speedup = t_compiled / t_flash

    print(f"  {label:40s}: eager={t_eager:.2f}ms compiled={t_compiled:.2f}ms flash={t_flash:.2f}ms  eager_speedup={eager_speedup:.1f}x compiled_speedup={compiled_speedup:.1f}x  err_vllm={err_vllm:.2e}  err_flash={err_flash:.2e}")

# ── Variable-length pairs (realistic: different queries have different lengths) ──
print(f"\n{'=' * 70}")
print("Variable-length pairs (mixed Lq and Ld)")
print("=" * 70)
print("Each pair has a different query/doc length — realistic serving scenario.\n")

import random
random.seed(42)

for num_pairs, lq_range, ld_range, d, label in [
    (50,  (16, 64),   (100, 500),  128, "50 pairs, text (Lq=16-64, Ld=100-500)"),
    (100, (16, 64),   (100, 500),  128, "100 pairs, text (Lq=16-64, Ld=100-500)"),
    (50,  (32, 128),  (256, 1024), 128, "50 pairs, mixed (Lq=32-128, Ld=256-1024)"),
    (100, (32, 128),  (256, 1024), 128, "100 pairs, mixed (Lq=32-128, Ld=256-1024)"),
    (50,  (128, 1024), (128, 1024), 128, "50 pairs, visual (Lq=128-1024, Ld=128-1024)"),
]:
    q_embs = [F.normalize(torch.randn(random.randint(*lq_range), d, device='cuda', dtype=torch.float16), dim=-1) for _ in range(num_pairs)]
    d_embs = [F.normalize(torch.randn(random.randint(*ld_range), d, device='cuda', dtype=torch.float16), dim=-1) for _ in range(num_pairs)]

    ref_fp32 = torch.stack([torch.einsum('qd,ld->ql', q.float(), dd.float()).max(1).values.sum() for q, dd in zip(q_embs, d_embs)])
    vllm_scores = vllm_maxsim(q_embs, d_embs)
    flash_scores = flash_maxsim_pairs_wrapper(q_embs, d_embs)
    err_vllm = (ref_fp32.cuda() - vllm_scores.cuda()).abs().max().item()
    err_flash = (ref_fp32.cuda() - flash_scores).abs().max().item()

    t_eager, t_compiled, t_flash = bench_interleaved([vllm_maxsim, compiled_vllm_maxsim, flash_maxsim_pairs_wrapper], [[q_embs, d_embs]])
    eager_speedup = t_eager / t_flash
    compiled_speedup = t_compiled / t_flash

    print(f"  {label:55s}: eager={t_eager:.2f}ms compiled={t_compiled:.2f}ms flash={t_flash:.2f}ms eager_speedup={eager_speedup:.1f}x compiled_speedup={compiled_speedup:.1f}x  err_vllm={err_vllm:.2e}  err_flash={err_flash:.2e}")

# ── Single query, multiple docs (the common re-ranking case) ──
print(f"\n{'=' * 70}")
print("Re-ranking: 1 query vs B documents")
print("=" * 70)
print("vLLM treats each (q,d) as a pair. Flash scores all docs at once.\n")

for B, Lq, Ld, d, label in [
    (100,  32,  300,  128, "ColBERT, B=100"),
    (1000, 32,  300,  128, "ColBERT, B=1000"),
    (100,  38,  729,  128, "Granite Vision, B=100"),
    (1000, 38,  729,  128, "Granite Vision, B=1000"),
    (100,  1024, 1024, 128, "Visual, B=100"),
    (1000, 1024, 1024, 128, "Visual, B=1000"),
]:
    Q = F.normalize(torch.randn(Lq, d, device='cuda', dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(B, Ld, d, device='cuda', dtype=torch.float16), dim=-1)

    # vLLM: each doc is a separate pair
    q_embs = [Q] * B
    d_embs = [D[i] for i in range(B)]

    # Flash: one call
    t_eager, t_compiled, t_flash = bench_interleaved([vllm_maxsim, compiled_vllm_maxsim, flash_maxsim],
                                                     [[q_embs, d_embs], [q_embs, d_embs], [Q, D]])
    eager_speedup = t_eager / t_flash
    compiled_speedup = t_compiled / t_flash

    print(f"  {label:35s}: eager={t_eager:.2f}ms compiled={t_compiled:.2f}ms flash={t_flash:.2f}ms eager_speedup={eager_speedup:.1f}x compiled_speedup={compiled_speedup:.1f}x")

print(f"\n{'=' * 70}")
print("DONE")
print("=" * 70)
