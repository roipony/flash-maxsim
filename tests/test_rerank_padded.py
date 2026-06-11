"""Parity + perf test: flash_maxsim_rerank_padded vs erikkaum's naive baseline.

flash_maxsim_rerank_padded mirrors erikkaum/maxsim's score_candidates_padded
signature exactly. This test verifies:

  1. Numerical parity vs the same naive reference Erik ships in his
     benchmarks/benchmark.py — _naive_maxsim_padded.
  2. Speedup on Erik's three published workloads (SmallRerank, HeavyRerank,
     LongDocStress).

Run:  python tests/test_rerank_padded.py
"""
from __future__ import annotations
import sys, time
import torch

sys.path.insert(0, "/path/to/data")
from flash_maxsim import flash_maxsim_rerank_padded

DEV = "cuda"


# ── Erik's exact naive baseline, copied verbatim from his benchmark.py ──

def _naive_maxsim_padded(queries, documents, query_lengths, doc_lengths):
    B, C, Ld, D = documents.shape
    Lq = queries.shape[1]
    sim = torch.einsum("bid,bcjd->bcij", queries.float(), documents.float())
    q_mask = (torch.arange(Lq, device=queries.device)[None, :]
              < query_lengths[:, None])
    d_mask = (torch.arange(Ld, device=queries.device)[None, None, :]
              < doc_lengths[:, :, None])
    sim = sim.masked_fill(~d_mask[:, :, None, :], float("-inf"))
    per_q_max = sim.max(dim=-1).values
    per_q_max = per_q_max.masked_fill(~q_mask[:, None, :], 0.0)
    return per_q_max.sum(dim=-1)


# ── helpers ────────────────────────────────────────────────────────────

def _make_inputs(B, C, Lq, Ld, D, seed=1234):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(B, Lq, D, generator=gen, dtype=torch.float16).to(DEV)
    d = torch.randn(B, C, Ld, D, generator=gen, dtype=torch.float16).to(DEV)
    q_lens = torch.full((B,),    Lq, dtype=torch.int32, device=DEV)
    d_lens = torch.full((B, C),  Ld, dtype=torch.int32, device=DEV)
    return q, d, q_lens, d_lens

def _bench(fn, *args, warmup=3, n=20):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn(*args)
    torch.cuda.synchronize()
    return (time.time() - t0) * 1000 / n   # ms


# ── Erik's three published workloads ───────────────────────────────────

WORKLOADS = [
    ("SmallRerank",   dict(B=32, C=10,  Lq=32, Ld=180,  D=128)),
    ("HeavyRerank",   dict(B=32, C=100, Lq=32, Ld=256,  D=128)),
    ("LongDocStress", dict(B=8,  C=16,  Lq=64, Ld=1024, D=128)),
]


print(f"GPU: {torch.cuda.get_device_name(0)}\n")
print(f"{'workload':<16} {'naive (ms)':>10} {'flash-API':>10} {'flash-fast':>11} "
      f"{'API ×':>7} {'fast ×':>8} {'max |Δ|':>9}")
print(f"{'':<16} {'':<10} {'+lengths':>10} {'lengths=None':>11} "
      f"{'':<7} {'':<8}")
print("─" * 86)

for name, cfg in WORKLOADS:
    q, d, q_lens, d_lens = _make_inputs(**cfg)

    # Correctness: with lengths (API-compat path)
    ref      = _naive_maxsim_padded(q, d, q_lens, d_lens)
    f_api    = flash_maxsim_rerank_padded(q, d, q_lens, d_lens)
    f_fast   = flash_maxsim_rerank_padded(q, d, None, None)
    abs_err  = (ref.float() - f_api.float()).abs()
    abs_err2 = (ref.float() - f_fast.float()).abs()
    max_abs  = max(abs_err.max().item(), abs_err2.max().item())

    # Timing
    t_naive = _bench(_naive_maxsim_padded, q, d, q_lens, d_lens)
    t_api   = _bench(flash_maxsim_rerank_padded, q, d, q_lens, d_lens)
    t_fast  = _bench(flash_maxsim_rerank_padded, q, d, None, None)

    print(f"{name:<16} {t_naive:>10.2f} {t_api:>10.2f} {t_fast:>11.2f} "
          f"{t_naive/t_api:>6.2f}x {t_naive/t_fast:>7.2f}x {max_abs:>9.5f}")


# ── varying-length sanity check (the masking actually fires) ────────────

print("\nVarying-length sanity check (masking active):")
q, d, _, _ = _make_inputs(B=8, C=16, Lq=64, Ld=1024, D=128, seed=42)
# Pick weird non-uniform lengths so masking really exercises:
q_lens = torch.tensor([12, 64, 33, 47, 64, 8, 25, 64],
                      dtype=torch.int32, device=DEV)
d_lens = torch.randint(100, 1025, (8, 16), dtype=torch.int32, device=DEV)

ref   = _naive_maxsim_padded(q, d, q_lens, d_lens)
flash = flash_maxsim_rerank_padded(q, d, q_lens, d_lens)
abs_err = (ref.float() - flash.float()).abs()
rel_err = abs_err / ref.float().abs().clamp_min(1e-6)
print(f"  max |Δ| = {abs_err.max().item():.5f}, "
      f"max rel err = {rel_err.max().item():.2%}, "
      f"{'✓ PASS' if abs_err.max().item() < 1e-2 else '✗ FAIL'}")
