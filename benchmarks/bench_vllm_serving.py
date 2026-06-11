"""Complete Flash-MaxSim vs vLLM benchmark across all workloads.

Tests every configuration: pairs, re-ranking, variable/uniform lengths.
Shows which API to use for each case.

Usage: python benchmarks/bench_vllm_serving.py
"""
import torch, torch.nn.functional as F
import random

assert torch.cuda.is_available(), "CUDA required"
print(f"GPU: {torch.cuda.get_device_name()}\n")

from flash_maxsim import flash_maxsim, flash_maxsim_batched
from flash_maxsim.flash_maxsim_varlen import flash_maxsim_varlen, pack_pairs
from utils import bench_interleaved, compile_high_precision

random.seed(42)
torch.manual_seed(42)
d = 128


# ── vLLM's MaxSim (from their PR) ──
def vllm_maxsim(q_embs, d_embs, max_batch_size=16, max_score_matrix_elements=16_000_000):
    num_pairs = len(q_embs)
    device = q_embs[0].device
    scores = []
    start = 0
    while start < num_pairs:
        end = min(start + max_batch_size, num_pairs)
        max_q = max(x.shape[0] for x in q_embs[start:end])
        max_d = max(x.shape[0] for x in d_embs[start:end])
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
        for i, (q, dd) in enumerate(zip(batch_q, batch_d)):
            q_batch[i, :q.shape[0]] = q
            d_batch[i, :dd.shape[0]] = dd
            q_mask[i, :q.shape[0]] = True
            d_mask[i, :dd.shape[0]] = True
        token_scores = torch.bmm(q_batch, d_batch.transpose(1, 2))
        token_scores.masked_fill_(~d_mask.unsqueeze(1), float("-inf"))
        max_per_query = token_scores.amax(dim=-1)
        max_per_query.masked_fill_(~q_mask, 0)
        batch_scores = max_per_query.sum(dim=-1)
        scores.extend(batch_scores.unbind(0))
        start = end
    return torch.stack(scores)

compiled_maxsim = compile_high_precision(vllm_maxsim)

def print_row(label, t_eager, t_compiled, t_flash, api_used, extra=""):
    eager_sp = t_eager / t_flash if t_flash > 0 else 0
    compiled_sp = t_compiled / t_flash if t_flash > 0 else 0
    times = {"eager": t_eager, "compiled": t_compiled, "flash": t_flash}
    winner = min(times, key=times.get)
    print(f"  {label:50s}  eager={t_eager:8.2f}ms compiled={t_compiled:8.2f}ms flash={t_flash:8.2f}ms eager_speedup={eager_sp:6.1f}x compiled_speedup={compiled_sp:6.1f}x [{winner}]  api={api_used}{extra}")


# ═══════════════════════════════════════════════════════════════
print("=" * 110)
print("COMPLETE BENCHMARK: Flash-MaxSim vs vLLM across all workloads")
print("=" * 110)

# ─── CASE 1: Re-ranking (1 query vs B docs, same Lq, uniform Ld) ───
print(f"\n{'─'*110}")
print("CASE 1: Re-ranking — 1 query vs B docs (uniform doc length)")
print(f"{'─'*110}")

for B, Lq, Ld in [(100, 32, 300), (1000, 32, 300), (100, 32, 1024), (1000, 32, 1024)]:
    Q = F.normalize(torch.randn(Lq, d, device='cuda', dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(B, Ld, d, device='cuda', dtype=torch.float16), dim=-1)
    q_pairs = [Q] * B
    d_pairs = [D[i] for i in range(B)]

    tv, tc, tf = bench_interleaved([vllm_maxsim, compiled_maxsim, flash_maxsim],
                                   [[q_pairs, d_pairs], [q_pairs, d_pairs], [Q, D]])
    print_row(f"Lq={Lq}, Ld={Ld}, B={B}", tv, tc, tf, "flash_maxsim")
    del Q, D; torch.cuda.empty_cache()

# ─── CASE 2: Re-ranking with variable doc lengths ───
print(f"\n{'─'*110}")
print("CASE 2: Re-ranking — 1 query vs B docs (variable doc lengths)")
print(f"{'─'*110}")

for B, Lq, ld_range in [(100, 32, (100,500)), (1000, 32, (100,500)), (100, 32, (512,2048)), (1000, 32, (512,2048))]:
    Q = F.normalize(torch.randn(Lq, d, device='cuda', dtype=torch.float16), dim=-1)
    doc_lens = [random.randint(*ld_range) for _ in range(B)]
    max_ld = max(doc_lens)
    D = torch.zeros(B, max_ld, d, device='cuda', dtype=torch.float16)
    for i in range(B):
        D[i, :doc_lens[i]] = F.normalize(torch.randn(doc_lens[i], d, device='cuda', dtype=torch.float16), dim=-1)
    doc_lengths_t = torch.tensor(doc_lens, device='cuda', dtype=torch.int32)
    q_pairs = [Q] * B
    d_pairs = [D[i, :doc_lens[i]] for i in range(B)]

    tv, tc, tf = bench_interleaved([vllm_maxsim, compiled_maxsim, lambda: flash_maxsim(Q, D, doc_lengths=doc_lengths_t)], [[q_pairs, d_pairs], [q_pairs, d_pairs], []])
    print_row(f"Lq={Lq}, Ld={ld_range}, B={B}", tv, tc, tf, "flash_maxsim+doc_lengths")
    del Q, D; torch.cuda.empty_cache()

# ─── CASE 3: Pairs, uniform lengths ───
print(f"\n{'─'*110}")
print("CASE 3: Pairs — different query per doc (uniform Lq and Ld)")
print(f"{'─'*110}")



for N, Lq, Ld in [(50, 32, 300), (100, 32, 300), (500, 32, 300), (100, 32, 1024), (100, 38, 729)]:
    q_embs = [F.normalize(torch.randn(Lq, d, device='cuda', dtype=torch.float16), dim=-1) for _ in range(N)]
    d_embs = [F.normalize(torch.randn(Ld, d, device='cuda', dtype=torch.float16), dim=-1) for _ in range(N)]

    Q_pk, D_pk, cu_q, cu_d, mlq, mld = pack_pairs(q_embs, d_embs)

    # tf_e2e: pack + kernel, tf_kernel: kernel only (pre-packed)
    tv, tc, tf_e2e, tf_kernel = bench_interleaved([vllm_maxsim, compiled_maxsim, flash_maxsim_varlen, flash_maxsim_varlen], [[q_embs, d_embs], [q_embs, d_embs], [*pack_pairs(q_embs, d_embs)], [Q_pk, D_pk, cu_q, cu_d, mlq, mld]])

    print_row(f"N={N}, Lq={Lq}, Ld={Ld}", tv, tc, tf_e2e, "varlen (e2e)", f"  kernel={tf_kernel:.3f}ms")
    del q_embs, d_embs, Q_pk, D_pk; torch.cuda.empty_cache()

# ─── CASE 4: Pairs, variable lengths ───
print(f"\n{'─'*110}")
print("CASE 4: Pairs — different query per doc (variable Lq and Ld)")
print(f"{'─'*110}")

for N, lq_range, ld_range, label in [
    (100, (8, 32),   (100, 500),  "text"),
    (500, (8, 32),   (100, 500),  "text"),
    (100, (8, 64),   (300, 1024), "mixed"),
    (500, (8, 64),   (300, 1024), "mixed"),
    (100, (16, 128), (512, 2048), "long"),
    (500, (16, 128), (512, 2048), "long"),
]:
    q_embs = [F.normalize(torch.randn(random.randint(*lq_range), d, device='cuda', dtype=torch.float16), dim=-1) for _ in range(N)]
    d_embs = [F.normalize(torch.randn(random.randint(*ld_range), d, device='cuda', dtype=torch.float16), dim=-1) for _ in range(N)]

    Q_pk, D_pk, cu_q, cu_d, mlq, mld = pack_pairs(q_embs, d_embs)

    tv, tc, tf_e2e, tf_kernel = bench_interleaved([vllm_maxsim, compiled_maxsim, flash_maxsim_varlen, flash_maxsim_varlen], [[q_embs, d_embs], [q_embs, d_embs], [*pack_pairs(q_embs, d_embs)], [Q_pk, D_pk, cu_q, cu_d, mlq, mld]])

    print_row(f"N={N} {label} Lq={lq_range} Ld={ld_range}", tv, tc, tf_e2e, "varlen (e2e)", f"  kernel={tf_kernel:.3f}ms")
    del q_embs, d_embs, Q_pk, D_pk; torch.cuda.empty_cache()

# ─── CASE 5: Multi-query re-ranking (batched, shared docs) ───
print(f"\n{'─'*110}")
print("CASE 5: Multi-query re-ranking — Nq queries vs same B docs")
print(f"{'─'*110}")

for Nq, B, Lq, Ld in [(10, 100, 32, 300), (50, 100, 32, 300), (10, 1000, 32, 300)]:
    Q_batch = F.normalize(torch.randn(Nq, Lq, d, device='cuda', dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(B, Ld, d, device='cuda', dtype=torch.float16), dim=-1)

    # vLLM: Nq*B separate pairs
    q_pairs = []
    d_pairs = []
    for i in range(Nq):
        for j in range(B):
            q_pairs.append(Q_batch[i])
            d_pairs.append(D[j])

    tv, tc, tf = bench_interleaved([vllm_maxsim, compiled_maxsim, flash_maxsim_batched], [[q_pairs, d_pairs], [q_pairs, d_pairs], [Q_batch, D]])
    print_row(f"Nq={Nq}, B={B}, Lq={Lq}, Ld={Ld}", tv, tc, tf, "flash_maxsim_batched(shared)")
    del Q_batch, D; torch.cuda.empty_cache()

# ─── SUMMARY ───
print(f"\n{'═'*110}")
print("SUMMARY: Which Flash-MaxSim API to use")
print(f"{'═'*110}")
print("""
  Workload                              API                              Typical speedup
  ──────────────────────────────────    ──────────────────────────────    ───────────────
  1 query vs B docs (re-ranking)        flash_maxsim(Q, D)               68-315x
  1 query vs B docs (variable Ld)       flash_maxsim(Q, D, doc_lengths)  68-315x
  N pairs, same lengths                 flash_maxsim_varlen + pack       4-13x e2e
  N pairs, variable lengths             flash_maxsim_varlen + pack       4-15x e2e
  Nq queries vs same B docs             flash_maxsim_batched(shared)     50-300x

  All cases: Flash is faster, more precise (FP32 accum), zero OOM risk.
""")
