"""Collect varlen benchmark data for the plot.

Run: python benchmarks/collect_varlen_data.py
"""
import json, sys, time, torch
import torch.nn.functional as F
sys.path.insert(0, '.')
from flash_maxsim import flash_maxsim_varlen, pack_pairs
from flash_maxsim.naive import maxsim_naive

device = 'cuda'
D_DIM = 128
DTYPE = torch.float16

def bench(fn, warmup=10, runs=15):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]

gpu = torch.cuda.get_device_name(0)
print(f"GPU: {gpu}")

results = []

# ColBERT skewed (most docs short, few long — realistic)
for N in [500, 1000, 5000, 10000, 20000, 50000, 100000]:
    torch.manual_seed(42)
    avg_ld = 49
    Lq_range = (20, 32)
    Ld_max = 300

    # Poisson-like lengths
    lens_d = torch.poisson(torch.full((N,), float(avg_ld))).int().clamp(5, Ld_max).tolist()
    lens_q = torch.randint(Lq_range[0], Lq_range[1]+1, (N,)).tolist()

    q_embs = [F.normalize(torch.randn(lq, D_DIM, device=device, dtype=DTYPE), dim=-1) for lq in lens_q]
    d_embs = [F.normalize(torch.randn(ld, D_DIM, device=device, dtype=DTYPE), dim=-1) for ld in lens_d]

    # Varlen
    Q_pk, D_pk, cu_q, cu_d, max_lq, max_ld = pack_pairs(q_embs, d_embs)
    varlen_ms = bench(lambda: flash_maxsim_varlen(Q_pk, D_pk, cu_q, cu_d, max_lq, max_ld))

    # Naive: pad to max and score
    max_lq_pad = max(lens_q)
    max_ld_pad = max(lens_d)
    Q_pad = torch.zeros(N, max_lq_pad, D_DIM, device=device, dtype=DTYPE)
    D_pad = torch.zeros(N, max_ld_pad, D_DIM, device=device, dtype=DTYPE)
    for i in range(N):
        Q_pad[i, :lens_q[i]] = q_embs[i]
        D_pad[i, :lens_d[i]] = d_embs[i]

    # Naive batched padded: pad all to max lengths, single batched einsum
    def naive_padded():
        S = torch.einsum('nqd,nld->nql', Q_pad, D_pad)
        return S.max(dim=2).values.sum(dim=1)

    try:
        naive_ms = bench(naive_padded) if N <= 20000 else None
    except torch.cuda.OutOfMemoryError:
        naive_ms = None
        torch.cuda.empty_cache()

    # D savings
    total_d = sum(lens_d)
    padded_d = N * max_ld_pad
    d_saved = 1 - total_d / padded_d

    row = {
        "tag": "colbert_skewed", "N": N,
        "avg_ld": round(sum(lens_d)/N, 1),
        "max_ld": max(lens_d),
        "varlen_ms": round(varlen_ms, 3),
        "naive_ms": round(naive_ms, 3) if naive_ms else None,
        "speedup": round(naive_ms / varlen_ms, 2) if naive_ms else None,
        "d_saved_pct": round(d_saved * 100, 1),
    }
    results.append(row)
    sp_str = f"{naive_ms/varlen_ms:.1f}x" if naive_ms else "—"
    naive_str = f"{naive_ms:.2f}" if naive_ms else "OOM"
    print(f"  ColBERT-skewed N={N}: varlen={varlen_ms:.2f}ms padded_naive={naive_str}ms sp={sp_str} d_saved={d_saved*100:.0f}%")

    del q_embs, d_embs, Q_pk, D_pk, Q_pad, D_pad
    torch.cuda.empty_cache()

# ColPali variable
for N in [100, 500, 1000, 2000]:
    torch.manual_seed(42)
    lens_q = torch.randint(512, 1025, (N,)).tolist()
    lens_d = torch.randint(256, 1025, (N,)).tolist()

    q_embs = [F.normalize(torch.randn(lq, D_DIM, device=device, dtype=DTYPE), dim=-1) for lq in lens_q]
    d_embs = [F.normalize(torch.randn(ld, D_DIM, device=device, dtype=DTYPE), dim=-1) for ld in lens_d]

    Q_pk, D_pk, cu_q, cu_d, max_lq, max_ld = pack_pairs(q_embs, d_embs)
    varlen_ms = bench(lambda: flash_maxsim_varlen(Q_pk, D_pk, cu_q, cu_d, max_lq, max_ld))

    # Padded batched naive
    max_lq_pad = max(lens_q)
    max_ld_pad = max(lens_d)
    Q_pad = torch.zeros(N, max_lq_pad, D_DIM, device=device, dtype=DTYPE)
    D_pad = torch.zeros(N, max_ld_pad, D_DIM, device=device, dtype=DTYPE)
    for i in range(N):
        Q_pad[i, :lens_q[i]] = q_embs[i]
        D_pad[i, :lens_d[i]] = d_embs[i]

    def naive_padded_colpali():
        S = torch.einsum('nqd,nld->nql', Q_pad, D_pad)
        return S.max(dim=2).values.sum(dim=1)

    try:
        naive_ms = bench(naive_padded_colpali)
    except torch.cuda.OutOfMemoryError:
        naive_ms = None
        torch.cuda.empty_cache()

    total_d = sum(lens_d)
    padded_d = N * max_ld_pad
    d_saved = 1 - total_d / padded_d

    row = {
        "tag": "colpali_varlen", "N": N,
        "avg_ld": round(sum(lens_d)/N, 1),
        "max_ld": max(lens_d),
        "varlen_ms": round(varlen_ms, 3),
        "naive_ms": round(naive_ms, 3) if naive_ms else None,
        "speedup": round(naive_ms / varlen_ms, 2) if naive_ms else None,
        "d_saved_pct": round(d_saved * 100, 1),
    }
    results.append(row)
    sp_str = f"{naive_ms/varlen_ms:.1f}x" if naive_ms else "—"
    naive_str = f"{naive_ms:.2f}" if naive_ms else "OOM"
    print(f"  ColPali-varlen N={N}: varlen={varlen_ms:.2f}ms naive={naive_str}ms sp={sp_str} d_saved={d_saved*100:.0f}%")

    del q_embs, d_embs, Q_pk, D_pk, Q_pad, D_pad
    torch.cuda.empty_cache()

outfile = "benchmarks/results_varlen_NVIDIA_A100-SXM4-80GB.json"
with open(outfile, "w") as f:
    json.dump({"gpu": gpu, "varlen": results}, f, indent=2)
print(f"\nSaved: {outfile}")
print("Done.")
