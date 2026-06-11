"""Benchmark Flash-MaxSim vs pylate's ColBERT scoring functions.

Tests all 4 pylate scoring functions against Flash equivalents.

Usage: python benchmarks/bench_pylate.py
"""
import torch, torch.nn.functional as F
import random

assert torch.cuda.is_available(), "CUDA required"
print(f"GPU: {torch.cuda.get_device_name()}\n")

from flash_maxsim import flash_maxsim, flash_maxsim_batched
from flash_maxsim.flash_maxsim_varlen import flash_maxsim_varlen, pack_pairs
from utils import bench_interleaved

random.seed(42)
torch.manual_seed(42)


# ── pylate's implementations (copied from their codebase) ──

def pylate_colbert_scores(queries_embeddings, documents_embeddings):
    """pylate: [Nq, Lq, d] × [B, Ld, d] → [Nq, B]"""
    scores = torch.einsum("ash,bth->abst", queries_embeddings, documents_embeddings)
    return scores.max(axis=-1).values.sum(axis=-1)


def pylate_colbert_scores_pairwise(queries_embeddings, documents_embeddings):
    """pylate: per-pair loop → [N]"""
    scores = []
    for q, d in zip(queries_embeddings, documents_embeddings):
        s = torch.einsum("sh,th->st", q, d)
        scores.append(s.max(axis=-1).values.sum())
    return torch.stack(scores, dim=0)


def pylate_colbert_kd_scores(queries_embeddings, documents_embeddings):
    """pylate: [Nq, Lq, d] × [Nq, B, Ld, d] → [Nq, B] (training/KD)"""
    scores = torch.einsum("ash,abth->abst", queries_embeddings, documents_embeddings)
    return scores.max(axis=-1).values.sum(axis=-1)


def print_row(label, t_pylate, t_flash, api):
    sp = t_pylate / t_flash if t_flash > 0 else 0
    winner = "FLASH" if sp > 1 else "pylate"
    print(f"  {label:55s}  pylate={t_pylate:8.2f}ms  flash={t_flash:8.3f}ms  {sp:6.1f}x  [{winner}]  api={api}")


# ═══════════════════════════════════════════════════════════════
print("=" * 120)
print("FLASH-MAXSIM vs PYLATE: All scoring functions")
print("=" * 120)

# ─── FUNCTION 1: colbert_scores (Nq queries × B shared docs) ───
print(f"\n{'─'*120}")
print("FUNCTION 1: colbert_scores — Nq queries × B shared docs → [Nq, B]")
print("pylate: einsum('ash,bth->abst') — materializes full [Nq,B,Lq,Ld] tensor")
print(f"{'─'*120}")

for Nq, B, Lq, Ld, d, label in [
    (5,   100,  32,  300,  128, "ColBERT (Nq=5, B=100)"),
    (5,   1000, 32,  300,  128, "ColBERT (Nq=5, B=1000)"),
    (10,  100,  32,  300,  128, "ColBERT (Nq=10, B=100)"),
    (5,   100,  32,  1024, 128, "Long-doc (Nq=5, B=100, Ld=1024)"),
    (5,   100,  128, 1024, 128, "Medium (Nq=5, Lq=128, Ld=1024)"),
    (5,   50,   1024, 1024, 128, "Visual (Nq=5, Lq=Ld=1024, B=50)"),
    (10,  100,  1024, 1024, 128, "Visual (Nq=10, Lq=Ld=1024, B=100)"),
]:
    Q = F.normalize(torch.randn(Nq, Lq, d, device='cuda', dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(B, Ld, d, device='cuda', dtype=torch.float16), dim=-1)

    # Check OOM for pylate
    sim_gb = Nq * B * Lq * Ld * 4 / 1e9
    if sim_gb > 40:
        print(f"  {label:55s}  pylate=OOM ({sim_gb:.0f}GB sim matrix)")
        tp = float('inf')
    else:
        try:
            tp, tf = bench_interleaved([lambda: pylate_colbert_scores(Q.float(), D.float()),
                                        flash_maxsim_batched(Q, D, shared_docs=True)], [[]])
        except RuntimeError:
            tf = bench_interleaved([lambda: flash_maxsim_batched(Q, D, shared_docs=True)], [[]]).item()
            tp = float('inf')
            print(f"  {label:55s}  pylate=OOM")


    if tp != float('inf'):
        # Correctness
        ref = pylate_colbert_scores(Q.float(), D.float())
        out = flash_maxsim_batched(Q, D, shared_docs=True)
        err = (ref - out).abs().max().item()
        print_row(label, tp, tf, f"flash_maxsim_batched(shared)  err={err:.1e}")
    else:
        print(f"  {label:55s}  pylate=OOM  flash={tf:8.3f}ms  [FLASH]  api=flash_maxsim_batched(shared)")

    del Q, D; torch.cuda.empty_cache()

# ─── FUNCTION 2: colbert_scores_pairwise (per-pair loop) ───
print(f"\n{'─'*120}")
print("FUNCTION 2: colbert_scores_pairwise — N pairs, Python loop")
print("pylate: for loop with einsum per pair")
print(f"{'─'*120}")

for N, Lq, Ld, d, label in [
    (50,  32,  300,  128, "ColBERT 50 pairs"),
    (100, 32,  300,  128, "ColBERT 100 pairs"),
    (500, 32,  300,  128, "ColBERT 500 pairs"),
    (100, 32,  1024, 128, "Long-doc 100 pairs"),
    (100, 128, 1024, 128, "Medium 100 pairs (Lq=128)"),
]:
    Q = F.normalize(torch.randn(N, Lq, d, device='cuda', dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(N, Ld, d, device='cuda', dtype=torch.float16), dim=-1)

    # pylate pairwise
    # Flash varlen
    q_list = [Q[i] for i in range(N)]
    d_list = [D[i] for i in range(N)]
    Q_pk, D_pk, cu_q, cu_d, mlq, mld = pack_pairs(q_list, d_list)
    tp, tf = bench_interleaved([pylate_colbert_scores_pairwise, flash_maxsim_varlen],
                               [[Q.float(), D.float()], [Q_pk, D_pk, cu_q, cu_d, mlq, mld]])

    # Correctness
    ref = pylate_colbert_scores_pairwise(Q.float(), D.float())
    out = flash_maxsim_varlen(Q_pk, D_pk, cu_q, cu_d, mlq, mld)
    err = (ref - out).abs().max().item()

    print_row(label, tp, tf, f"flash_maxsim_varlen  err={err:.1e}")
    del Q, D, Q_pk, D_pk; torch.cuda.empty_cache()

# ─── FUNCTION 2b: pairwise with VARIABLE lengths ───
print(f"\n{'─'*120}")
print("FUNCTION 2b: colbert_scores_pairwise — variable lengths per pair")
print(f"{'─'*120}")

for N, lq_range, ld_range, d, label in [
    (100, (8, 64),   (100, 500),  128, "100 text pairs (Lq=8-64, Ld=100-500)"),
    (500, (8, 64),   (100, 500),  128, "500 text pairs"),
    (100, (16, 128), (300, 1024), 128, "100 mixed pairs (Lq=16-128, Ld=300-1024)"),
    (500, (16, 128), (300, 1024), 128, "500 mixed pairs"),
]:
    q_list = [F.normalize(torch.randn(random.randint(*lq_range), d, device='cuda', dtype=torch.float16), dim=-1) for _ in range(N)]
    d_list = [F.normalize(torch.randn(random.randint(*ld_range), d, device='cuda', dtype=torch.float16), dim=-1) for _ in range(N)]

    # tp - pylate pairwise, tf - flash varlen
    Q_pk, D_pk, cu_q, cu_d, mlq, mld = pack_pairs(q_list, d_list)
    tp, tf = bench_interleaved([lambda: pylate_colbert_scores_pairwise(
        [q.float() for q in q_list], [dd.float() for dd in d_list]),
                            lambda: flash_maxsim_varlen(Q_pk, D_pk, cu_q, cu_d, mlq, mld)], [[]])

    print_row(label, tp, tf, "flash_maxsim_varlen")
    del q_list, d_list, Q_pk, D_pk; torch.cuda.empty_cache()

# ─── FUNCTION 3: colbert_kd_scores (training / knowledge distillation) ───
print(f"\n{'─'*120}")
print("FUNCTION 3: colbert_kd_scores — training with KD, each query has own docs")
print("pylate: einsum('ash,abth->abst') — [Nq, Lq, d] × [Nq, B, Ld, d] → [Nq, B]")
print(f"{'─'*120}")

for Nq, B, Lq, Ld, d, label in [
    (16, 8,  32,  300,  128, "ColBERT training (Nq=16, B=8)"),
    (32, 8,  32,  300,  128, "ColBERT training (Nq=32, B=8)"),
    (16, 16, 32,  300,  128, "ColBERT training (Nq=16, B=16)"),
    (8,  8,  128, 1024, 128, "Medium training (Lq=128, Ld=1024)"),
]:
    Q = F.normalize(torch.randn(Nq, Lq, d, device='cuda', dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(Nq, B, Ld, d, device='cuda', dtype=torch.float16), dim=-1)

    sim_gb = Nq * B * Lq * Ld * 4 / 1e9
    if sim_gb > 40:
        print(f"  {label:55s}  pylate=OOM ({sim_gb:.0f}GB)")
        continue

    tp, tf = bench_interleaved([lambda: pylate_colbert_kd_scores(Q.float(), D.float()),
                                lambda: flash_maxsim_batched(Q, D, shared_docs=False)], [[]])

    # Correctness
    ref = pylate_colbert_kd_scores(Q.float(), D.float())
    out = flash_maxsim_batched(Q, D, shared_docs=False)
    err = (ref - out).abs().max().item()

    print_row(label, tp, tf, f"flash_maxsim_batched(shared=False)  err={err:.1e}")
    del Q, D; torch.cuda.empty_cache()

# ─── FUNCTION 4: rerank (per-query loop + pad) ───
print(f"\n{'─'*120}")
print("FUNCTION 4: rerank — 1 query vs B docs (the main use case)")
print("pylate: pad_sequence + colbert_scores per query")
print(f"{'─'*120}")

for B, Lq, Ld, d, label in [
    (100,  32,  300,  128, "ColBERT B=100"),
    (1000, 32,  300,  128, "ColBERT B=1000"),
    (100,  32,  1024, 128, "Long-doc B=100"),
    (1000, 32,  1024, 128, "Long-doc B=1000"),
    (100,  128, 1024, 128, "Medium B=100 (Lq=128)"),
    (100,  1024, 1024, 128, "Visual B=100 (Lq=Ld=1024)"),
    (1000, 1024, 1024, 128, "Visual B=1000 (Lq=Ld=1024)"),
]:
    Q = F.normalize(torch.randn(Lq, d, device='cuda', dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(B, Ld, d, device='cuda', dtype=torch.float16), dim=-1)

    # pylate rerank: unsqueeze Q + colbert_scores
    sim_gb = B * Lq * Ld * 4 / 1e9
    if sim_gb > 40:
        tp = float('inf')
    else:
        try:
            tp, tf = bench_interleaved([lambda: pylate_colbert_scores(Q.unsqueeze(0).float(), D.float()),
                                        lambda: flash_maxsim(Q, D)], [[]])
        except RuntimeError:
            tf = bench_interleaved([lambda: flash_maxsim(Q, D)], []).item()
            tp = float('inf')

    if tp != float('inf'):
        print_row(label, tp, tf, "flash_maxsim")
    else:
        print(f"  {label:55s}  pylate=OOM  flash={tf:8.3f}ms  [FLASH]  api=flash_maxsim")

    del Q, D; torch.cuda.empty_cache()

# ─── SUMMARY ───
print(f"\n{'═'*120}")
print("SUMMARY: Flash-MaxSim drop-in replacements for pylate")
print(f"{'═'*120}")
print("""
  pylate function              Flash replacement                          Notes
  ───────────────────────────  ─────────────────────────────────────────   ──────────────────────
  colbert_scores               flash_maxsim_batched(shared_docs=True)     Nq queries × B shared docs
  colbert_scores_pairwise      flash_maxsim_varlen + pack_pairs           N independent pairs
  colbert_kd_scores            flash_maxsim_batched(shared_docs=False)    Training with KD
  rerank                       flash_maxsim(Q, D)                         1 query vs B docs
""")
