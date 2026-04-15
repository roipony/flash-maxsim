"""Flash-MaxSim v0.2.0 — Complete API Examples.

All public APIs with runnable examples. Each section is self-contained.

Usage:
    python examples/demo_v020.py
"""

import torch
import torch.nn.functional as F
import time


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# ============================================================================
# 1. Basic Scoring: flash_maxsim
# ============================================================================

def demo_basic():
    section("1. Basic Scoring — flash_maxsim")
    from flash_maxsim import flash_maxsim

    # Score one query against 1000 documents
    Q = torch.randn(32, 128, device="cuda", dtype=torch.float16)   # [Lq=32, d=128]
    D = torch.randn(1000, 300, 128, device="cuda", dtype=torch.float16)  # [B=1000, Ld=300, d=128]

    scores = flash_maxsim(Q, D)  # [1000]
    print(f"Scores shape: {scores.shape}")
    print(f"Top-5 indices: {scores.topk(5).indices.tolist()}")
    print(f"Top-5 scores:  {scores.topk(5).values.tolist()}")

    # With variable-length documents (some docs shorter than 300)
    doc_lengths = torch.randint(50, 301, (1000,), device="cuda", dtype=torch.int32)
    scores_vl = flash_maxsim(Q, D, doc_lengths=doc_lengths)
    print(f"\nWith variable lengths — scores shape: {scores_vl.shape}")

    # ColPali-scale: long query (1024 tokens) — automatic chunking
    Q_long = torch.randn(1024, 128, device="cuda", dtype=torch.float16)
    scores_long = flash_maxsim(Q_long, D, query_chunk_size=128)
    print(f"ColPali query (Lq=1024) — scores shape: {scores_long.shape}")


# ============================================================================
# 2. Batched Scoring: flash_maxsim_batched
# ============================================================================

def demo_batched():
    section("2. Batched Scoring — flash_maxsim_batched")
    from flash_maxsim import flash_maxsim_batched

    # 10 different queries, all scored against the same 500 documents
    Q = torch.randn(10, 32, 128, device="cuda", dtype=torch.float16)  # [Nq=10, Lq=32, d=128]
    D = torch.randn(500, 300, 128, device="cuda", dtype=torch.float16)  # [B=500, Ld=300, d=128]

    # shared_docs=True: all queries share the same document set (reranking)
    scores = flash_maxsim_batched(Q, D, shared_docs=True)  # [10, 500]
    print(f"Shared docs — scores shape: {scores.shape}")

    # With per-query variable query lengths
    query_lengths = torch.randint(20, 33, (10,), device="cuda", dtype=torch.int32)
    scores_ql = flash_maxsim_batched(Q, D, shared_docs=True, query_lengths=query_lengths)
    print(f"Variable query lengths — scores shape: {scores_ql.shape}")


# ============================================================================
# 3. Variable-Length Pairs: flash_maxsim_varlen
# ============================================================================

def demo_varlen():
    section("3. Variable-Length Pairs — flash_maxsim_varlen + pack_pairs")
    from flash_maxsim import flash_maxsim_varlen, pack_pairs

    # 50 query-document pairs, each with different lengths
    d = 128
    q_embs = [torch.randn(torch.randint(10, 64, (1,)).item(), d, device="cuda", dtype=torch.float16)
              for _ in range(50)]
    d_embs = [torch.randn(torch.randint(50, 300, (1,)).item(), d, device="cuda", dtype=torch.float16)
              for _ in range(50)]

    print(f"Query lengths: {[q.shape[0] for q in q_embs[:5]]}...")
    print(f"Doc lengths:   {[d.shape[0] for d in d_embs[:5]]}...")

    # Pack into contiguous buffers with cu_seqlens (like FlashAttention varlen API)
    Q_packed, D_packed, cu_q, cu_d = pack_pairs(q_embs, d_embs)
    max_lq = max(q.shape[0] for q in q_embs)
    max_ld = max(d.shape[0] for d in d_embs)

    print(f"Packed Q: {Q_packed.shape}, Packed D: {D_packed.shape}")
    print(f"cu_seqlens_q: {cu_q.shape}, cu_seqlens_d: {cu_d.shape}")

    # Score all pairs in one kernel launch — zero padding waste
    scores = flash_maxsim_varlen(Q_packed, D_packed, cu_q, cu_d, max_lq, max_ld)
    print(f"Scores: {scores.shape} — {scores[:5].tolist()}")


# ============================================================================
# 4. Zero-Copy Reranking: flash_maxsim_rerank / rerank_direct
# ============================================================================

def demo_rerank():
    section("4. Zero-Copy Reranking — flash_maxsim_rerank_direct")
    from flash_maxsim import flash_maxsim_rerank, flash_maxsim_rerank_direct

    Lq, d, B = 32, 128, 100
    Q = torch.randn(Lq, d, device="cuda", dtype=torch.float16)

    # --- 4a. Packed docs (from torch.cat) ---
    doc_lens = [torch.randint(50, 200, (1,)).item() for _ in range(B)]
    docs = [torch.randn(dl, d, device="cuda", dtype=torch.float16) for dl in doc_lens]

    D_packed = torch.cat(docs, dim=0)  # [total_tokens, d]
    cu_seqlens = torch.zeros(B + 1, dtype=torch.int32, device="cuda")
    for i in range(B):
        cu_seqlens[i + 1] = cu_seqlens[i] + doc_lens[i]
    max_ld = max(doc_lens)

    scores_packed = flash_maxsim_rerank(Q, D_packed, cu_seqlens, max_ld)
    print(f"Packed rerank — scores: {scores_packed.shape}")

    # --- 4b. TRUE zero-copy: scattered docs in a batch tensor ---
    # Simulate a model output tensor where docs are at various positions
    # (mixed with query tokens, padding, other requests)
    total_tokens = sum(doc_lens)
    gap = 10  # gaps between docs (simulating other content)
    batch_tensor = torch.randn(total_tokens + B * gap, d, device="cuda", dtype=torch.float16)

    # Place docs at scattered positions
    doc_offsets = torch.zeros(B, dtype=torch.int32, device="cuda")
    doc_lengths = torch.tensor(doc_lens, dtype=torch.int32, device="cuda")
    pos = 0
    for i in range(B):
        doc_offsets[i] = pos
        batch_tensor[pos:pos + doc_lens[i]] = docs[i]
        pos += doc_lens[i] + gap  # gap for other content

    # Score directly from batch tensor — ZERO additional memory
    scores_direct = flash_maxsim_rerank_direct(
        Q, batch_tensor, doc_offsets, doc_lengths, max_ld)
    print(f"Zero-copy rerank — scores: {scores_direct.shape}")

    # Verify: both methods produce identical results
    max_err = (scores_packed - scores_direct).abs().max().item()
    print(f"Packed vs direct max error: {max_err:.2e} (should be ~0)")

    # Memory comparison
    stacked_mem = B * max_ld * d * 2  # torch.stack would allocate this
    print(f"\nMemory for {B} docs:")
    print(f"  torch.stack would allocate: {stacked_mem / 1e6:.1f} MB")
    print(f"  zero-copy allocates:        0 MB")


# ============================================================================
# 5. INT8 Quantized Scoring: flash_maxsim_int8
# ============================================================================

def demo_int8():
    section("5. INT8 Scoring — flash_maxsim_int8")
    from flash_maxsim import flash_maxsim, flash_maxsim_int8, quantize_int8

    Q = torch.randn(32, 128, device="cuda", dtype=torch.float16)
    D = torch.randn(1000, 300, 128, device="cuda", dtype=torch.float16)

    # Quantize documents to INT8 (2x storage compression)
    D_q, scales, mins = quantize_int8(D)
    print(f"Original D: {D.shape}, dtype={D.dtype}, size={D.nelement() * 2 / 1e6:.1f} MB")
    print(f"Quantized:  {D_q.shape}, dtype={D_q.dtype}, size={D_q.nelement() / 1e6:.1f} MB")
    print(f"Compression: {D.nelement() * 2 / D_q.nelement():.1f}x")

    # Score with fused dequantization (dequant happens in SRAM, never written to HBM)
    scores_int8 = flash_maxsim_int8(Q, D_q, scales, mins)
    scores_fp16 = flash_maxsim(Q, D)

    # Compare rankings
    topk_int8 = set(scores_int8.topk(5).indices.tolist())
    topk_fp16 = set(scores_fp16.topk(5).indices.tolist())
    overlap = len(topk_int8 & topk_fp16) / 5
    print(f"\nTop-5 overlap (INT8 vs FP16): {overlap:.0%}")
    print(f"Max score difference: {(scores_int8 - scores_fp16).abs().max().item():.4f}")


# ============================================================================
# 6. Training: flash_maxsim_train
# ============================================================================

def demo_training():
    section("6. Training — flash_maxsim_train")
    from flash_maxsim import flash_maxsim_train

    Lq, d, B, Ld = 32, 128, 50, 100
    Q = torch.randn(Lq, d, device="cuda", dtype=torch.float16, requires_grad=True)
    D = torch.randn(B, Ld, d, device="cuda", dtype=torch.float16, requires_grad=True)

    # Forward + backward
    scores = flash_maxsim_train(Q, D)  # [B]
    loss = scores.sum()
    loss.backward()

    print(f"Scores: {scores.shape}")
    print(f"Q gradient: {Q.grad.shape}, norm={Q.grad.norm().item():.4f}")
    print(f"D gradient: {D.grad.shape}, norm={D.grad.norm().item():.4f}")
    print("Autograd works — use in contrastive/distillation training loops")


# ============================================================================
# 7. Performance Comparison
# ============================================================================

def demo_performance():
    section("7. Performance — Flash vs Naive")
    from flash_maxsim import flash_maxsim

    configs = [
        ("ColBERT",  32, 300, 128, 1000),
        ("Long-doc", 32, 1024, 128, 1000),
        ("ColPali", 512, 1024, 128, 1000),
    ]

    for name, Lq, Ld, d, B in configs:
        Q = torch.randn(Lq, d, device="cuda", dtype=torch.float16)
        D = torch.randn(B, Ld, d, device="cuda", dtype=torch.float16)

        # Warmup
        _ = flash_maxsim(Q, D)
        torch.cuda.synchronize()

        # Flash
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            _ = flash_maxsim(Q, D)
        torch.cuda.synchronize()
        flash_ms = (time.perf_counter() - t0) / 20 * 1000

        # Naive
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            S = torch.einsum('qd,bld->bql', Q.float(), D.float())
            _ = S.max(dim=2).values.sum(dim=1)
        torch.cuda.synchronize()
        naive_ms = (time.perf_counter() - t0) / 20 * 1000

        print(f"{name:>10} (Lq={Lq}, Ld={Ld}): "
              f"naive={naive_ms:.2f}ms, flash={flash_ms:.2f}ms, "
              f"speedup={naive_ms/flash_ms:.1f}x")

    # Memory
    print(f"\nMemory comparison (B=10000, Lq=1024, Ld=1024):")
    print(f"  Naive similarity matrix: {10000 * 1024 * 1024 * 2 / 1e9:.1f} GB")
    print(f"  Flash-MaxSim:            <0.1 MB (never materialized)")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print(f"Flash-MaxSim v0.2.0 Demo")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")

    demos = [
        demo_basic,
        demo_batched,
        demo_varlen,
        demo_rerank,
        demo_int8,
        demo_training,
        demo_performance,
    ]

    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("  Done! All APIs demonstrated.")
    print(f"{'='*60}")
