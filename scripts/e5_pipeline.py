"""E5 — end-to-end two-stage retrieval pipeline (vNrM Q3).

Complete pipeline on HotpotQA-500K with real ColBERTv2 embeddings + qrels:

  query encoding -> stage-1 candidate generation (exact single-vector
  retrieval: normalized mean-pooled ColBERT vectors, GPU matmul over the
  full 500K corpus) -> gather top-K candidates -> stage-2 late-interaction
  rerank (Flash-MaxSim vs chunked-FP16-eager) -> final ranking.

An ANN/PLAID index would only make stage 1 FASTER, which increases the
rerank share of the budget; the exact first stage is the conservative
choice and removes index-quality as a confound.

Per candidate depth K we report: per-stage latency, end-to-end ms/query
and QPS for both rerankers, and retrieval quality (nDCG@10 / recall@10)
of stage-1-only vs reranked — i.e. what deeper reranking buys and what
it costs.

Output: analysis/e1_results/e5_pipeline.json
"""
from __future__ import annotations
import gc, json, math, os, time
from pathlib import Path

import torch

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
OUT = HERE.parent / "data" / "e1_results" / "e5_pipeline.json"
PT = Path(os.environ.get('E5_EMBEDDINGS', 'hotpotqa_500K.pt'))  # pre-encoded ColBERTv2 HotpotQA-500K embeddings + qrels (see docstring)
DEV = "cuda"
KS = [1000, 5000, 10000, 50000, 100000]
N_TIME_QUERIES = 32          # timing subset (quality uses all queries)


def chunked_fp16_scores(Q, D, tile=4096):
    """Deployed chunked-eager baseline: fp16 einsum over doc tiles."""
    Nq = Q.shape[0]
    out = torch.empty(Nq, D.shape[0], device=Q.device, dtype=torch.float32)
    for s in range(0, D.shape[0], tile):
        e = min(s + tile, D.shape[0])
        sim = torch.einsum('nqd,bld->nbql', Q, D[s:e])
        out[:, s:e] = sim.max(dim=-1).values.sum(dim=-1).float()
        del sim
    return out


def ndcg_recall(rank_cols, qid, qrels, pid_of_col, k=10):
    rel = qrels.get(qid, {})
    gains = [rel.get(pid_of_col[c], 0) for c in rank_cols[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    n_rel = min(len([r for r in rel.values() if r > 0]), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_rel))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    rec = float(any(g > 0 for g in gains))
    return ndcg, rec


def main():
    from flash_maxsim import flash_maxsim_batched

    print("loading corpus ...", flush=True)
    t0 = time.time()
    d = torch.load(PT, map_location='cpu', weights_only=False)
    meta = d['meta']
    qrels = meta['qrels']
    pids = [str(p) for p in meta['passage_ids']]
    qids = [str(q) for q in meta['query_ids']]
    Q = torch.stack(d['query_embeddings']).to(DEV, torch.float16)   # [Nq,Lq,dm]
    chunks = d['doc_embeddings']
    max_ld = max(c.shape[1] for c in chunks)
    chunks = [torch.nn.functional.pad(c, (0, 0, 0, max_ld - c.shape[1]))
              if c.shape[1] < max_ld else c for c in chunks]
    D = torch.cat(chunks, dim=0).to(DEV, torch.float16)
    del chunks
    Nq, Lq, dm = Q.shape
    B, Ld, _ = D.shape
    print(f"loaded {time.time()-t0:.0f}s  Q{tuple(Q.shape)} D{tuple(D.shape)}",
          flush=True)

    res = {"gpu": torch.cuda.get_device_name(0), "corpus": "hotpotqa-500K",
           "model": meta["model"], "Nq": Nq, "B": B, "Lq": Lq, "Ld": Ld}

    # ---- query encoding cost (serving-style, batch of 1) ----------------
    try:
        from datasets import load_dataset
        qtext = {r["_id"]: r["text"] for r in
                 load_dataset("BeIR/hotpotqa", "queries", split="queries")}
        texts = [qtext[q] for q in qids[:64] if q in qtext]
        from pylate import models
        enc = models.ColBERT(model_name_or_path="colbert-ir/colbertv2.0")
        enc.eval().to(DEV)
        enc.encode(texts[:4], is_query=True, show_progress_bar=False)  # warm
        torch.cuda.synchronize(); t0 = time.time()
        for t in texts[:32]:
            enc.encode([t], is_query=True, show_progress_bar=False)
        torch.cuda.synchronize()
        res["encode_ms_per_query_b1"] = round((time.time() - t0) / 32 * 1e3, 2)
        del enc; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001
        res["encode_ms_per_query_b1"] = f"unavailable: {type(e).__name__}: {e}"
    print("encode:", res["encode_ms_per_query_b1"], flush=True)

    # ---- stage 1: exact single-vector retrieval over 500K ---------------
    dvec = torch.nn.functional.normalize(D.float().mean(dim=1), dim=-1).half()
    qvec = torch.nn.functional.normalize(Q.float().mean(dim=1), dim=-1).half()
    torch.cuda.synchronize(); t0 = time.time()
    s1 = (qvec @ dvec.T).float()                                    # [Nq, B]
    torch.cuda.synchronize()
    res["stage1_ms_per_query"] = round((time.time() - t0) / Nq * 1e3, 3)
    max_k = max(KS)
    top_idx = s1.topk(max_k, dim=1).indices                         # [Nq,maxK]
    print("stage1 done", flush=True)

    # warm up flash at this shape family
    flash_maxsim_batched(Q[:1], D[:1024])

    rows = []
    for K in KS:
        row = {"K": K}
        # --- timing on a subset, per query -------------------------------
        for name, scorer in (("flash", lambda q, c: flash_maxsim_batched(q, c)),
                             ("chunked", chunked_fp16_scores)):
            gather_s = score_s = 0.0
            ok = True
            try:
                # per-shape warmup: autotune + allocator, untimed
                for _ in range(3):
                    cand = D.index_select(0, top_idx[0, :K])
                    scorer(Q[0:1], cand)
                    del cand
                torch.cuda.synchronize()
                for qi in range(N_TIME_QUERIES):
                    torch.cuda.synchronize(); t0 = time.time()
                    cand = D.index_select(0, top_idx[qi, :K])       # gather
                    torch.cuda.synchronize(); t1 = time.time()
                    scorer(Q[qi:qi + 1], cand)
                    torch.cuda.synchronize(); t2 = time.time()
                    gather_s += t1 - t0; score_s += t2 - t1
                    del cand
            except torch.cuda.OutOfMemoryError:
                ok = False
                gc.collect(); torch.cuda.empty_cache()
            if ok:
                row[f"gather_ms"] = round(gather_s / N_TIME_QUERIES * 1e3, 2)
                row[f"rerank_ms_{name}"] = round(score_s / N_TIME_QUERIES * 1e3, 2)
            else:
                row[f"rerank_ms_{name}"] = "OOM"
        # --- quality on ALL queries via flash ----------------------------
        nd1 = nd2 = rc1 = rc2 = 0.0
        for qi in range(Nq):
            cols = top_idx[qi, :K]
            s2 = flash_maxsim_batched(Q[qi:qi + 1], D.index_select(0, cols))
            order2 = cols[s2[0].float().argsort(descending=True)].tolist()
            order1 = cols[:10].tolist()
            a, b = ndcg_recall(order1, qids[qi], qrels, pids)
            c, e = ndcg_recall(order2, qids[qi], qrels, pids)
            nd1 += a; rc1 += b; nd2 += c; rc2 += e
        row.update({
            "ndcg10_stage1": round(nd1 / Nq, 4),
            "ndcg10_reranked": round(nd2 / Nq, 4),
            "recall10_stage1": round(rc1 / Nq, 4),
            "recall10_reranked": round(rc2 / Nq, 4),
        })
        # --- end-to-end ---------------------------------------------------
        enc_ms = res["encode_ms_per_query_b1"]
        enc_ms = enc_ms if isinstance(enc_ms, (int, float)) else 0.0
        for name in ("flash", "chunked"):
            r = row.get(f"rerank_ms_{name}")
            if isinstance(r, (int, float)):
                tot = enc_ms + res["stage1_ms_per_query"] + row["gather_ms"] + r
                row[f"total_ms_{name}"] = round(tot, 2)
                row[f"qps_{name}"] = round(1e3 / tot, 1)
        rows.append(row)
        print(json.dumps(row), flush=True)

    res["rows"] = rows
    OUT.write_text(json.dumps(res, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
