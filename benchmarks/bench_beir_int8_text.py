"""INT8 end-to-end retrieval quality on TEXT BEIR (ArguAna + SciDocs).

Reviewer Q7: the long-doc INT8 top-20 overlap is 95% at the operator
level; the paper bounds the end-to-end impact only on vision
(REAL-MM-RAG). This script mirrors tab:beir_ndcg's protocol exactly
(same loader, same FP32 vectorized baseline) and adds the INT8x8 path,
so the textual nDCG@10 / Recall@10 delta of INT8 is measured directly.

Scorers per query:
  baseline : FP32 einsum + mask + max + sum (the tab:beir_ndcg reference)
  flash    : flash_maxsim_batched, FP16 in / FP32 accum (exact path)
  int8     : flash_maxsim_int8x8, D pre-quantized at index time,
             Q quantized per query (the serving pattern)

Output: bench_beir_int8_text_<GPU>.json
"""
import json, os, sys, time, math
import numpy as np
import torch
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p) != HERE]
sys.path.insert(0, "/path/to/embedding_registry")
from utils.dataset_registry import load_bundle, load_prepared  # noqa: E402
sys.path.insert(0, ".")
from flash_maxsim import (flash_maxsim_batched, flash_maxsim_int8x8,  # noqa: E402
                          quantize_int8_symmetric, quantize_query_int8)

torch.set_float32_matmul_precision('high')
DATASETS = os.environ.get("BEIR_DATASETS", "arguana,scidocs").split(",")
ENCODER = "lightonai_colbertv2"
KS = [5, 10, 20]


def _first(d, *names):
    for n in names:
        if n in d:
            return d[n]
    raise KeyError(names)


def _to_list(emb, lengths=None):
    if isinstance(emb, torch.Tensor) and emb.dim() == 3:
        L = lengths.tolist() if isinstance(lengths, torch.Tensor) else (lengths or [emb.shape[1]] * emb.shape[0])
        return [np.ascontiguousarray(emb[i, :int(L[i])].float().numpy()) for i in range(emb.shape[0])]
    return [np.ascontiguousarray(t.float().numpy() if isinstance(t, torch.Tensor) else t, dtype=np.float32) for t in emb]


def _dcg(rels, k):
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels[:k]) if r > 0)


def _ndcg(rels, q, k):
    idcg = _dcg(sorted(q.values(), reverse=True), k)
    return _dcg(rels, k) / idcg if idcg > 0 else 0.0


def _recall(rels, q, k):
    n = sum(1 for r in q.values() if r > 0)
    return sum(1 for r in rels[:k] if r > 0) / n if n else 0.0


def build_corpus(docs):
    B = len(docs); d = docs[0].shape[1]; Ld_max = max(x.shape[0] for x in docs)
    D = torch.zeros(B, Ld_max, d, dtype=torch.float32, device='cuda')
    lens = torch.zeros(B, dtype=torch.int32, device='cuda')
    for i, doc in enumerate(docs):
        L = doc.shape[0]
        D[i, :L] = torch.from_numpy(doc).cuda()
        lens[i] = L
    D = D / D.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return D.to(torch.float16), D, lens


def eval_dataset(name):
    print(f"\n========== {name} ==========", flush=True)
    b = load_bundle(name, encoder=ENCODER)
    doc_embs, q_embs = load_prepared(name, encoder=ENCODER)
    docs = _to_list(doc_embs); queries = _to_list(q_embs)
    doc_ids = [str(x) for x in _first(b, "passage_ids", "doc_ids", "corpus_ids")]
    qids = [str(x) for x in _first(b, "query_ids", "qids")]
    qrels_raw = _first(b, "qrels", "qrels_dict")
    if isinstance(qrels_raw, dict):
        qrels = {str(q): {str(dd): int(s) for dd, s in v.items()} for q, v in qrels_raw.items()}
    else:
        qrels = defaultdict(dict)
        for q, x, s in qrels_raw:
            qrels[str(q)][str(x)] = int(s)

    D16, D32, lens = build_corpus(docs)
    print(f"  N={len(docs)} Q={len(queries)} Ld_max={D16.shape[1]}", flush=True)

    # Index-time INT8 quantization of D (once)
    t0 = time.time()
    D_int8, d_scales = quantize_int8_symmetric(D16)
    print(f"  D quantized to INT8 in {time.time()-t0:.2f}s "
          f"({D_int8.nbytes/1e9:.2f} GB vs {D16.nbytes/1e9:.2f} GB FP16)", flush=True)

    metrics = {k: {s: defaultdict(float) for s in ("baseline", "flash", "int8")}
               for k in KS}
    overlap = {k: {"flash": 0.0, "int8": 0.0} for k in KS}
    n = 0
    Kmax = max(KS)
    for qi, qid in enumerate(qids):
        if qid not in qrels or not any(r > 0 for r in qrels[qid].values()):
            continue
        Qf = torch.from_numpy(queries[qi]).cuda().float()
        Qf = Qf / Qf.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        Q16 = Qf.to(torch.float16)

        sim = torch.einsum('id,bjd->bij', Qf, D32)
        mask = (torch.arange(D32.shape[1], device='cuda')[None, :] < lens[:, None])
        sim = sim.masked_fill(~mask[:, None, :], float('-inf'))
        s_b = sim.max(dim=-1).values.sum(dim=-1).cpu().numpy()
        del sim

        s_f = flash_maxsim_batched(Q16.unsqueeze(0), D16, doc_lengths=lens
                                   ).cpu().numpy().squeeze(0)

        Q_i8, q_sc = quantize_query_int8(Q16.contiguous())
        s_q = flash_maxsim_int8x8(Q16.contiguous(), D_int8, d_scales,
                                  Q_int8=Q_i8, q_scales=q_sc,
                                  doc_lengths=lens).float().cpu().numpy()

        rb = np.argsort(-s_b); rf = np.argsort(-s_f); rq = np.argsort(-s_q)
        for K in KS:
            for tag, rr in (("baseline", rb), ("flash", rf), ("int8", rq)):
                top = [doc_ids[i] for i in rr[:K]]
                rels = [qrels[qid].get(did, 0) for did in top]
                metrics[K][tag]["ndcg"] += _ndcg(rels, qrels[qid], K)
                metrics[K][tag]["recall"] += _recall(rels, qrels[qid], K)
            overlap[K]["flash"] += len(set(rb[:K].tolist()) & set(rf[:K].tolist())) / K
            overlap[K]["int8"] += len(set(rb[:K].tolist()) & set(rq[:K].tolist())) / K
        n += 1
        if n % 200 == 0:
            print(f"  q{n}...", flush=True)

    row = {"dataset": name, "n_eval": n, "corpus": len(docs), "per_K": {}}
    print(f"  evaluated {n} queries", flush=True)
    for K in KS:
        r = {}
        for tag in ("baseline", "flash", "int8"):
            r[f"ndcg_{tag}"] = round(metrics[K][tag]["ndcg"] / n, 4)
            r[f"recall_{tag}"] = round(metrics[K][tag]["recall"] / n, 4)
        r["topk_overlap_flash_pct"] = round(100 * overlap[K]["flash"] / n, 2)
        r["topk_overlap_int8_pct"] = round(100 * overlap[K]["int8"] / n, 2)
        row["per_K"][K] = r
        print(f"  K={K}: nDCG base={r['ndcg_baseline']:.4f} flash={r['ndcg_flash']:.4f} "
              f"int8={r['ndcg_int8']:.4f} (Δint8={r['ndcg_int8']-r['ndcg_baseline']:+.4f})  "
              f"int8 top-{K} overlap={r['topk_overlap_int8_pct']:.1f}%", flush=True)
    del D16, D32, D_int8
    torch.cuda.empty_cache()
    return row


results = [eval_dataset(ds) for ds in DATASETS]
gpu = torch.cuda.get_device_name()
p = os.path.join(HERE, f"bench_beir_int8_text_{gpu.replace(' ', '_').replace('/', '_')}.json")
json.dump({"gpu": gpu, "encoder": ENCODER, "results": results}, open(p, "w"), indent=2)
print(f"\nwrote {p}")
