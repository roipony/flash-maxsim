"""End-to-end nDCG/Recall on REAL-MM-RAG with ColPali and GVE encoders.

Uses qrels from the GVE .pt files (meta.qrels = {qid: {doc_id: rel}},
one positive per query, with 4 rephrases per gold page). For ColPali
we test the row-alignment assumption (queries 0..N-1 of ColPali ==
queries 0..N-1 of GVE).
"""
from __future__ import annotations
import gc, json, math, os, time
from pathlib import Path
import torch

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
EMBED_ROOT = Path('/path/to/data')
SUBSETS = [('finreport', '2.7K'), ('finslides', '2.3K'),
           ('techreport', '1.7K'), ('techslides', '2K')]
DEV = 'cuda'


def naive_scores_fp32_chunked(Q, D, target_tile_gb=8.0):
    Nq, Lq, _ = Q.shape; B, Ld, _ = D.shape
    cells_budget = int(target_tile_gb * (1024**3) / (Lq * Ld * 4))
    if Nq * 64 <= cells_budget:
        chunk_q, chunk_b = Nq, max(1, min(B, cells_budget // Nq))
    else:
        chunk_q = max(1, int(cells_budget ** 0.5))
        chunk_b = max(1, cells_budget // chunk_q)
    scores = torch.empty(Nq, B, device=Q.device, dtype=torch.float32)
    Q32 = Q.to(torch.float32)
    for qs in range(0, Nq, chunk_q):
        qe = min(qs + chunk_q, Nq); Qc = Q32[qs:qe]
        for ds in range(0, B, chunk_b):
            de = min(ds + chunk_b, B); Dc = D[ds:de].to(torch.float32)
            S = torch.einsum('nqd,bld->nbql', Qc, Dc)
            scores[qs:qe, ds:de] = S.max(dim=-1).values.sum(dim=-1)
            del Dc, S
    return scores


def flash_scores(Q, D):
    from flash_maxsim import flash_maxsim_batched
    return flash_maxsim_batched(Q.to(torch.float16), D.to(torch.float16)).to(torch.float32)


def compute_metrics(scores, qrels, query_ids, doc_ids, ks=(5, 10)):
    Nq, B = scores.shape
    doc_id_to_col = {str(d): i for i, d in enumerate(doc_ids)}
    max_k = max(ks)
    topk_idx = scores.topk(max_k, dim=1).indices.cpu().tolist()
    out = {f'ndcg@{k}': 0.0 for k in ks}
    out.update({f'recall@{k}': 0.0 for k in ks})
    n_eval = 0
    for i in range(min(Nq, len(query_ids))):
        qid = str(query_ids[i])
        rel_map = qrels.get(qid)
        if not rel_map: continue
        rel_cols = {doc_id_to_col[str(d)]: r for d, r in rel_map.items()
                    if str(d) in doc_id_to_col and r > 0}
        if not rel_cols: continue
        n_eval += 1
        topk = topk_idx[i]
        for k in ks:
            hits_in_topk = [c for c in topk[:k] if c in rel_cols]
            out[f'recall@{k}'] += float(len(hits_in_topk) > 0)
            dcg = sum(rel_cols[c] / math.log2(rank + 2)
                      for rank, c in enumerate(topk[:k]) if c in rel_cols)
            n_rel = min(k, sum(1 for v in rel_cols.values() if v > 0))
            idcg = sum(1.0 / math.log2(r + 2) for r in range(n_rel))
            out[f'ndcg@{k}'] += (dcg / idcg) if idcg > 0 else 0.0
    if n_eval > 0:
        for k in ks:
            out[f'ndcg@{k}']   /= n_eval
            out[f'recall@{k}'] /= n_eval
    out['n_eval'] = n_eval
    return out


def run_cell(encoder, dataset, n_str, gve_cache):
    p = EMBED_ROOT / encoder / f'{dataset}_{n_str}.pt'
    if not p.exists():
        return {'error': f'missing: {p}'}
    print(f"\n=== {encoder} / {dataset} ===", flush=True)
    d = torch.load(p, map_location='cpu', weights_only=False)
    Q = d['query_embeddings'].to(DEV); D = d['doc_embeddings'].to(DEV)
    q_lens = d.get('query_lengths')
    Nq, Lq, _ = Q.shape; B, Ld, _ = D.shape

    if q_lens is not None:
        q_lens = q_lens.to(DEV).long()
        pos = torch.arange(Lq, device=DEV).unsqueeze(0)
        mask = (pos < q_lens.unsqueeze(1)).to(Q.dtype).unsqueeze(-1)
        Q = Q * mask

    qrels = query_ids = doc_ids = None
    qrel_src = 'none'
    if encoder == 'gve':
        m = d['meta']
        qrels, query_ids, doc_ids = m['qrels'], m['query_ids'], m['doc_ids']
        gve_cache[dataset] = (qrels, query_ids, doc_ids)
        qrel_src = 'self'
    elif dataset in gve_cache:
        qrels, query_ids, doc_ids = gve_cache[dataset]
        if len(doc_ids) != B:
            print(f"  WARN: ColPali B={B} but GVE meta has {len(doc_ids)} docs", flush=True)
        qrel_src = 'gve-aligned'

    t0 = time.time(); s_naive = naive_scores_fp32_chunked(Q, D)
    torch.cuda.synchronize(); t_naive = time.time() - t0
    gc.collect(); torch.cuda.empty_cache()
    t0 = time.time(); s_flash = flash_scores(Q, D)
    torch.cuda.synchronize(); t_flash = time.time() - t0

    parity = {'max_abs': float((s_naive - s_flash).abs().max()),
              'mean_abs': float((s_naive - s_flash).abs().mean())}
    row = {'encoder': encoder, 'dataset': dataset, 'Nq': Nq, 'B': B,
           'Lq': Lq, 'Ld': Ld, 't_naive_s': round(t_naive, 2),
           't_flash_s': round(t_flash, 2), 'parity': parity,
           'qrel_source': qrel_src}

    if qrels:
        qids = query_ids[:Nq] if encoder == 'colpali' else query_ids
        row['naive_metrics'] = compute_metrics(s_naive, qrels, qids, doc_ids)
        row['flash_metrics'] = compute_metrics(s_flash, qrels, qids, doc_ids)
        nm, fm = row['naive_metrics'], row['flash_metrics']
        print(f"  naive: nDCG@5={nm['ndcg@5']:.4f}  nDCG@10={nm['ndcg@10']:.4f}  "
              f"Recall@5={nm['recall@5']:.4f}  Recall@10={nm['recall@10']:.4f}  "
              f"(n_eval={nm['n_eval']}/{Nq})", flush=True)
        print(f"  flash: nDCG@5={fm['ndcg@5']:.4f}  nDCG@10={fm['ndcg@10']:.4f}  "
              f"Recall@5={fm['recall@5']:.4f}  Recall@10={fm['recall@10']:.4f}  "
              f"(n_eval={fm['n_eval']}/{Nq})", flush=True)
    print(f"  parity: max|Δ|={parity['max_abs']:.3e}  mean|Δ|={parity['mean_abs']:.3e}", flush=True)
    return row


def main():
    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}", flush=True)
    import flash_maxsim
    print(f"flash_maxsim {flash_maxsim.__version__}\n", flush=True)
    rows = []; gve_cache = {}
    for enc in ['gve', 'colpali']:
        for ds, n in SUBSETS:
            try:
                rows.append(run_cell(enc, ds, n, gve_cache))
            except Exception as e:
                import traceback; traceback.print_exc()
                rows.append({'encoder': enc, 'dataset': ds, 'error': str(e)})
            gc.collect(); torch.cuda.empty_cache()
    out = {'gpu': gpu, 'flash_maxsim_version': flash_maxsim.__version__,
           'note': 'GVE qrels reused for ColPali via row-i query alignment',
           'results': rows}
    op = HERE / f'bench_real_mm_rag_ndcg_{gpu.replace(" ", "_").replace("/", "_")}.json'
    op.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {op}", flush=True)


if __name__ == '__main__':
    main()
