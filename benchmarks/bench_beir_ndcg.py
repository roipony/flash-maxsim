"""End-to-end nDCG@10 / Recall@10 on BEIR with the ColBERTv2 encoder
for Flash vs FP32 naive reference.

Extends the current §5.6 BEIR result (ArguAna + SciDocs, both small)
with three larger publicly-cited BEIR subsets that have pre-computed
ColBERTv2 embeddings available:

  - HotpotQA   500K docs (22 GB FP16)
  - Quora-Full 523K docs (8.7 GB FP16)
  - NQ         2.68M docs (~116 GB FP16) → OOC streaming only

The first two run both backends (naive FP32 chunked reference vs
Flash FP16) for parity. NQ is far too large for any GPU-resident
naive baseline; we run Flash via streaming (host→device per doc
chunk) and report Flash's nDCG@10 vs the published ColBERTv2
number as the closest proxy for ranking quality.

Output: bench_beir_ndcg_<GPU>.json next to this script.
"""
from __future__ import annotations
import gc, json, math, os, time
from pathlib import Path
import torch

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
EMBED_ROOT = Path('/path/to/data')
SUBSETS = [
    # (name,        path,                              run_naive_ref)
    ('hotpotqa-500K', EMBED_ROOT / 'hotpotqa_500K.pt', True),
    ('quora-523K',    EMBED_ROOT / 'quora_523K.pt',    True),
    ('nq-2.68M',      EMBED_ROOT / 'nq_2.68M.pt',      False),  # OOC, no naive ref
]
DEV = 'cuda'


def load_dataset(p: Path):
    print(f"  loading {p.name}...", flush=True)
    t0 = time.time()
    d = torch.load(p, map_location='cpu', weights_only=False)
    q_list = d['query_embeddings']
    Nq = len(q_list)
    Lq = q_list[0].shape[0]
    dim = q_list[0].shape[1]
    Q = torch.stack(q_list, dim=0)
    doc_chunks = d['doc_embeddings']
    B_total = sum(c.shape[0] for c in doc_chunks)
    meta = d['meta']
    print(f"  loaded in {time.time()-t0:.1f}s — Nq={Nq}, B={B_total}, Lq={Lq}, "
          f"chunks={len(doc_chunks)}, dim={dim}", flush=True)
    return Q, doc_chunks, meta, B_total


def flash_scores_streaming(Q_cpu, doc_chunks, dtype=torch.float16):
    """Stream doc chunks H→D, score with Flash, accumulate scores [Nq,B] on CPU."""
    from flash_maxsim import flash_maxsim_batched
    Nq = Q_cpu.shape[0]
    Q_dev = Q_cpu.to(DEV, dtype=dtype, non_blocking=True)
    chunk_scores = []
    for ci, D_chunk in enumerate(doc_chunks):
        D_dev = D_chunk.to(DEV, dtype=dtype, non_blocking=True)
        s = flash_maxsim_batched(Q_dev, D_dev).to(torch.float32).cpu()
        chunk_scores.append(s)
        del D_dev
        if ci % 200 == 0 and ci > 0:
            torch.cuda.empty_cache()
            print(f"    flash chunk {ci}/{len(doc_chunks)}", flush=True)
    del Q_dev
    torch.cuda.empty_cache()
    return torch.cat(chunk_scores, dim=1)


def naive_fp32_scores_streaming(Q_cpu, doc_chunks, target_tile_gb=8.0):
    """FP32 reference, chunked over docs to bound the materialised
    [Nq, B_chunk, Lq, Ld] tile under target_tile_gb."""
    Nq, Lq, dim = Q_cpu.shape
    Q_dev = Q_cpu.to(DEV, torch.float32)
    chunk_scores = []
    for ci, D_chunk in enumerate(doc_chunks):
        B_j, Ld_j, _ = D_chunk.shape
        bytes_per_cell = Nq * Lq * Ld_j * 4
        sub_b = max(1, min(B_j, int(target_tile_gb * (1024**3) / bytes_per_cell)))
        c_scores = torch.empty(Nq, B_j, dtype=torch.float32)
        for s_b in range(0, B_j, sub_b):
            e_b = min(s_b + sub_b, B_j)
            D_sub = D_chunk[s_b:e_b].to(DEV, torch.float32)
            S = torch.einsum('nqd,bld->nbql', Q_dev, D_sub)
            c_scores[:, s_b:e_b] = S.max(dim=-1).values.sum(dim=-1).cpu()
            del D_sub, S
        chunk_scores.append(c_scores)
        if ci % 200 == 0 and ci > 0:
            torch.cuda.empty_cache()
            print(f"    naive chunk {ci}/{len(doc_chunks)}", flush=True)
    del Q_dev
    torch.cuda.empty_cache()
    return torch.cat(chunk_scores, dim=1)


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


def run_subset(name, path, run_naive):
    if not path.exists():
        return {'name': name, 'error': f'missing: {path}'}
    print(f"\n=== {name} ===", flush=True)
    Q, doc_chunks, meta, B_total = load_dataset(path)
    qrels = meta.get('qrels')
    query_ids = meta.get('query_ids')
    doc_ids = meta.get('passage_ids') or meta.get('doc_ids')

    row = {'name': name, 'Nq': Q.shape[0], 'B': B_total,
           'Lq': Q.shape[1], 'dim': Q.shape[2], 'chunks': len(doc_chunks)}

    # Flash
    t0 = time.time()
    s_flash = flash_scores_streaming(Q, doc_chunks)
    row['flash_wall_s'] = round(time.time() - t0, 1)
    print(f"  flash done in {row['flash_wall_s']}s", flush=True)

    # naive FP32 (skip for NQ-scale)
    s_naive = None
    if run_naive:
        gc.collect(); torch.cuda.empty_cache()
        t0 = time.time()
        s_naive = naive_fp32_scores_streaming(Q, doc_chunks)
        row['naive_wall_s'] = round(time.time() - t0, 1)
        print(f"  naive done in {row['naive_wall_s']}s", flush=True)
        row['parity'] = {'max_abs': float((s_naive - s_flash).abs().max()),
                         'mean_abs': float((s_naive - s_flash).abs().mean())}
    else:
        row['naive_wall_s'] = None
        row['parity'] = None
        row['note'] = 'naive FP32 reference skipped — corpus exceeds GPU memory'

    if qrels and query_ids and doc_ids:
        row['flash_metrics'] = compute_metrics(s_flash, qrels, query_ids, doc_ids)
        fm = row['flash_metrics']
        print(f"  flash: nDCG@10={fm['ndcg@10']:.4f}  Recall@10={fm['recall@10']:.4f}  "
              f"n_eval={fm['n_eval']}/{Q.shape[0]}", flush=True)
        if s_naive is not None:
            row['naive_metrics'] = compute_metrics(s_naive, qrels, query_ids, doc_ids)
            nm = row['naive_metrics']
            print(f"  naive: nDCG@10={nm['ndcg@10']:.4f}  Recall@10={nm['recall@10']:.4f}", flush=True)
    if s_naive is not None:
        del s_naive
    del s_flash, Q, doc_chunks
    gc.collect(); torch.cuda.empty_cache()
    return row


def main():
    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}", flush=True)
    import flash_maxsim
    print(f"flash_maxsim {flash_maxsim.__version__}\n", flush=True)
    rows = []
    for name, path, run_naive in SUBSETS:
        try:
            rows.append(run_subset(name, path, run_naive))
        except Exception as e:
            import traceback; traceback.print_exc()
            rows.append({'name': name, 'error': str(e)})
        gc.collect(); torch.cuda.empty_cache()
    out = {'gpu': gpu, 'flash_maxsim_version': flash_maxsim.__version__,
           'encoder': 'ColBERTv2',
           'note': 'BEIR end-to-end nDCG@10 / Recall@10. Naive FP32 chunked '
                   'reference is the same kernel as tab:beir_ndcg; NQ-2.68M '
                   'is OOC (Flash only, naive ref skipped).',
           'results': rows}
    op = HERE / f'bench_beir_ndcg_{gpu.replace(" ", "_").replace("/", "_")}.json'
    op.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {op}", flush=True)


if __name__ == '__main__':
    main()
