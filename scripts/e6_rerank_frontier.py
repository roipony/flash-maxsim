"""E6 — rerank-stage candidate-capacity frontier at ColPali shapes (vNrM Q3).

The pipeline experiment (E5, HotpotQA text shapes) isolates LATENCY; this one
measures the MEMORY-enabled operating point: the maximum candidate set K each
exact MaxSim implementation can rerank on one A100-80GB at visual (ColPali)
shapes, plus latency where feasible.

Regimes (candidates are ColPali pages: Ld=1030, d=128, fp16 resident):
  text-1   : one text query  [1, 32, 128]      (single-stream serving)
  text-32  : 32 concurrent text queries        (batched serving)
  page-1   : one page-as-query [1, 1030, 128]  (similar-page retrieval)

Methods (all exact):
  unchunked : eager einsum, full [Nq,K,Lq,Ld] block in fp16 (paper's
              unchunked FP16 eager baseline)
  chunked   : deployed baseline — doc tiles, running max-sum, block capped
  flash     : Flash-MaxSim (no block at all)

K sweep: 16K..200K (doubling + frontier refinement). Per cell: OK/OOM,
peak GB, median latency of 3 runs.

Output: analysis/e1_results/e6_rerank_frontier.json
"""
from __future__ import annotations
import gc, json, os, time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
OUT = HERE / "e1_results" / "e6_rerank_frontier.json"
DEV = "cuda"
Ld, DM = 1030, 128
KS = [16_000, 32_000, 64_000, 96_000, 128_000, 160_000, 200_000]
CHUNK_TILE = 4096          # deployed chunked baseline tile (docs per tile)
REGIMES = [("text-1", 1, 32), ("text-32", 32, 32), ("page-1", 1, 1030)]


def unchunked(Q, D):
    sim = torch.einsum("nqd,bld->nbql", Q, D)           # fp16 block
    return sim.max(dim=-1).values.sum(dim=-1).float()


def chunked(Q, D):
    Nq = Q.shape[0]
    out = torch.empty(Nq, D.shape[0], device=DEV, dtype=torch.float32)
    for s in range(0, D.shape[0], CHUNK_TILE):
        e = min(s + CHUNK_TILE, D.shape[0])
        sim = torch.einsum("nqd,bld->nbql", Q, D[s:e])
        out[:, s:e] = sim.max(dim=-1).values.sum(dim=-1).float()
        del sim
    return out


def flash(Q, D):
    from flash_maxsim import flash_maxsim_batched
    return flash_maxsim_batched(Q, D)


def make_docs(K):
    # build in slabs to avoid a single giant randn workspace
    D = torch.empty(K, Ld, DM, device=DEV, dtype=torch.float16)
    slab = 20_000
    for s in range(0, K, slab):
        e = min(s + slab, K)
        D[s:e] = F.normalize(
            torch.randn(e - s, Ld, DM, device=DEV, dtype=torch.float16), dim=-1)
    return D


def main():
    res = {"gpu": torch.cuda.get_device_name(0), "Ld": Ld, "d": DM,
           "chunk_tile": CHUNK_TILE, "rows": []}
    from flash_maxsim import flash_maxsim_batched  # noqa: F401 (warm import)

    for reg_name, Nq, Lq in REGIMES:
        Q = F.normalize(torch.randn(Nq, Lq, DM, device=DEV,
                                    dtype=torch.float16), dim=-1)
        dead = set()
        for K in KS:
            gc.collect(); torch.cuda.empty_cache()
            try:
                D = make_docs(K)
            except torch.cuda.OutOfMemoryError:
                res["rows"].append({"regime": reg_name, "K": K,
                                    "note": "embeddings alone exceed HBM"})
                print(json.dumps(res["rows"][-1]), flush=True)
                gc.collect(); torch.cuda.empty_cache()
                break
            emb_gb = D.numel() * D.element_size() / 1e9
            row = {"regime": reg_name, "K": K, "emb_gb": round(emb_gb, 1)}
            for name, fn in (("unchunked", unchunked), ("chunked", chunked),
                             ("flash", flash)):
                if name in dead:
                    row[name] = {"ok": False, "skipped": "OOM at smaller K"}
                    continue
                try:
                    fn(Q, D)                      # warmup / autotune
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    ts = []
                    for _ in range(3):
                        torch.cuda.synchronize(); t0 = time.time()
                        fn(Q, D)
                        torch.cuda.synchronize(); ts.append(time.time() - t0)
                    row[name] = {
                        "ok": True,
                        "ms": round(sorted(ts)[1] * 1e3, 1),
                        "peak_gb": round(torch.cuda.max_memory_allocated()/1e9, 1),
                    }
                except torch.cuda.OutOfMemoryError:
                    row[name] = {"ok": False, "oom": True}
                    dead.add(name)
                    gc.collect(); torch.cuda.empty_cache()
            res["rows"].append(row)
            print(json.dumps(row), flush=True)
            del D
            gc.collect(); torch.cuda.empty_cache()
        del Q
        gc.collect(); torch.cuda.empty_cache()

    OUT.write_text(json.dumps(res, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
