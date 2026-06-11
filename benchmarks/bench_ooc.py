"""Out-of-core corpus scaling for Flash-MaxSim (the Flash-KMeans analogue).

Scores one ColPali query (Lq=Ld=1024, d=128) against a document corpus that is
held in CPU memory and STREAMED to the GPU in blocks. Flash scores each block
with peak GPU memory ~= one block, so it scales to corpora far larger than VRAM,
whereas the naive materialized path OOMs at B~20K and even on-GPU flash is bounded
by the corpus that fits in VRAM (~B=300K at 80GB). Reports wall-clock latency and
document throughput vs corpus size, including corpus sizes that exceed GPU memory.
"""
import json, os, sys, time, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from flash_maxsim import flash_maxsim

d, Lq, Ld = 128, 1024, 1024
gpu = torch.cuda.get_device_name()
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
Q = F.normalize(torch.randn(Lq, d, device='cuda', dtype=torch.float16), dim=-1)
BLOCK = 20000  # docs per streamed block (~5 GB GPU on visual shape)
tile = F.normalize(torch.randn(BLOCK, Ld, d, dtype=torch.float16), dim=-1)  # reused content


def stream_score(B):
    Dcpu = torch.empty(B, Ld, d, dtype=torch.float16)        # pageable CPU corpus
    for i in range(0, B, BLOCK):
        n = min(BLOCK, B - i)
        Dcpu[i:i + n] = tile[:n]
    corpus_gb = Dcpu.numel() * 2 / 1e9
    torch.cuda.synchronize(); t0 = time.time()
    scores = torch.empty(B, device='cuda')
    for i in range(0, B, BLOCK):
        n = min(BLOCK, B - i)
        Dblk = Dcpu[i:i + n].to('cuda', non_blocking=True)
        scores[i:i + n] = flash_maxsim(Q, Dblk)
        del Dblk
    torch.cuda.synchronize(); dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    del Dcpu; return dt, corpus_gb, peak


print(f"GPU: {gpu} ({vram_gb:.0f} GB VRAM)")
out = {"gpu": gpu, "vram_gb": round(vram_gb, 1), "block": BLOCK, "rows": []}
for B in [100_000, 250_000, 500_000, 750_000]:
    try:
        torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
        dt, corpus_gb, peak = stream_score(B)
        rec = {"B": B, "corpus_gb": round(corpus_gb, 1), "latency_s": round(dt, 2),
               "docs_per_s": round(B / dt), "gpu_peak_gb": round(peak, 2),
               "exceeds_vram": corpus_gb > vram_gb}
        out["rows"].append(rec); print("  ", rec)
    except Exception as e:
        print(f"  B={B} FAIL: {e.__class__.__name__}: {e}")
        out["rows"].append({"B": B, "status": f"FAIL {e.__class__.__name__}"})
        torch.cuda.empty_cache()

p = os.path.join(HERE, f"bench_ooc_{gpu.replace(' ', '_').replace('/', '_')}.json")
json.dump(out, open(p, "w"), indent=2); print("wrote", p)
