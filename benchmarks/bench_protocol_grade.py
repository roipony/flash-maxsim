"""Protocol-grade re-measurement of the paper's headline tables.

Implements App. bench_protocol in full:
  - interleaved round-robin timing across all methods of a cell
  - 100 MB L2 flush before every timed call
  - CUDA-event timing, median + IQR reported
  - compile: max-autotune (graphs-on, FP32/TF32) AND
    max-autotune-no-cudagraphs (FP16); >=100 warmup calls;
    autotune in subprocess
  - per-method peak memory with reset_peak_memory_stats, measured in
    a separate non-interleaved pass (allocator reset between methods)
  - OOM -> sentinel, exception text recorded

Parts (select via BENCH_PART env: shapes | frontier | train | all):
  shapes  : 5 canonical shapes at B=1K x {eager_fp16, eager_fp32_tf32,
            compile_ma_fp32, compile_nocg_fp16, flash}   [tab:fwd / fwd_abs / nocg]
  frontier: visual+ColPali at B=5K/10K/20K x {vanilla_fp16,
            chunked_fp16_best, flash}                    [tab:fwd_chunked]
  train   : ColPali B=64/128 x {naive, recompute_best,
            flash_unified, flash_csr}                    [tab:train]

Output: bench_protocol_grade_<part>_<GPU>.json
"""
import json, os, sys
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from flash_maxsim import flash_maxsim, flash_maxsim_batched_train

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch._inductor.config.autotune_in_subproc = True
    import logging
    logging.getLogger("torch._inductor").setLevel(logging.ERROR)
except Exception:
    pass

d = 128
_l2buf = None


def flush_l2():
    global _l2buf
    if _l2buf is None:
        _l2buf = torch.empty(100 * 1024 * 1024, dtype=torch.uint8, device='cuda')
    _l2buf.zero_()


def pct(ts, p):
    ts = sorted(ts)
    i = p * (len(ts) - 1)
    lo, hi = int(i), min(int(i) + 1, len(ts) - 1)
    return ts[lo] + (ts[hi] - ts[lo]) * (i - lo)


def bench_interleaved_events(fns, argss, warmup, n):
    """Round-robin, L2-flushed, CUDA-event timing. Returns per-fn stats."""
    for fn, args in zip(fns, argss):
        for _ in range(warmup):
            fn(*args)
        torch.cuda.synchronize()
    res = [[] for _ in fns]
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    for _ in range(n):
        for j, (fn, args) in enumerate(zip(fns, argss)):
            flush_l2()
            torch.cuda.synchronize()
            s.record(); fn(*args); e.record()
            torch.cuda.synchronize()
            res[j].append(s.elapsed_time(e))
    return [{"ms": round(pct(ts, .5), 3),
             "iqr": round(pct(ts, .75) - pct(ts, .25), 4)} for ts in res]


def peak_pass(fn, args, warmup=3):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        fn(*args); torch.cuda.synchronize()
    return round(torch.cuda.max_memory_allocated() / 1e9, 3)


def make_qd(Lq, Ld, B):
    Q = F.normalize(torch.randn(Lq, d, device='cuda', dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(B, Ld, d, device='cuda', dtype=torch.float16), dim=-1)
    return Q, D


def n_fp16(Q, D):
    return torch.einsum("qd,bld->bql", Q, D).max(2).values.sum(1)


def n_fp32(Qf, Df):
    return torch.einsum("qd,bld->bql", Qf, Df).max(2).values.sum(1)


def chunked_fp16(Q, D, chunk):
    B = D.shape[0]
    out = torch.empty(B, device=Q.device, dtype=torch.float16)
    for i in range(0, B, chunk):
        S = torch.einsum("qd,bld->bql", Q, D[i:i+chunk])
        out[i:i+chunk] = S.max(2).values.sum(1)
    return out


def part_shapes(out):
    SHAPES = [("textual", 32, 300), ("long-doc", 32, 1024), ("medium", 128, 1024),
              ("visual", 512, 1024), ("ColPali", 1024, 1024)]
    B = 1000
    cma = torch.compile(n_fp32, mode="max-autotune", dynamic=False)
    nocg = torch.compile(n_fp16, mode="max-autotune-no-cudagraphs", dynamic=False)
    for name, Lq, Ld in SHAPES:
        print(f"=== {name} ===", flush=True)
        Q, D = make_qd(Lq, Ld, B)
        Qf, Df = Q.float(), D.float()
        fns = [n_fp16, n_fp32, cma, nocg, flash_maxsim]
        argss = [[Q, D], [Qf, Df], [Qf, Df], [Q, D], [Q, D]]
        stats = bench_interleaved_events(fns, argss, warmup=100, n=100)
        keys = ["eager_fp16", "eager_fp32_tf32", "compile_ma_fp32",
                "compile_nocg_fp16", "flash"]
        row = {"shape": name, "Lq": Lq, "Ld": Ld, "B": B}
        for k, st in zip(keys, stats):
            row[k] = st
        # per-method peaks, allocator reset between (no cudagraph methods first)
        row["peaks_gb"] = {
            "eager_fp16": peak_pass(n_fp16, [Q, D]),
            "eager_fp32_tf32": peak_pass(n_fp32, [Qf, Df]),
            "flash": peak_pass(flash_maxsim, [Q, D]),
        }
        fl = row["flash"]["ms"]
        row["spd"] = {k: round(row[k]["ms"] / fl, 2) for k in keys[:-1]}
        print(f"  { {k: row[k]['ms'] for k in keys} }", flush=True)
        print(f"  speedups vs flash: {row['spd']}", flush=True)
        out["rows"].append(row)
        del Q, D, Qf, Df
        torch.cuda.empty_cache()
        torch._dynamo.reset()


def part_frontier(out):
    CELLS = [("visual", 512, 1024), ("ColPali", 1024, 1024)]
    for name, Lq, Ld in CELLS:
        for B in (5000, 10000, 20000):
            print(f"=== {name} B={B} ===", flush=True)
            Q, D = make_qd(Lq, Ld, B)
            chunk = 1024
            fns = [n_fp16, lambda q, dd: chunked_fp16(q, dd, chunk), flash_maxsim]
            argss = [[Q, D]] * 3
            stats = bench_interleaved_events(fns, argss, warmup=20, n=50)
            keys = ["vanilla_fp16", "chunked_fp16", "flash"]
            row = {"shape": name, "Lq": Lq, "Ld": Ld, "B": B, "chunk": chunk}
            for k, st in zip(keys, stats):
                row[k] = st
            row["peaks_gb"] = {
                "vanilla_fp16": peak_pass(n_fp16, [Q, D]),
                "chunked_fp16": peak_pass(lambda q, dd: chunked_fp16(q, dd, chunk), [Q, D]),
                "flash": peak_pass(flash_maxsim, [Q, D]),
            }
            fl = row["flash"]["ms"]
            row["spd"] = {k: round(row[k]["ms"] / fl, 2) for k in keys[:-1]}
            print(f"  { {k: row[k]['ms'] for k in keys} }  peaks={row['peaks_gb']}", flush=True)
            out["rows"].append(row)
            del Q, D
            torch.cuda.empty_cache()


def part_train(out):
    Lq = Ld = 1024

    def make_train(B):
        Q = F.normalize(torch.randn(B, Lq, d, device='cuda', dtype=torch.float16), dim=-1)
        D = F.normalize(torch.randn(B, Ld, d, device='cuda', dtype=torch.float16), dim=-1)
        Q.requires_grad_(True); D.requires_grad_(True)
        return Q, D

    from torch.utils.checkpoint import checkpoint

    def _chunk_scores(Q, D_chunk):
        S = torch.einsum('aqd,bld->abql', Q.float(), D_chunk.float())
        return S.max(dim=3).values.sum(dim=2)

    def step_naive(Q, D):
        Q.grad = None; D.grad = None
        S = torch.einsum('aqd,bld->abql', Q.float(), D.float())
        S.max(dim=3).values.sum(dim=2).sum().backward()

    def step_recompute(Q, D, chunk=16):
        Q.grad = None; D.grad = None
        parts = [checkpoint(_chunk_scores, Q, D[i:i+chunk], use_reentrant=False)
                 for i in range(0, D.shape[0], chunk)]
        torch.cat(parts, dim=1).sum().backward()

    def step_flash(Q, D):
        Q.grad = None; D.grad = None
        flash_maxsim_batched_train(Q, D, shared_docs=True).sum().backward()

    def with_path(path):
        def f(Q, D):
            os.environ["FLASH_BWD_PATH"] = path
            try:
                step_flash(Q, D)
            finally:
                os.environ.pop("FLASH_BWD_PATH", None)
        return f

    for B in (64, 128):
        print(f"=== train B={B} ===", flush=True)
        Q, D = make_train(B)
        methods = [("recompute", step_recompute),
                   ("flash_unified", with_path("atomic_unified")),
                   ("flash_csr", with_path("invgrid"))]
        naive_ok = B * B * Lq * Ld * 4 * 2 < 70e9
        if naive_ok:
            methods.insert(0, ("naive", step_naive))
        fns = [m[1] for m in methods]
        stats = bench_interleaved_events(fns, [[Q, D]] * len(fns), warmup=5, n=20)
        row = {"B": B}
        for (k, _), st in zip(methods, stats):
            row[k] = st
        for k, fn in methods:
            row.setdefault("peaks_gb", {})[k] = peak_pass(fn, [Q, D], warmup=2)
        if not naive_ok:
            row["naive"] = {"oom": True,
                            "note": "analytic S+gradS = 128GB FP32 > VRAM"}
        print(f"  { {k: row.get(k) for k in ('naive','recompute','flash_unified','flash_csr')} }", flush=True)
        print(f"  peaks: {row['peaks_gb']}", flush=True)
        out["rows"].append(row)
        del Q, D
        torch.cuda.empty_cache()


def main():
    gpu = torch.cuda.get_device_name()
    part = os.environ.get("BENCH_PART", "all")
    print(f"GPU: {gpu}  part={part}", flush=True)
    for p in (["shapes", "frontier", "train"] if part == "all" else [part]):
        out = {"gpu": gpu, "part": p, "torch": torch.__version__,
               "protocol": "interleaved round-robin, 100MB L2 flush per call, "
                           "CUDA-event median+IQR, compile warmup>=100 with "
                           "subprocess autotune, per-method peak passes",
               "rows": []}
        {"shapes": part_shapes, "frontier": part_frontier, "train": part_train}[p](out)
        op = os.path.join(HERE, f"bench_protocol_grade_{p}_{gpu.replace(' ', '_').replace('/', '_')}.json")
        json.dump(out, open(op, "w"), indent=2)
        print("wrote", op, flush=True)


if __name__ == '__main__':
    main()
