"""Split pylate vs flash by phase:
  - inference  : forward only, no requires_grad (pure scoring)
  - forward    : forward with grad graph built (saves argmax / S)
  - backward   : backward pass only (given a prebuilt graph)
  - full step  : fwd + bwd (like training)
"""
import sys, time, gc, torch
import torch.nn.functional as F

sys.path.insert(0, '/path/to/data')
sys.path.insert(0, '/path/to/data')

from flash_maxsim.flash_maxsim_batched_train import flash_maxsim_batched_train
from flash_maxsim import flash_maxsim_batched
from pylate.scores import colbert_scores

device = 'cuda'
DTYPE = torch.float16


def reset():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()


def bench_one(fn, warmup=3, runs=5):
    for _ in range(warmup): fn()
    reset(); torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times)//2]


def measure(name, B, Lq, Ld, d):
    torch.manual_seed(B)
    Q0 = F.normalize(torch.randn(B, Lq, d, dtype=DTYPE, device=device), dim=-1)
    D0 = F.normalize(torch.randn(B, Ld, d, dtype=DTYPE, device=device), dim=-1)

    results = {"B": B, "Lq": Lq, "Ld": Ld}

    # -------- inference (no grad graph) --------
    with torch.no_grad():
        results["pyt_inf"] = bench_one(lambda: colbert_scores(Q0, D0))
        results["fl_inf"]  = bench_one(lambda: flash_maxsim_batched(Q0, D0, shared_docs=True))

    # -------- forward w/ grad graph (same cost as training fwd) --------
    def pyt_fwd():
        Qp = Q0.clone().requires_grad_(True); Dp = D0.clone().requires_grad_(True)
        _ = colbert_scores(Qp, Dp)
    def fl_fwd():
        Qf = Q0.clone().requires_grad_(True); Df = D0.clone().requires_grad_(True)
        _ = flash_maxsim_batched_train(Qf, Df, shared_docs=True)
    results["pyt_fwd"] = bench_one(pyt_fwd)
    results["fl_fwd"]  = bench_one(fl_fwd)

    # -------- backward only (fwd outside the timer) --------
    def pyt_bwd():
        Qp = Q0.clone().requires_grad_(True); Dp = D0.clone().requires_grad_(True)
        s = colbert_scores(Qp, Dp)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        s.diagonal().sum().backward()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000

    def fl_bwd():
        Qf = Q0.clone().requires_grad_(True); Df = D0.clone().requires_grad_(True)
        s = flash_maxsim_batched_train(Qf, Df, shared_docs=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        s.diagonal().sum().backward()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000

    # warmup
    for _ in range(3): pyt_bwd(); fl_bwd()
    pts = sorted([pyt_bwd() for _ in range(5)])
    fts = sorted([fl_bwd()  for _ in range(5)])
    results["pyt_bwd"] = pts[len(pts)//2]
    results["fl_bwd"]  = fts[len(fts)//2]

    # -------- full step (fwd + bwd) --------
    def full_pyt():
        Qp = Q0.clone().requires_grad_(True); Dp = D0.clone().requires_grad_(True)
        colbert_scores(Qp, Dp).diagonal().sum().backward()
    def full_fl():
        Qf = Q0.clone().requires_grad_(True); Df = D0.clone().requires_grad_(True)
        flash_maxsim_batched_train(Qf, Df, shared_docs=True).diagonal().sum().backward()
    results["pyt_step"] = bench_one(full_pyt)
    results["fl_step"]  = bench_one(full_fl)

    return results


print(f"GPU: {torch.cuda.get_device_name(0)}")
print("Methodology: 3 warmup + median of 5 runs. Times in ms.\n")


CASES = [
    ("ColBERT",  32,   180, 128, [128, 512, 1024, 2048]),
    ("ColBERT",  32,  1024, 128, [128, 512]),
    ("Medium",   128, 1024, 128, [64, 128, 256]),
    ("ColPali",  1024, 1024, 128, [8, 16, 32]),
]

for name, Lq, Ld, d, Bs in CASES:
    print(f"=" * 110)
    print(f"  {name}   Lq={Lq}  Ld={Ld}")
    print(f"=" * 110)
    hdr = ("B",
           "pyt_inf", "fl_inf",  "x",
           "pyt_fwd", "fl_fwd",  "x",
           "pyt_bwd", "fl_bwd",  "x",
           "pyt_step","fl_step", "x")
    print(f"  {'B':>5}   "
          f"{'pyt_inf':>7} {'fl_inf':>7} {'×':>5}   "
          f"{'pyt_fwd':>7} {'fl_fwd':>7} {'×':>5}   "
          f"{'pyt_bwd':>7} {'fl_bwd':>7} {'×':>5}   "
          f"{'pyt_step':>8} {'fl_step':>7} {'×':>5}")
    for B in Bs:
        r = measure(name, B, Lq, Ld, d)
        ratio = lambda a,b: f"{a/b:>4.2f}×" if b > 0 else "   —"
        print(f"  {B:>5}   "
              f"{r['pyt_inf']:>7.2f} {r['fl_inf']:>7.2f} {ratio(r['pyt_inf'], r['fl_inf']):>5}   "
              f"{r['pyt_fwd']:>7.2f} {r['fl_fwd']:>7.2f} {ratio(r['pyt_fwd'], r['fl_fwd']):>5}   "
              f"{r['pyt_bwd']:>7.2f} {r['fl_bwd']:>7.2f} {ratio(r['pyt_bwd'], r['fl_bwd']):>5}   "
              f"{r['pyt_step']:>8.2f} {r['fl_step']:>7.2f} {ratio(r['pyt_step'], r['fl_step']):>5}")
    print()

print("Done.")
