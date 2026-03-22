"""Flash-MaxSim benchmark. Run: python benchmarks/bench.py"""
import time, torch, torch.nn.functional as F

assert torch.cuda.is_available(), "CUDA required"
print(f"GPU: {torch.cuda.get_device_name()}\n")

from flash_maxsim import flash_maxsim, flash_maxsim_batched, flash_maxsim_train, maxsim_naive
from flash_maxsim import flash_maxsim_int8, quantize_int8

def bench(fn, *a, warmup=10, n=50):
    for _ in range(warmup): fn(*a)
    torch.cuda.synchronize()
    t = []
    for _ in range(n):
        torch.cuda.synchronize(); s = time.perf_counter(); fn(*a)
        torch.cuda.synchronize(); t.append((time.perf_counter()-s)*1000)
    t.sort(); return t[len(t)//2]

def sim(B, Lq, Ld, d):
    Q = F.normalize(torch.randn(Lq, d, device='cuda', dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(B, Ld, d, device='cuda', dtype=torch.float16), dim=-1)
    return Q, D

def naive_fp32(Q, D):
    return torch.einsum('qd,bld->bql', Q.float(), D.float()).max(2).values.sum(1)

# ── Correctness ──
print("=" * 65)
print("CORRECTNESS")
print("=" * 65)
for B,Lq,Ld,d in [(50,32,300,128),(20,64,512,128),(10,32,300,96)]:
    Q,D = sim(B,Lq,Ld,d)
    ref = naive_fp32(Q,D)
    out = flash_maxsim(Q,D)
    err = (ref - out).abs().max().item()
    rank_ok = (ref.argsort(descending=True)[:10] == out.argsort(descending=True)[:10]).all()
    print(f"  B={B},Lq={Lq},Ld={Ld},d={d}: err={err:.2e} rank={'OK' if rank_ok else 'FAIL'}")

# ── Single query speedup (vs FP32 einsum — what users replace) ──
print(f"\n{'=' * 65}")
print("SINGLE QUERY SPEEDUP (vs naive FP32 einsum)")
print("=" * 65)
for B,Lq,Ld,d,label in [
    (1000,32,300,128,"ColBERT (Lq=32,Ld=300)"),
    (500,64,300,128,"ColBERT-long (Lq=64)"),
    (500,32,1024,128,"ColPali text (Ld=1024)"),
    (2000,32,300,128,"Large B=2000"),
    (1000,64,1024,128,"ColPali Lq=64,Ld=1024"),
]:
    Q,D = sim(B,Lq,Ld,d)
    n = bench(naive_fp32, Q, D)
    f = bench(flash_maxsim, Q, D)
    print(f"  {label:35s}: naive={n:.2f}ms  flash={f:.2f}ms  {n/f:.1f}x")

# ── Lq=1024 regime (the big numbers) ──
print(f"\n{'=' * 65}")
print("Lq=1024 REGIME (ColPali image-to-image)")
print("=" * 65)
for B,Lq,Ld,d,label in [
    (100,1024,1024,128,"1q x 100p (Lq=Ld=1024)"),
    (1000,1024,1024,128,"1q x 1000p (Lq=Ld=1024)"),
    (5000,1024,1024,128,"1q x 5000p (Lq=Ld=1024)"),
]:
    Q,D = sim(B,Lq,Ld,d)
    n = bench(naive_fp32, Q, D, warmup=5, n=20)
    f = bench(flash_maxsim, Q, D, warmup=5, n=20)
    print(f"  {label:35s}: naive={n:.2f}ms  flash={f:.2f}ms  {n/f:.1f}x")

# ── INT8 fused dequant ──
print(f"\n{'=' * 65}")
print("INT8 FUSED DEQUANT")
print("=" * 65)
for B,Lq,Ld,d,label in [
    (1000,32,300,128,"ColBERT"),
    (500,32,1024,128,"ColPali"),
    (5000,32,300,128,"Large B=5000"),
]:
    Q,D = sim(B,Lq,Ld,d)
    Dq,s,m = quantize_int8(D)
    def naive_int8():
        Df = Dq.float()*s.float()+m.float()
        return torch.einsum('qd,bld->bql',Q.float(),Df).max(2).values.sum(1)
    ni = bench(naive_int8, warmup=5, n=30)
    fi = bench(flash_maxsim_int8, Q, Dq, s, m, warmup=5, n=30)
    nf = bench(naive_fp32, Q, D, warmup=5, n=30)
    print(f"  {label:15s}: naive_fp32={nf:.2f}ms  naive_int8={ni:.2f}ms  fused={fi:.2f}ms")
    print(f"  {'':15s}  vs naive_int8: {ni/fi:.1f}x  vs naive_fp32: {nf/fi:.1f}x")

# ── Batched throughput ──
print(f"\n{'=' * 65}")
print("BATCHED THROUGHPUT")
print("=" * 65)
for Nq,B,Lq,Ld,d,label in [
    (100,1000,32,300,128,"100q x 1000d ColBERT"),
    (50,500,32,1024,128,"50q x 500p ColPali"),
    (100,100,1024,1024,128,"100q x 100p Lq=Ld=1024"),
]:
    Q = F.normalize(torch.randn(Nq,Lq,d,device='cuda',dtype=torch.float16),dim=-1)
    D = F.normalize(torch.randn(B,Ld,d,device='cuda',dtype=torch.float16),dim=-1)
    def naive_loop():
        return torch.stack([naive_fp32(Q[i],D) for i in range(Nq)])
    nl = bench(naive_loop, warmup=3, n=10)
    fb = bench(flash_maxsim_batched, Q, D, None, True, warmup=5, n=20)
    tp = Nq*B/(fb/1000)
    print(f"  {label:30s}: naive={nl:.1f}ms  flash={fb:.2f}ms  {nl/fb:.1f}x  ({tp/1e6:.1f}M pairs/s)")

# ── Training ──
print(f"\n{'=' * 65}")
print("TRAINING (fwd+bwd)")
print("=" * 65)
for B,Lq,Ld,d,label in [(500,32,300,128,"ColBERT"),(1000,32,300,128,"ColBERT-large")]:
    Qi = F.normalize(torch.randn(Lq,d,device='cuda',dtype=torch.float16),dim=-1)
    Di = F.normalize(torch.randn(B,Ld,d,device='cuda',dtype=torch.float16),dim=-1)
    def run_naive():
        Q2=Qi.clone().float().requires_grad_(True); D2=Di.clone().float().requires_grad_(True)
        s=torch.einsum('qd,bld->bql',Q2,D2).max(2).values.sum(1); s.sum().backward()
    Qf=Qi.clone().requires_grad_(True); Df=Di.clone().requires_grad_(True)
    def run_flash():
        s=flash_maxsim_train(Qf,Df); s.sum().backward(); Qf.grad=None; Df.grad=None
    n = bench(run_naive, warmup=5, n=20)
    f = bench(run_flash, warmup=5, n=20)
    print(f"  {label:15s}: naive={n:.2f}ms  flash={f:.2f}ms  {n/f:.1f}x")

# ── Peak memory ──
print(f"\n{'=' * 65}")
print("PEAK MEMORY")
print("=" * 65)
for Nq,B,Lq,Ld,d,label in [
    (1,1000,32,1024,128,"1q x 1000p Ld=1024"),
    (1,1000,1024,1024,128,"1q x 1000p Lq=Ld=1024"),
    (10,1000,1024,1024,128,"10q x 1000p Lq=Ld=1024"),
]:
    Q = F.normalize(torch.randn(Nq,Lq,d,device='cuda',dtype=torch.float16),dim=-1)
    D = F.normalize(torch.randn(B,Ld,d,device='cuda',dtype=torch.float16),dim=-1)
    sim_gb = Nq*B*Lq*Ld*4/1e9
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    try:
        _ = torch.einsum('nqd,bld->nbql',Q.float(),D.float()).max(3).values.sum(2); torch.cuda.synchronize()
        naive_gb = (torch.cuda.max_memory_allocated()-base)/1e9
        del _
    except RuntimeError:
        naive_gb = sim_gb
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    _ = flash_maxsim_batched(Q, D, shared_docs=True); torch.cuda.synchronize()
    flash_gb = (torch.cuda.max_memory_allocated()-base)/1e9
    del _; torch.cuda.empty_cache()
    print(f"  {label:30s}: sim={sim_gb:.1f}GB  naive={naive_gb:.1f}GB  flash={flash_gb:.1f}GB  ({max(1,naive_gb/max(flash_gb,0.01)):.0f}x less)")

print(f"\n{'=' * 65}")
print("DONE")
print("=" * 65)
