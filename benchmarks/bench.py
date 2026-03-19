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

# ── Correctness ──
print("Correctness:")
for B,Lq,Ld,d in [(50,32,300,128),(20,64,512,128),(10,32,300,96)]:
    Q,D = sim(B,Lq,Ld,d)
    err = (maxsim_naive(Q,D) - flash_maxsim(Q,D)).abs().max().item()
    print(f"  B={B},Lq={Lq},Ld={Ld},d={d}: err={err:.2e} {'PASS' if err<1 else 'FAIL'}")

# ── FP16 speedup ──
print("\nFP16 speedup:")
for B,Lq,Ld,d,label in [
    (1000,32,300,128,"ColBERT"), (500,64,300,128,"ColBERT-long"),
    (500,32,1024,128,"ColPali"), (2000,32,300,128,"Large-B")]:
    Q,D = sim(B,Lq,Ld,d)
    n = bench(lambda: torch.einsum('qd,bld->bql',Q.float(),D.float()).max(2).values.sum(1))
    f = bench(flash_maxsim, Q, D)
    print(f"  {label:15s}: naive={n:.2f}ms  flash={f:.2f}ms  {n/f:.1f}x")

# ── INT8 ──
print("\nINT8 fused dequant:")
for B,Lq,Ld,d,label in [(1000,32,300,128,"ColBERT"),(500,32,1024,128,"ColPali")]:
    Q,D = sim(B,Lq,Ld,d)
    Dq,s,m = quantize_int8(D)
    ni = bench(lambda: (lambda Df: torch.einsum('qd,bld->bql',Q.float(),Df.float()).max(2).values.sum(1))(Dq.float()*s.float()+m.float()))
    fi = bench(flash_maxsim_int8, Q, Dq, s, m)
    print(f"  {label:15s}: naive_int8={ni:.2f}ms  fused={fi:.2f}ms  {ni/fi:.1f}x")

# ── Batched ──
print("\nBatched throughput:")
for Nq,B,Lq,Ld,d,label in [(100,1000,32,300,128,"100q×1000d"),(50,500,32,1024,128,"50q×500p ColPali")]:
    Q = F.normalize(torch.randn(Nq,Lq,d,device='cuda',dtype=torch.float16),dim=-1)
    D = F.normalize(torch.randn(B,Ld,d,device='cuda',dtype=torch.float16),dim=-1)
    f = bench(flash_maxsim_batched, Q, D, None, True)
    tp = Nq*B/(f/1000)
    print(f"  {label:20s}: {f:.2f}ms  ({tp/1e6:.1f}M pairs/s)")

# ── Training ──
print("\nTraining fwd+bwd:")
for B,Lq,Ld,d,label in [(500,32,300,128,"ColBERT"),(100,64,300,128,"ColBERT-long")]:
    Q = F.normalize(torch.randn(Lq,d,device='cuda',dtype=torch.float16),dim=-1).requires_grad_(True)
    D = F.normalize(torch.randn(B,Ld,d,device='cuda',dtype=torch.float16),dim=-1).requires_grad_(True)
    def run():
        s=flash_maxsim_train(Q,D); s.sum().backward(); Q.grad=None; D.grad=None
    def run_naive():
        Q2=Q.detach().float().requires_grad_(True); D2=D.detach().float().requires_grad_(True)
        s=torch.einsum('qd,bld->bql',Q2,D2).max(2).values.sum(1); s.sum().backward()
    n = bench(run_naive, warmup=5, n=20)
    f = bench(run, warmup=5, n=20)
    print(f"  {label:15s}: naive={n:.2f}ms  flash={f:.2f}ms  {n/f:.1f}x")

# ── Memory ──
print("\nPeak memory:")
Q,D = sim(10, 1024, 1024, 128)
torch.cuda.reset_peak_memory_stats()
_ = torch.einsum('nqd,pld->npql',Q[None].float(),D.float()).max(3).values.sum(2)
naive_peak = torch.cuda.max_memory_allocated()/1e9
del _; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
_ = flash_maxsim(Q, D)
flash_peak = torch.cuda.max_memory_allocated()/1e9
print(f"  10×1000 (Lq=Ld=1024): naive={naive_peak:.1f}GB  flash={flash_peak:.1f}GB  ({naive_peak/flash_peak:.0f}x less)")
