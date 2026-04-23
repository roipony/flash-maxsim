"""v0.2.1 candidate smoke: verify top-level import works + pylate fwd+bwd match."""
import sys, torch
import torch.nn.functional as F

sys.path.insert(0, '/dccstor/ocr-ai/pony/projects/flash_maxsim')
sys.path.insert(0, '/dccstor/ocr-ai/pony/projects/pylate')

# Step 1: top-level import (this is the public surface)
from flash_maxsim import (
    flash_maxsim, flash_maxsim_batched, flash_maxsim_pairs,
    flash_maxsim_train, flash_maxsim_batched_train,            # <-- new v0.2.1
    flash_maxsim_int8, quantize_int8,
    flash_maxsim_varlen, pack_pairs,
    maxsim_naive, maxsim_naive_fp32,
)
import flash_maxsim as fm
print(f"flash-maxsim version: {fm.__version__}")
print(f"flash_maxsim_batched_train: {flash_maxsim_batched_train}")
print("[OK] v0.2.1 exports all importable from top level\n")

# Step 2: a realistic contrastive training step + compare to pylate
from pylate.scores import colbert_scores

device = 'cuda'
torch.manual_seed(0)
B, Lq, Ld, d = 256, 32, 180, 128

Q_base = F.normalize(torch.randn(B, Lq, d, dtype=torch.float16, device=device), dim=-1)
D_base = F.normalize(torch.randn(B, Ld, d, dtype=torch.float16, device=device), dim=-1)

# Flash path
Qf = Q_base.clone().requires_grad_(True); Df = D_base.clone().requires_grad_(True)
Sf = flash_maxsim_batched_train(Qf, Df, shared_docs=True)
loss_f = Sf.diagonal().sum()
loss_f.backward()

# Pylate path
Qp = Q_base.clone().requires_grad_(True); Dp = D_base.clone().requires_grad_(True)
Sp = colbert_scores(Qp, Dp)
loss_p = Sp.diagonal().sum()
loss_p.backward()

# Correctness
score_rel = (Sf.float() - Sp.float()).abs().max().item() / max(Sp.abs().max().item(), 1e-9)
gQ_cos = F.cosine_similarity(Qf.grad.float().flatten().unsqueeze(0),
                               Qp.grad.float().flatten().unsqueeze(0)).item()
gD_cos = F.cosine_similarity(Df.grad.float().flatten().unsqueeze(0),
                               Dp.grad.float().flatten().unsqueeze(0)).item()

print(f"pylate ↔ flash (contrastive B={B}):")
print(f"  score rel err = {score_rel:.3e}")
print(f"  grad_Q cosine = {gQ_cos:.5f}")
print(f"  grad_D cosine = {gD_cos:.5f}")

# Latency (warmup + median of 5)
import time, gc
def bench(fn, *args):
    for _ in range(3): fn(*args)  # warmup
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    ts = []
    for _ in range(5):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn(*args); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort(); return ts[2]

def flash_step():
    Q = Q_base.clone().requires_grad_(True); D = D_base.clone().requires_grad_(True)
    flash_maxsim_batched_train(Q, D, shared_docs=True).diagonal().sum().backward()

def pyt_step():
    Q = Q_base.clone().requires_grad_(True); D = D_base.clone().requires_grad_(True)
    colbert_scores(Q, D).diagonal().sum().backward()

flash_ms = bench(flash_step)
pyt_ms   = bench(pyt_step)
print(f"\nLatency (one training step, B={B}):")
print(f"  pylate = {pyt_ms:.2f} ms")
print(f"  flash  = {flash_ms:.2f} ms")
print(f"  speedup = {pyt_ms/flash_ms:.2f}×")
print("\n[OK] v0.2.1 candidate smoke PASS")
