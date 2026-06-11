"""Run the new batched-training demo cell and the new demo_v020.py section."""
import sys, torch
import torch.nn.functional as F

sys.path.insert(0, '/path/to/data')

# --- 1. Notebook cell equivalent ---
from flash_maxsim import flash_maxsim_batched_train
Nq, B, Lq, Ld, d = 32, 32, 1024, 1024, 128      # ColPali — where memory story shows
Q = F.normalize(torch.randn(Nq, Lq, d, dtype=torch.float16, device='cuda'), dim=-1).requires_grad_(True)
D = F.normalize(torch.randn(B, Ld, d, dtype=torch.float16, device='cuda'), dim=-1).requires_grad_(True)

torch.cuda.reset_peak_memory_stats(); baseline = torch.cuda.memory_allocated()
scores = flash_maxsim_batched_train(Q, D, shared_docs=True)
scores.diagonal().sum().backward()
flash_peak = (torch.cuda.max_memory_allocated() - baseline) / 1e6
gQ_flash = Q.grad.detach().clone(); gD_flash = D.grad.detach().clone()

Q.grad = None; D.grad = None
torch.cuda.reset_peak_memory_stats(); baseline = torch.cuda.memory_allocated()
S = torch.einsum('nqd,bld->nbql', Q, D).float()
scores_ref = S.max(dim=-1).values.sum(dim=-1)
scores_ref.diagonal().sum().backward()
naive_peak = (torch.cuda.max_memory_allocated() - baseline) / 1e6

gQ_cos = F.cosine_similarity(gQ_flash.float().flatten().unsqueeze(0),
                             Q.grad.float().flatten().unsqueeze(0)).item()
gD_cos = F.cosine_similarity(gD_flash.float().flatten().unsqueeze(0),
                             D.grad.float().flatten().unsqueeze(0)).item()

print(f"scores shape: {tuple(scores.shape)}  dtype: {scores.dtype}")
print(f"grad_Q cos:   {gQ_cos:.5f}   grad_D cos: {gD_cos:.5f}")
print(f"naive peak:   {naive_peak:>6.1f} MB  (stores [Nq, B, Lq, Ld])")
print(f"flash peak:   {flash_peak:>6.1f} MB  (only argmax)")
print(f"reduction:    {naive_peak/max(flash_peak,1e-6):>6.1f}×")
print()

# --- 2. demo_v020.py's new demo_batched_training ---
sys.path.insert(0, '/path/to/data')
from demo_v020 import demo_batched_training
demo_batched_training()
