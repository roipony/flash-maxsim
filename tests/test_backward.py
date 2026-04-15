"""Verify backward pass correctness: analytical vs naive gradients."""
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '.')
from flash_maxsim import flash_maxsim_train

print('=== Backward pass gradient verification ===')
print()

# Manual check: compare flash backward vs naive backward (FP16)
Q = F.normalize(torch.randn(16, 128, device='cuda', dtype=torch.float16), dim=-1).requires_grad_(True)
D = F.normalize(torch.randn(10, 64, 128, device='cuda', dtype=torch.float16), dim=-1).requires_grad_(True)

s = flash_maxsim_train(Q, D)
s.sum().backward()

print(f'Q.grad: shape={Q.grad.shape}, norm={Q.grad.norm():.4f}, nan={Q.grad.isnan().any()}')
print(f'D.grad: shape={D.grad.shape}, norm={D.grad.norm():.4f}, nan={D.grad.isnan().any()}')

# Naive backward
Q2 = Q.detach().clone().float().requires_grad_(True)
D2 = D.detach().clone().float().requires_grad_(True)
S_naive = torch.einsum('qd,bld->bql', Q2, D2)
s_naive = S_naive.max(dim=2).values.sum(dim=1)
s_naive.sum().backward()

q_err = (Q.grad.float() - Q2.grad.half().float()).abs().max().item()
d_err = (D.grad.float() - D2.grad.half().float()).abs().max().item()
print(f'Q.grad max error vs naive: {q_err:.6f} {"OK" if q_err < 0.05 else "BAD"}')
print(f'D.grad max error vs naive: {d_err:.6f} {"OK" if d_err < 0.05 else "BAD"}')

# Test ColPali-size
print()
Q3 = F.normalize(torch.randn(1024, 128, device='cuda', dtype=torch.float16), dim=-1).requires_grad_(True)
D3 = F.normalize(torch.randn(50, 1024, 128, device='cuda', dtype=torch.float16), dim=-1).requires_grad_(True)
s3 = flash_maxsim_train(Q3, D3)
s3.sum().backward()
print(f'ColPali (Lq=1024, B=50): Q.grad norm={Q3.grad.norm():.4f}, D.grad norm={D3.grad.norm():.4f}')
print(f'  Q.grad nan={Q3.grad.isnan().any()}, D.grad nan={D3.grad.isnan().any()}')

# Compare with naive
Q4 = Q3.detach().clone().float().requires_grad_(True)
D4 = D3.detach().clone().float().requires_grad_(True)
S4 = torch.einsum('qd,bld->bql', Q4, D4)
s4 = S4.max(dim=2).values.sum(dim=1)
s4.sum().backward()
q_err = (Q3.grad.float() - Q4.grad.half().float()).abs().max().item()
d_err = (D3.grad.float() - D4.grad.half().float()).abs().max().item()
print(f'  Q.grad max error: {q_err:.6f} {"OK" if q_err < 0.1 else "BAD"}')
print(f'  D.grad max error: {d_err:.6f} {"OK" if d_err < 0.1 else "BAD"}')

print()
print('Done.')
