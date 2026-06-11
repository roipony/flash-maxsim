"""Minimal repro: invgrid varlen → bincount illegal memory access.

Hypothesis: forward kernel returns argmax values >= Ld for some position
when varlen masking interacts oddly with the BLOCK_D tile. We sweep argmax
to verify whether it's actually out-of-range, and propose a fix.
"""
import sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/path/to/data')

from flash_maxsim.flash_maxsim_batched_train import flash_maxsim_batched_train
from flash_maxsim.flash_maxsim import _launch_fwd, _next_pow2, _default_lengths

device = 'cuda'

torch.manual_seed(124)
Nq, B, Lq, Ld, d = 8, 8, 28, 175, 128
Q = F.normalize(torch.randn(Nq, Lq, d, device=device, dtype=torch.float16), dim=-1)
D = F.normalize(torch.randn(B,  Ld, d, device=device, dtype=torch.float16), dim=-1)
doc_lengths   = torch.tensor([169,106,85,175,90,157,165,58], dtype=torch.int32, device=device)
query_lengths = torch.tensor([26, 17,19, 21,28, 12,15,20], dtype=torch.int32, device=device)

# Run forward to get argmax
Q_c = Q.contiguous().half()
D_c = D.contiguous().half()
lengths = _default_lengths(B, Ld, device, doc_lengths)
scores, argmax = _launch_fwd(
    Q_c, D_c, lengths, Nq, B, Lq, Ld, d,
    True, True, q_lengths=query_lengths,
)
torch.cuda.synchronize()

print(f"argmax shape: {tuple(argmax.shape)}")
print(f"argmax dtype: {argmax.dtype}")
print(f"argmax min:   {argmax.min().item()}")
print(f"argmax max:   {argmax.max().item()}")
print(f"Ld:           {Ld}")
print(f"argmax >= Ld? count: {(argmax >= Ld).sum().item()} / {argmax.numel()}")
oob = argmax >= Ld
if oob.any():
    print(f"OOB positions (sample): {oob.nonzero()[:5].tolist()}")
    print(f"OOB values:             {argmax[oob][:5].tolist()}")
print(f"argmax < 0?  count: {(argmax < 0).sum().item()}")

# Diagnose by pair_idx and lq_idx
flat_pair = torch.arange(B * Nq, device=device)
b_of_pair = flat_pair // Nq
q_of_pair = flat_pair % Nq

# For each pair, check: does it have any argmax >= doc_lengths[b]?
# This means argmax exceeded the actual valid range, even if within Ld.
violations = []
for p in range(B * Nq):
    b_idx = b_of_pair[p].item()
    q_idx = q_of_pair[p].item()
    dlen = doc_lengths[b_idx].item()
    qlen = query_lengths[q_idx].item()
    am_row = argmax[p].cpu()
    # for valid q positions (lq < qlen), argmax should be < dlen
    for lq in range(Lq):
        if lq < qlen:
            if am_row[lq] >= dlen:
                violations.append((p, b_idx, q_idx, lq, am_row[lq].item(), dlen))

print(f"\n# argmax >= doc_lengths[b]: {len(violations)} violations")
for v in violations[:10]:
    print(f"  pair={v[0]} b={v[1]} q={v[2]} lq={v[3]}: argmax={v[4]} >= dlen={v[5]}")
