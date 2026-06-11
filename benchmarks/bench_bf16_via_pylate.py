"""Bisect raphaelsty's bf16+masked anomaly through pylate's actual call path.

The previous bisect (bench_bf16_masked_bisect.py) called flash directly and saw
only normal bf16 noise (~0.01). The remaining suspect is pylate's wrapper, which
does `Q = Q * queries_mask.to(Q.dtype)` in user dtype before calling flash, and
converts masks to lengths via `_mask_to_lengths` (which itself had a bf16 bug
that was fixed in 79e444a but may have left a sibling bug behind).

Goals:
  1. Reproduce the through-pylate setup at long-doc and ColPali shapes.
  2. Sweep light vs heavy masking (mean ~50% real tokens vs ~16% real tokens).
  3. Compare pylate's torch backend vs flash backend at bf16 against an FP32
     reference computed the way pylate would (apply masks pre-einsum).
  4. If flash backend diverges much more from FP32 than torch backend does,
     the bug lives in `_flash_backend.py`'s mask handling.

We test 8 cells: {Lq=32/1024} × {fp16/bf16} × {light/heavy mask}.
"""
import os, sys, json, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/path/to/data")
sys.path.insert(0, HERE)
from pylate.scores import colbert_scores

torch.set_float32_matmul_precision('high')

B, Nq, d = 4, 4, 128


def make(Lq, Ld, dtype, seed=0):
    g = torch.Generator(device='cuda').manual_seed(seed)
    Q32 = F.normalize(torch.randn(Nq, Lq, d, device='cuda', dtype=torch.float32, generator=g), dim=-1)
    D32 = F.normalize(torch.randn(B, Ld, d, device='cuda', dtype=torch.float32, generator=g), dim=-1)
    return Q32.to(dtype), D32.to(dtype), Q32, D32


def make_masks(Lq, Ld, mode, dtype):
    """0/1 masks in user dtype (the pylate convention).
    mode='light' -> ~50% of doc tokens real; mode='heavy' -> ~16% real
    (modeling ColBERT/ColPali ragged distributions)."""
    g = torch.Generator(device='cpu').manual_seed(13)
    q_lens = torch.randint(max(1, Lq // 2), Lq + 1, (Nq,), generator=g)
    if mode == 'light':
        d_lens = torch.randint(max(1, Ld // 2), Ld + 1, (B,), generator=g)
    else:  # heavy
        d_lens = torch.randint(max(1, Ld // 12), Ld // 4 + 1, (B,), generator=g)
    ar_q = torch.arange(Lq); ar_d = torch.arange(Ld)
    q_mask = (ar_q[None, :] < q_lens[:, None]).to(dtype).cuda()
    d_mask = (ar_d[None, :] < d_lens[:, None]).to(dtype).cuda()
    return q_mask, d_mask


def fp32_ref(Q32, D32, q_mask32, d_mask32):
    """Reference computed the way pylate's torch path does it: mask Q and D in
    fp32 (precision-preserving), einsum, post-multiply by masks again, max, sum."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        Qm = Q32 * q_mask32.unsqueeze(-1)
        Dm = D32 * d_mask32.unsqueeze(-1)
        sim = torch.einsum("nqd,bld->nbql", Qm, Dm)
        sim = sim * q_mask32[:, None, :, None] * d_mask32[None, :, None, :]
        return sim.max(dim=3).values.sum(dim=2)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def err(a, ref):
    diff = (a.float() - ref.float()).abs()
    return float(diff.max()), float(diff.mean())


results = []
print(f"Nq={Nq} B={B} d={d}\n")
for Lq, Ld, label in [(32, 1024, "long-doc"), (1024, 1024, "ColPali")]:
    for dtype, dt_label in [(torch.float16, "fp16"), (torch.bfloat16, "bf16")]:
        for mode in ["light", "heavy"]:
            Q, D, Q32, D32 = make(Lq, Ld, dtype, seed=0)
            q_mask, d_mask = make_masks(Lq, Ld, mode, dtype)
            q_mask32 = q_mask.float(); d_mask32 = d_mask.float()
            ref = fp32_ref(Q32, D32, q_mask32, d_mask32)
            # pylate torch backend (sanity: should match ref to noise)
            torch_out = colbert_scores(Q, D, q_mask, d_mask, backend="torch")
            tmx, tmn = err(torch_out, ref)
            # pylate flash backend (the path raphaelsty audited)
            flash_out = colbert_scores(Q, D, q_mask, d_mask, backend="flash")
            fmx, fmn = err(flash_out, ref)
            ratio = fmx / tmx if tmx > 0 else float('inf')
            tag = "(flash 10x worse!)" if ratio > 10 else ""
            print(f"  {label:8s} {dt_label}  mask={mode:5s}  "
                  f"torch=[max {tmx:.2e}, mean {tmn:.2e}]  "
                  f"flash=[max {fmx:.2e}, mean {fmn:.2e}]  "
                  f"f/t={ratio:5.2f}x  {tag}")
            results.append({
                "shape": label, "Lq": Lq, "Ld": Ld, "dtype": dt_label,
                "mask_mode": mode,
                "torch_max": tmx, "torch_mean": tmn,
                "flash_max": fmx, "flash_mean": fmn,
                "flash_over_torch": ratio,
            })

gpu = torch.cuda.get_device_name()
p = os.path.join(HERE, f"bench_bf16_via_pylate_{gpu.replace(' ', '_').replace('/', '_')}.json")
json.dump({"gpu": gpu, "results": results}, open(p, "w"), indent=2)
print(f"\nwrote {p}")
