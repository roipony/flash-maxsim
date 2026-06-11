"""Bisect the bf16+masked precision bug raphaelsty reported in #224.

His findings:
  - bf16 + unmasked: ~fp32-equivalent (correct)
  - bf16 +   masked: max|Δ| 0.13 (long-doc) -> 0.29 (ColPali)
  - fp16 +   masked: correct

So the bug is the bf16 * masking interaction, not bf16 alone or masking alone.

This script tests the 4 corners of {dtype × masking} against an FP32 reference
on ColPali shape (Lq=Ld=1024, d=128, B=4 to keep wall-clock low), then runs
three "manual repair" probes that progressively bypass suspected layers:

  probe_A: pylate-style pre-mask multiplication done OUTSIDE the kernel
           (Q *= q_mask, D *= d_mask in user dtype before calling flash)
           -> tells us whether the bug lives in pylate's mask path or upstream

  probe_B: bf16 inputs but cast to fp16 by user BEFORE calling flash
           (Q.half(), D.half() done explicitly)
           -> tells us whether the bug is in flash's wrapper cast or kernel

  probe_C: bf16 inputs but cast to fp32 by user BEFORE calling flash
           -> tells us whether the bug survives ANY masking at fp32 precision

Each result is max|Δ| and mean|Δ| vs the FP32 einsum reference.
"""
import json, os, sys, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from flash_maxsim import flash_maxsim_batched_train

# Precision config: TF32 on for tensor cores, but we'll compute the FP32
# reference with true fp32 (TF32 off temporarily) so the reference itself is exact.
torch.set_float32_matmul_precision('high')

Lq, Ld, d = 1024, 1024, 128
B = 4


def make(dtype, seed=0):
    g = torch.Generator(device='cuda').manual_seed(seed)
    Q = F.normalize(torch.randn(B, Lq, d, device='cuda', dtype=torch.float32, generator=g), dim=-1)
    D = F.normalize(torch.randn(B, Ld, d, device='cuda', dtype=torch.float32, generator=g), dim=-1)
    return Q.to(dtype), D.to(dtype)


def make_lengths():
    """Per-document lengths in [128, Ld]. Identical across runs."""
    g = torch.Generator(device='cpu').manual_seed(7)
    return torch.randint(128, Ld + 1, (B,), generator=g).to('cuda', dtype=torch.int32)


def fp32_reference(Q32, D32, d_lengths, q_lengths=None):
    """True FP32 reference: dense einsum + mask to -inf + max + sum."""
    # Force true FP32 matmul (no TF32) for the reference.
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        sim = torch.einsum("nqd,bld->nbql", Q32, D32)  # [N, B, Lq, Ld]
        # Doc-side mask
        ar_d = torch.arange(Ld, device=Q32.device)
        d_keep = ar_d[None, :] < d_lengths[:, None]  # [B, Ld]
        sim = sim.masked_fill(~d_keep[None, :, None, :], float("-inf"))
        # Query-side mask
        m = sim.max(dim=3).values  # [N, B, Lq]
        if q_lengths is not None:
            ar_q = torch.arange(Lq, device=Q32.device)
            q_keep = ar_q[None, :] < q_lengths[:, None]  # [N, Lq]
            m = m.masked_fill(~q_keep[:, None, :], 0.0)
        scores = m.sum(dim=2)  # [N, B]
        return scores
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def err(a, ref):
    """Return (max|abs|, mean|abs|) of a vs ref, both as float32."""
    diff = (a.float() - ref.float()).abs()
    return float(diff.max()), float(diff.mean())


# Reference computed in true FP32 from the fp32 originals
Q32, D32 = make(torch.float32, seed=0)
d_lens = make_lengths()

print(f"Shape: B={B} Nq={B} Lq={Lq} Ld={Ld} d={d}")
print(f"doc_lengths sample: {d_lens.tolist()}\n")

ref_masked = fp32_reference(Q32, D32, d_lens)
ref_unmasked = fp32_reference(Q32, D32, torch.full((B,), Ld, dtype=torch.int32, device='cuda'))

results = []

def trial(label, Q, D, d_lens_arg, q_lens_arg=None, ref=ref_masked):
    """Call flash and compare to the requested reference."""
    out = flash_maxsim_batched_train(
        Q, D, shared_docs=True, doc_lengths=d_lens_arg, query_lengths=q_lens_arg,
    )
    mx, mn = err(out, ref)
    print(f"  {label:55s}  max|Δ|={mx:.3e}  mean|Δ|={mn:.3e}")
    results.append({"label": label, "max_abs": mx, "mean_abs": mn})


print("=== Four-corner baseline (no manual repair) ===")
trial("fp16, unmasked  (d_lens=full)",
      *make(torch.float16, seed=0),
      torch.full((B,), Ld, dtype=torch.int32, device='cuda'), ref=ref_unmasked)
trial("fp16,   masked",
      *make(torch.float16, seed=0), d_lens)
trial("bf16, unmasked  (d_lens=full)",
      *make(torch.bfloat16, seed=0),
      torch.full((B,), Ld, dtype=torch.int32, device='cuda'), ref=ref_unmasked)
trial("bf16,   masked  <-- raphaelsty's bug",
      *make(torch.bfloat16, seed=0), d_lens)

print("\n=== Probes localizing the bug ===")
# Probe A: pre-mask D in user dtype outside flash (zero out padded doc positions)
Qb, Db = make(torch.bfloat16, seed=0)
ar = torch.arange(Ld, device='cuda')
d_keep = (ar[None, :] < d_lens[:, None]).to(Db.dtype)
Db_premask = Db * d_keep.unsqueeze(-1)
trial("probe_A: bf16, D pre-masked outside flash, d_lens=full",
      Qb, Db_premask, torch.full((B,), Ld, dtype=torch.int32, device='cuda'))

# Probe B: explicit .half() cast by caller, then flash with the original d_lens
Qb, Db = make(torch.bfloat16, seed=0)
trial("probe_B: caller-side .half() cast, then masked flash",
      Qb.half(), Db.half(), d_lens)

# Probe C: caller-side .float() cast, then masked flash
Qb, Db = make(torch.bfloat16, seed=0)
trial("probe_C: caller-side .float() cast, then masked flash",
      Qb.float(), Db.float(), d_lens)

print()
gpu = torch.cuda.get_device_name()
p = os.path.join(HERE, f"bench_bf16_masked_bisect_{gpu.replace(' ', '_').replace('/', '_')}.json")
json.dump({"gpu": gpu, "shape": {"B": B, "Lq": Lq, "Ld": Ld, "d": d},
           "results": results}, open(p, "w"), indent=2)
print(f"wrote {p}")
