"""Empirical correctness check for the invgrid backward proposal.

Compares the gradients produced by:
    (a) flash_maxsim_invgrid_train  (the new atomic-free dD path under test)
    (b) flash_maxsim_batched_train  (current FP32-atomic backward — assumed-correct baseline)
    (c) FP32 PyTorch reference (TF32 disabled)

Verdict per shape uses cosine similarity > 0.999 vs the FP32 reference for both
grad_Q and grad_D.  Also reports max abs error and the (a)-vs-(b) gap.

Run on a single GPU.
"""
from __future__ import annotations

import os
import sys
import time
import math
import traceback

import torch
import torch.nn.functional as F

sys.path.insert(0, "/path/to/data")

# Hard-disable TF32 — the FP32 reference must be the *true* FP32 result.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
try:
    torch.set_float32_matmul_precision("highest")
except Exception:
    pass

from flash_maxsim.flash_maxsim_batched_train import flash_maxsim_batched_train
from flash_maxsim.flash_maxsim_invgrid_proposal import flash_maxsim_invgrid_train

DEVICE = "cuda"
DTYPE = torch.float16


# ---------------------------------------------------------------------------
# Reference (FP32 ground truth).  einsum + max + sum, TF32 OFF.
# ---------------------------------------------------------------------------
def maxsim_ref_fp32(Q: torch.Tensor, D: torch.Tensor, shared_docs: bool) -> torch.Tensor:
    """FP32 reference scores.

    shared_docs=True : Q[Nq, Lq, d], D[B, Ld, d]    -> S[Nq, B]
    shared_docs=False: Q[Nq, Lq, d], D[Nq, B, Ld, d] -> S[Nq, B]
    """
    Qf = Q.float()
    Df = D.float()
    if shared_docs:
        # [Nq, B, Lq, Ld]
        sim = torch.einsum("ash,bth->abst", Qf, Df)
    else:
        # [Nq, B, Lq, Ld]
        sim = torch.einsum("ash,abth->abst", Qf, Df)
    return sim.max(-1).values.sum(-1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    na = a.norm()
    nb = b.norm()
    if na.item() == 0 and nb.item() == 0:
        return 1.0
    if na.item() == 0 or nb.item() == 0:
        return float("nan")
    return torch.dot(a, b).item() / (na.item() * nb.item())


def maxabs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def make_inputs(Nq, B, Lq, Ld, d, shared_docs, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    Q = F.normalize(torch.randn(Nq, Lq, d, generator=g, device=DEVICE, dtype=torch.float32), dim=-1)
    if shared_docs:
        D = F.normalize(torch.randn(B, Ld, d, generator=g, device=DEVICE, dtype=torch.float32), dim=-1)
    else:
        D = F.normalize(torch.randn(Nq, B, Ld, d, generator=g, device=DEVICE, dtype=torch.float32), dim=-1)
    return Q.to(DTYPE), D.to(DTYPE)


def clone_grad(*tensors):
    out = []
    for t in tensors:
        c = t.detach().clone().requires_grad_(True)
        out.append(c)
    return out


def grads_for(impl, Q, D, shared_docs, doc_lengths=None, query_lengths=None, fp32_ref=False):
    """Returns (gQ, gD) using `loss = S.diagonal().sum()`."""
    Qc, Dc = clone_grad(Q, D)
    if fp32_ref:
        # FP32 reference: cast to fp32, run pure pytorch.
        Qc_f = Qc.float()
        Dc_f = Dc.float()
        # We need grad on the original FP16 tensors for fair comparison with kernels.
        # Simpler: do everything in FP32 right from the leaves.
        Qf = Q.detach().float().requires_grad_(True)
        Df = D.detach().float().requires_grad_(True)
        S = maxsim_ref_fp32(Qf, Df, shared_docs)
        # NOTE: variable-length is not modelled in the reference's einsum unless we
        # mask.  Apply masks if provided.
        if doc_lengths is not None or query_lengths is not None:
            S = _masked_ref(Qf, Df, shared_docs, doc_lengths, query_lengths)
        loss = S.diagonal().sum()
        loss.backward()
        return Qf.grad.detach(), Df.grad.detach(), S.detach()
    else:
        if doc_lengths is None and query_lengths is None:
            S = impl(Qc, Dc, shared_docs=shared_docs)
        else:
            S = impl(Qc, Dc, shared_docs=shared_docs,
                     doc_lengths=doc_lengths, query_lengths=query_lengths)
        loss = S.diagonal().sum()
        loss.backward()
        return Qc.grad.detach(), Dc.grad.detach(), S.detach()


def _masked_ref(Q, D, shared_docs, doc_lengths, query_lengths):
    """FP32 reference that respects per-doc and per-query lengths via -inf masking."""
    Nq, Lq, d = Q.shape
    if shared_docs:
        B = D.shape[0]
        Ld = D.shape[1]
        sim = torch.einsum("ash,bth->abst", Q, D)
    else:
        B = D.shape[1]
        Ld = D.shape[2]
        sim = torch.einsum("ash,abth->abst", Q, D)

    if doc_lengths is not None:
        dl = doc_lengths.to(DEVICE).long()
        # dl shape: [B] (shared) or [Nq*B] reshaped to [Nq, B]
        if shared_docs:
            assert dl.numel() == B
            valid_d = (torch.arange(Ld, device=DEVICE)[None, :] < dl[:, None])  # [B, Ld]
            mask_d = valid_d[None, :, None, :].expand(Nq, B, Lq, Ld)
        else:
            assert dl.numel() == Nq * B
            dl = dl.view(Nq, B)
            valid_d = (torch.arange(Ld, device=DEVICE)[None, None, :] < dl[:, :, None])  # [Nq, B, Ld]
            mask_d = valid_d[:, :, None, :].expand(Nq, B, Lq, Ld)
        sim = sim.masked_fill(~mask_d, float("-inf"))

    # max over Ld
    mx = sim.max(-1).values  # [Nq, B, Lq]
    # If a row has all -inf (zero-length doc), max is -inf — clamp to 0.
    mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))

    if query_lengths is not None:
        ql = query_lengths.to(DEVICE).long()
        assert ql.numel() == Nq
        valid_q = (torch.arange(Lq, device=DEVICE)[None, :] < ql[:, None])  # [Nq, Lq]
        mx = mx * valid_q[:, None, :].to(mx.dtype)

    return mx.sum(-1)  # [Nq, B]


# ---------------------------------------------------------------------------
# Per-shape evaluator.
# ---------------------------------------------------------------------------
def evaluate_shape(name, Nq, B, Lq, Ld, d, shared_docs=True, seed=0,
                   doc_lengths=None, query_lengths=None):
    Q, D = make_inputs(Nq, B, Lq, Ld, d, shared_docs, seed=seed)
    try:
        gQ_inv, gD_inv, S_inv = grads_for(flash_maxsim_invgrid_train, Q, D, shared_docs,
                                          doc_lengths=doc_lengths, query_lengths=query_lengths)
        gQ_atm, gD_atm, S_atm = grads_for(flash_maxsim_batched_train, Q, D, shared_docs,
                                          doc_lengths=doc_lengths, query_lengths=query_lengths)
        gQ_ref, gD_ref, S_ref = grads_for(None, Q, D, shared_docs,
                                          doc_lengths=doc_lengths, query_lengths=query_lengths,
                                          fp32_ref=True)
    except Exception as e:
        print(f"  [{name}] EXCEPTION while running: {e}")
        traceback.print_exc()
        return None

    res = {
        "name": name,
        "shape": (Nq, B, Lq, Ld, d),
        "shared_docs": shared_docs,
        "S_cos_inv_vs_ref": cos(S_inv, S_ref),
        "S_max_inv_vs_ref": maxabs(S_inv, S_ref),
        "gQ_cos_inv_vs_ref": cos(gQ_inv, gQ_ref),
        "gD_cos_inv_vs_ref": cos(gD_inv, gD_ref),
        "gQ_cos_atm_vs_ref": cos(gQ_atm, gQ_ref),
        "gD_cos_atm_vs_ref": cos(gD_atm, gD_ref),
        "gQ_max_inv_vs_atm": maxabs(gQ_inv, gQ_atm),
        "gD_max_inv_vs_atm": maxabs(gD_inv, gD_atm),
        "gQ_max_inv_vs_ref": maxabs(gQ_inv, gQ_ref),
        "gD_max_inv_vs_ref": maxabs(gD_inv, gD_ref),
    }
    return res


def fmt_row(r):
    return (f"  {r['name']:<32}  shared={str(r['shared_docs'])[:1]}  "
            f"gQ_cos={r['gQ_cos_inv_vs_ref']:.5f}  gD_cos={r['gD_cos_inv_vs_ref']:.5f}  "
            f"(atm: gQ={r['gQ_cos_atm_vs_ref']:.5f} gD={r['gD_cos_atm_vs_ref']:.5f})  "
            f"max|inv-atm|: gQ={r['gQ_max_inv_vs_atm']:.2e} gD={r['gD_max_inv_vs_atm']:.2e}")


def verdict(r, threshold=0.999):
    return (r["gQ_cos_inv_vs_ref"] > threshold) and (r["gD_cos_inv_vs_ref"] > threshold)


# ---------------------------------------------------------------------------
# Test sequence.
# ---------------------------------------------------------------------------
def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch={torch.__version__}, cuda={torch.version.cuda}")
    print(f"TF32 matmul: {torch.backends.cuda.matmul.allow_tf32}")
    print()

    # ------------------------------------------------------------
    # 1) Reproduce the file's __main__ check first.
    # ------------------------------------------------------------
    print("=" * 100)
    print(" 1) Reproduce file's own __main__ smoke (Nq=4 B=8 Lq=16 Ld=24 d=64)")
    print("=" * 100)
    for sd in [True, False]:
        r = evaluate_shape(f"smoke shared_docs={sd}", 4, 8, 16, 24, 64, shared_docs=sd, seed=0)
        if r is not None:
            print(fmt_row(r))
    print()

    # ------------------------------------------------------------
    # 2) Shape sweep, shared_docs=True
    # ------------------------------------------------------------
    print("=" * 100)
    print(" 2) Shape sweep — shared_docs=True (cosine vs FP32 ref must be > 0.999)")
    print("=" * 100)
    shapes_shared = [
        ("(1,1,1,1,128) degenerate",       1,   1, 1,   1, 128),
        ("(1,1,32,180,128) single q",      1,   1, 32, 180, 128),
        ("(8,8,32,180,128) small",         8,   8, 32, 180, 128),
        ("(64,64,32,180,128) typical",     64, 64, 32, 180, 128),
        ("(128,128,32,180,128) bigger",    128,128,32, 180, 128),
        ("(16,16,1024,1024,128) ColPaliS", 16, 16, 1024, 1024, 128),
        ("(32,32,1024,1024,128) ColPali",  32, 32, 1024, 1024, 128),
    ]
    results = []
    for tup in shapes_shared:
        name, Nq, B, Lq, Ld, d = tup
        r = evaluate_shape(name, Nq, B, Lq, Ld, d, shared_docs=True, seed=0)
        if r is not None:
            print(fmt_row(r))
            results.append(r)
    print()

    # ------------------------------------------------------------
    # 3) Nq != B (KD-like, but still shared_docs=True)
    # ------------------------------------------------------------
    print("=" * 100)
    print(" 3) Nq != B (shared_docs=True): (Nq=4, B=8, Lq=32, Ld=180, d=128)")
    print("=" * 100)
    r = evaluate_shape("Nq!=B shared", 4, 8, 32, 180, 128, shared_docs=True, seed=0)
    if r is not None:
        print(fmt_row(r))
        results.append(r)
    print()

    # ------------------------------------------------------------
    # 4) shared_docs=False (KD)
    # ------------------------------------------------------------
    print("=" * 100)
    print(" 4) shared_docs=False (KD): (Nq=8, B=16, Lq=32, Ld=180, d=128)")
    print("=" * 100)
    r = evaluate_shape("KD non-shared", 8, 16, 32, 180, 128, shared_docs=False, seed=0)
    if r is not None:
        print(fmt_row(r))
        results.append(r)
    # also a few small ones
    for tup in [("KD small (4,4,16,24,64)", 4, 4, 16, 24, 64),
                ("KD medium (8,8,32,180,128)", 8, 8, 32, 180, 128)]:
        name, Nq, B, Lq, Ld, d = tup
        r = evaluate_shape(name, Nq, B, Lq, Ld, d, shared_docs=False, seed=0)
        if r is not None:
            print(fmt_row(r))
            results.append(r)
    print()

    # ------------------------------------------------------------
    # 5) Variable-length doc/query lengths
    # ------------------------------------------------------------
    print("=" * 100)
    print(" 5) Variable-length doc_lengths + query_lengths (shared_docs=True)")
    print("=" * 100)
    Nq, B, Lq, Ld, d = 8, 8, 32, 180, 128
    g = torch.Generator(device=DEVICE).manual_seed(123)
    doc_lens = torch.randint(low=max(1, Ld // 4), high=Ld + 1, size=(B,), generator=g, device=DEVICE, dtype=torch.int32)
    query_lens = torch.randint(low=max(1, Lq // 4), high=Lq + 1, size=(Nq,), generator=g, device=DEVICE, dtype=torch.int32)
    print(f"  doc_lens={doc_lens.tolist()}")
    print(f"  query_lens={query_lens.tolist()}")
    r = evaluate_shape("varlen doc+query (shared)", Nq, B, Lq, Ld, d, shared_docs=True,
                       seed=0, doc_lengths=doc_lens, query_lengths=query_lens)
    if r is not None:
        print(fmt_row(r))
        results.append(r)

    # also non-shared varlen
    Nq, B, Lq, Ld, d = 4, 8, 32, 180, 128
    g = torch.Generator(device=DEVICE).manual_seed(124)
    doc_lens = torch.randint(low=max(1, Ld // 4), high=Ld + 1, size=(Nq * B,), generator=g, device=DEVICE, dtype=torch.int32)
    query_lens = torch.randint(low=max(1, Lq // 4), high=Lq + 1, size=(Nq,), generator=g, device=DEVICE, dtype=torch.int32)
    r = evaluate_shape("varlen doc+query (KD)", Nq, B, Lq, Ld, d, shared_docs=False,
                       seed=0, doc_lengths=doc_lens, query_lengths=query_lens)
    if r is not None:
        print(fmt_row(r))
        results.append(r)
    print()

    # ------------------------------------------------------------
    # 6) Hammer test: 100 seeds at one fixed shape.
    # ------------------------------------------------------------
    print("=" * 100)
    print(" 6) Hammer test — 100 random seeds at (Nq=8, B=8, Lq=32, Ld=180, d=128) shared_docs=True")
    print("=" * 100)
    Nq, B, Lq, Ld, d = 8, 8, 32, 180, 128
    failures = []
    min_gQ = 1.0
    min_gD = 1.0
    for seed in range(100):
        r = evaluate_shape(f"hammer seed={seed}", Nq, B, Lq, Ld, d, shared_docs=True, seed=seed)
        if r is None:
            failures.append((seed, "exception"))
            continue
        min_gQ = min(min_gQ, r["gQ_cos_inv_vs_ref"])
        min_gD = min(min_gD, r["gD_cos_inv_vs_ref"])
        if not verdict(r):
            failures.append((seed, r))
    print(f"  hammer: 100 seeds, min gQ_cos={min_gQ:.6f}, min gD_cos={min_gD:.6f}, failures={len(failures)}")
    if failures:
        for seed, info in failures[:5]:
            print(f"    FAIL seed={seed}: {info}")
    print()

    # ------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------
    print("=" * 100)
    print(" SUMMARY TABLE (cosine sim vs FP32 reference; verdict at threshold 0.999)")
    print("=" * 100)
    print(f"  {'shape':<32}  {'shared':>6}  {'gQ_cos':>10}  {'gD_cos':>10}  {'verdict':>10}")
    print("  " + "-" * 80)
    any_fail = False
    for r in results:
        ok = verdict(r)
        any_fail = any_fail or (not ok)
        print(f"  {r['name']:<32}  {str(r['shared_docs'])[:1]:>6}  "
              f"{r['gQ_cos_inv_vs_ref']:>10.6f}  {r['gD_cos_inv_vs_ref']:>10.6f}  "
              f"{('PASS' if ok else 'FAIL'):>10}")
    print()
    print(f"VERDICT: {'BROKEN — see failures above' if any_fail or failures else 'ALL CORRECT'}")


if __name__ == "__main__":
    main()
