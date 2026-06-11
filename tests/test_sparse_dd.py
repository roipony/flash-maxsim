import os

import pytest
import torch
import torch.nn.functional as F

from flash_maxsim.flash_maxsim_batched_train import flash_maxsim_batched_train
from flash_maxsim.flash_maxsim_sparse_dd import build_sparse_dd_tables


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_inputs(Nq, B, Lq, Ld, d, shared_docs, dtype=torch.float16, seed=0):
    torch.manual_seed(seed)
    Q = F.normalize(torch.randn(Nq, Lq, d, device="cuda", dtype=dtype), dim=-1)
    if shared_docs:
        D = F.normalize(torch.randn(B, Ld, d, device="cuda", dtype=dtype), dim=-1)
    else:
        D = F.normalize(torch.randn(Nq, B, Ld, d, device="cuda", dtype=dtype), dim=-1)
    return Q, D


def _grads_for(path, Q, D, shared_docs, grad_scores, doc_lengths=None, query_lengths=None):
    old = os.environ.get("FLASH_BWD_PATH")
    os.environ["FLASH_BWD_PATH"] = path
    try:
        Qc = Q.detach().clone().requires_grad_(True)
        Dc = D.detach().clone().requires_grad_(True)
        scores = flash_maxsim_batched_train(
            Qc, Dc, shared_docs=shared_docs,
            doc_lengths=doc_lengths, query_lengths=query_lengths,
        )
        scores.backward(grad_scores)
        torch.cuda.synchronize()
        return scores.detach(), Qc.grad.detach(), Dc.grad.detach()
    finally:
        if old is None:
            os.environ.pop("FLASH_BWD_PATH", None)
        else:
            os.environ["FLASH_BWD_PATH"] = old


@pytest.mark.parametrize("shared_docs", [False, True])
def test_sparse_dd_matches_atomic(shared_docs):
    Nq, B, Lq, Ld, d = 4, 5, 7, 19, 32
    Q, D = _make_inputs(Nq, B, Lq, Ld, d, shared_docs, seed=11)
    grad_scores = torch.randn(Nq, B, device="cuda", dtype=torch.float32)

    s_ref, gq_ref, gd_ref = _grads_for("atomic", Q, D, shared_docs, grad_scores)
    s_sp, gq_sp, gd_sp = _grads_for("sparse_dd", Q, D, shared_docs, grad_scores)

    assert torch.allclose(s_sp, s_ref, atol=0, rtol=0)
    assert torch.allclose(gq_sp.float(), gq_ref.float(), atol=2e-3, rtol=2e-3)
    assert torch.allclose(gd_sp.float(), gd_ref.float(), atol=2e-3, rtol=2e-3)


def test_sparse_dd_matches_atomic_nonshared_varlen():
    Nq, B, Lq, Ld, d = 3, 4, 9, 23, 32
    Q, D = _make_inputs(Nq, B, Lq, Ld, d, shared_docs=False, seed=12)
    q_lens = torch.tensor([5, 7, 9], device="cuda", dtype=torch.int32)
    d_lens = torch.tensor(
        [17, 21, 15, 23, 20, 19, 13, 22, 18, 16, 23, 14],
        device="cuda", dtype=torch.int32,
    )
    pad = torch.arange(Lq, device="cuda")[None, :] >= q_lens[:, None]
    Q = torch.where(pad[..., None], torch.randn_like(Q), Q)
    grad_scores = torch.randn(Nq, B, device="cuda", dtype=torch.float32)

    s_ref, gq_ref, gd_ref = _grads_for(
        "atomic", Q, D, False, grad_scores, doc_lengths=d_lens, query_lengths=q_lens,
    )
    s_sp, gq_sp, gd_sp = _grads_for(
        "sparse_dd", Q, D, False, grad_scores, doc_lengths=d_lens, query_lengths=q_lens,
    )

    assert torch.allclose(s_sp, s_ref, atol=0, rtol=0)
    assert torch.allclose(gq_sp.float(), gq_ref.float(), atol=2e-3, rtol=2e-3)
    assert torch.allclose(gd_sp.float(), gd_ref.float(), atol=2e-3, rtol=2e-3)


def test_sparse_dd_table_count_matches_unique_destinations():
    Nq, B, Lq, Ld = 3, 2, 5, 11
    argmax = torch.tensor(
        [
            [1, 1, 2, 2, 2],
            [3, 4, 3, 4, 3],
            [0, 0, 0, 1, 1],
            [9, 9, 8, 8, 8],
            [6, 7, 7, 7, 6],
            [5, 5, 5, 5, 5],
        ],
        device="cuda", dtype=torch.int32,
    )
    _, _, seg_count, uniq_db, uniq_j = build_sparse_dd_tables(
        argmax, Nq, B, Lq, Ld, shared_docs=False,
    )
    got = set(zip(uniq_db.cpu().tolist(), uniq_j.cpu().tolist()))
    expected = set()
    for pair in range(Nq * B):
        for j in argmax[pair].cpu().tolist():
            expected.add((pair, j))
    assert got == expected
    assert int(seg_count.sum().item()) == Nq * B * Lq
