import pytest
import torch
import torch.nn.functional as F


def _sim(B, Lq, Ld, d):
    Q = F.normalize(torch.randn(Lq, d, device="cuda", dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(B, Ld, d, device="cuda", dtype=torch.float16), dim=-1)
    return Q, D


@pytest.fixture(autouse=True)
def skip_no_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")


def test_correctness():
    from flash_maxsim import flash_maxsim, maxsim_naive
    Q, D = _sim(50, 32, 300, 128)
    assert torch.allclose(maxsim_naive(Q, D), flash_maxsim(Q, D), atol=1.0)


def test_variable_lengths():
    from flash_maxsim import flash_maxsim, maxsim_naive
    Q, D = _sim(20, 32, 256, 128)
    lengths = torch.randint(64, 256, (20,), device="cuda", dtype=torch.int32)
    ref = maxsim_naive(Q, D, doc_lengths=lengths)
    out = flash_maxsim(Q, D, doc_lengths=lengths)
    assert torch.allclose(ref, out, atol=1.0)


def test_batched():
    from flash_maxsim import flash_maxsim_batched
    Q = F.normalize(torch.randn(5, 32, 128, device="cuda", dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(20, 300, 128, device="cuda", dtype=torch.float16), dim=-1)
    scores = flash_maxsim_batched(Q, D, shared_docs=True)
    assert scores.shape == (5, 20)


def test_int8():
    from flash_maxsim import flash_maxsim_int8, quantize_int8
    Q, D = _sim(50, 32, 300, 128)
    D_q, s, m = quantize_int8(D)
    scores = flash_maxsim_int8(Q, D_q, s, m)
    assert scores.shape == (50,)


def test_train():
    from flash_maxsim import flash_maxsim_train
    Q = F.normalize(torch.randn(16, 128, device="cuda", dtype=torch.float16), dim=-1).requires_grad_(True)
    D = F.normalize(torch.randn(10, 64, 128, device="cuda", dtype=torch.float16), dim=-1)
    s = flash_maxsim_train(Q, D)
    s.sum().backward()
    assert Q.grad is not None


def test_chamfer_backward():
    P = torch.randn(100, 3, device="cuda", requires_grad=True)
    Q = torch.randn(200, 3, device="cuda", requires_grad=True)
    from flash_maxsim.flash_chamfer import flash_chamfer_train
    cd = flash_chamfer_train(P, Q)
    cd.backward()
    assert P.grad is not None and Q.grad is not None


def test_varlen_correctness():
    from flash_maxsim import flash_maxsim, maxsim_naive
    from flash_maxsim.flash_maxsim_varlen import flash_maxsim_varlen, pack_pairs
    # Variable-length pairs
    q_embs = [
        F.normalize(torch.randn(8, 128, device="cuda", dtype=torch.float16), dim=-1),
        F.normalize(torch.randn(32, 128, device="cuda", dtype=torch.float16), dim=-1),
        F.normalize(torch.randn(16, 128, device="cuda", dtype=torch.float16), dim=-1),
    ]
    d_embs = [
        F.normalize(torch.randn(300, 128, device="cuda", dtype=torch.float16), dim=-1),
        F.normalize(torch.randn(100, 128, device="cuda", dtype=torch.float16), dim=-1),
        F.normalize(torch.randn(500, 128, device="cuda", dtype=torch.float16), dim=-1),
    ]
    # Reference: per-pair naive
    ref = torch.stack([maxsim_naive(q, d.unsqueeze(0)).squeeze(0) for q, d in zip(q_embs, d_embs)])
    # Varlen
    Q_pk, D_pk, cu_q, cu_d, mlq, mld = pack_pairs(q_embs, d_embs)
    out = flash_maxsim_varlen(Q_pk, D_pk, cu_q, cu_d, mlq, mld)
    assert torch.allclose(ref, out, atol=1.0), f"varlen err={((ref-out).abs().max().item()):.4f}"


def test_varlen_uniform():
    from flash_maxsim import flash_maxsim
    from flash_maxsim.flash_maxsim_varlen import flash_maxsim_varlen, pack_pairs
    # Uniform lengths — should match flash_maxsim
    Q = F.normalize(torch.randn(32, 128, device="cuda", dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(50, 300, 128, device="cuda", dtype=torch.float16), dim=-1)
    ref = flash_maxsim(Q, D)
    # Pack as pairs (same Q, different D)
    q_embs = [Q] * 50
    d_embs = [D[i] for i in range(50)]
    Q_pk, D_pk, cu_q, cu_d, mlq, mld = pack_pairs(q_embs, d_embs)
    out = flash_maxsim_varlen(Q_pk, D_pk, cu_q, cu_d, mlq, mld)
    assert torch.allclose(ref, out, atol=1.0), f"varlen uniform err={((ref-out).abs().max().item()):.4f}"


@pytest.mark.parametrize("d", [256, 512, 1024, 2048])
def test_large_embedding_dim(d):
    from flash_maxsim import flash_maxsim, maxsim_naive
    Q, D = _sim(20, 32, 128, d)
    ref = maxsim_naive(Q, D)
    out = flash_maxsim(Q, D)
    assert torch.allclose(ref, out, atol=1.0), f"d={d}: max err={((ref-out).abs().max().item()):.4f}"


@pytest.mark.parametrize("keep", [0.25, 0.5, 0.75])
def test_sparse_maxsim(keep):
    from flash_maxsim import flash_maxsim_sparse, maxsim_sparse_naive
    Q, D = _sim(50, 32, 300, 128)
    K = max(1, int(32 * keep))
    importance = torch.randn(50, 32, device="cuda")
    indices = importance.topk(K, dim=1).indices.to(torch.int32)
    ref = maxsim_sparse_naive(Q, D, indices)
    out = flash_maxsim_sparse(Q, D, indices)
    assert torch.allclose(ref, out, atol=1.0), f"keep={keep}: max err={((ref-out).abs().max().item()):.4f}"


@pytest.mark.parametrize("d", [256, 512, 1024])
def test_large_dim_int8(d):
    from flash_maxsim import flash_maxsim_int8, flash_maxsim, quantize_int8
    Q, D = _sim(20, 32, 128, d)
    D_q, s, m = quantize_int8(D)
    scores_fp = flash_maxsim(Q, D)
    scores_q8 = flash_maxsim_int8(Q, D_q, s, m)
    # INT8 has quantization error, check top-1 matches
    assert scores_fp.argmax() == scores_q8.argmax(), f"d={d}: top-1 mismatch fp={scores_fp.argmax().item()} q8={scores_q8.argmax().item()}"
