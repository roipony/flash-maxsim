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


@pytest.mark.parametrize("d", [256, 512, 1024, 2048])
def test_large_embedding_dim(d):
    from flash_maxsim import flash_maxsim, maxsim_naive
    Q, D = _sim(20, 32, 128, d)
    ref = maxsim_naive(Q, D)
    out = flash_maxsim(Q, D)
    assert torch.allclose(ref, out, atol=1.0), f"d={d}: max err={((ref-out).abs().max().item()):.4f}"


@pytest.mark.parametrize("keep", [0.25, 0.5, 0.75])
def test_sparse_maxsim(keep):
    from flash_maxsim import flash_maxsim, flash_maxsim_sparse
    Q, D = _sim(50, 32, 300, 128)
    k = max(1, int(32 * keep))
    # Uniform query_lengths: same as dense with truncated Q
    qlens = torch.full((50,), k, device="cuda", dtype=torch.int32)
    ref = flash_maxsim(Q[:k], D)
    out = flash_maxsim_sparse(Q, D, qlens)
    assert torch.allclose(ref, out, atol=0.01), f"keep={keep}: max err={((ref-out).abs().max().item()):.4f}"


def test_prepare_sparse():
    from flash_maxsim import flash_maxsim_sparse, prepare_sparse
    Q, D = _sim(50, 64, 300, 128)
    importance = torch.randn(50, 64, device="cuda")
    Q_sorted, qlens, order = prepare_sparse(Q, importance, keep_ratio=0.5)
    scores = flash_maxsim_sparse(Q_sorted, D, qlens)
    assert scores.shape == (50,)


@pytest.mark.parametrize("d", [256, 512, 1024])
def test_large_dim_int8(d):
    from flash_maxsim import flash_maxsim_int8, flash_maxsim, quantize_int8
    Q, D = _sim(20, 32, 128, d)
    D_q, s, m = quantize_int8(D)
    scores_fp = flash_maxsim(Q, D)
    scores_q8 = flash_maxsim_int8(Q, D_q, s, m)
    # INT8 has quantization error, just check rankings match for top-5
    fp_rank = scores_fp.argsort(descending=True)[:5]
    q8_rank = scores_q8.argsort(descending=True)[:5]
    assert (fp_rank == q8_rank).all(), f"d={d}: ranking mismatch"
