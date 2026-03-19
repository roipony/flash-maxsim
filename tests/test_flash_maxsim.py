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
