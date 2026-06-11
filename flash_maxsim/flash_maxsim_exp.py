"""Experimental: cache-aware heuristic kernel + chunked streaming.

Inspired by Flash-KMeans (arxiv 2603.09229):
- FlashAssign uses a hand-picked config from L1/L2 cache + problem shape (no autotune)
- Chunked stream overlap: double-buffered H→D copies for out-of-core corpora
"""

import torch
import triton
import triton.language as tl

from .flash_maxsim import _next_pow2


# ---------------------------------------------------------------------------
# Cache-aware heuristic — picks (BLOCK_Q, BLOCK_D, num_warps, num_stages)
# ---------------------------------------------------------------------------

def _gpu_family():
    if not torch.cuda.is_available():
        return "generic"
    name = torch.cuda.get_device_name().lower()
    if "h200" in name or "h100" in name:
        return "hopper"
    if "a100" in name:
        return "a100"
    return "generic"


def pick_config(Lq: int, Ld: int, d: int, gpu: str = None):
    """Pick (BLOCK_Q, BLOCK_D, num_warps, num_stages) from problem shape + hw.

    Adapted from Flash-KMeans `_heuristic_euclid_config`
    (github.com/svg-project/flash-kmeans):
      Mapping K-means → MaxSim: N→Lq (points), K→Ld (centroids), D→d.
      K-means has one CTA per BLOCK_N points; we have one CTA per (q, doc) and
      chunk Lq internally, so BLOCK_Q caps at Lq (else it's wasted work).

    Their A100 heuristic for D=128 (our ColBERT/ColPali regime):
      BLOCK_N=128, BLOCK_K=32, stages=2, warps=4;  BLOCK_K=64 if N<=65536.
    Their H100/H200 add more per-shape dispatch; we follow their structure.
    """
    gpu = gpu or _gpu_family()

    # Round Lq to its bucket (matches _round_up_lq behavior).
    Lq_eff = Lq
    for b in (32, 64, 128, 256, 512, 1024):
        if Lq <= b:
            Lq_eff = b
            break

    # ---- Large-d: SMEM-constrained, cap tiles hard. ----
    if d >= 512:
        return {"BLOCK_Q": min(16, Lq_eff), "BLOCK_D": 16,
                "num_warps": 2, "num_stages": 2}
    if d >= 256:
        return {"BLOCK_Q": min(32, Lq_eff), "BLOCK_D": 32,
                "num_warps": 4, "num_stages": 2}

    # ---- d <= 128: their main regime. Keep their BLOCK_K=64 default (D=128). ----
    bd = 64
    warps = 4
    stages = 2

    # Hopper prefers larger BLOCK_K for large corpora; mirror their H100 dispatch.
    if gpu == "hopper":
        if Ld >= 4096:
            bd = 128
            warps = 8
            stages = 2
        elif Ld >= 1024:
            bd = 64
            warps = 4
            stages = 2
        else:
            bd = 64
            warps = 4
            stages = 2

    # A100 follows the simpler branch: BLOCK_K=64 for D=128, N<=65536.
    # (For our regime B is always the effective "N" outer dim and we already
    # parallelize over it via grid — per-CTA we just need BLOCK_Q vs Lq.)

    # BLOCK_Q: their BLOCK_N=128 default; cap at Lq_eff so we don't waste.
    bq = min(128, Lq_eff)
    # Lq-small corner: tiny Lq means low warp utilization; drop warps.
    if bq <= 16:
        warps = 2
    elif bq <= 32:
        warps = 4

    return {"BLOCK_Q": bq, "BLOCK_D": bd, "num_warps": warps, "num_stages": stages}


# ---------------------------------------------------------------------------
# Heuristic-driven kernel: same body as _maxsim_fwd_kernel but configured
# directly at launch (no @triton.autotune), so cold start compiles ONE kernel.
# ---------------------------------------------------------------------------

@triton.jit
def _maxsim_heuristic_kernel(
    Q_ptr, D_ptr, lens_ptr, scores_ptr,
    Nq: tl.constexpr, B: tl.constexpr,
    Lq: tl.constexpr, Ld: tl.constexpr, d: tl.constexpr, d_pad: tl.constexpr,
    stride_q_n, stride_q_l, stride_q_d,
    stride_d_b, stride_d_l, stride_d_d,
    stride_s_n, stride_s_b,
    BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    q_idx = pid // B
    doc_idx = pid % B
    if q_idx >= Nq:
        return

    doc_len = tl.load(lens_ptr + doc_idx).to(tl.int32)

    k_off = tl.arange(0, d_pad)
    k_mask = k_off < d
    score_acc = tl.zeros([], dtype=tl.float32)

    for q_start in tl.static_range(0, Lq, BLOCK_Q):
        q_off = q_start + tl.arange(0, BLOCK_Q)
        q_valid = q_off < Lq

        Q_block = tl.load(
            Q_ptr + q_idx * stride_q_n + q_off[:, None] * stride_q_l + k_off[None, :] * stride_q_d,
            mask=q_valid[:, None] & k_mask[None, :], other=0.0,
        ).to(tl.float16)

        m = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)

        for d_start in range(0, doc_len, BLOCK_D):
            d_off = d_start + tl.arange(0, BLOCK_D)
            d_valid = d_off < doc_len

            D_block = tl.load(
                D_ptr + doc_idx * stride_d_b + d_off[:, None] * stride_d_l + k_off[None, :] * stride_d_d,
                mask=d_valid[:, None] & k_mask[None, :], other=0.0,
            ).to(tl.float16)

            S = tl.dot(Q_block, tl.trans(D_block))
            S = tl.where(d_valid[None, :], S, float("-inf"))
            m = tl.maximum(m, tl.max(S, axis=1))

        m = tl.where(q_valid, m, 0.0)
        score_acc += tl.sum(m)

    tl.store(scores_ptr + q_idx * stride_s_n + doc_idx * stride_s_b, score_acc)


def flash_maxsim_heuristic(Q: torch.Tensor, D: torch.Tensor, doc_lengths=None) -> torch.Tensor:
    """MaxSim with heuristic-picked config (no autotune → fast cold start)."""
    assert Q.dim() == 2 and D.dim() == 3 and Q.shape[1] == D.shape[2]
    Lq, d = Q.shape
    B, Ld, _ = D.shape
    Q2 = Q.unsqueeze(0).contiguous().half()
    D2 = D.contiguous().half()

    d_pad = _next_pow2(d)
    lengths = (doc_lengths.to(torch.int32).contiguous() if doc_lengths is not None
               else torch.full((B,), Ld, device=D.device, dtype=torch.int32))
    scores = torch.empty(1, B, device=Q.device, dtype=torch.float32)

    cfg = pick_config(Lq, Ld, d)

    _maxsim_heuristic_kernel[(1 * B,)](
        Q2, D2, lengths, scores,
        1, B,
        Lq, Ld, d, d_pad,
        Q2.stride(0), Q2.stride(1), Q2.stride(2),
        D2.stride(0), D2.stride(1), D2.stride(2),
        scores.stride(0), scores.stride(1),
        BLOCK_Q=cfg["BLOCK_Q"], BLOCK_D=cfg["BLOCK_D"],
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
    )
    return scores.squeeze(0)


# ---------------------------------------------------------------------------
# Chunked streaming: double-buffered H→D copy overlapped with compute.
# Enables scoring corpora larger than VRAM.
# ---------------------------------------------------------------------------

# Module-level singleton copy stream — creating a new torch.cuda.Stream per
# call caused illegal-memory-access at interpreter teardown when used in a
# benchmark loop. Reusing one stream is cleaner and matches what Flash-KMeans
# does in kmeans_large.py.
_COPY_STREAM = {}

def _get_copy_stream(device):
    key = str(device)
    if key not in _COPY_STREAM:
        _COPY_STREAM[key] = torch.cuda.Stream(device=device)
    return _COPY_STREAM[key]


def flash_maxsim_streamed(Q: torch.Tensor, D_cpu: torch.Tensor,
                           chunk_size: int = 10_000,
                           scorer=None) -> torch.Tensor:
    """Score a CPU-resident corpus via double-buffered chunked streaming.

    Args:
        Q: [Lq, d] on CUDA.
        D_cpu: [B, Ld, d] CPU tensor (must be pinned for async copy).
        chunk_size: docs per chunk. Tune to fit in VRAM.
        scorer: function (Q, D_chunk) → [chunk] scores. Default flash_maxsim.

    Pipeline (two streams, ping-pong buffers):
        compute stream:  [compute ck-0 ][compute ck-1 ][compute ck-2 ]...
        copy stream:     [copy ck-0][copy ck-1][copy ck-2][copy ck-3]...
        (copy of ck-(i+1) overlaps with compute of ck-i)
    """
    from .flash_maxsim import flash_maxsim
    if scorer is None:
        scorer = flash_maxsim

    assert Q.is_cuda, "Q must be on CUDA"
    B, Ld, d = D_cpu.shape
    device = Q.device

    if chunk_size >= B:
        D_gpu = D_cpu.to(device, non_blocking=True)
        return scorer(Q, D_gpu)

    buf = [torch.empty(chunk_size, Ld, d, device=device, dtype=D_cpu.dtype),
           torch.empty(chunk_size, Ld, d, device=device, dtype=D_cpu.dtype)]
    out = torch.empty(B, device=device, dtype=torch.float32)

    compute_stream = torch.cuda.current_stream()
    copy_stream = _get_copy_stream(device)

    n_chunks = (B + chunk_size - 1) // chunk_size

    def slice_range(i):
        s = i * chunk_size
        e = min(s + chunk_size, B)
        return s, e

    # Pre-load chunk 0 into buf[0].
    s0, e0 = slice_range(0)
    with torch.cuda.stream(copy_stream):
        buf[0][: e0 - s0].copy_(D_cpu[s0:e0], non_blocking=True)
    copy_event = [copy_stream.record_event(), None]

    for i in range(n_chunks):
        curr_buf = i % 2
        next_buf = (i + 1) % 2
        s, e = slice_range(i)

        # Prefetch chunk i+1 into the other buffer on the copy stream.
        if i + 1 < n_chunks:
            ns, ne = slice_range(i + 1)
            # The copy into `next_buf` may overwrite data from TWO iters ago.
            # That earlier compute used `next_buf` at iter i-1, so wait for it.
            with torch.cuda.stream(copy_stream):
                if i >= 1:
                    copy_stream.wait_stream(compute_stream)
                buf[next_buf][: ne - ns].copy_(D_cpu[ns:ne], non_blocking=True)
            copy_event[next_buf] = copy_stream.record_event()

        # Compute stream must wait for our buffer's copy to complete.
        compute_stream.wait_event(copy_event[curr_buf])
        out[s:e] = scorer(Q, buf[curr_buf][: e - s])

    # Ensure the copy stream finishes any outstanding work before local
    # references go out of scope; otherwise late stream completions can
    # reference freed tensors and raise "illegal memory access" on teardown.
    compute_stream.wait_stream(copy_stream)
    copy_stream.synchronize()

    return out
