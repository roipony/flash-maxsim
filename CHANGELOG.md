# Changelog

All notable changes to flash-maxsim. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.3.0] — 2026-06-11

### Added
- **Split-d forward** (`flash_maxsim_splitd`) — tiles the embedding
  dimension for d > 512, removing the SRAM-spill latency cliff
  measured on A100 and H100. Auto-dispatched from the public API at
  d > 512; fat-embedding encoders (Jina v2 d=512, Voyage v2 d=1024,
  NV-Embed d=4096) now run without the 3–18x cliff.
- **Unified backward kernel** — fused dQ+dD in a single launch with
  Q register hoisting; one D load serves both gradients. Dispatcher
  default for in-batch-negatives contrastive training; the inverse-grid
  CSR path remains available via `FLASH_BWD_PATH=invgrid` for
  bitwise-stable scatter.
- **`flash_maxsim.warmup()`** — pre-compiles every kernel
  specialization the dispatcher can select, so the first request
  pays no Triton JIT cost. CLI: `python -m flash_maxsim.warmup`.
- **Dense-pairs fast path** (`flash_maxsim_pairs`) — tensor inputs or
  equal-shape lists route to a grid-per-pair kernel, skipping the
  cu_seqlens packing tax; ragged lists fall through to varlen.
- **`flash_maxsim_rerank_padded`** — batched padded rerank with
  `score_candidates_padded`-compatible signature.
- **INT8-D training** (`flash_maxsim_int8_batched_train`) — saves D
  as INT8 in the autograd context; grad_Q cosine 0.999979 / grad_D
  cosine 1.000000 vs the FP16-saved reference.
- **Per-arch heuristic launch table** replaces Triton autotune —
  deterministic launch config every call (CUDA-graph friendly), zero
  first-call trial overhead.
- **int32 CSR build** for the inverse-grid backward — 25–35% lower
  CSR transient memory at ColPali B=128.
- Benchmark suite expansion: chunked-FP16-eager production baseline
  (`bench_chunked_fp16_eager.py`), dispersion/IQR audit
  (`bench_dispersion.py`), corpus-scale BEIR parity
  (`bench_beir_ndcg.py`: HotpotQA-500K identical nDCG@10 to the FP32
  reference; NQ-2.68M scored out-of-core), fat-d probe, split-d
  sweeps, INT8 suite, backward component breakdown, multi-query
  serving sweep.

### Honest non-wins documented (so users don't adopt them expecting
something we measured doesn't happen)
- INT8-D training (Phase 1 and Phase 2 of the kernel-side dequant)
  does not reduce peak GPU memory in typical training loops because
  the user's original FP16 D stays alive throughout. The feature
  ships for gradient-quality validation, the quantize-time memory
  fix, and niche use cases (custom training loops that release the
  FP16 D between forward and backward). See
  `flash_maxsim/flash_maxsim_int8_train.py` module docstring and
  `benchmarks/bench_int8_train_mem.py` for the measurement.

## [0.2.1] — 2026-04-23

Released to PyPI. Headline: pylate PR #212 (flash backend), training-CSR
backward, INT8×INT8.