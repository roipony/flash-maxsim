# Paper-Numbers Audit Map

Every quantitative claim in `main.tex` is mapped here to its source benchmark
JSON (or analytical derivation). Reviewers can audit every cell by opening the
listed file at the listed key path.

**Where each JSON lives.** Most benchmark JSONs live in the implementation
repo at `github.com/roipony/flash-maxsim/benchmarks/` and are re-derivable from
the matching `benchmarks/bench_*.py` script. **Two end-to-end retrieval JSONs
ship in `data/` here rather than being re-derivable offline, because they were computed
from pre-existing IBM-internal embedding files (`/dccstor/.../colbandit_embeddings/`)
rather than from inputs the bench script downloads itself.
Specifically:

- `data/bench_beir_ndcg_arguana_NVIDIA_A100-SXM4-80GB.json` — ArguAna nDCG row of `tab:beir_ndcg` (ColBERTv2 embeddings prepared by `colbert_pac_prune`)
- `data/bench_beir_ndcg_scidocs_NVIDIA_A100-SXM4-80GB.json` — SciDocs row of `tab:beir_ndcg`

For these two cells, the bench script `benchmarks/bench_beir_ndcg.py` is the
exact code that produced them, but reproduction requires the upstream
embedding files; the JSONs are shipped here so reviewers can audit without
re-encoding.

Hardware: NVIDIA A100-SXM4-80GB (and -40GB where noted) and NVIDIA H100 80GB
HBM3. Software: PyTorch 2.8 + Triton 3.5/3.6.
`torch.set_float32_matmul_precision('high')` (TF32) unless stated. Casts
hoisted out of the timed region in all naive baselines.

## Reproduction

`bash scripts/reproduce_paper.sh` runs the **core six benchmarks** that back
`tab:fwd`, `tab:traffic`, `tab:ooc`, `tab:train`, `tab:varlen`, and
`fig:scaling`. The remaining tables (the 3-path backward ablation, INT8
sweeps, split-d sweeps, training-parity 500-step run, REAL-MM-RAG, BEIR)
are each backed by their own `benchmarks/bench_*.py` script listed per-table
below; the per-table source pointer is the authoritative reproduction
recipe.

---

## Abstract claims

| Claim | Value | Source |
|---|---|---|
| ColPali sim tensor (10K docs, FP16) | 21 GB | analytic: 10^4 · 1024² · 2 bytes = 20.97 GB |
| FP32 version | 42 GB | analytic: 10^4 · 1024² · 4 = 41.94 GB |
| Forward speedup, A100 | up to 3.9× | `bench_fwd_fair_NVIDIA_A100-SXM4-80GB.json:rows[4].spd_fp32_tf32 = 3.86` |
| Forward speedup, H100 | up to 4.7× | `bench_fwd_fair_NVIDIA_H100_80GB_HBM3.json:rows[4].spd_fp32_tf32 = 4.70` |
| Inference memory | up to 16× less | analytic from `tab:mem`: 47.2 / 2.9 GB = 16.3× (matched-FP32 at B=10K) |
| Training memory | ~130× less | `bench_train_paper_NVIDIA_A100-SXM4-80GB.json:rows[0].mem_ratio = 129.3` (B=64) |
| ~4× faster than `torch.compile(max-autotune)` on dense forward | 4.47× | `data/bench_compile_ma_audit_NVIDIA_A100-SXM4-80GB.json:rows[0].{flash.ms = 1.685, compile_ma.ms = 7.529}` (ColPali B=1K, `mode='max-autotune'` with CUDA graphs enabled, `dynamic=False`; protocol matches `bench_compile_ma_audit.py`) |
| Top-20 ranking | 100% | manual correctness check (FP32 reference); see §5.5 |

## §1 Introduction

| Claim | Value | Source |
|---|---|---|
| Sim tensor at ColPali 10K | 21 GB FP16, 42 GB FP32 | analytic (as above) |
| OOMs 40 GB GPU | yes | `bench_blog_40gb_NVIDIA_A100-SXM4-40GB.json:rows[1].naive_matched_fp32.error = "OutOfMemoryError"` (B=7K matched-FP32 OOM) |
| Forward contribution: up to 3.9× / 4.7× | as abstract | as abstract |
| INT8 Spearman ρ = 0.999 | VERIFIED: measured 0.9996 | `data/bench_correctness_NVIDIA_A100-SXM4-80GB.json:flash_int8.spearman_rho = 0.999591588973999` (ColPali shape; see §5.5 audit row below) |
| Varlen up to 5× | 4.59× (rounds to ~5×) | `bench_varlen_buckets_NVIDIA_A100-SXM4-80GB.json:buckets[2].speedup` |

## §3 Background & Motivation

| Claim | Value | Source |
|---|---|---|
| AI_naive ≈ 64 FLOPs/byte | derived | analytic (paper §3.2): 2·L_q·L_d·d / (4·L_q·L_d) = d/2 |
| H100 BF16 ridge ≈ 295 FLOPs/byte | vendor spec | NVIDIA H100 datasheet (publicly available) |
| AI_fused ≈ 512 FLOPs/byte | derived | analytic (paper §4.1.2): L_q·L_d / (L_q+L_d) ≈ 512 at L_q=L_d=1024 |

## §4 Method (Theorem & Proposition)

Theorem 1 (IO complexity): derivation in §4.1.2 proof sketch.
Proposition 1 (Exactness): derivation in §4.1 prose. Both are math, no JSON.

## Table tab:fwd — Forward latency (A100 ms; speedup on A100 + H100, B=1K)

| Shape | naive | flash | A100× | H100× | Source |
|---|---|---|---|---|---|
| textual (32,300) | 0.26 | 0.19 | 1.4× | 1.2× | `bench_fwd_fair_*:rows[0].{naive_fp32_tf32_ms, flash_ms, spd_fp32_tf32}` (A100 + H100) |
| long-doc (32,1024) | 0.59 | 0.30 | 2.0× | 1.8× | `rows[1]` |
| medium (128,1024) | 1.09 | 0.36 | 3.0× | 3.3× | `rows[2]` |
| visual (512,1024) | 3.31 | 0.94 | 3.5× | 4.2× | `rows[3]` |
| ColPali (1024²) | 6.58 | 1.70 | 3.9× | 4.7× | `rows[4]` |

## Table tab:traffic — HBM traffic at B=1K

| Shape | naive HBM | flash HBM | ratio | Source |
|---|---|---|---|---|
| medium | 1.31 GB | 0.26 GB | 5× | `bench_paper_io_*:traffic[0].{naive_GB, flash_GB, traffic_ratio}` |
| visual | 4.46 GB | 0.26 GB | 17× | `traffic[1]` |
| ColPali | 8.65 GB | 0.26 GB | 33× | `traffic[2]` |

Prose: "naive saturates ~1.3 TB/s" from `traffic[2].naive_achieved_GBs = 1297`.
ColPali compute floor 1.72 ms from `traffic[2].compute_floor_ms = 1.721` (analytic
2·B·L_q·L_d·d / TF32 peak).

## Table tab:mem — Corpus scaling at ColPali shape

| B | naive 40GB (FP16) | naive 80GB (FP32) | flash | Source |
|---|---|---|---|---|
| 10K | 23.9 GB | 47.2 GB | 2.9 GB | naive 80GB-FP32: `results_landing_NVIDIA_A100-SXM4-80GB.json:peak_memory[14].naive_gb = 47.19`; naive 40GB-FP16: `bench_blog_40gb_NVIDIA_A100-SXM4-40GB.json:rows[2].naive_fp16.peak_gb = 23.87`; flash: analytic doc-embedding bound B·L_d·d·2 + Q + overhead, also `bench_blog_40gb:rows[2].flash.peak_gb = 2.63` (≈2.9 with autotune scratch) |
| 20K | OOM | OOM | 5.2 GB | naive both: OOM (40 GB FP16 sim = 42 GB > 40; 80 GB FP32 sim = 84 GB > 80); flash: `bench_blog_40gb:rows[5].flash.peak_gb = 5.25` |
| 50K | OOM | OOM | 13.1 GB | flash: `bench_blog_40gb:rows[6].flash.peak_gb = 13.12` |
| 100K | OOM | OOM | 26.2 GB | flash: analytic 100K · 1024 · 128 · 2 = 26.21 GB; B=100K in single-process sweep OOM'd at make() due to fragmentation, but D-only fits |

## Table tab:train — Contrastive training step (ColPali in-batch-negatives, A100-80GB)

| B | naive ms | flash ms | naive peak | flash peak | mem ratio | Source |
|---|---|---|---|---|---|---|
| 64 | 81.6 | 12.3 | 51.7 GB | 0.40 GB | 129× | `bench_train_paper_NVIDIA_A100-SXM4-80GB.json:rows[0].{naive.ms, flash.ms, naive.peak_gb, flash.peak_gb, mem_ratio}` |
| 128 | OOM | 48.6 | OOM | 1.43 GB | unlocked | `rows[1].{naive.error="OutOfMemoryError", flash.ms, flash.peak_gb}` |

Prose "129× memory reduction": `rows[0].mem_ratio = 129.3` (51.72 / 0.40).

## Table tab:bwd_abl — Backward 3-path ablation: CSR / atomic / atomic-unified (H100)

Three backward paths at the same forward shape; the production dispatcher
picks `atomic_unified` by default for `shared_docs=True`.

| Shape | CSR ms | CSR peak | atomic ms | atomic peak | unified ms | unified peak | Source |
|---|---|---|---|---|---|---|---|
| ColBERT B=128  (Lq=32)  | 0.96  | 0.04 GB | 3.95   | 0.03 GB | 0.76  | 0.03 GB | `bench_bwd_paths_3way_NVIDIA_H100_80GB_HBM3.json:rows[0].{invgrid, atomic, atomic_unified}` |
| ColBERT B=512  (Lq=32)  | 5.76  | 0.43 GB | 18.62  | 0.13 GB | 6.21  | 0.13 GB | `rows[1]` |
| ColBERT B=1024 (Lq=32)  | 21.21 | 1.56 GB | 50.85  | 0.33 GB | 23.53 | 0.33 GB | `rows[2]` |
| ColPali B=32   (Lq=1024)| 1.87  | 0.09 GB | 28.48  | 0.06 GB | 1.86  | 0.06 GB | `rows[3]` |
| ColPali B=64   (Lq=1024)| 5.81  | 0.27 GB | 58.13  | 0.12 GB | 6.05  | 0.12 GB | `rows[4]` |
| ColPali B=128  (Lq=1024)| 21.02 | 0.88 GB | 136.86 | 0.28 GB | 22.65 | 0.28 GB | `rows[5]` |

Bench script: `benchmarks/bench_bwd_paths_3way.py` (toggles
`FLASH_BWD_PATH={invgrid, atomic, atomic_unified}` per cell).
**Dispatcher default** for `shared_docs=True` is `atomic_unified` (no shape
threshold); CSR and plain atomic are reachable only via env override.

The original 2-path CSR-vs-atomic bench is preserved at
`benchmarks/bench_csr_vs_atomic.py` (renamed from `bench_beat_lik.py` in
prior cleanup) and its result JSONs at
`bench_csr_vs_atomic_NVIDIA_H100_80GB_HBM3.json` for historical reference.

## Table tab:varlen — Variable-length scoring (B=1K, L_d^max=512, A100-80GB)

| Bucket | ρ | speedup | Source |
|---|---|---|---|
| uniform [256,512] | 0.76 | 3.24× | `bench_varlen_buckets_NVIDIA_A100-SXM4-80GB.json:buckets[0].{rho, speedup}` |
| HotpotQA-like | 0.24 | 4.27× | `buckets[1]` |
| highly ragged | 0.14 | 4.59× | `buckets[2]` |

## Table tab:ooc — Out-of-core corpus scaling (A100-80GB, 20K-doc blocks)

| B | embeddings | exceeds VRAM | flash peak | Source |
|---|---|---|---|---|
| 100K | 26 GB | no | 5.5 GB | `bench_ooc_NVIDIA_A100-SXM4-80GB.json:rows[0].{corpus_gb, exceeds_vram, gpu_peak_gb}` |
| 250K | 66 GB | no | 5.2 GB | `rows[1]` (`gpu_peak_gb = 5.24`) |
| 500K | 131 GB | yes | 5.3 GB | `rows[2]` (`gpu_peak_gb = 5.25`, round-half-up) |
| 750K | 197 GB | yes | 5.3 GB | `rows[3]` (`gpu_peak_gb = 5.25`, round-half-up) |

Prose "~52K docs/s steady state": `rows[1..3].docs_per_s ∈ {53.6K, 51.2K, 52.7K}`.
Row 0 (`docs_per_s = 11289`) excluded because it includes one-time autotuning.

## App. A (sec:compile) — Forward dense vs `torch.compile(max-autotune)`

Audited 2026-06-07 (job 1667280) on A100-80GB with the canonical
`torch.compile(naive_fp32_precast, mode='max-autotune', dynamic=False)`
(CUDA graphs ON — the production-strength `max-autotune` flavor).
Bench script: `benchmarks/bench_compile_ma_audit.py`.

| Claim | Value | Source |
|---|---|---|
| Flash B=1K | 1.7 ms | `data/bench_compile_ma_audit_NVIDIA_A100-SXM4-80GB.json:rows[0].flash.ms = 1.685` |
| compile-MA B=1K | 7.5 ms | `rows[0].compile_ma.ms = 7.529` |
| eager FP32 B=1K | 6.5 ms | `rows[0].eager_fp32_tf32.ms = 6.535` |
| Flash vs compile-MA at B=1K | **4.47×** | 7.529 / 1.685 |
| Flash B=10K | 16 ms | `rows[1].flash.ms = 16.028` |
| compile-MA B=10K | OOM on 80 GB | `rows[1].compile_ma.error = "OutOfMemoryError"` (CUDA-graph private pool pre-allocation request of 39 GiB on top of eager residuals exceeds the budget) |
| eager FP32 B=10K | 67 ms / 50 GB peak | `rows[1].eager_fp32_tf32.{ms = 67.275, peak_gb = 50.104}` |

**Note on revision.** Previous App. A prose claimed compile-MA = 3.8 ms
at B=1K and ~36 ms fitting at B=10K (no corresponding JSON-of-record).
Those numbers were not reproducible under the canonical `max-autotune`
protocol (CUDA graphs ON) on A100-80GB at our torch version (2.8.0+cu128);
the audit above is the new source of record and the prose has been
updated. Flash and eager-FP32 numbers from the previous claim DO
reproduce within 1-3% noise — only compile-MA differs.

## Figure fig:scaling — A100-40GB ColPali OOM cliffs (fair-bench protocol)

Bench rewrite 2026-06-07 (job 1667510) under reviewer-feedback fairness
protocol: each (B, method) runs in an isolated allocator state, inputs
generated in the method's native dtype (no FP16 inputs kept alive while
measuring FP32 / compile-MA), narrow OOM-only exception catch, startup
correctness check (Flash vs FP32-precast naive at B=4: max abs err
6.1e-5). Bench script `benchmarks/bench_blog_40gb.py`; JSON-of-record
`data/bench_blog_40gb_NVIDIA_A100-SXM4-40GB.json`.

Empirical OOM cliffs (`error` field on the first OOM cell per series):

| Series | First-OOM B | Why |
|---|---|---|
| `naive_matched_fp32` | 10K | `[B,Lq,Ld]` FP32 intermediate (40 GB at B=10K) exceeds VRAM |
| `compile_ma_fp32` | 10K | same materialised intermediate (compile doesn't remove it) |
| `naive_fp16` | 20K | same intermediate at FP16 (20 GB at B=20K → OOM with workspace + context) |
| `flash` | 175K | **dense-D allocation cap**: D fp16 at B=175K is 44.8 GB, exceeds the 42.4 GB physical VRAM. Flash runs cleanly through B=150K (peak 40.3 GB). The previous iteration of this bench reported B=100K as the empirical OOM because `F.normalize(...)` returned a new same-shape tensor (transiently doubling peak); the in-place L2-norm in `make_inputs` removes this artifact and the empirical cap now matches the analytic physics. Beyond 175K, the kernel can score arbitrarily large corpora via out-of-core streaming (`tab:ooc`: 750K docs at 5.3 GB flat peak). |

Caption clarifications that match this bench:
- `compile-MA` is labelled a "materialising baseline" (it fuses surrounding ops but cannot remove the `[B,Lq,Ld]` intermediate; `no_cudagraphs` on this 40 GB protocol because cudagraph private pool inflates peak — see `bench_compile_ma_audit_*.json` for the cudagraphs-on numbers).
- Flash's "OOM cliff" is the dense-$D$ allocation cap (not a kernel cap); points the reader at `tab:ooc` for the streaming-$D$ case (750K docs at flat 5.3 GB peak).

Headline ratio at B=1K (Flash vs `compile-MA`): 7.62 / 1.69 = **4.51×**
(matches the audit on A100-80GB: 7.529 / 1.685 = 4.47×; same story
under both protocols).

## Figure fig:scaling — A100-40GB ColPali OOM cliffs (legacy header for backref)

PNG file: `fig_corpus_scaling.png` (committed). Generator: `make_scaling_fig.py`
reading `bench_blog_40gb_NVIDIA_A100-SXM4-40GB.json`. The three OOM cliffs are:

- matched FP32 at B≈7K: `bench_blog_40gb:rows[1].naive_matched_fp32.error = "OutOfMemoryError"`
- pure FP16 at B≈20K: `bench_blog_40gb:rows[5].naive_fp16.error = "OutOfMemoryError"`
- flash at B≈175K: analytic, D = 175K · 1024 · 128 · 2 = 45.9 GB > 40 GB VRAM (the kernel itself uses ≤5.5 GB scratch per `bench_ooc:rows[*].gpu_peak_gb`).

## §5.5 bf16 paragraph — flash 35× to 87× tighter than naive bf16 path

Added to back the production-relevance claim ("at bf16 + ColPali scale, the
einsum baseline is numerically broken and Flash is the only correct path").

| Shape | mask | torch backend max\|Δ\| vs FP32 ref | flash backend max\|Δ\| | flash÷torch (lower is better) | Source |
|---|---|---|---|---|---|
| long-doc (Lq=32, Ld=1024) | fp16 light | 2.43e-3 | 2.30e-4 | **0.09×** | `bench_bf16_via_pylate_NVIDIA_A100-SXM4-80GB.json:results[0]` |
| long-doc | fp16 heavy | 1.83e-3 | 2.63e-4 | 0.14× | `results[1]` |
| long-doc | bf16 light | 1.83e-2 | 2.07e-3 | 0.11× | `results[2]` |
| long-doc | bf16 heavy | 1.56e-2 | 2.83e-3 | 0.18× | `results[3]` |
| ColPali (Lq=Ld=1024) | fp16 light | 5.40e-2 | 1.63e-3 | 0.03× | `results[4]` |
| ColPali | fp16 heavy | 6.01e-2 | 1.22e-3 | 0.02× | `results[5]` |
| **ColPali** | **bf16 light** | **7.64e-1** | **8.77e-3** | **0.011× (flash 87× tighter)** | `results[6]` |
| **ColPali** | **bf16 heavy** | **4.94e-1** | **1.41e-2** | **0.029× (flash 35× tighter)** | `results[7]` |

Bench script: `benchmarks/bench_bf16_via_pylate.py` (script + JSON in
`flash-maxsim/benchmarks/`). The 0.5 to 0.76 max-error magnitude on the torch
backend matches what `references/bench_summary.md` §6 already reported for
"FP16-mat eager 8.91e-1 max error" — it's the same bf16-accumulator-saturation
phenomenon, surfaced here with a clean controlled experiment.

## Table tab:fat_emb_cliff + tab:splitd_perf — Fat-embedding split-d forward

Added per the §5.8 split-d coverage section. Two tables:

(1) tab:fat_emb_cliff documents the in-one-shot forward latency cliff
at d > 512 on both arches (the kernel doesn't crash; correctness is
preserved; only latency degrades because the d-wide register tile
exceeds per-SM SRAM budget and Triton spills to local memory).

| d | A100 in-shot ms | H100 in-shot ms | Source |
|---|---|---|---|
| 128 | 0.36 | 0.36 | `data/bench_fat_embeddings_*.json:shapes[7]` (ColPali B=128) |
| 256 | 0.59 | 0.55 | `shapes[8]` |
| 384 | 1.28 | 0.96 | `shapes[9]` |
| 512 | 1.29 | 0.93 | `shapes[10]` |
| 768 | **5.19** | **2.68** | `shapes[11]` ← cliff |
| 1024 | **5.59** | **2.58** | `shapes[12]` |
| 2048 | **99.7** | **17.3** | `shapes[13]` |

Bench: `benchmarks/bench_fat_embeddings.py`, A100 job 1500463, H100 job 1500464.

(2) tab:splitd_perf compares the split-d kernel (new) against the
spilling in-one-shot kernel at the cliff shapes.

| d | A100 in→split ms | speedup | H100 in→split ms | speedup | Source |
|---|---|---|---|---|---|
| 768 | 5.19→2.43 | 2.13× | 2.68→0.76 | 3.53× | `data/bench_splitd_perf_*.json:rows[11]` |
| 1024 | 5.59→2.60 | 2.15× | 2.58→1.07 | 2.41× | `rows[12]` |
| 2048 | 99.7→4.91 | **20.3×** | 17.3→2.06 | **8.40×** | `rows[13]` |

Bench: `benchmarks/bench_splitd_perf.py`, A100 job 1500457, H100 job 1500458.

Correctness: max |Δ| vs naive FP32 einsum in [1e-5, 1e-4] at every
d in {384, 512, 768, 1024, 2048}, matching the in-one-shot kernel's
FP16-cast noise. See `flash_maxsim/flash_maxsim_splitd.py:__main__`
self-test (jobs 1500453, 1500460) and the end-to-end probe
(`benchmarks/bench_fat_embeddings.py` shows |Δ| at every d).

Implementation commit: flash-maxsim repo d657b9b. Dispatched in
`flash_maxsim/flash_maxsim.py:_launch_fwd` at d > _SPLITD_THRESH (512).

## §5.9 Determinism and cold-start (warmup numbers)

Added as a new subsection in §5 covering production-readiness concerns
not directly visible in the perf tables. Two empirical numbers:

| measurement | value | Source |
|---|---|---|
| Warmup wall-clock, empty Triton cache (A100, 8 fwd + 2 train shapes) | 126.4 s | `benchmarks/_warmup_test.sh`, A100 job 1502920 |
| Warmup wall-clock, populated Triton 3.4+ disk cache | 9.05 s | same |

The 126 s is what a fresh-install user pays once per machine; the 9 s
is what every subsequent process on the same machine pays. The
deterministic per-arch config table (commit 3461a66 in flash-maxsim
repo) is what guarantees the first request after the cache is hot
uses the same launch params as the thousandth — Triton autotune
gives no such guarantee. Heuristic matches autotune within ~2% at
every measured shape per `benchmarks/bench_config_sweep_*.json`.

## Table tab:fatd_vs_lik — Flash split-d vs LIK v0.3.0 at fat embedding dim (H100 + A100)

Added under §5.8 split-d coverage. Confirms that LIK shares the in-one-
shot kernel pattern of our prior release and pays the same SRAM-spill
cliff at d > 512 — Flash with split-d auto-dispatched eliminates it.

ColPali shape (B=128 contrastive, Nq=128, Lq=Ld=1024), forward-only,
fp16 inputs, fp32 accum, median of 10 post-warmup runs:

| d | A100 Flash ms | A100 LIK ms | Flash adv | H100 Flash ms | H100 LIK ms | Flash adv | Source |
|---|---|---|---|---|---|---|---|
| 128 | 25.0 | 31.6 | 1.26× | 9.63 | 10.32 | 1.07× | `data/bench_vs_lik_fatd_*.json:rows[0]` |
| 512 | 153 | 261 | 1.71× | 92.6 | 109.5 | 1.18× | `rows[1]` |
| 768 | 234 | 924 | **3.96×** | 101 | 320 | **3.17×** | `rows[2]` ← cliff |
| 1024 | 330 | 991 | 3.01× | 174 | 346 | 1.99× | `rows[3]` |
| 2048 | 737 | **13991** | **19.0×** | 364 | 1951 | **5.37×** | `rows[4]` |

Peak GPU memory is identical for both implementations at every cell
(same Q, D, scores tensor accounting); omitted from the table.

Bench: `benchmarks/bench_vs_lik_fatd.py`, H100 job 1501020, A100 job 1501021.
The A100 d=2048 LIK cell takes ~14 seconds per call (10 runs ≈ 2.3 min);
that's why the A100 job ran for ~15 min total vs ~9 min on H100.

## Table tab:lik_v030 — Head-to-head vs LIK v0.3.0 (H100 + A100)

Added per R2 #15 (concurrent work) and to address the head-to-head
comparison thread on PyLate issue #224. LIK v0.3.0 published 2026-05-28
(commits 81699862, 6151c44) with sentinel-gated argmax + SRAM-aware
autotune model.

Bench script: `benchmarks/bench_vs_lik_v030.py` (in implementation repo).
Measures fwd+bwd step latency (median of 30 post-warmup runs) and steady-
state peak GPU memory (peak reset post-warmup to exclude autotune trial
allocations).

H100 80GB HBM3, contrastive in-batch-negatives, FP16 inputs / FP32 accum:

| Shape | LIK ms / GB | Flash ms / GB | Speedup | Mem ratio | Source |
|---|---|---|---|---|---|
| ColBERT B=128 (Lq=32) | 0.79 / 0.097 | 0.56 / 0.096 | 1.40× | 1.01× | `data/bench_vs_lik_v030_NVIDIA_H100_80GB_HBM3.json:rows[0]` |
| ColBERT B=512 | 8.35 / 0.213 | 5.99 / 0.209 | 1.39× | 1.02× | `rows[1]` |
| ColBERT B=1024 | 32.7 / 0.430 | 23.1 / 0.421 | 1.41× | 1.02× | `rows[2]` |
| ColPali B=32 (Lq=Ld=1024) | 2.02 / 0.138 | 1.65 / 0.130 | 1.23× | 1.06× | `rows[3]` |
| ColPali B=64 | 7.34 / 0.218 | 5.81 / 0.201 | 1.26× | 1.09× | `rows[4]` |
| ColPali B=128 | 29.4 / 0.403 | 22.4 / 0.369 | 1.31× | 1.09× | `rows[5]` |

A100 80GB SXM4:

| Shape | LIK ms / GB | Flash ms / GB | Speedup | Mem ratio | Source |
|---|---|---|---|---|---|
| ColBERT B=128 | 1.18 / 0.048 | 0.95 / 0.047 | 1.24× | 1.02× | `data/bench_vs_lik_v030_NVIDIA_A100-SXM4-80GB.json:rows[0]` |
| ColBERT B=512 | 14.8 / 0.165 | 12.9 / 0.161 | 1.14× | 1.03× | `rows[1]` |
| ColBERT B=1024 | 73.8 / 0.381 | 54.9 / 0.373 | 1.35× | 1.02× | `rows[2]` |
| ColPali B=32 | 4.12 / 0.090 | 3.57 / 0.081 | 1.15× | 1.11× | `rows[3]` |
| ColPali B=64 | 16.2 / 0.169 | 13.5 / 0.153 | 1.20× | 1.11× | `rows[4]` |
| ColPali B=128 | 71.7 / 0.354 | 54.4 / 0.320 | 1.32× | 1.11× | `rows[5]` |

Flash reports lower latency and lower peak memory than LIK at every cell on
both axes (12 cells). Three implementation deltas vs LIK explain the
deltas, all in flash-maxsim commits 618a372 + the per-arch heuristic
commit:
1. int32 throughout build_inverse_csr (CSR transients down 25-35%).
2. Fused dQ+dD unified kernel with Q register hoisting (one launch instead
   of two, single D load reused for both gradient computations).
3. Per-arch heuristic launch-config table (no Triton autotune trial
   overhead on first call; deterministic config every step). Pattern
   inspired by Flash-KMeans `_heuristic_euclid_config`.

## Table tab:int8 — INT8 cross-shape evaluation (A100)

Added per R2 #11 ("strengthen or demote INT8"). Reports INT8 vs three
baselines (naive FP32 einsum, dequant-then-naive, FP16 flash) so the
reader can compare against whatever they have in production. Earlier
single-baseline (FP16 flash only) reframing has been replaced — the
original paper's "4.7x" claim against dequant-then-naive is real, see
tab:int8_B for the B-sweep that includes the dequant baseline.

| Shape | B | FP16 ms | INT8 ms | vs naive | vs FP16 | Spearman | top-20 | Source |
|---|---|---|---|---|---|---|---|---|
| textual | 1024 | 0.14 | 0.15 | 2.1× | 0.93× | 0.9997 | 100% | `data/bench_int8_shapes_NVIDIA_A100-SXM4-80GB.json:results[0]` |
| long-doc | 1024 | 0.26 | 0.21 | **6.0×** | 1.27× | 0.9995 | 95% | `results[1]` |
| medium | 512 | 0.23 | 0.24 | 3.7× | 0.98× | 0.9996 | 100% | `results[2]` |
| visual | 256 | 0.35 | 0.34 | 3.0× | 1.04× | 0.9992 | 100% | `results[3]` |
| ColPali | 128 | 0.35 | 0.30 | 3.0× | 1.07× | 0.9993 | 100% | `results[4]` |

Storage savings 1.97× uniformly.

## Table tab:int8_B — INT8 corpus-size sweep at ColPali shape (A100)

Added in response to "are we sure about the int8 speedup reduction" — at B=128
ColPali the D tensor (33 MB FP16) fits in A100's 40 MB L2, so the bench is
compute-bound and INT8 only wins by the dequant-vs-FP16 throughput delta. At
B>=256 D exceeds L2, INT8's halved storage becomes a memory-BW advantage,
and the original "4.7x" paper claim against dequant-then-naive becomes the
B=512 cell (4.9x).

| B | naive | dequant | FP16 | INT8 | INT8/naive | INT8/dequant | INT8/FP16 | Source |
|---|---|---|---|---|---|---|---|---|
| 128   | 0.94 | 1.03 | 0.34 | 0.31 | 3.0× | 3.3× | 1.07× | `data/bench_int8_B_sweep_NVIDIA_A100-SXM4-80GB.json:results[0]` |
| 256   | 1.84 | 2.02 | 0.57 | 0.48 | 3.8× | 4.2× | 1.18× | `results[1]` |
| 512   | 3.62 | 3.95 | 0.98 | 0.80 | 4.5× | **4.9×** | 1.22× | `results[2]` ← matches original paper's "4.7x" claim |
| 1024  | 7.34 | 8.01 | 1.75 | 1.45 | 5.0× | 5.5× | 1.21× | `results[3]` |
| 2048  | 14.83 | 16.13 | 3.40 | 2.75 | 5.4× | 5.9× | 1.24× | `results[4]` |
| 4096  | 29.87 | 32.28 | 6.71 | 5.37 | 5.6× | 6.0× | 1.25× | `results[5]` |
| 10000 | 74.35 | 80.73 | 16.19 | 12.77 | **5.8×** | **6.3×** | **1.27×** | `results[6]` |

Bench: `benchmarks/bench_int8_B_sweep.py`, job 1488209.

## Table tab:op_abl — Operator-design ablation (A100)

Added per R2 #4: unbundle the forward win into fused-only vs +query-chunking.

| Shape | naive ms | fused ms | +chunk ms | fused/naive | +chunk/naive | delta from chunk | Source |
|---|---|---|---|---|---|---|---|
| textual (32,180)   | 0.30 | 0.13 | 0.13 | 2.4× | 2.5× | 1.02× | `data/bench_operator_design_NVIDIA_A100-SXM4-80GB.json:results[0]` |
| long-doc (32,1024) | 1.23 | 0.25 | 0.25 | 4.9× | 5.0× | 1.01× | `results[1]` |
| medium (128,1024)  | 0.89 | 0.23 | 0.22 | 3.9× | 4.1× | 1.05× | `results[2]` |
| visual (512,1024)  | 1.03 | 0.84 | 0.36 | 1.2× | 2.9× | 2.4× | `results[3]` |
| ColPali (1024²)    | 0.94 | 0.76 | 0.39 | 1.2× | 2.4× | 1.9× | `results[4]` |

Bench: `benchmarks/bench_operator_design.py`. Naive baseline here is an
in-script `einsum + max + sum` with FP32 cast inside the timed region; it's
about 1.3× faster than tab:fwd's naive column because tab:fwd hoists the
cast to a different timed region. Relative ordering is unchanged.

## Table tab:bwd_breakdown — Backward component decomposition (A100)

Added per R2 #3: per-component CSR backward timing to justify the dispatch
heuristic and show where time goes.

| Shape | fwd | CSR build | dQ | dD | bwd pieces | autograd | Source |
|---|---|---|---|---|---|---|---|
| ColBERT B=128 (Lq=32) | 0.19 | 0.45 (76%) | 0.07 (12%) | 0.08 (13%) | 0.58 | 0.55 | `data/bench_bwd_breakdown_NVIDIA_A100-SXM4-80GB.json:results[0]` |
| ColBERT B=512 (Lq=32) | 0.20 | 0.49 (66%) | 0.09 (12%) | 0.13 (17%) | 0.75 | 0.63 | `results[1]` |
| ColBERT B=1024 (Lq=32) | 0.20 | 0.55 (54%) | 0.15 (15%) | 0.23 (23%) | 1.02 | 0.76 | `results[2]` |
| ColPali B=32 (Lq=1024) | 0.46 | 0.51 (78%) | 0.07 (10%) | 0.08 (13%) | 0.65 | 1.33 | `results[3]` |
| ColPali B=64 (Lq=1024) | 0.46 | 0.52 (75%) | 0.07 (10%) | 0.12 (18%) | 0.70 | 1.16 | `results[4]` |
| ColPali B=128 (Lq=1024) | 0.80 | 0.53 (67%) | 0.08 (10%) | 0.19 (24%) | 0.79 | 1.74 | `results[5]` |

Bench: `benchmarks/bench_bwd_breakdown.py` with FLASH_BWD_PATH unset (uses dispatch).
Times are CUDA-event medians over 30 post-warmup runs.

## Table tab:multiquery — Multi-query serving sweep (A100)

Added per R2 #8: throughput at Nq ∈ {1,4,8,16,32} × {textual, long-doc, ColPali}.

| Cell | Nq | flash ms | naive ms | spd | QPS | docs/s | Source |
|---|---|---|---|---|---|---|---|
| textual | 1-32 | 0.14-0.61 | 0.27-1.96 | 2.0-3.2× | 7K-52K | 7M-54M | `data/bench_multiquery_serving_NVIDIA_A100-SXM4-80GB.json:results[0..4]` |
| long-doc | 1-32 | 0.26-2.52 | 1.23-7.29 | 4.7-2.9× | 4K-13K | 4M-13M | `results[5..9]` |
| ColPali | 1-32 | 0.35-6.30 | 0.94-28.24 | 2.7-4.5× | 3K-5K | 0.37M-0.65M | `results[10..14]` |

Bench: `benchmarks/bench_multiquery_serving.py`.

## Table tab:beir_ndcg — BEIR end-to-end retrieval parity (ArguAna + SciDocs)

Added per R1 #1 / R2 #1 (top reviewer ask): closes the "exact in theory but
verify in practice" gap by showing baseline-vs-flash nDCG / Recall / MRR on
real BEIR test sets with the ColBERTv2 encoder.

ArguAna:

| K | baseline nDCG | flash nDCG | baseline Recall | flash Recall | top-K overlap | Source |
|---|---|---|---|---|---|---|
| 5 | 0.2844 | 0.2844 | 0.5277 | 0.5277 | 100% | `data/bench_beir_ndcg_arguana_*.json:results[0].per_K["5"]` |
| 10 | 0.3308 | 0.3308 | 0.6707 | 0.6707 | 100% | `results[0].per_K["10"]` |

Prose "1405/1406 ArguAna queries match exactly" from `ranking_exact_match_top10 = 1405` over `N_queries = 1406`.

SciDocs:

| K | baseline nDCG | flash nDCG | top-K overlap | Source |
|---|---|---|---|---|
| 5 | 0.1341 | 0.1341 | 100% | `data/bench_beir_ndcg_scidocs_*.json:results[0].per_K["5"]` |
| 10 | 0.1565 | 0.1565 | 100% | `results[0].per_K["10"]` |

Prose "1000/1000 SciDocs queries match exactly" from `ranking_exact_match_top10 = 1000` over `N_queries = 1000`.

Prose "per-query mean |Δ| ~ 2e-4" from the in-line print samples (q 0..1400):
{1.45e-4, 1.11e-4, 2.47e-4, 1.70e-4, 3.40e-4, 1.32e-4, 1.27e-4, 1.79e-4,
1.30e-4, 1.24e-4, 2.29e-4, 1.18e-4, 1.85e-4, 3.13e-4, 2.71e-4} — mean 1.95e-4.

Bench script: `benchmarks/bench_beir_ndcg.py`. Vectorized dense baseline
(GPU einsum + max + sum in FP32) replaced an earlier per-doc numpy loop that
was ~100x slower; the two implementations produce identical scores to FP32
reassociation noise.

## Table tab:colpali_ndcg — REAL-MM-RAG end-to-end retrieval parity (ColPali v1.2 + Granite Vision)

Two JSONs back this table. The nDCG@10 and Recall@10 columns come from
`bench_real_mm_rag_ndcg_*.json`; the Spearman $\rho$ and top-10 overlap
columns come from `bench_real_mm_rag_parity_*.json` (the parity sweep
runs the two backends side-by-side without spending wall-clock on the
qrel-evaluation pass and is therefore where the per-cell parity stats
live). `data/bench_real_mm_rag_ndcg_*.json:note` documents the split.

**Indexing note:** the two JSONs list encoders in opposite order.
`bench_real_mm_rag_ndcg_*.json` is GVE at `results[0..3]`, ColPali at
`results[4..7]`. `bench_real_mm_rag_parity_*.json` is ColPali at
`results[0..3]`, GVE at `results[4..7]`. Pointers below reflect that.

ColPali v1.2 (rows 1-4 of the table):

| Subset | $B$ | naive nDCG@10 | flash nDCG@10 | naive Recall@10 | flash Recall@10 | $\rho$ | top-10 overlap | Source |
|---|---|---|---|---|---|---|---|---|
| FinReport  | 2687 | 0.6019 | 0.6019 | 0.7600 | 0.7600 | 1.000000 | 100% | `data/bench_real_mm_rag_ndcg_NVIDIA_A100-SXM4-80GB.json:results[4]` (ColPali/finreport) + `data/bench_real_mm_rag_parity_NVIDIA_A100-SXM4-80GB.json:results[0].parity` |
| FinSlides  | 2280 | 0.6385 | 0.6385 | 0.7900 | 0.7900 | 1.000000 | 100% | `ndcg results[5]` + `parity results[1]` |
| TechReport | 1674 | 0.8209 | 0.8209 | 0.9300 | 0.9300 | 1.000000 | 100% | `ndcg results[6]` + `parity results[2]` |
| TechSlides | 1963 | 0.8659 | 0.8659 | 0.9300 | 0.9300 | 1.000000 | 100% | `ndcg results[7]` + `parity results[3]` |

Granite Vision (rows 5-8):

| Subset | $B$ | naive nDCG@10 | flash nDCG@10 | naive Recall@10 | flash Recall@10 | $\rho$ | top-10 overlap | Source |
|---|---|---|---|---|---|---|---|---|
| FinReport  | 2687 | 0.7562 | 0.7566 | 0.9472 | 0.9484 | 0.999999 | 99.89% | `ndcg results[0]` (gve/finreport) + `parity results[4]` |
| FinSlides  | 2280 | 0.8054 | 0.8049 | 0.9762 | 0.9762 | 0.999999 | 99.95% | `ndcg results[1]` + `parity results[5]` |
| TechReport | 1674 | 0.8764 | 0.8765 | 0.9784 | 0.9784 | 0.999999 | 99.91% | `ndcg results[2]` + `parity results[6]` |
| TechSlides | 1963 | 0.9321 | 0.9321 | 0.9904 | 0.9904 | 0.999999 | 99.93% | `ndcg results[3]` + `parity results[7]` |

Bench scripts: `benchmarks/bench_real_mm_rag_ndcg.py` (nDCG/Recall over
qrels) and `benchmarks/bench_real_mm_rag_parity.py` (per-query Spearman
and top-K overlap). Reproduction requires IBM-internal multi-modal
embeddings; the two JSONs are shipped in `data/` so reviewers can
audit without re-encoding.

## §5.5 Numerical Correctness

| Claim | Value | Source |
|---|---|---|
| Flash max relative error vs FP32 reference | **4×10⁻⁷** (paper rounded; measured 4.25e-7) | `data/bench_correctness_NVIDIA_A100-SXM4-80GB.json:flash_fp16.max_rel_err = 4.25e-7` (ColPali shape, B=1K, true-FP32 reference with TF32 OFF) |
| Top-20 ranking overlap | 100% | `data/bench_correctness_*.json:flash_fp16.top20_overlap = 1.0` |
| Top-50 ranking overlap | 100% | `data/bench_correctness_*.json:flash_fp16.top50_overlap = 1.0` |
| Spearman ρ (Flash vs FP32) | 0.999999 | `data/bench_correctness_*.json:flash_fp16.spearman_rho = 0.9999998807907104` |
| INT8×INT8 Spearman ρ | 0.999 (measured 0.9996) | `data/bench_correctness_NVIDIA_A100-SXM4-80GB.json:flash_int8.spearman_rho = 0.999591588973999` (via `flash_maxsim_int8x8` with `quantize_int8_symmetric`+`quantize_query_int8`, ColPali shape) |
| INT8×INT8 top-20 overlap | 100% | `data/bench_correctness_*.json:flash_int8.top20_overlap = 1.0` |

## §4.3.1 INT8 paragraph

| Claim | Value | Source |
|---|---|---|
| 4.7× faster than naive dequant + score | yes | reported, internal INT8 bench at ColPali shape; closest committed evidence is `results_landing_NVIDIA_A100-SXM4-80GB.json:int8_fused[*]` (small-shape regime only). Caveat in paper: "at the tested shapes; not yet a net win at full ColPali scale on A100" |

## Table tab:fwd_chunked — Inference frontier vs chunked FP16 eager (A100-80GB) [main text]
## Table tab:chunked_sweep — Chunked FP16 eager full sweep (App: chunked-eager baseline)

| Claim | Value | Source |
|---|---|---|
| visual B=10K: flash 8.4 ms vs chunked-best 21.8 ms (2.6x), peaks 2.6/13.2 GB | yes | `benchmarks/bench_chunked_fp16_eager_NVIDIA_A100-SXM4-80GB.json:rows[label=visual,B=10000]` (flash.ms=8.407, chunked_fp16_eager_best.ms=21.84 @chunk=1024, flash.peak_gb=2.63, best.peak_gb=13.25) |
| ColPali B=10K: flash 16.4 ms vs chunked-best 43.0 ms (2.6x), peaks 2.6/23.9 GB | yes | same JSON, rows[label=ColPali,B=10000] (16.387 / 42.98 @1024 / 2.631 / 23.868) |
| ColPali B=20K: flash 32.6 ms vs chunked-best 86.3 ms (2.6x), peaks 5.3/47.7 GB | yes | same JSON, rows[label=ColPali,B=20000] |
| unchunked FP16 eager fits ColPali B=20K on 80 GB at 47.7 GB peak | yes | same JSON, rows[label=ColPali,B=20000].vanilla_fp16_eager (ms=92.939, peak_gb=47.728) |
| best chunk = 256 at B=1K, 1024 at B>=5K | yes | same JSON, chunked_fp16_eager_best.chunk per row |
| headline 2.6x per-candidate, 5-9x lower peak | yes | derived from the rows above (peak ratios 5.0x-9.1x) |

Reproduce: `python benchmarks/bench_chunked_fp16_eager.py` (single A100-80GB; sweeps chunk sizes {1..16384} per cell, reports best non-OOM median over 10 CUDA-event-timed runs after 3 warmups).

## Table tab:beir_scale — Corpus-scale BEIR parity (App: BEIR at Corpus Scale)

| Claim | Value | Source |
|---|---|---|
| HotpotQA-500K: flash == FP32 ref, nDCG@10=.5681, Recall@10=.6957 | yes | `benchmarks/bench_beir_ndcg_NVIDIA_A100-SXM4-80GB.json:results[hotpotqa-500K]` (flash_metrics == naive_metrics to all decimals shown; n_eval=322/500) |
| HotpotQA parity max abs delta = 6.4e-4 | yes | same JSON: parity.max_abs = 0.000638 |
| HotpotQA wall: flash 26s vs naive 104s (4.0x) | yes | same JSON: flash_wall_s=26.3, naive_wall_s=103.9 |
| NQ-2.68M (116 GB embeddings) OOC: nDCG@10=.5223, Recall@10=.7460, 103s | yes | same JSON: results[nq-2.68M].flash_metrics, flash_wall_s=102.8; naive ref skipped (corpus exceeds VRAM) |
| Quora-523K excluded | n/a | same JSON: metrics are 0 for BOTH backends with parity 5e-6 — qrel/doc_id mapping mismatch in the precomputed bundle, not a scoring issue; excluded from the paper table |

Reproduce: `python benchmarks/bench_beir_ndcg.py` (requires the precomputed ColBERTv2 embedding bundles; institution-internal paths in the script header).

## Table tab:nocg — compile-MA-no-cudagraphs, strongest per-cell config (App. A)

| Claim | Value | Source |
|---|---|---|
| nocg-FP16 at tab:fwd shapes B=1K: 0.19/0.32/0.57/1.94/3.68 ms | yes | `benchmarks/bench_compile_ma_nocg_NVIDIA_A100-SXM4-80GB.json:rows[0..4].ms` |
| nocg-FP16 does NOT OOM at ColPali B=10K (23.7 GB) or B=20K (47.3 GB) | yes | same JSON rows[5..6] (ok=true) |
| flash vs strongest compile: 1.0x (textual) to 2.2x (ColPali 1K), 2.1x at 10K/20K | yes | derived: flash ms from bench_fwd_fair / bench_chunked_fp16_eager |
| nocg beats canonical graphs-on compile-MA (3.68 vs 7.5 ms at ColPali 1K) | yes | nocg JSON vs bench_compile_ma_audit rows[0].compile_ma.ms=7.529 |

Reproduce: `python benchmarks/bench_compile_ma_nocg.py` (single A100-80GB; one compile flavour per process).

## §5.6 INT8-on-text-BEIR sentence

| Claim | Value | Source |
|---|---|---|
| ArguAna: int8 nDCG@5/10/20 delta -0.0001/+0.0002/-0.0004, top-20 overlap 99.8% | yes | `benchmarks/bench_beir_int8_text_NVIDIA_A100-SXM4-80GB.json:results[arguana].per_K` |
| SciDocs: int8 delta -0.0002/-0.0001/+0.0001, top-20 overlap 99.7% | yes | same JSON, results[scidocs].per_K |

## App. chunked-eager cost paragraph

| Claim | Value | Source |
|---|---|---|
| 1.6 vs 4.3 us/candidate -> 0.44 vs 1.19 GPU-s per 1M -> $0.19 vs $0.50 per 1B at $1.50/h | derived | latencies: bench_chunked_fp16_eager (ColPali 10K/20K cells); arithmetic: p x t/3600 |

## §5.5 chunked-recompute paragraph (re-review item 3, measured)

| Claim | Value | Source |
|---|---|---|
| B=64: recompute 105.9 ms / 17.3 GB (best chunk 32) vs flash 13.1 ms / 0.34 GB (8.1x) | yes | `benchmarks/bench_recompute_train_NVIDIA_A100-SXM4-80GB.json:rows[B=64]` |
| B=128: naive OOM (tried 64 GiB); recompute fits 425.1 ms / 17.4 GB (chunk 16); flash 53.7 ms / 0.29 GB (7.9x, 61x peak) | yes | same JSON rows[B=128] |
| protocol | mean-loss fwd+bwd step, torch.utils.checkpoint(use_reentrant=False) per doc block, 10 CUDA-event medians | script header |

## PROTOCOL-GRADE CAMPAIGN (2026-06-11) — authoritative source for Tabs. fwd / fwd_abs / nocg / fwd_chunked / train

All five headline tables re-measured under App. bench_protocol in one
campaign per GPU: interleaved round-robin, 100MB L2 flush per timed call,
CUDA-event median+IQR, compile warmup>=100 with subprocess autotune,
per-method peak passes. Sources:
- `benchmarks/bench_protocol_grade_shapes_NVIDIA_A100-SXM4-80GB.json`
- `benchmarks/bench_protocol_grade_shapes_NVIDIA_H100_80GB_HBM3.json`
- `benchmarks/bench_protocol_grade_frontier_NVIDIA_A100-SXM4-80GB.json`
- `benchmarks/bench_protocol_grade_train_NVIDIA_A100-SXM4-80GB.json`
Reproduce: `BENCH_PART={shapes|frontier|train} python benchmarks/bench_protocol_grade.py`

Key values now in the paper:
| Claim | Value |
|---|---|
| tab:fwd matched A100 | 1.21 / 1.95 / 2.82 / 3.39 / 3.76 (ColPali bold 3.8x) |
| tab:fwd matched H100 | 1.12 / 1.67 / 3.07 / 3.99 / 4.65 (bold 4.7x) |
| tab:fwd vs compile-MA-FP32 A100 | 2.35 / 4.28 / 4.67 / 3.93 / 3.86 |
| tab:nocg strongest-config band | 0.97-1.95 ("1.0-2.0x") |
| frontier ColPali 10K | flash 16.2ms/2.7GB, chunked 42.8ms/7.0GB (2.6x lat, 2.6x peak), vanilla 45.7ms/24.0GB (9x peak) |
| tab:train B=64 | naive 81.3/51.8GB; recompute 109.2/8.8GB; flash-uni 13.4/0.24GB (6.1x, 217x); CSR 11.7/0.39GB |
| tab:train B=128 | naive OOM; recompute 426.6/17.5GB; flash-uni 53.6/0.39GB (8.0x vs rec); CSR 46.0/1.04GB |

SUPERSEDED for these tables (kept for audit): bench_fwd_fair_*,
bench_compile_ma_shapes_*, bench_chunked_fp16_eager_* (its chunked
peak_gb at ColPali 10K/20K was allocator-contaminated: 23.9/47.7GB;
clean per-method value is 7.0/9.7GB), bench_train_paper_v2_*.
Independent cross-checks: results_indep_harness_*.log (current kernels,
colleague harness) and the merged-dev run (old-line kernels: flash
2.15ms vs 1.77ms at A100 ColPali B=1K — v0.3 launch-table tuning).

## Verification status (all four original UNVERIFIED items now resolved)

1. ~~Flash max relative error 2×10⁻⁶~~ → **VERIFIED**: measured **4.25×10⁻⁷** (paper underclaimed by 5×; tightened to 4×10⁻⁷). `data/bench_correctness_*.json:flash_fp16.max_rel_err`.
2. ~~Top-20 ranking overlap 100%~~ → **VERIFIED**: 100.0% measured at ColPali shape, both FP16 and INT8×INT8. `data/bench_correctness_*.json:flash_fp16.top20_overlap = 1.0` and `flash_int8.top20_overlap = 1.0`.
3. ~~INT8×INT8 Spearman ρ = 0.999~~ → **VERIFIED**: measured **0.9996** at ColPali shape via `flash_maxsim_int8x8`. `data/bench_correctness_*.json:flash_int8.spearman_rho`.
4. Figure `fig_corpus_scaling.png` provenance → **RESOLVED**: `make_scaling_fig.py` regenerates an equivalent from `data/bench_blog_40gb_*.json` (committed PNG is the original styling; the script is the audit pathway).

## Minor residuals (cosmetic only, not blocking)

- tab:mem flash cell at B=10K shows 2.9 GB; `bench_blog_40gb:rows[2].flash.peak_gb = 2.63` (difference is autotune scratch from a cold first-call sweep). Either round paper to 2.6 GB to match JSON, or document the autotune-scratch caveat. Not material for reviewers.

## Camera-ready additions (rebuttal experiment campaign, Jul 2026)

Six tables/result blocks were added at camera-ready from the OpenReview
rebuttal campaign. Raw JSONs live in `data/e1_results/` in this repo; each is
re-derivable from the matching `scripts/e*.py`.

| Paper element | Source JSON (`data/e1_results/`) | Key path | Script |
|---|---|---|---|
| `tab:fullstep` (complete-step peaks) | `b{64,128,256,512}_{flash,torch}_s0_lr-sqrt.json`, `b{1024,2048}_*_lr-sqrt_gc.json` | `train.train_peak_gb` | `scripts/e1_batch_quality.py` |
| `tab:fullstep` ColPali rows | `e3b_probe.json`, `e3b_probe_paq.json` | `steps[].peak_gb` | `scripts/e3b_probe.py` |
| `tab:fullstep` wall-clock ¶ | same `b*_lr-sqrt*.json` | `train.train_wall_s` | `scripts/e1_batch_quality.py` |
| `tab:batchquality` (5-seed nDCG) | `b{64,128,256,512}_flash_s{0..4}_lr-sqrt.json` (+ `s{0..2}_lr-fixed` control) | `eval.mean` | `scripts/e1_batch_quality.py` |
| `tab:ddp` (W=1/2/4) | `e2_ddp_W{1,2,4}.json` | `rows[].{conventional,fm-flash-tiled}.{peak_gb,step_ms}`; gradient check `gradient_verify.*` | `scripts/e2_ddp_negatives.py` |
| `tab:mem_audit` (byte audit) | `e4_mem_audit.json` | `rows[].{peak_gb,baseline_QD_gb,grads_gb,analytic_argmax_gb,residual_transient_gb}` | `scripts/e4_mem_audit.py` |
| App. serving frontier (K→200K) | `e6_rerank_frontier.json` | `rows[].{unchunked,chunked,flash}.{ms,peak_gb}` | `scripts/e6_rerank_frontier.py` |
| App. complete pipeline (HotpotQA-500K) | `e5_pipeline.json` | `encode_ms_per_query_b1`, `rows[].{rerank_ms_flash,rerank_ms_chunked,qps_flash,qps_chunked,ndcg10_reranked}` | `scripts/e5_pipeline.py` (needs `E5_EMBEDDINGS` env var — pre-encoded ColBERTv2 embeddings; JSON shipped so no re-encode needed) |
| `tab:e2e_train` (main-text summary) | derived: rows 1–2 from `tab:fullstep`, row 3 from `tab:batchquality`, row 4 from `tab:ddp` B=2048 | — | — |

Note: the paper's `tab:ddp` "ours" columns are the `fm-flash-tiled` arm (stated
in the caption); the JSONs also contain the `fm-sharded` arm. The DDP W=2
conventional peaks were re-measured after the rebuttal was posted; the paper
matches the shipped JSONs (9.21/36.5 GB), which differ from the posted rebuttal
comment (9.17/36.4 GB) by <0.5% run-to-run variance.
