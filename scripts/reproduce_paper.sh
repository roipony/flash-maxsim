#!/usr/bin/env bash
# Reproduce the SIX CORE paper tables from the public flash-maxsim repo.
#
# Scope (six tables/figures only — not the full paper):
#   - tab:fwd      forward latency vs naive at matched precision
#   - tab:traffic  HBM traffic + chunk ablation + backward bandwidth
#   - tab:ooc      out-of-core corpus scaling
#   - tab:train    contrastive training step
#   - tab:varlen   variable-length scoring (three fill-ratio buckets)
#   - fig:scaling  A100-40GB OOM cliffs (JSON source for the figure)
#
# Tables NOT reproduced by this script — each has its own bench_*.py listed
# in PAPER_NUMBERS.md under the per-table source pointer:
#   tab:bwd_abl         (3-path backward: bench_bwd_paths_3way.py)
#   tab:bwd_breakdown   (bench_bwd_breakdown.py)
#   tab:int8 / int8_B   (bench_int8_shapes.py / bench_int8_B_sweep.py)
#   tab:fat_emb_cliff   (bench_fat_embeddings.py)
#   tab:splitd_perf     (bench_splitd_perf.py)
#   tab:op_abl          (bench_operator_design.py)
#   tab:beir_ndcg       (bench_beir_ndcg.py; shipped JSONs in paper/data/)
#   tab:colpali_ndcg    (bench_real_mm_rag_ndcg.py + bench_real_mm_rag_parity.py)
#   fig:training_parity (bench_training_parity_curve.py — 500-step training loop)
#   tab:bwd_mem         (analytical, not measured)
#   tab:precision / tab:baselines / tab:shapes (protocol / glossary, not measured)
#
# Speedup numbers may differ by ~1-5% across hardware/runs (CUDA-event timing
# variance, autotune choices); the ratios and OOM cliffs should reproduce
# exactly. This script does NOT diff-check outputs against shipped JSONs;
# users should compare manually against PAPER_NUMBERS.md.
#
# Requires:
#   - An NVIDIA GPU (A100 or H100 recommended) with PyTorch 2.8 + Triton 3.5+
#   - `pip install -e .`
#   - This repo cloned at $FM_REPO (defaults to ../flash-maxsim)
#
# Usage:
#   FM_REPO=/path/to/flash-maxsim PY=/path/to/python bash reproduce_paper.sh

set -euo pipefail

FM_REPO="${FM_REPO:-../flash-maxsim}"
PY="${PY:-python}"
BD="${FM_REPO}/benchmarks"

cd "${BD}"
echo "==> reproducing paper tables from ${BD}"
echo "==> using python: $(${PY} -c 'import sys; print(sys.executable, sys.version)' )"

run() {
    local label="$1"; shift
    echo
    echo "------ ${label} (${*}) ------"
    "${PY}" "$@"
}

run "tab:fwd (forward latency, both TF32 modes)"        bench_fwd_fair.py
run "tab:traffic + chunk ablation + backward bandwidth"  bench_paper_io.py
run "tab:ooc (out-of-core corpus scaling)"               bench_ooc.py
run "tab:train (truly-naive autograd vs Flash)"          bench_train_paper.py
run "tab:varlen (three fill-ratio buckets)"              bench_varlen_buckets.py
run "fig:scaling source (A100-40GB OOM cliffs)"          bench_blog_40gb.py

echo
echo "==> done. New JSONs in ${BD}/bench_*.json and results_*.json"
echo "==> compare against PAPER_NUMBERS.md to audit."
