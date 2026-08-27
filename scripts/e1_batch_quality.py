"""E1 — does the batch size unlocked by Flash-MaxSim produce a BETTER retriever?

Reviewer vNrM: "the practical benefit is currently demonstrated as feasibility,
not improved model quality." This script converts the feasibility claim into a
measurable quality claim, or falsifies it.

Design
------
Batch size in in-batch-negative contrastive training IS the negative-pool size
(B queries x N docs => 2B-1 negatives per query). Flash-MaxSim removes the
[Nq,B,Lq,Ld] similarity tensor, which is the term that caps B on one GPU. So:

    arm := (batch_size B, scoring backend, seed)

Every arm runs IDENTICAL code except the MaxSim implementation, selected via
pylate's `ColBERTScores(backend=...)` knob. The `torch` arms OOM above some B;
the `flash` arms keep going. The question is whether the extra negatives buy
retrieval quality.

Controls (the two things a reviewer will attack)
------------------------------------------------
1. DATA BUDGET. Arms see the same number of training *examples*, not the same
   number of *steps* -- otherwise large-B arms would simply see more data and
   the comparison would be confounded. max_steps = TOTAL_EXAMPLES // B.
2. LEARNING RATE. Comparing batch sizes at one fixed LR invites "you just needed
   to retune". Default policy scales LR as sqrt(B/B_ref) (standard for
   contrastive objectives); `--lr-mode fixed` reruns the extremes without
   scaling as a robustness check. Both are reported.

Evaluation is deliberately run through the *torch reference* scoring path for
every arm, so the quality number never depends on our own kernel.

Usage
-----
  python e1_batch_quality.py probe                 # OOM frontier, both backends
  python e1_batch_quality.py run --batch-size 256 --backend flash --seed 0
  python e1_batch_quality.py collect               # merge arm JSONs -> summary

Output: analysis/e1_results/<arm>.json, one per arm.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
RESULTS = HERE / "e1_results"
CKPT_ROOT = Path(os.environ.get("E1_CKPT_ROOT", HERE / "e1_ckpt"))

# ---------------------------------------------------------------- experiment
TOTAL_EXAMPLES = int(os.environ.get("E1_TOTAL_EXAMPLES", 400_000))
B_REF = 32                      # LR reference batch size
BASE_LR = 3e-6 * (32 / B_REF)   # pylate contrastive example LR at B_REF
QUERY_LEN = 32
DOC_LEN = 180
BASE_MODEL = "bert-base-uncased"
TRAIN_DATASET = ("sentence-transformers/msmarco-bm25", "triplet")

# NanoBEIR: 50 queries / ~3K docs each -> exact eval in seconds, 4 domains.
NANO_BEIR = [
    "zeta-alpha-ai/NanoNFCorpus",
    "zeta-alpha-ai/NanoSciFact",
    "zeta-alpha-ai/NanoNQ",
    "zeta-alpha-ai/NanoFiQA2018",
]

PROBE_BATCHES = [32, 64, 128, 256, 512, 1024, 2048]


def arm_name(batch_size: int, backend: str, seed: int, lr_mode: str,
             grad_checkpointing: bool = False) -> str:
    gc_tag = "_gc" if grad_checkpointing else ""
    return f"b{batch_size}_{backend}_s{seed}_lr-{lr_mode}{gc_tag}"


def resolve_lr(batch_size: int, lr_mode: str) -> float:
    if lr_mode == "fixed":
        return BASE_LR
    if lr_mode == "sqrt":
        return BASE_LR * math.sqrt(batch_size / B_REF)
    raise ValueError(f"unknown lr_mode {lr_mode!r}")


# ------------------------------------------------------------ backend control
def fixed_backend_scores(backend: str):
    """ColBERTScores pinned to one MaxSim implementation.

    This is the ONLY difference between a `torch` arm and a `flash` arm.
    """
    from pylate.scores import ColBERTScores

    class _Pinned(ColBERTScores):
        def __call__(
            self,
            queries_embeddings,
            documents_embeddings,
            queries_mask=None,
            documents_mask=None,
            backend=None,
        ):
            return super().__call__(
                queries_embeddings,
                documents_embeddings,
                queries_mask=queries_mask,
                documents_mask=documents_mask,
                backend=_Pinned._pinned,
            )

    _Pinned._pinned = backend
    return _Pinned()


# ------------------------------------------------------------------- training
def train_arm(batch_size: int, backend: str, seed: int, lr_mode: str,
              grad_checkpointing: bool, out_dir: Path) -> dict:
    from datasets import load_dataset
    from sentence_transformers import (
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from pylate import losses, models, utils

    lr = resolve_lr(batch_size, lr_mode)
    max_steps = TOTAL_EXAMPLES // batch_size

    model = models.ColBERT(
        model_name_or_path=BASE_MODEL,
        query_length=QUERY_LEN,
        document_length=DOC_LEN,
    )

    dataset = load_dataset(*TRAIN_DATASET, split="train")
    # Deterministic slice: every arm draws from the same prefix, so arms differ
    # only in how those examples are grouped into batches.
    n_needed = min(len(dataset), TOTAL_EXAMPLES)
    train_dataset = dataset.select(range(n_needed))

    train_loss = losses.Contrastive(
        model=model,
        score_metric=fixed_backend_scores(backend),
    )

    args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir),
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        warmup_steps=max(1, int(0.05 * max_steps)),
        lr_scheduler_type="linear",
        bf16=True,
        fp16=False,
        seed=seed,
        data_seed=seed,
        logging_steps=max(1, max_steps // 20),
        save_strategy="no",
        report_to=[],
        gradient_checkpointing=grad_checkpointing,
        dataloader_num_workers=4,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=train_loss,
        data_collator=utils.ColBERTCollator(model.tokenize),
    )

    # sentence-transformers 5.1 calls Trainer._nested_gather, which transformers
    # 5.x removed. Single process here, so gathering is the identity.
    if not hasattr(trainer, "_nested_gather"):
        trainer._nested_gather = lambda tensors, *a, **kw: tensors

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    train_out = trainer.train()
    wall = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    model.save(str(out_dir))

    hist = [h for h in trainer.state.log_history if "loss" in h]
    return {
        "max_steps": max_steps,
        "examples_seen": max_steps * batch_size,
        "negatives_per_query": 2 * batch_size - 1,
        "learning_rate": lr,
        "train_wall_s": wall,
        "train_peak_gb": peak_gb,
        "final_loss": train_out.training_loss,
        "loss_curve": [{"step": h["step"], "loss": h["loss"]} for h in hist],
    }


# ----------------------------------------------------------------- evaluation
def ndcg_at_k(ranked_ids: list[str], relevant: set[str], k: int = 10) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, doc_id in enumerate(ranked_ids[:k])
        if doc_id in relevant
    )
    ideal = sum(1.0 / math.log2(r + 2) for r in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def _pad_stack(embs, device: str):
    """pylate encode() returns a list of [Li, H] tensors; pad-stack + mask."""
    if torch.is_tensor(embs):
        return embs.to(device), None
    lens = torch.tensor([e.shape[0] for e in embs])
    T = torch.nn.utils.rnn.pad_sequence(
        [e.cpu() for e in embs], batch_first=True
    ).to(device)
    mask = (torch.arange(T.shape[1])[None, :] < lens[:, None]).to(device)
    return T, mask


@torch.no_grad()
def evaluate_nano_beir(model_dir: Path, doc_chunk: int = 512) -> dict:
    """Exact nDCG@10 on NanoBEIR, scored through the TORCH reference path.

    Using the reference scorer (not our kernel) for every arm keeps the quality
    measurement independent of the thing under test.
    """
    from datasets import load_dataset
    from pylate import models
    from pylate.scores import colbert_scores

    model = models.ColBERT(model_name_or_path=str(model_dir))
    model.eval().to("cuda")

    per_dataset = {}
    for ds_name in NANO_BEIR:
        corpus = load_dataset(ds_name, "corpus", split="train")
        queries = load_dataset(ds_name, "queries", split="train")
        qrels = load_dataset(ds_name, "qrels", split="train")

        rel: dict[str, set[str]] = {}
        for row in qrels:
            rel.setdefault(row["query-id"], set()).add(row["corpus-id"])

        doc_ids = corpus["_id"]
        q_ids = queries["_id"]

        Q = model.encode(
            queries["text"], is_query=True, batch_size=64,
            convert_to_numpy=False, convert_to_tensor=True,
            show_progress_bar=False,
        )
        # queries are expanded to a fixed query_length, so no mask needed
        Q, _ = _pad_stack(Q, "cuda")

        scores = []
        for i in range(0, len(doc_ids), doc_chunk):
            D = model.encode(
                corpus["text"][i:i + doc_chunk], is_query=False, batch_size=64,
                convert_to_numpy=False, convert_to_tensor=True,
                show_progress_bar=False,
            )
            D, d_mask = _pad_stack(D, "cuda")
            scores.append(
                colbert_scores(
                    Q, D, documents_mask=d_mask, backend="torch"
                ).float().cpu()
            )
            del D
        scores = torch.cat(scores, dim=1)          # [Nq, Ndocs]

        topk = scores.topk(k=min(10, scores.shape[1]), dim=1).indices
        vals = [
            ndcg_at_k([doc_ids[j] for j in topk[qi].tolist()],
                      rel.get(qid, set()), k=10)
            for qi, qid in enumerate(q_ids)
        ]
        per_dataset[ds_name.split("/")[-1]] = sum(vals) / len(vals)
        del scores, Q
        gc.collect()
        torch.cuda.empty_cache()

    per_dataset["mean"] = sum(
        v for k, v in per_dataset.items() if k != "mean"
    ) / len(NANO_BEIR)
    return per_dataset


# ---------------------------------------------------------------------- probe
def probe() -> dict:
    """Where does each backend hit the memory wall, at real training shapes?

    Isolates the scoring op (encoder excluded) so the frontier is attributable
    to the similarity tensor rather than to encoder activations.
    """
    from pylate.scores import colbert_scores

    d = 128
    # Warm up both backends first: the flash kernel's first call pays Triton
    # autotune workspace, which otherwise shows up as a phantom peak on the
    # smallest probed shape.
    for backend in ("torch", "flash"):
        Qw = torch.nn.functional.normalize(
            torch.randn(8, QUERY_LEN, d, device="cuda", dtype=torch.bfloat16),
            dim=-1).requires_grad_(True)
        Dw = torch.nn.functional.normalize(
            torch.randn(8, DOC_LEN, d, device="cuda", dtype=torch.bfloat16),
            dim=-1).requires_grad_(True)
        colbert_scores(Qw, Dw, backend=backend).sum().backward()
    torch.cuda.synchronize()
    Qw = Dw = None
    gc.collect()
    torch.cuda.empty_cache()

    out = []
    for B in PROBE_BATCHES:
        row = {"batch_size": B}
        for backend in ("torch", "flash"):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                Q = torch.nn.functional.normalize(
                    torch.randn(B, QUERY_LEN, d, device="cuda",
                                dtype=torch.bfloat16), dim=-1
                ).requires_grad_(True)
                D = torch.nn.functional.normalize(
                    torch.randn(B, DOC_LEN, d, device="cuda",
                                dtype=torch.bfloat16), dim=-1
                ).requires_grad_(True)
                s = colbert_scores(Q, D, backend=backend)
                s.sum().backward()
                torch.cuda.synchronize()
                row[backend] = {
                    "ok": True,
                    "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
                }
            except torch.cuda.OutOfMemoryError:
                row[backend] = {"ok": False, "peak_gb": None, "error": "OOM"}
            except Exception as e:  # noqa: BLE001
                row[backend] = {"ok": False, "peak_gb": None,
                                "error": f"{type(e).__name__}: {e}"}
            finally:
                Q = D = s = None
                gc.collect()
                torch.cuda.empty_cache()
        out.append(row)
        print(json.dumps(row), flush=True)
    return {"probe": out, "query_len": QUERY_LEN, "doc_len": DOC_LEN, "dim": d}


# ----------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["probe", "run", "eval", "collect"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--backend", choices=["torch", "flash"], default="flash")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr-mode", choices=["sqrt", "fixed"], default="sqrt")
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--model-dir", type=str, default=None)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    if args.mode == "probe":
        res = probe()
        res["gpu"] = gpu
        (RESULTS / "probe.json").write_text(json.dumps(res, indent=2))
        print(f"wrote {RESULTS / 'probe.json'}")
        return

    if args.mode == "eval":
        res = evaluate_nano_beir(Path(args.model_dir))
        print(json.dumps(res, indent=2))
        return

    if args.mode == "collect":
        rows = [json.loads(p.read_text())
                for p in sorted(RESULTS.glob("b*_*.json"))]
        summary = sorted(
            ({"arm": r["arm"], "batch_size": r["batch_size"],
              "backend": r["backend"], "seed": r["seed"],
              "lr_mode": r["lr_mode"],
              "ndcg10_mean": r["eval"]["mean"],
              "train_peak_gb": r["train"]["train_peak_gb"],
              "final_loss": r["train"]["final_loss"]} for r in rows),
            key=lambda x: (x["lr_mode"], x["batch_size"], x["backend"], x["seed"]),
        )
        (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return

    # mode == "run": train + eval one arm
    name = arm_name(args.batch_size, args.backend, args.seed, args.lr_mode,
                    args.grad_checkpointing)
    out_json = RESULTS / f"{name}.json"
    ckpt = CKPT_ROOT / name
    ckpt.mkdir(parents=True, exist_ok=True)

    record = {
        "arm": name,
        "batch_size": args.batch_size,
        "backend": args.backend,
        "seed": args.seed,
        "lr_mode": args.lr_mode,
        "gpu": gpu,
        "base_model": BASE_MODEL,
        "total_examples_budget": TOTAL_EXAMPLES,
        "grad_checkpointing": args.grad_checkpointing,
    }

    try:
        record["train"] = train_arm(
            args.batch_size, args.backend, args.seed, args.lr_mode,
            args.grad_checkpointing, ckpt,
        )
    except torch.cuda.OutOfMemoryError as e:
        record["train"] = {"oom": True, "error": str(e)[:400]}
        out_json.write_text(json.dumps(record, indent=2))
        print(f"OOM at B={args.batch_size} backend={args.backend} -> {out_json}")
        return

    gc.collect()
    torch.cuda.empty_cache()
    record["eval"] = evaluate_nano_beir(ckpt)
    out_json.write_text(json.dumps(record, indent=2))
    print(json.dumps(record["eval"], indent=2))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
