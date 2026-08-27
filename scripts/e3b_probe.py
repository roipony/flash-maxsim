"""E3b feasibility probe: one full ColPali LoRA contrastive training step.

Round 2: colpali_engine's load_adapter hits a transformers-5.5.4 bug
(KeyError 'llava' in the MoE peft-config conversion), so this version uses
the NATIVE transformers ColPali port (vidore/colpali-v1.2-hf) and attaches
fresh LoRA adapters directly with peft (the standard ColPali training
recipe: r=32 on the language-model attention projections).

Measures ONE full training step (encoder fwd for B pages + B queries,
in-batch contrastive loss, backward, AdamW step) at B = 4..32 for the
torch vs flash scoring backends: peak GB + wall-clock.

Text-query regime (real ColPali fine-tuning): queries are FinTab questions,
docs are page images (Ld ~= 1030 patches).
"""
from __future__ import annotations
import gc, json, os, time
from pathlib import Path

import torch

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
OUT = HERE / "e1_results" / "e3b_probe.json"
DEV = "cuda"
MODEL_ID = "vidore/colpali-v1.2-hf"


def attach_lora(model):
    """Fresh LoRA on the LM attention projections (standard ColPali recipe)."""
    try:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(
            r=32, lora_alpha=32, lora_dropout=0.1, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, cfg)
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return model, f"peft-lora r=32 ({n/1e6:.1f}M trainable)"
    except Exception as e:  # noqa: BLE001
        # fallback: train only the last 4 LM layers + projection
        for p in model.parameters():
            p.requires_grad = False
        n = 0
        for name, p in model.named_parameters():
            if any(f"layers.{i}." in name for i in range(14, 18)) \
                    or "embedding_proj" in name or "custom_text_proj" in name:
                p.requires_grad = True
                n += p.numel()
        return model, f"fallback last-4-layers ({n/1e6:.1f}M); peft failed: {e}"


def get_batches(processor, n: int, page_as_query: bool = False):
    from datasets import load_dataset
    ds = load_dataset("ibm-research/REAL-MM-RAG_FinTabTrainSet", split="train")
    cols = ds.column_names
    print("FinTab columns:", cols, "n=", len(ds), flush=True)
    img_col = next(c for c in cols if c.lower() in
                   ("image", "img", "page_image", "page"))
    txt_col = next(c for c in cols if "quer" in c.lower()
                   or "question" in c.lower() or c.lower() == "text")
    rows = ds.select(range(2 * n if page_as_query else n))
    images = [rows[i][img_col] for i in range(n)]
    batch_docs = processor(images=images, return_tensors="pt")
    if page_as_query:
        # page-as-query regime (paper flagship): the query is itself a page
        q_images = [rows[n + i][img_col] for i in range(n)]
        batch_qs = processor(images=q_images, return_tensors="pt")
    else:
        queries = [str(rows[i][txt_col]) for i in range(n)]
        batch_qs = processor(text=queries, return_tensors="pt")
    return batch_docs, batch_qs


def embed(model, batch):
    out = model(**batch)
    emb = getattr(out, "embeddings", None)
    if emb is None:
        emb = out[0] if isinstance(out, tuple) else out.last_hidden_state
    return emb


def one_step(model, batch_docs, batch_qs, backend, optimizer):
    from pylate.scores import colbert_scores

    torch.cuda.synchronize()
    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    docs = {k: v.to(DEV) for k, v in batch_docs.items()}
    qs = {k: v.to(DEV) for k, v in batch_qs.items()}
    D = embed(model, docs)
    Q = embed(model, qs)
    scores = colbert_scores(Q, D, backend=backend)
    labels = torch.arange(scores.shape[0], device=DEV)
    loss = torch.nn.functional.cross_entropy(scores / 0.02, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return {
        "loss": float(loss),
        "wall_s": round(time.time() - t0, 2),
        "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "Lq": int(Q.shape[1]), "Ld": int(D.shape[1]), "dim": int(D.shape[2]),
    }


def main():
    from transformers import ColPaliForRetrieval, ColPaliProcessor

    res = {"gpu": torch.cuda.get_device_name(0), "model": MODEL_ID}
    print("loading", MODEL_ID, flush=True)
    model = ColPaliForRetrieval.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEV)
    processor = ColPaliProcessor.from_pretrained(MODEL_ID)
    # the top-level retrieval wrapper rejects gradient checkpointing; enable it
    # on every inner module that supports it (LM + vision tower)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    gc_on = []
    for name, mod in model.named_modules():
        if getattr(mod, "supports_gradient_checkpointing", False):
            try:
                mod.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
                gc_on.append(name or "<root>")
            except Exception:  # noqa: BLE001
                pass
    res["grad_ckpt"] = gc_on or "none supported"
    print("grad-checkpointing on:", gc_on, flush=True)

    model, lora_note = attach_lora(model)
    res["lora"] = lora_note
    print(lora_note, flush=True)
    model.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=5e-5)

    page_as_query = os.environ.get("E3B_PAGE_AS_QUERY", "0") == "1"
    res["page_as_query"] = page_as_query
    batches = (8, 16, 32, 64, 96) if page_as_query else (8, 16, 32, 64, 128)

    res["steps"] = []
    for B in batches:
        try:
            batch_docs, batch_qs = get_batches(processor, B, page_as_query)
        except Exception as e:  # noqa: BLE001
            res["steps"].append({"B": B, "error": f"data: {type(e).__name__}: {e}"})
            break
        for backend in ("torch", "flash"):
            try:
                r = one_step(model, batch_docs, batch_qs, backend, optimizer)
                r.update({"B": B, "backend": backend})
                res["steps"].append(r)
                print(json.dumps(r), flush=True)
            except torch.cuda.OutOfMemoryError:
                res["steps"].append({"B": B, "backend": backend, "oom": True})
                print(f"B={B} {backend}: OOM", flush=True)
                gc.collect(); torch.cuda.empty_cache()
            except Exception as e:  # noqa: BLE001
                res["steps"].append({"B": B, "backend": backend,
                                     "error": f"{type(e).__name__}: {e}"})
                print(f"B={B} {backend}: {type(e).__name__}: {e}", flush=True)

    out = OUT.with_name("e3b_probe_paq.json") if page_as_query else OUT
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
