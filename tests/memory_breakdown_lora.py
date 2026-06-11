"""LoRA contrastive training — memory breakdown, torch vs flash MaxSim.

Builds a realistic LoRA fine-tune setup at ColPali shape:
  • frozen MLP "backbone" (~20 M params, FP16)
  • LoRA adapters (~0.5 M trainable params, rank 16)
  • encoder produces query / doc token embeddings [B, L, d_emb]
  • contrastive in-batch-negatives loss (cross-entropy, diagonal positive)
  • Adam only over LoRA params

Runs ONE training step end-to-end (encoder fwd → scoring → loss → backward
→ Adam) for each scoring path. Reports peak GPU memory broken down by
named phase, so you can see exactly where the extra ~16 GB in the torch
path comes from. Produces a side-by-side stacked bar plot.

Run:   python tests/memory_breakdown_lora.py
"""
from __future__ import annotations
import sys, gc
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/path/to/data")
from flash_maxsim import flash_maxsim_batched_train

DEV = "cuda"

# ── ColPali-ish regime ──────────────────────────────────────────────────
B          = 64        # batch (queries == documents — in-batch negatives)
L          = 1024      # token-sequence length for both Q and D
D_IN       = 768       # backbone hidden width
D_OUT      = 128       # final projection (MaxSim embedding dim)
N_BLOCKS   = 4         # transformer-like blocks (MLP-only here, for speed)
LORA_RANK  = 16
SEED       = 42
TEMP       = float(L) ** 0.5


# ── LoRA layer ──────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Linear(W + (B @ A)) — W frozen, A & B trainable. peft-equivalent."""
    def __init__(self, in_d, out_d, rank=16):
        super().__init__()
        self.base = nn.Linear(in_d, out_d, bias=False)
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(rank, in_d) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_d, rank))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B)


class Block(nn.Module):
    """MLP block with LoRA on both projections, base frozen FP16."""
    def __init__(self, d, mult=4, rank=16):
        super().__init__()
        self.up   = LoRALinear(d, d * mult, rank)
        self.down = LoRALinear(d * mult, d, rank)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        h = self.norm(x)
        h = F.gelu(self.up(h))
        h = self.down(h)
        return x + h


class TinyEncoder(nn.Module):
    """A small LoRA-augmented MLP encoder. Backbone frozen in FP16,
    LoRA adapters trainable in FP32 (the standard mixed-precision LoRA
    convention). Output projected to D_OUT for MaxSim scoring."""
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(
            [Block(D_IN, mult=4, rank=LORA_RANK) for _ in range(N_BLOCKS)])
        self.proj = nn.Linear(D_IN, D_OUT, bias=False)
        for p in self.proj.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.proj(x)                       # [B, L, D_OUT]


# ── scoring functions ───────────────────────────────────────────────────

def torch_scores(Q_fp16, D_fp16):
    sim = torch.einsum("iqd,jld->ijql", Q_fp16, D_fp16)
    return sim.max(dim=-1).values.sum(dim=-1)

def flash_scores(Q_fp16, D_fp16):
    return flash_maxsim_batched_train(Q_fp16, D_fp16, shared_docs=True)


# ── memory probes ───────────────────────────────────────────────────────

def gb(): return torch.cuda.memory_allocated() / 1024**3
def peak_gb(): return torch.cuda.max_memory_allocated() / 1024**3

def sync_reset_peak():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


# ── one full training step, fully instrumented ──────────────────────────

def step_breakdown(score_fn, label):
    """Returns dict of memory deltas at each phase, all in GB."""
    torch.manual_seed(SEED)
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

    # ----- 1. build model -----
    enc = TinyEncoder().to(DEV)
    # Frozen backbone → FP16. LoRA adapters stay FP32.
    for n, p in enc.named_parameters():
        if not p.requires_grad and p.dtype == torch.float32:
            p.data = p.data.to(torch.float16)
    trainable = [p for p in enc.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=1e-3)
    torch.cuda.synchronize()
    mem_model = gb()
    n_frozen   = sum(p.numel() for p in enc.parameters() if not p.requires_grad)
    n_lora     = sum(p.numel() for p in trainable)

    # ----- 2. simulate inputs -----
    # Random token inputs (FP16, what comes out of an upstream tokenizer/embedder).
    q_in = torch.randn(B, L, D_IN, device=DEV, dtype=torch.float16)
    d_in = torch.randn(B, L, D_IN, device=DEV, dtype=torch.float16)
    mem_inputs = gb()

    # ----- 3. encoder forward (autocast FP16) -----
    sync_reset_peak()
    base_for_fwd = gb()
    with torch.amp.autocast("cuda", dtype=torch.float16):
        Q_emb = enc(q_in)                        # [B, L, D_OUT]
        D_emb = enc(d_in)
        Q_emb = F.normalize(Q_emb.float(), dim=-1).to(torch.float16)
        D_emb = F.normalize(D_emb.float(), dim=-1).to(torch.float16)
    torch.cuda.synchronize()
    peak_encoder_fwd = peak_gb() - base_for_fwd  # delta from before encoder fwd
    mem_after_enc    = gb()                       # activations still alive

    # ----- 4. scoring forward -----
    sync_reset_peak()
    base_for_score = gb()
    scores = score_fn(Q_emb, D_emb) / TEMP
    torch.cuda.synchronize()
    peak_score_fwd = peak_gb() - base_for_score
    mem_after_score = gb()

    # ----- 5. loss + backward -----
    sync_reset_peak()
    base_for_bwd = gb()
    target = torch.arange(B, device=DEV)
    loss   = F.cross_entropy(scores, target)
    loss.backward()
    torch.cuda.synchronize()
    peak_bwd = peak_gb() - base_for_bwd
    mem_after_bwd = gb()

    # ----- 6. Adam step -----
    sync_reset_peak()
    base_for_opt = gb()
    opt.step()
    opt.zero_grad()
    torch.cuda.synchronize()
    peak_opt = peak_gb() - base_for_opt
    mem_after_opt = gb()

    total_peak = max(mem_after_enc + peak_score_fwd,
                     mem_after_score + (peak_bwd - 0),
                     mem_after_bwd + peak_opt)
    # Simpler: full-step peak measured from a clean start:
    sync_reset_peak()
    torch.cuda.reset_peak_memory_stats()
    # we already ran the step; pull the high-water mark we recorded across phases:
    full_peak = max(mem_model, mem_inputs, mem_after_enc + peak_score_fwd,
                    mem_after_score + peak_bwd, mem_after_bwd + peak_opt)

    print(f"\n── {label.upper()} ───────────────────────────────────────────")
    print(f"  Frozen backbone params  : {n_frozen/1e6:6.2f} M  (FP16)")
    print(f"  LoRA trainable params   : {n_lora/1e6:6.2f} M  (FP32)")
    print(f"  Model + Adam state      : {mem_model:6.3f} GB")
    print(f"  + random Q,D inputs     : {mem_inputs:6.3f} GB")
    print(f"  + encoder fwd peak Δ    : {peak_encoder_fwd:6.3f} GB  "
          f"(activations saved for backward = {mem_after_enc-mem_inputs:.3f} GB)")
    print(f"  + scoring fwd peak Δ    : {peak_score_fwd:6.3f} GB  "
          f"(sim matrix in torch / nothing in flash)")
    print(f"  + backward peak Δ       : {peak_bwd:6.3f} GB  "
          f"(grad_sim + LoRA grads)")
    print(f"  + Adam peak Δ           : {peak_opt:6.3f} GB")
    print(f"  ──────────────────────────────")
    print(f"  Full-step PEAK          : {full_peak:6.3f} GB")

    return dict(
        label=label,
        model=mem_model,
        encoder_acts=mem_after_enc - mem_inputs,
        score_extra=peak_score_fwd,
        bwd_extra=peak_bwd,
        opt_extra=peak_opt,
        full_peak=full_peak,
        n_frozen=n_frozen, n_lora=n_lora,
    )


# ── main ────────────────────────────────────────────────────────────────

print(f"GPU: {torch.cuda.get_device_name(0)}  "
      f"({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB)")
print(f"LoRA contrastive regime: B={B}  L={L}  D_in={D_IN}  D_out={D_OUT}  "
      f"blocks={N_BLOCKS}  rank={LORA_RANK}")

t = step_breakdown(torch_scores, "torch")
f = step_breakdown(flash_scores, "flash")

print("\n══ Side-by-side full-step peak ════════════════════════════════")
print(f"  torch peak  = {t['full_peak']:6.2f} GB")
print(f"  flash peak  = {f['full_peak']:6.2f} GB  "
      f"({t['full_peak']/max(f['full_peak'],1e-6):.1f}× less)")


# ── stacked bar plot ────────────────────────────────────────────────────

cats = ["Model + Adam state",
        "Encoder activations",
        "Scoring fwd extra (sim matrix)",
        "Backward extra (grad_sim + grads)",
        "Optimizer step extra"]
vals_t = [t["model"], t["encoder_acts"], t["score_extra"], t["bwd_extra"], t["opt_extra"]]
vals_f = [f["model"], f["encoder_acts"], f["score_extra"], f["bwd_extra"], f["opt_extra"]]
colors = ["#888", "#1976D2", "#D81B60", "#F4511E", "#43A047"]

fig, ax = plt.subplots(figsize=(8, 6))
x = np.array([0, 1.0])
bottoms_t = 0; bottoms_f = 0
for v_t, v_f, c, lbl in zip(vals_t, vals_f, colors, cats):
    ax.bar(x[0], v_t, 0.6, bottom=bottoms_t, color=c, label=lbl)
    ax.bar(x[1], v_f, 0.6, bottom=bottoms_f, color=c)
    bottoms_t += v_t; bottoms_f += v_f

ax.set_xticks(x); ax.set_xticklabels(["torch", "flash"])
ax.set_ylabel("peak memory contribution (GB)")
ax.set_title(f"LoRA contrastive training — memory breakdown\n"
             f"(B={B}, L={L}, D={D_OUT}, frozen={t['n_frozen']/1e6:.0f}M, "
             f"LoRA={t['n_lora']/1e6:.1f}M @ rank {LORA_RANK})")
ax.grid(alpha=0.3, axis="y")
ax.legend(loc="upper right", fontsize=9)
for i, v in enumerate([t["full_peak"], f["full_peak"]]):
    ax.text(x[i], v + 0.3, f"{v:.2f} GB", ha="center",
            fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig("~/memory_breakdown_lora.png", dpi=140, bbox_inches="tight")
print("\nwrote ~/memory_breakdown_lora.png")
