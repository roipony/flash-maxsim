"""Generate landing page figure from continuous sweep data.

Usage:
    python benchmarks/make_figure.py benchmarks/results_*.json
    # produces benchmarks/flash_maxsim_benchmarks.png
"""
import json, sys, glob
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ── Load data ──
files = sys.argv[1:] or sorted(glob.glob("benchmarks/results_*.json"))
assert files, "No result files found. Run collect_data.py first."

all_data = []
for f in files:
    with open(f) as fh:
        all_data.append(json.load(fh))
    print(f"Loaded: {f} ({all_data[-1]['gpu']})")

data = all_data[0]
gpu = data["gpu"]

# ── Style ──
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.2,
})

# Bold, high-contrast colors
LQ_COLORS = {
    32:   "#1565C0",  # strong blue
    128:  "#2E7D32",  # green
    512:  "#E65100",  # orange
    1024: "#C62828",  # red
}

TAG_COLORS = {
    "textual":  "#1565C0",
    "long_doc": "#6A1B9A",
    "visual":   "#C62828",
}
TAG_LABELS = {
    "textual":  "Textual (Lq=32, Ld=300)",
    "long_doc": "Long-doc (Lq=32, Ld=1024)",
    "visual":   "Visual (Lq=1024, Ld=1024)",
}

BG = "#F8F9FA"

fig = plt.figure(figsize=(20, 12), facecolor="white")

fig.suptitle("Flash-MaxSim", fontsize=28, fontweight="bold", y=0.97)
torch_ver = data.get("torch", "")
triton_ver = data.get("triton", "")
ver_str = f"  ·  torch {torch_ver}  ·  triton {triton_ver}" if torch_ver else ""
fig.text(0.5, 0.932,
         f"Fused GPU kernel for late-interaction retrieval  ·  {gpu}{ver_str}",
         ha="center", fontsize=14, color="#555", style="italic")

# ═══════════════════════════════════════════════════════════════════
# Panel 1: Speedup landscape — log scale so Flash lines are visible
# ═══════════════════════════════════════════════════════════════════
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_facecolor(BG)

sweep = data["sweep_seq"]
lq_values = sorted(set(d["Lq"] for d in sweep))

# Shade regions
ax1.axvspan(0, 400, alpha=0.07, color="#1565C0", zorder=0)
ax1.axvspan(400, 1100, alpha=0.05, color="#E65100", zorder=0)
ax1.text(200, 0.97, "Textual", ha="center", fontsize=11, color="#1565C0", alpha=0.6,
         fontweight="bold", transform=ax1.get_xaxis_transform(), va="top")
ax1.text(750, 0.97, "Visual / Long-doc", ha="center", fontsize=11, color="#E65100", alpha=0.6,
         fontweight="bold", transform=ax1.get_xaxis_transform(), va="top")

# Annotate speedup at first and last Ld points
ANNOTATE_LDS = {128, 1024}

for Lq in lq_values:
    pts = sorted([d for d in sweep if d["Lq"] == Lq], key=lambda d: d["Ld"])
    lds = [d["Ld"] for d in pts]
    naive = [d["naive_ms"] for d in pts]
    flash = [d["flash_ms"] for d in pts]
    speedup = [d["speedup"] for d in pts]
    c = LQ_COLORS.get(Lq, "#333")

    ax1.plot(lds, naive, '--', color=c, alpha=0.7, lw=2.5)
    ax1.plot(lds, flash, '-', color=c, lw=3.5, marker='o', markersize=7,
             markeredgecolor="white", markeredgewidth=1, label=f"Lq={Lq}", zorder=5)

    for i, ld in enumerate(lds):
        if ld in ANNOTATE_LDS:
            # Place label to the right at last point, above at first point
            if ld == lds[-1]:
                xytext, ha = (8, 0), "left"
            else:
                xytext, ha = (0, 10), "center"
            ax1.annotate(f"{speedup[i]}x", (ld, flash[i]),
                         fontsize=9, fontweight="bold", color=c, ha=ha,
                         xytext=xytext, textcoords="offset points",
                         bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=c, alpha=0.85, lw=0.8))

ax1.set_yscale("log")
ax1.set_xlabel("Document length (Ld)", fontsize=12)
ax1.set_ylabel("Latency (ms, log scale)", fontsize=12)
ax1.set_title("Single Query · Vary Sequence Length (B=1000)\nsolid = Flash, dashed = Naive",
              fontsize=14, fontweight="bold", pad=14)
ax1.legend(loc="upper left", fontsize=10, title="Query length", title_fontsize=11,
           framealpha=0.9, edgecolor="#ccc")
ax1.set_xlim(100, 1120)
ax1.grid(axis="y", alpha=0.3, which="both")

# ═══════════════════════════════════════════════════════════════════
# Panel 2: Corpus scaling — log scale
# ═══════════════════════════════════════════════════════════════════
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_facecolor(BG)

corpus = data["sweep_corpus"]

ANNOTATE_BS = {500, 5000}

for tag in ["textual", "long_doc", "visual"]:
    pts = sorted([d for d in corpus if d["tag"] == tag], key=lambda d: d["B"])
    Bs = [d["B"] for d in pts]
    naive = [d["naive_ms"] for d in pts]
    flash = [d["flash_ms"] for d in pts]
    speedup = [d["speedup"] for d in pts]
    c = TAG_COLORS[tag]

    valid_naive = [(b, n) for b, n in zip(Bs, naive) if n == n]
    if valid_naive:
        ax2.plot(*zip(*valid_naive), '--', color=c, alpha=0.7, lw=2.5)
    ax2.plot(Bs, flash, '-', color=c, lw=3.5, marker='o', markersize=7,
             markeredgecolor="white", markeredgewidth=1, label=TAG_LABELS[tag], zorder=5)

    for i, b in enumerate(Bs):
        if b in ANNOTATE_BS and speedup[i] == speedup[i]:
            if b == Bs[-1] or b == max(ANNOTATE_BS):
                xytext, ha = (8, 0), "left"
            else:
                xytext, ha = (0, 10), "center"
            ax2.annotate(f"{speedup[i]}x", (b, flash[i]),
                         fontsize=9, fontweight="bold", color=c, ha=ha,
                         xytext=xytext, textcoords="offset points",
                         bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=c, alpha=0.85, lw=0.8))

ax2.set_yscale("log")
ax2.set_xlabel("Corpus size (B docs)", fontsize=12)
ax2.set_ylabel("Latency (ms, log scale)", fontsize=12)
ax2.set_title("Single Query · Scaling with Corpus Size\nsolid = Flash, dashed = Naive",
              fontsize=14, fontweight="bold", pad=14)
ax2.legend(loc="upper left", fontsize=10, framealpha=0.9, edgecolor="#ccc")
ax2.grid(axis="y", alpha=0.3, which="both")

# ═══════════════════════════════════════════════════════════════════
# Panel 3: Memory — log scale to show the gap
# ═══════════════════════════════════════════════════════════════════
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_facecolor(BG)

mem = data["sweep_mem"]
for tag in ["textual", "long_doc", "visual"]:
    pts = sorted([d for d in mem if d["tag"] == tag], key=lambda d: d["B"])
    if not pts:
        continue
    Bs = [d["B"] for d in pts]
    naive_gb = [d["naive_gb"] for d in pts]
    flash_gb = [max(d["flash_gb"], 0.001) for d in pts]  # avoid log(0)
    c = TAG_COLORS[tag]

    ax3.plot(Bs, naive_gb, '--', color=c, alpha=0.7, lw=3, marker='s', markersize=6,
             markeredgecolor="white", markeredgewidth=1)
    ax3.plot(Bs, flash_gb, '-', color=c, lw=3.5, marker='o', markersize=7,
             markeredgecolor="white", markeredgewidth=1, label=TAG_LABELS[tag], zorder=5)

    # Annotate ratio at largest B
    if naive_gb[-1] > 0 and flash_gb[-1] > 0:
        ratio = naive_gb[-1] / max(flash_gb[-1], 0.001)
        ax3.annotate(f"{ratio:.0f}× less",
                     (Bs[-1], naive_gb[-1]),
                     fontsize=11, fontweight="bold", color=c,
                     xytext=(12, 0), textcoords="offset points",
                     arrowprops=dict(arrowstyle="->", color=c, lw=1.5, alpha=0.6))

ax3.set_yscale("log")
ax3.set_xlabel("Corpus size (B docs)", fontsize=12)
ax3.set_ylabel("Peak GPU Memory (GB, log scale)", fontsize=12)
ax3.set_title("Memory: Naive Grows, Flash Stays Flat\nsolid = Flash, dashed = Naive",
              fontsize=14, fontweight="bold", pad=14)
ax3.legend(loc="center right", fontsize=10, framealpha=0.9, edgecolor="#ccc")
ax3.grid(axis="y", alpha=0.3, which="both")

# ═══════════════════════════════════════════════════════════════════
# Panel 4: INT8 — log scale
# ═══════════════════════════════════════════════════════════════════
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_facecolor(BG)

int8 = data["sweep_int8"]
for tag in ["textual", "long_doc"]:
    pts = sorted([d for d in int8 if d["tag"] == tag], key=lambda d: d["B"])
    if not pts:
        continue
    Bs = [d["B"] for d in pts]
    naive = [d["naive_fp32_ms"] for d in pts]
    flash_fp16 = [d["flash_fp16_ms"] for d in pts]
    flash_q8 = [d["flash_q8_ms"] for d in pts]
    c = TAG_COLORS[tag]

    ax4.plot(Bs, naive, '--', color=c, alpha=0.7, lw=2.5, marker='s', markersize=5,
             markeredgecolor="white", markeredgewidth=1)
    ax4.plot(Bs, flash_fp16, '-', color=c, lw=3.5, marker='o', markersize=7,
             markeredgecolor="white", markeredgewidth=1, label=f"Flash FP16 ({tag})", zorder=5)
    ax4.plot(Bs, flash_q8, '-.', color=c, lw=3, marker='D', markersize=6,
             markeredgecolor="white", markeredgewidth=1, alpha=0.8,
             label=f"Flash Q8 ({tag})", zorder=5)

    # Annotate
    ax4.annotate("2× compression\nsame speed",
                 (Bs[-1], flash_q8[-1]),
                 fontsize=10, color="#2E7D32", fontweight="bold",
                 xytext=(12, 15), textcoords="offset points",
                 arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=1.5))

ax4.set_yscale("log")
ax4.set_xlabel("Corpus size (B docs)", fontsize=12)
ax4.set_ylabel("Latency (ms, log scale)", fontsize=12)
ax4.set_title("INT8 Fused Dequantization\ndashed = Naive FP32, solid = Flash FP16, dash-dot = Flash Q8",
              fontsize=14, fontweight="bold", pad=14)
ax4.legend(loc="upper left", fontsize=10, framealpha=0.9, edgecolor="#ccc")
ax4.grid(axis="y", alpha=0.3, which="both")

# ═══════════════════════════════════════════════════════════════════
# Global legend
# ═══════════════════════════════════════════════════════════════════
legend_elements = [
    Line2D([0], [0], color="#555", lw=3, linestyle="--", label="Naive PyTorch"),
    Line2D([0], [0], color="#555", lw=3.5, linestyle="-", label="Flash-MaxSim"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=12,
           frameon=True, fancybox=True, shadow=False, bbox_to_anchor=(0.3, 0.008),
           edgecolor="#ccc")

# Notation box
notation = "Lq = query tokens  ·  Ld = document tokens  ·  B = corpus size (docs)  ·  d = embedding dim (128)"
fig.text(0.75, 0.015, notation, ha="center", fontsize=10, color="#777", style="italic")

# ── Save ──
plt.tight_layout(rect=[0, 0.04, 1, 0.915])
out_hi = "benchmarks/flash_maxsim_benchmarks.png"
out_lo = "benchmarks/flash_maxsim_benchmarks_small.png"
fig.savefig(out_hi, dpi=150, bbox_inches="tight", facecolor="white")
fig.savefig(out_lo, dpi=100, bbox_inches="tight", facecolor="white")
print(f"\nSaved: {out_hi}")
print(f"Saved: {out_lo}")
plt.show()
