"""Generate landing page figure from continuous sweep data.

Usage:
    python benchmarks/make_figure.py benchmarks/results_*.json
    # produces benchmarks/flash_maxsim_benchmarks.png
"""
import json, sys, glob, math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

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
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Colors
C_NAIVE = "#D32F2F"
C_FLASH = "#1565C0"
C_Q8 = "#2E7D32"
C_SPEEDUP = "#FF8F00"
C_REGION_TEXT = "#E3F2FD"
C_REGION_VIS = "#FFF3E0"
BG = "#FAFAFA"

fig = plt.figure(figsize=(20, 12), facecolor="white")

fig.suptitle("Flash-MaxSim", fontsize=26, fontweight="bold", y=0.97,
             fontfamily="sans-serif")
fig.text(0.5, 0.935,
         f"Fused GPU kernel for late-interaction retrieval  ·  {gpu}",
         ha="center", fontsize=13, color="#555", style="italic")

# ═══════════════════════════════════════════════════════════════════
# Panel 1: Speedup landscape — Ld on x-axis, lines per Lq
# ═══════════════════════════════════════════════════════════════════
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_facecolor(BG)
ax1_r = ax1.twinx()

sweep = data["sweep_seq"]
lq_values = sorted(set(d["Lq"] for d in sweep))
ld_values = sorted(set(d["Ld"] for d in sweep))

# Shade regions
ax1.axvspan(0, 400, alpha=0.08, color="#1565C0", zorder=0)
ax1.axvspan(400, 1100, alpha=0.06, color="#FF8F00", zorder=0)
ax1.text(200, 0, "Textual", ha="center", fontsize=10, color="#1565C0", alpha=0.5,
         fontweight="bold", transform=ax1.get_xaxis_transform(), va="bottom")
ax1.text(750, 0, "Visual / Long-doc", ha="center", fontsize=10, color="#FF8F00", alpha=0.5,
         fontweight="bold", transform=ax1.get_xaxis_transform(), va="bottom")

cmap = plt.cm.viridis
lq_colors = {lq: cmap(i / max(1, len(lq_values) - 1)) for i, lq in enumerate(lq_values)}

for Lq in lq_values:
    pts = sorted([d for d in sweep if d["Lq"] == Lq], key=lambda d: d["Ld"])
    lds = [d["Ld"] for d in pts]
    naive = [d["naive_ms"] for d in pts]
    flash = [d["flash_ms"] for d in pts]
    speedup = [d["speedup"] for d in pts]
    c = lq_colors[Lq]

    ax1.plot(lds, naive, '--', color=c, alpha=0.4, lw=1.5)
    ax1.plot(lds, flash, '-', color=c, lw=2.5, marker='o', markersize=5, label=f"Lq={Lq}")
    ax1_r.plot(lds, speedup, ':', color=c, alpha=0.6, lw=1.5)
    # Annotate peak speedup
    peak_idx = np.argmax(speedup)
    ax1_r.annotate(f"{speedup[peak_idx]}x", (lds[peak_idx], speedup[peak_idx]),
                   fontsize=9, fontweight="bold", color=c, ha="center",
                   xytext=(0, 8), textcoords="offset points")

ax1.set_xlabel("Document length (Ld)")
ax1.set_ylabel("Latency (ms)")
ax1_r.set_ylabel("Speedup (×)", color=C_SPEEDUP)
ax1_r.tick_params(axis="y", colors=C_SPEEDUP)
ax1_r.spines["right"].set_visible(True)
ax1_r.spines["right"].set_color(C_SPEEDUP)
ax1.set_title("Speedup vs Sequence Length\n(B=1000, solid=Flash, dashed=Naive)", fontsize=13, fontweight="bold", pad=12)
ax1.legend(loc="upper left", fontsize=8, title="Query length", title_fontsize=9)
ax1.set_xlim(100, 1080)
ax1.grid(axis="y", alpha=0.2)

# ═══════════════════════════════════════════════════════════════════
# Panel 2: Corpus scaling — B on x-axis, lines per config
# ═══════════════════════════════════════════════════════════════════
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_facecolor(BG)
ax2_r = ax2.twinx()

corpus = data["sweep_corpus"]
tag_style = {
    "textual":  {"color": "#1565C0", "label": "Textual (Lq=32, Ld=300)"},
    "long_doc": {"color": "#7B1FA2", "label": "Long-doc (Lq=32, Ld=1024)"},
    "visual":   {"color": "#E65100", "label": "Visual (Lq=1024, Ld=1024)"},
}

for tag, style in tag_style.items():
    pts = sorted([d for d in corpus if d["tag"] == tag], key=lambda d: d["B"])
    Bs = [d["B"] for d in pts]
    naive = [d["naive_ms"] for d in pts]
    flash = [d["flash_ms"] for d in pts]
    speedup = [d["speedup"] for d in pts]
    c = style["color"]

    # Filter NaN for naive (OOM)
    valid_naive = [(b, n) for b, n in zip(Bs, naive) if n == n]
    if valid_naive:
        ax2.plot(*zip(*valid_naive), '--', color=c, alpha=0.4, lw=1.5)
    ax2.plot(Bs, flash, '-', color=c, lw=2.5, marker='o', markersize=5, label=style["label"])

    valid_sp = [(b, s) for b, s in zip(Bs, speedup) if s == s]
    if valid_sp:
        ax2_r.plot(*zip(*valid_sp), ':', color=c, alpha=0.6, lw=1.5)
        # Label last valid speedup
        b, s = valid_sp[-1]
        ax2_r.annotate(f"{s}x", (b, s), fontsize=9, fontweight="bold", color=c,
                       xytext=(5, 5), textcoords="offset points")

ax2.set_xlabel("Corpus size (B docs)")
ax2.set_ylabel("Latency (ms)")
ax2_r.set_ylabel("Speedup (×)", color=C_SPEEDUP)
ax2_r.tick_params(axis="y", colors=C_SPEEDUP)
ax2_r.spines["right"].set_visible(True)
ax2_r.spines["right"].set_color(C_SPEEDUP)
ax2.set_title("Scaling with Corpus Size\n(solid=Flash, dashed=Naive)", fontsize=13, fontweight="bold", pad=12)
ax2.legend(loc="upper left", fontsize=8)
ax2.grid(axis="y", alpha=0.2)

# ═══════════════════════════════════════════════════════════════════
# Panel 3: Memory — B on x-axis, naive grows, flash stays flat
# ═══════════════════════════════════════════════════════════════════
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_facecolor(BG)

mem = data["sweep_mem"]
for tag, style in tag_style.items():
    pts = sorted([d for d in mem if d["tag"] == tag], key=lambda d: d["B"])
    if not pts:
        continue
    Bs = [d["B"] for d in pts]
    naive_gb = [d["naive_gb"] for d in pts]
    flash_gb = [d["flash_gb"] for d in pts]
    c = style["color"]

    ax3.plot(Bs, naive_gb, '--', color=c, alpha=0.5, lw=2, marker='s', markersize=4)
    ax3.plot(Bs, flash_gb, '-', color=c, lw=2.5, marker='o', markersize=5, label=style["label"])

    # Annotate ratio at largest B
    if naive_gb[-1] > 0 and flash_gb[-1] > 0:
        ratio = naive_gb[-1] / max(flash_gb[-1], 0.001)
        ax3.annotate(f"{ratio:.0f}×\nless",
                     (Bs[-1], naive_gb[-1]),
                     fontsize=9, fontweight="bold", color=c,
                     xytext=(10, -5), textcoords="offset points",
                     arrowprops=dict(arrowstyle="->", color=c, alpha=0.5))

ax3.set_xlabel("Corpus size (B docs)")
ax3.set_ylabel("Peak GPU Memory (GB)")
ax3.set_title("Memory: Naive Grows, Flash Stays Flat\n(solid=Flash, dashed=Naive)", fontsize=13, fontweight="bold", pad=12)
ax3.legend(loc="upper left", fontsize=8)
ax3.grid(axis="y", alpha=0.2)

# ═══════════════════════════════════════════════════════════════════
# Panel 4: INT8 — grouped by B, comparing naive_fp32 vs flash_fp16 vs flash_q8
# ═══════════════════════════════════════════════════════════════════
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_facecolor(BG)

int8 = data["sweep_int8"]
for tag, style in [("textual", tag_style["textual"]), ("long_doc", tag_style["long_doc"])]:
    pts = sorted([d for d in int8 if d["tag"] == tag], key=lambda d: d["B"])
    if not pts:
        continue
    Bs = [d["B"] for d in pts]
    naive = [d["naive_fp32_ms"] for d in pts]
    flash_fp16 = [d["flash_fp16_ms"] for d in pts]
    flash_q8 = [d["flash_q8_ms"] for d in pts]
    c = style["color"]

    ax4.plot(Bs, naive, '--', color=c, alpha=0.4, lw=1.5, marker='s', markersize=4)
    ax4.plot(Bs, flash_fp16, '-', color=c, lw=2, marker='o', markersize=5, label=f"Flash FP16 ({tag})")
    ax4.plot(Bs, flash_q8, '-', color=c, lw=2, marker='D', markersize=5, alpha=0.7,
             linestyle='-.', label=f"Flash Q8 ({tag})")

    # Annotate: Q8 has same speed as FP16
    ax4.annotate("2× compression\nsame speed",
                 (Bs[-1], flash_q8[-1]),
                 fontsize=8, color=C_Q8, fontweight="bold",
                 xytext=(10, 10), textcoords="offset points")

ax4.set_xlabel("Corpus size (B docs)")
ax4.set_ylabel("Latency (ms)")
ax4.set_title("INT8 Fused Dequantization\n(dashed=Naive FP32, solid=Flash FP16, dash-dot=Flash Q8)", fontsize=13, fontweight="bold", pad=12)
ax4.legend(loc="upper left", fontsize=8)
ax4.grid(axis="y", alpha=0.2)

# ═══════════════════════════════════════════════════════════════════
# Global legend
# ═══════════════════════════════════════════════════════════════════
legend_elements = [
    Line2D([0], [0], color="gray", lw=2, linestyle="--", label="Naive PyTorch"),
    Line2D([0], [0], color="gray", lw=2.5, linestyle="-", label="Flash-MaxSim"),
    Line2D([0], [0], color=C_SPEEDUP, lw=1.5, linestyle=":", label="Speedup (right axis)"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=11,
           frameon=True, fancybox=True, shadow=False, bbox_to_anchor=(0.5, 0.01))

# ── Save ──
plt.tight_layout(rect=[0, 0.04, 1, 0.92])
out_hi = "benchmarks/flash_maxsim_benchmarks.png"
out_lo = "benchmarks/flash_maxsim_benchmarks_small.png"
fig.savefig(out_hi, dpi=150, bbox_inches="tight", facecolor="white")
fig.savefig(out_lo, dpi=100, bbox_inches="tight", facecolor="white")
print(f"\nSaved: {out_hi}")
print(f"Saved: {out_lo}")
plt.show()
