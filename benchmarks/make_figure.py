"""Generate landing page figure from benchmark JSON files.

Usage:
    python benchmarks/make_figure.py benchmarks/results_*.json
    # produces benchmarks/flash_maxsim_benchmarks.png
"""
import json, sys, glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Load data ──
files = sys.argv[1:] or sorted(glob.glob("benchmarks/results_*.json"))
assert files, "No result files found. Run collect_data.py first."

all_data = []
for f in files:
    with open(f) as fh:
        all_data.append(json.load(fh))
    print(f"Loaded: {f} ({all_data[-1]['gpu']})")

# Use first GPU's data (or combine if multiple)
data = all_data[0]
gpu = data["gpu"]

# ── Style ──
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
NAIVE_COLOR = "#FF7043"
FLASH_COLOR = "#1E88E5"
Q8_COLOR = "#2E7D32"
MEM_NAIVE = "#FF8A65"
MEM_FLASH = "#42A5F5"
BG_COLOR = "#FAFAFA"

fig = plt.figure(figsize=(18, 10), facecolor="white")
fig.suptitle(f"Flash-MaxSim  —  Fused GPU Kernel for ColBERT/ColPali MaxSim",
             fontsize=20, fontweight="bold", y=0.97)
fig.text(0.5, 0.935, f"Benchmarked on {gpu}  |  Similarity matrix never touches HBM",
         ha="center", fontsize=12, color="#666", style="italic")

# ── Panel 1: Speedup ──
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_facecolor(BG_COLOR)
sp = data["speedup"]
labels = [d["label"] for d in sp]
naive_ms = [d["naive_ms"] for d in sp]
flash_ms = [d["flash_ms"] for d in sp]
speedups = [d["speedup"] for d in sp]

x = np.arange(len(labels)); w = 0.32
bars_n = ax1.bar(x - w/2, naive_ms, w, label="Naive FP32", color=NAIVE_COLOR, edgecolor="white", lw=0.8, zorder=3)
bars_f = ax1.bar(x + w/2, flash_ms, w, label="Flash-MaxSim", color=FLASH_COLOR, edgecolor="white", lw=0.8, zorder=3)

for i, s in enumerate(speedups):
    ymax = max(naive_ms[i], flash_ms[i])
    ax1.text(i, ymax * 1.08, f"{s}x", ha="center", fontsize=13, fontweight="bold", color=FLASH_COLOR, zorder=4)

ax1.set_ylabel("Latency (ms)")
ax1.set_title("Kernel Speedup", fontsize=14, fontweight="bold", pad=10)
ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=9)
ax1.legend(loc="upper left", fontsize=9)
ax1.set_ylim(0, max(naive_ms) * 1.25)
ax1.grid(axis="y", alpha=0.3, zorder=0)

# ── Panel 2: Memory ──
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_facecolor(BG_COLOR)
mem = data["memory"]
labels_m = [d["label"] for d in mem]
naive_gb = [d["naive_gb"] for d in mem]
flash_gb = [d["flash_gb"] for d in mem]
ratios = [d["ratio"] for d in mem]

x = np.arange(len(labels_m)); w = 0.32
ax2.bar(x - w/2, naive_gb, w, label="Naive einsum", color=MEM_NAIVE, edgecolor="white", lw=0.8, zorder=3)
ax2.bar(x + w/2, flash_gb, w, label="Flash-MaxSim", color=MEM_FLASH, edgecolor="white", lw=0.8, zorder=3)

for i, r in enumerate(ratios):
    ymax = max(naive_gb[i], flash_gb[i])
    ax2.text(i, ymax * 1.08, f"{r:.0f}x less", ha="center", fontsize=12, fontweight="bold", color=Q8_COLOR, zorder=4)

ax2.set_ylabel("Peak GPU Memory (GB)")
ax2.set_title("Memory Reduction", fontsize=14, fontweight="bold", pad=10)
ax2.set_xticks(x); ax2.set_xticklabels(labels_m, fontsize=9)
ax2.legend(loc="upper left", fontsize=9)
ax2.set_ylim(0, max(naive_gb) * 1.3)
ax2.grid(axis="y", alpha=0.3, zorder=0)

# ── Panel 3: INT8 ──
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_facecolor(BG_COLOR)
q8 = data["int8"]
labels_q = [d["label"] for d in q8]
naive_q = [d["naive_ms"] for d in q8]
flash_q = [d["flash_ms"] for d in q8]
sp_q = [d["speedup"] for d in q8]

x = np.arange(len(labels_q)); w = 0.32
ax3.bar(x - w/2, naive_q, w, label="Naive FP32", color=NAIVE_COLOR, edgecolor="white", lw=0.8, zorder=3)
ax3.bar(x + w/2, flash_q, w, label="Flash Q8 (fused)", color=Q8_COLOR, edgecolor="white", lw=0.8, zorder=3)

for i, s in enumerate(sp_q):
    ymax = max(naive_q[i], flash_q[i])
    ax3.text(i, ymax * 1.08, f"{s}x", ha="center", fontsize=13, fontweight="bold", color=Q8_COLOR, zorder=4)

ax3.set_ylabel("Latency (ms)")
ax3.set_title("INT8 Fused Dequantization", fontsize=14, fontweight="bold", pad=10)
ax3.set_xticks(x); ax3.set_xticklabels(labels_q, fontsize=9)
ax3.legend(loc="upper left", fontsize=9)
ax3.set_ylim(0, max(naive_q) * 1.25)
ax3.grid(axis="y", alpha=0.3, zorder=0)

# ── Panel 4: Batched throughput ──
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_facecolor(BG_COLOR)
bat = data["batched"]
labels_b = [d["label"] for d in bat]
naive_b = [d["naive_ms"] for d in bat]
flash_b = [d["flash_ms"] for d in bat]
sp_b = [d["speedup"] for d in bat]
tp_b = [d["pairs_per_sec"] for d in bat]

x = np.arange(len(labels_b)); w = 0.32
ax3b = ax4.bar(x - w/2, naive_b, w, label="Naive loop", color=NAIVE_COLOR, edgecolor="white", lw=0.8, zorder=3)
ax4b = ax4.bar(x + w/2, flash_b, w, label="Flash batched", color=FLASH_COLOR, edgecolor="white", lw=0.8, zorder=3)

for i, (s, tp) in enumerate(zip(sp_b, tp_b)):
    ymax = max(naive_b[i], flash_b[i])
    ax4.text(i, ymax * 1.08, f"{s}x", ha="center", fontsize=13, fontweight="bold", color=FLASH_COLOR, zorder=4)
    ax4.text(i, ymax * 1.20, f"{tp}M pairs/s", ha="center", fontsize=9, color="#666", zorder=4)

ax4.set_ylabel("Latency (ms)")
ax4.set_title("Batched Multi-Query Throughput", fontsize=14, fontweight="bold", pad=10)
ax4.set_xticks(x); ax4.set_xticklabels(labels_b, fontsize=9)
ax4.legend(loc="upper left", fontsize=9)
ax4.set_ylim(0, max(naive_b) * 1.35)
ax4.grid(axis="y", alpha=0.3, zorder=0)

# ── Save ──
plt.tight_layout(rect=[0, 0, 1, 0.92])
out = "benchmarks/flash_maxsim_benchmarks.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"\nSaved: {out}")

# Also save a smaller version for README
fig.savefig("benchmarks/flash_maxsim_benchmarks_small.png", dpi=100, bbox_inches="tight", facecolor="white")
print("Saved: benchmarks/flash_maxsim_benchmarks_small.png")

plt.show()
