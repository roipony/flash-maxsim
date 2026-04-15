"""Generate v0.2.0 benchmark figure from collected A100 data.

Run: MPLBACKEND=Agg python benchmarks/make_v020_figure.py
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

with open("benchmarks/results_v020_NVIDIA_A100-SXM4-80GB.json") as f:
    data = json.load(f)

gpu = data["gpu"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

TAG_COLORS = {
    "textual":  "#1565C0",
    "long_doc": "#6A1B9A",
    "medium":   "#2E7D32",
    "visual":   "#C62828",
    "short_d":  "#E65100",
}
TAG_LABELS = {
    "textual":  "ColBERT (Lq=32, Ld=300)",
    "long_doc": "Long-doc (Lq=32, Ld=1024)",
    "medium":   "Medium (Lq=128, Ld=1024)",
    "visual":   "ColPali (Lq=1024, Ld=1024)",
    "short_d":  "ColPali-ShortD (Lq=1024, Ld=512)",
}

fig, axes = plt.subplots(1, 4, figsize=(22, 5))

# ── Panel 1: Speedup vs FP16 naive ──
ax = axes[0]
for row in data["sweep"]:
    tag = row["tag"]
    if tag not in TAG_COLORS:
        continue
    Bs = [p["B"] for p in row["points"]]
    sps = [p["speedup_fp16"] for p in row["points"]]
    ax.plot(Bs, sps, 'o-', color=TAG_COLORS[tag], label=TAG_LABELS[tag], lw=2, ms=5)
    # Label last point
    ax.annotate(f'{sps[-1]:.1f}x', (Bs[-1], sps[-1]), textcoords="offset points",
                xytext=(8, 0), fontsize=9, fontweight='bold', color=TAG_COLORS[tag])
ax.set_xscale('log')
ax.set_xlabel('Number of documents (B)')
ax.set_ylabel('Speedup vs FP16 naive')
ax.set_title(f'Speedup vs FP16 naive\n({gpu})', fontweight='bold')
ax.legend(fontsize=8, loc='upper left')
ax.axhline(1, color='gray', ls='--', lw=0.8)

# ── Panel 2: Speedup vs matched precision ──
ax = axes[1]
for row in data["sweep"]:
    tag = row["tag"]
    if tag not in TAG_COLORS:
        continue
    Bs = [p["B"] for p in row["points"]]
    sps = [p["speedup_matched"] for p in row["points"]]
    ax.plot(Bs, sps, 'o-', color=TAG_COLORS[tag], label=TAG_LABELS[tag], lw=2, ms=5)
    ax.annotate(f'{sps[-1]:.1f}x', (Bs[-1], sps[-1]), textcoords="offset points",
                xytext=(8, 0), fontsize=9, fontweight='bold', color=TAG_COLORS[tag])
ax.set_xscale('log')
ax.set_xlabel('Number of documents (B)')
ax.set_ylabel('Speedup vs matched-precision naive')
ax.set_title(f'Speedup at matched precision\n(FP16 matmul + FP32 reduction)', fontweight='bold')
ax.legend(fontsize=8, loc='upper left')
ax.axhline(1, color='gray', ls='--', lw=0.8)

# ── Panel 3: Memory ──
ax = axes[2]
tags = ["textual", "long_doc", "visual", "visual"]
labels = []
naive_mems = []
for m in data["memory"]:
    labels.append(f'{m["tag"]}\nB={m["B"]//1000}K')
    naive_mems.append(m["sim_matrix_mb"] / 1024)  # GB

x = np.arange(len(labels))
bars = ax.bar(x, naive_mems, color=['#1565C0', '#6A1B9A', '#C62828', '#C62828'],
              edgecolor='black', lw=0.5, alpha=0.85)
for i, (bar, gb) in enumerate(zip(bars, naive_mems)):
    ax.text(bar.get_x() + bar.get_width()/2, gb + 0.3, f'{gb:.1f} GB',
            ha='center', fontsize=10, fontweight='bold')
# Flash line at ~0
ax.axhline(0, color='#1E88E5', lw=3, label='Flash-MaxSim: 0 MB')
ax.set_ylabel('Sim matrix size (GB)')
ax.set_title('Memory: similarity matrix\n(eliminated by flash)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.legend(fontsize=10, loc='upper left')

# ── Panel 4: Varlen speedup (vs padded naive) ──
ax = axes[3]

# Verified data from bench_full (varlen vs padded batched naive)
varlen_data = {
    "ColBERT-skewed\n(avg_Ld~49)": {
        "Ns": [1000, 5000, 20000, 100000],
        "speedups": [1.2, 2.6, 4.0, 4.8],
        "color": "#1565C0",
    },
    "ColPali-uniform\n(Ld~U[256,1024])": {
        "Ns": [500, 2000],
        "speedups": [3.9, 1.6],
        "color": "#C62828",
    },
    "ColPali-skewed\n(avg_Ld~196)": {
        "Ns": [500, 2000, 10000],
        "speedups": [2.0, 1.9, 2.8],
        "color": "#E65100",
    },
}

for label, vd in varlen_data.items():
    ax.plot(vd["Ns"], vd["speedups"], 'o-', color=vd["color"], label=label, lw=2, ms=6)
    ax.annotate(f'{vd["speedups"][-1]:.1f}x', (vd["Ns"][-1], vd["speedups"][-1]),
                textcoords="offset points", xytext=(8, 0), fontsize=9,
                fontweight='bold', color=vd["color"])

ax.set_xscale('log')
ax.set_xlabel('Number of pairs (N)')
ax.set_ylabel('Speedup vs padded naive')
ax.set_title('Variable-length pairs\n(zero padding waste)', fontweight='bold')
ax.legend(fontsize=8, loc='upper left')
ax.axhline(1, color='gray', ls='--', lw=0.8)

plt.tight_layout()
outfile = "benchmarks/flash_maxsim_benchmarks.png"
plt.savefig(outfile, dpi=150)
print(f"Saved: {outfile}")
print("Done.")
