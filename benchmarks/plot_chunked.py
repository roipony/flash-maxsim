"""Plot the chunked naive vs Flash-MaxSim figure for paper.

Usage: MPLBACKEND=Agg python benchmarks/plot_chunked.py
"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.2,
})

# Data from bench_chunked.py (H100 80GB, triton 3.6)
configs = [
    {
        "label": "Textual\n(Lq=32, Ld=300, B=1K)",
        "flash_ms": 0.090,
        "chunks":  [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
        "latency": [79.655, 40.034, 16.070, 8.135, 4.104, 1.661, 0.849, 0.447, 0.310, 0.286],
        "oom_at": None,
    },
    {
        "label": "Medium\n(Lq=128, Ld=1024, B=1K)",
        "flash_ms": 0.161,
        "chunks":  [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
        "latency": [80.569, 40.383, 16.156, 8.089, 4.041, 2.147, 1.898, 1.732, 1.600, 1.561],
        "oom_at": None,
    },
    {
        "label": "Visual\n(Lq=1024, Ld=1024, B=1K)",
        "flash_ms": 0.906,
        "chunks":  [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
        "latency": [80.254, 40.912, 16.290, 12.994, 11.570, 10.241, 9.787, 9.541, 9.389, 9.434],
        "oom_at": None,
    },
    {
        "label": "Visual\n(Lq=1024, Ld=1024, B=5K)",
        "flash_ms": 4.080,
        "chunks":  [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
        "latency": [403.265, 204.377, 80.614, 64.459, 57.387, 50.920, 48.708, 47.617, 46.796, 46.988, 47.676, 46.902],
        "oom_at": None,
    },
    {
        "label": "Visual\n(Lq=1024, Ld=1024, B=10K)",
        "flash_ms": 8.713,
        "chunks":  [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
        "latency": [802.738, 408.467, 162.024, 128.656, 114.400, 101.200, 96.751, 94.613, 93.575, 94.111, 95.623, 93.171],
        "oom_at": 10000,  # OOM at chunk=10000 (42GB)
    },
]

colors = ["#1565C0", "#2E7D32", "#E65100", "#C62828", "#6A1B9A"]

fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor="white")

# ── Left: Absolute latency ──
ax = axes[0]
ax.set_facecolor("#F8F9FA")

for i, data in enumerate(configs):
    c = colors[i]
    chunks = data["chunks"]
    latency = data["latency"]
    flash = data["flash_ms"]
    label = data["label"].replace('\n', ' ')

    ax.plot(chunks, latency, '-o', color=c, lw=2.5, markersize=6,
            markeredgecolor="white", markeredgewidth=1, label=f"Chunked: {label}", zorder=5)

    # OOM marker
    if data["oom_at"]:
        ax.scatter([data["oom_at"]], [latency[-1]], marker='X', s=200, color=c,
                   edgecolors="red", linewidths=2, zorder=10)
        ax.annotate("OOM", (data["oom_at"], latency[-1]),
                    fontsize=10, fontweight="bold", color="red",
                    xytext=(10, 10), textcoords="offset points")

    # Flash line
    ax.axhline(y=flash, color=c, linestyle='--', lw=2, alpha=0.5)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Chunk size (docs per batch)", fontsize=13)
ax.set_ylabel("Latency (ms, log scale)", fontsize=13)
ax.set_title("Chunked Naive MaxSim vs Flash-MaxSim\nNo chunk size beats Flash — large chunks OOM",
             fontsize=14, fontweight="bold", pad=12)
ax.grid(axis="both", alpha=0.2, which="both")
ax.legend(fontsize=8, loc="upper right", framealpha=0.9, edgecolor="#ccc")

# Flash annotation
ax.annotate("Flash-MaxSim\n(dashed lines, parameter-free)",
            xy=(3, 0.3), fontsize=11, fontweight="bold", color="#333",
            ha="left", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.9))

# ── Right: Slowdown ratio ──
ax2 = axes[1]
ax2.set_facecolor("#F8F9FA")

for i, data in enumerate(configs):
    c = colors[i]
    chunks = data["chunks"]
    latency = data["latency"]
    flash = data["flash_ms"]
    label = data["label"].replace('\n', ' ')

    ratio = [l / flash for l in latency]
    ax2.plot(chunks, ratio, '-o', color=c, lw=2.5, markersize=6,
             markeredgecolor="white", markeredgewidth=1, label=label, zorder=5)

    # Label best ratio
    best_ratio = min(ratio)
    best_idx = ratio.index(best_ratio)
    ax2.annotate(f"{best_ratio:.0f}x", (chunks[best_idx], best_ratio),
                 fontsize=10, fontweight="bold", color=c, ha="center",
                 xytext=(0, -16), textcoords="offset points",
                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=c, alpha=0.85, lw=0.8))

    # OOM marker
    if data["oom_at"]:
        ax2.scatter([data["oom_at"]], [ratio[-1]], marker='X', s=200, color=c,
                    edgecolors="red", linewidths=2, zorder=10)
        ax2.annotate("OOM\n(42GB)", (data["oom_at"], ratio[-1]),
                     fontsize=9, fontweight="bold", color="red",
                     xytext=(10, 10), textcoords="offset points")

# Flash = 1x line
ax2.axhline(y=1.0, color="#333", linestyle='--', lw=2.5, alpha=0.6)
ax2.text(1.2, 1.15, "Flash-MaxSim (1x)", fontsize=11, color="#333", fontweight="bold")

ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("Chunk size (docs per batch)", fontsize=13)
ax2.set_ylabel("Slowdown vs Flash-MaxSim (x)", fontsize=13)
ax2.set_title("Even the Best Chunk Size is 3-11x Slower\nFlash-MaxSim needs zero tuning",
              fontsize=14, fontweight="bold", pad=12)
ax2.grid(axis="both", alpha=0.2, which="both")
ax2.legend(fontsize=8, loc="upper right", framealpha=0.9, edgecolor="#ccc")

# Bottom text
fig.text(0.5, 0.01,
         "Chunked naive requires manual batch size tuning per GPU / corpus / config.  "
         "Flash-MaxSim is parameter-free — same code on any GPU, any corpus size.",
         ha="center", fontsize=11, color="#555", style="italic")

plt.tight_layout(rect=[0, 0.04, 1, 1])
out = "benchmarks/chunked_vs_flash.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
plt.show()
