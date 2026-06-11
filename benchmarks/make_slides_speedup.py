"""Slide-ready speedup plots using measured H100 benchmark data.

Two plots:
  1. Speedup by configuration (bar chart) — shows 2x–12x range across regimes
  2. Speedup vs N — varlen HotpotQA conditions, speedup grows with corpus size

Usage:
    python benchmarks/make_slides_speedup.py [--save figs/]
"""
import argparse, os
import numpy as np

# -------------------------------------------------------------------------
# Measured data — H100 80GB, Triton 3.6.0, FP16 einsum baseline
# Source: bench_release_v020.py + bench_varlen.py
# -------------------------------------------------------------------------

# --- Plot 1: speedup by config (N=1000) ---
CONFIGS = [
    # (label, speedup, color, regime)
    ('ColBERT\nT=32, Ld=300',   3.9,  '#64B5F6', 'text'),
    ('Long-doc\nT=32, Ld=1024', 6.0,  '#1976D2', 'text'),
    ('Medium\nT=128, Ld=1024',  10.0, '#558B2F', 'image'),
    ('Visual\nT=512, Ld=1024',  11.6, '#2E7D32', 'image'),
    ('ColPali\nT=1024, Ld=1024',11.0, '#1B5E20', 'image'),
]

# --- Plot 2: speedup vs N — HotpotQA varlen (T=32, max_Ld=174, avg_Ld≈22)
# Flash exits inner loop at actual doc_len (~22 tokens out of 174)
# Source: bench_varlen.py on H100 (vs FP16 einsum Ld=22)
VARLEN_N   = [1_000, 5_000, 10_000, 50_000, 100_000, 200_000]
VARLEN_SPD = [2.0,   3.8,   5.2,    7.5,    8.9,     9.1]   # measured medians

# --- Bonus annotated speedup numbers for slides ---
# Corpus scaling (T=Ld=1024, H100, vs naive)
COLPALI_N  = [500,   1_000, 2_000,  5_000]
COLPALI_SPD= [10.4,  11.2,  11.9,   11.7]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', default='figs')
    args = parser.parse_args()
    os.makedirs(args.save, exist_ok=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    FONT = 13
    TITLE = 14
    plt.rcParams.update({'font.size': FONT})

    # =====================================================================
    # Plot 1: Horizontal bar chart — speedup by config
    # =====================================================================
    fig, ax = plt.subplots(figsize=(9, 5))

    labels  = [c[0] for c in CONFIGS]
    speedup = [c[1] for c in CONFIGS]
    colors  = [c[2] for c in CONFIGS]
    y       = np.arange(len(labels))

    bars = ax.barh(y, speedup, height=0.55, color=colors,
                   edgecolor='white', linewidth=1.2)

    # Annotate speedup values
    for bar, sp in zip(bars, speedup):
        ax.text(sp + 0.15, bar.get_y() + bar.get_height() / 2,
                f'{sp:.1f}×', va='center', fontsize=12, fontweight='bold',
                color='#212121')

    ax.axvline(1.0, color='#B0BEC5', linestyle='--', linewidth=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel('Latency speedup  (× vs FP16 einsum)', fontsize=FONT)
    ax.set_title('Flash-MaxSim Speedup by Regime\n(N=1,000 documents, H100 80GB)',
                 fontsize=TITLE, fontweight='bold')
    ax.set_xlim(0, 14)
    ax.set_ylim(-0.5, len(labels) - 0.5)

    legend = [Patch(facecolor='#1976D2', label='Text retrieval (ColBERT)'),
              Patch(facecolor='#2E7D32', label='Vision retrieval (ColPali)')]
    ax.legend(handles=legend, fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.2, axis='x')
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    out = os.path.join(args.save, 'slides_speedup_by_config.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')

    # =====================================================================
    # Plot 2: Speedup vs N — two lines (varlen HotpotQA + ColPali)
    # =====================================================================
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(VARLEN_N, VARLEN_SPD, 'o-',
            color='#1976D2', linewidth=2.5, markersize=8,
            markeredgecolor='white', markeredgewidth=1.5,
            label='HotpotQA varlen (T=32, avg_Ld≈22/174)')

    ax.plot(COLPALI_N, COLPALI_SPD, 's-',
            color='#2E7D32', linewidth=2.5, markersize=8,
            markeredgecolor='white', markeredgewidth=1.5,
            label='ColPali full (T=Ld=1024)')

    # Annotate key points
    ax.annotate('2×', xy=(VARLEN_N[0], VARLEN_SPD[0]),
                xytext=(VARLEN_N[0] * 1.3, VARLEN_SPD[0] - 0.5),
                fontsize=10, color='#1976D2', fontweight='bold')
    ax.annotate('9×', xy=(VARLEN_N[-1], VARLEN_SPD[-1]),
                xytext=(VARLEN_N[-1] * 0.45, VARLEN_SPD[-1] + 0.3),
                fontsize=10, color='#1976D2', fontweight='bold')
    ax.annotate('12×', xy=(COLPALI_N[-1], COLPALI_SPD[-1]),
                xytext=(COLPALI_N[-1] * 0.45, COLPALI_SPD[-1] + 0.3),
                fontsize=10, color='#2E7D32', fontweight='bold')

    ax.axhline(1.0, color='#B0BEC5', linestyle='--', linewidth=1.5)
    ax.text(1200, 1.35, 'einsum baseline', fontsize=9, color='gray')

    ax.set_xscale('log')
    ax.set_xlabel('Corpus size  N', fontsize=FONT)
    ax.set_ylabel('Latency speedup  (×)', fontsize=FONT)
    ax.set_title('Speedup Grows with Corpus Size\n(varlen: IO elim + early exit; ColPali: IO elim)',
                 fontsize=TITLE, fontweight='bold')
    ax.set_ylim(0, 14)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.2)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    out = os.path.join(args.save, 'slides_speedup_vs_N.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')

    # =====================================================================
    # Plot 3: Memory unlock — ColPali OOM visualization
    # =====================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    fig.suptitle('Flash-MaxSim Eliminates the Similarity Matrix\n(ColPali: T=1024, Ld=1024, d=128)',
                 fontsize=TITLE, fontweight='bold')

    N_bars = [500, 1_000, 2_000, 5_000, 10_000, 50_000]
    d = 128
    D_gb   = [N * 1024 * d * 2 / 1e9 for N in N_bars]
    sim_gb = [N * 1024 * 1024 * 4 / 1e9 for N in N_bars]  # FP32 sim matrix
    GPU_GB = 40

    x = np.arange(len(N_bars))
    w = 0.6
    xlabels = [f'{N//1000}K' for N in N_bars]

    for ax, show_sim, title, clr in [
        (ax1, True,  'Naive einsum', '#EF5350'),
        (ax2, False, 'Flash-MaxSim', '#42A5F5'),
    ]:
        ax.bar(x, D_gb, w, color='#42A5F5', edgecolor='white', lw=0.8,
               label='D embeddings (FP16)')
        if show_sim:
            sim_capped = [min(s, GPU_GB * 1.1 - d_) for s, d_ in zip(sim_gb, D_gb)]
            ax.bar(x, sim_capped, w, bottom=D_gb,
                   color='#EF5350', edgecolor='white', lw=0.8, alpha=0.88,
                   label='Sim matrix (FP32)')

        ax.axhline(GPU_GB, color='#B71C1C', linestyle=':', linewidth=2, alpha=0.8)
        ax.text(len(N_bars) - 0.5, GPU_GB + 0.5, f'{GPU_GB}GB OOM',
                fontsize=8, color='#B71C1C', ha='right', fontweight='bold')

        for i, N in enumerate(N_bars):
            total = D_gb[i] + (sim_gb[i] if show_sim else 0)
            if total > GPU_GB + 1:
                ax.text(i, min(total, GPU_GB * 1.08) + 0.5, 'OOM', ha='center',
                        fontsize=7.5, color='#B71C1C', fontweight='bold')
            else:
                ax.text(i, (D_gb[i] + (sim_capped[i] if show_sim else 0)) + 0.5,
                        f'{total:.0f}GB', ha='center', fontsize=7, color='#37474F')

        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold',
                     color='#B71C1C' if show_sim else '#1565C0')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.2, axis='y')
        ax.spines[['top', 'right']].set_visible(False)

    ax1.set_ylabel('GPU Memory (GB)', fontsize=FONT)
    ax1.set_ylim(0, GPU_GB * 1.15)
    ax1.set_xlabel('Number of documents N')
    ax2.set_xlabel('Number of documents N')

    plt.tight_layout()
    out = os.path.join(args.save, 'slides_memory_oom.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')

    print('\nAll 3 slides figures saved.')


if __name__ == '__main__':
    main()
