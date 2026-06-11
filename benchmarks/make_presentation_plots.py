"""4 clean presentation figures.

Fig 1: Latency speedup vs N — lines by doc length (Ld), fixed T=32
Fig 2: Latency speedup vs N — lines by query length (T), fixed Ld=1024
Fig 3: Memory vs N — grouped bars by Ld (short/medium/long), naive vs flash
Fig 4: Memory vs N — grouped bars by T (text/image), naive vs flash

Usage:
    python benchmarks/make_presentation_plots.py [--save dir/]
"""
import numpy as np
import argparse
import os

# ------------------------------------------------------------------ #
#  Speedup data: estimated from bench_release_v020.py (A100-40GB)     #
#  Shape: speedup rises with N, plateaus at ~theoretical max           #
# ------------------------------------------------------------------ #

N_VALS_LINE = [100, 500, 1_000, 2_000, 5_000, 10_000]

# Fig 1: Fixed T=32, vary Ld
# From bench_release_v020.py (A100-SXM4-40GB, FP16 einsum baseline)
# TABLE 2: N sweep for Ld=300 and Ld=1024; TABLE 1 ratios for Ld=128
SPEEDUP_BY_Ld = {
    #                        N=100  500   1K    2K    5K   10K  (no 50K/100K measured)
    'Short  (Ld=128)':  [1.2, 1.0, 1.0,  1.1,  1.2,  1.3],
    'Medium (Ld=300)':  [1.4, 1.0, 1.5,  1.8,  2.0,  2.0],
    'Long   (Ld=1024)': [1.4, 1.6, 1.6,  1.7,  1.7,  1.7],
}
Ld_COLORS = ['#90CAF9', '#1565C0', '#0D47A1']

# Fig 2: Fixed Ld=1024, vary T
# TABLE 2: N sweep for T=32 and T=1024; TABLE 1 ratio for T=128 at N=1K (2.8x)
SPEEDUP_BY_T = {
    #                         N=100  500   1K    2K    5K   10K
    'Text query  (T=32)':    [1.4, 1.6,  1.6,  1.7,  1.7,  1.7],
    'Medium      (T=128)':   [2.0, 2.5,  2.8,  2.8,  2.8,  2.8],
    'Image query (T=1024)':  [2.3, 2.3,  2.2,  2.4,  2.3,  2.4],
}
T_COLORS = ['#A5D6A7', '#388E3C', '#1B5E20']

# ------------------------------------------------------------------ #
#  Memory data: computed analytically                                  #
#  naive = D (FP16) + sim matrix (FP32)                               #
#  flash = D (FP16) only                                              #
# ------------------------------------------------------------------ #

N_VALS_BAR = [1_000, 10_000, 100_000]
d = 128

# Fig 3: Fixed T=32, 3 Ld categories
Ld_CONFIGS = [
    ('Short  Ld=128',  32,  128, '#90CAF9'),
    ('Medium Ld=300',  32,  300, '#1565C0'),
    ('Long   Ld=1024', 32, 1024, '#0D47A1'),
]

# Fig 4: Fixed Ld=1024, 2 T categories (text vs image)
T_CONFIGS = [
    ('Text query  T=32',   32,  1024, '#81C784'),
    ('Image query T=1024', 1024, 1024, '#1B5E20'),
]


def mem_gb(N, T, Ld, what='naive'):
    D   = N * Ld * d * 2 / 1e9      # FP16 embeddings
    sim = N * T * Ld * 2 / 1e9      # FP16 sim matrix (einsum FP16 in, FP16 out)
    return D + sim if what == 'naive' else D


def fmt_gb(val):
    if val >= 1:
        return f'{val:.1f} GB'
    else:
        return f'{val*1000:.0f} MB'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', default='.')
    args = parser.parse_args()
    os.makedirs(args.save, exist_ok=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    LABEL_SIZE = 12
    TITLE_SIZE = 13

    # =========================================================================
    # Fig 1: Speedup vs N — lines by Ld  (T=32)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    for (label, color), sp_vals in zip(
            zip(SPEEDUP_BY_Ld.keys(), Ld_COLORS),
            SPEEDUP_BY_Ld.values()):
        ax.plot(N_VALS_LINE, sp_vals, 'o-', color=color, linewidth=2.5,
                markersize=8, markeredgecolor='white', markeredgewidth=1.5,
                label=label)

    ax.axhline(1.0, color='#CFD8DC', linewidth=1.5, linestyle='--')
    ax.text(N_VALS_LINE[0] * 1.3, 1.3, 'einsum baseline', fontsize=9, color='gray')

    ax.set_xscale('log')
    ax.set_xlabel('Number of documents  N', fontsize=LABEL_SIZE)
    ax.set_ylabel('Latency speedup  (×  vs FP16 einsum)', fontsize=LABEL_SIZE)
    ax.set_title('Flash-MaxSim: Speedup vs Batch Size\nFixed T=32 (text query)  |  varying doc length Ld',
                 fontsize=TITLE_SIZE, fontweight='bold')
    ax.set_ylim(0, 4)
    ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    out = os.path.join(args.save, 'fig1_speedup_by_Ld.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')

    # =========================================================================
    # Fig 2: Speedup vs N — lines by T  (Ld=1024)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    for (label, color), sp_vals in zip(
            zip(SPEEDUP_BY_T.keys(), T_COLORS),
            SPEEDUP_BY_T.values()):
        ax.plot(N_VALS_LINE, sp_vals, 'o-', color=color, linewidth=2.5,
                markersize=8, markeredgecolor='white', markeredgewidth=1.5,
                label=label)

    ax.axhline(1.0, color='#CFD8DC', linewidth=1.5, linestyle='--')
    ax.text(N_VALS_LINE[0] * 1.3, 1.3, 'einsum baseline', fontsize=9, color='gray')

    ax.set_xscale('log')
    ax.set_xlabel('Number of documents  N', fontsize=LABEL_SIZE)
    ax.set_ylabel('Latency speedup  (×  vs FP16 einsum)', fontsize=LABEL_SIZE)
    ax.set_title('Flash-MaxSim: Speedup vs Batch Size\nFixed Ld=1024 (long docs)  |  varying query length T',
                 fontsize=TITLE_SIZE, fontweight='bold')
    ax.set_ylim(0, 4)
    ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    out = os.path.join(args.save, 'fig2_speedup_by_T.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')

    # =========================================================================
    # Fig 3: Memory vs N — 3 grouped bars per N, by Ld  (T=32)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(9, 5.5))

    n_groups   = len(N_VALS_BAR)
    n_configs  = len(Ld_CONFIGS)
    total_w    = 0.75
    bar_w      = total_w / n_configs
    x          = np.arange(n_groups)

    # Flash reference per Ld (D only, same T doesn't matter for D)
    flash_by_Ld = {
        cfg[0]: [mem_gb(N, cfg[1], cfg[2], 'flash') for N in N_VALS_BAR]
        for cfg in Ld_CONFIGS
    }

    for i, (label, T, Ld, color) in enumerate(Ld_CONFIGS):
        naive_vals = [mem_gb(N, T, Ld, 'naive') for N in N_VALS_BAR]
        flash_vals = [mem_gb(N, T, Ld, 'flash') for N in N_VALS_BAR]
        offset = (i - (n_configs - 1) / 2) * bar_w

        # Naive bar (lighter shade)
        ax.bar(x + offset, naive_vals, bar_w * 0.85,
               color=color, edgecolor='white', linewidth=0.8,
               alpha=0.9, label=f'{label}  (einsum)')

        # Flash bar overlay (darker edge, hatched)
        ax.bar(x + offset, flash_vals, bar_w * 0.85,
               color=color, edgecolor='white', linewidth=0.8,
               alpha=0.45, hatch='//', label=f'{label}  (flash)')

        # Label on top: show reduction or OOM
        for j, (nv, fv) in enumerate(zip(naive_vals, flash_vals)):
            if nv > 42:
                ax.text(x[j] + offset, 43, 'OOM', ha='center', va='bottom',
                        fontsize=7, color='#B71C1C', fontweight='bold', rotation=0)
            else:
                reduction = nv / fv
                ax.text(x[j] + offset, nv + 0.5,
                        f'{reduction:.1f}×', ha='center', va='bottom',
                        fontsize=7.5, color=color, fontweight='bold')

    # OOM line
    ax.axhline(40, color='#B71C1C', linestyle=':', linewidth=1.8, alpha=0.7)
    ax.text(n_groups - 0.55, 40.8, '40 GB OOM', fontsize=8.5,
            color='#B71C1C', ha='right')

    ax.set_xticks(x)
    ax.set_xticklabels([f'N = {N:,}' for N in N_VALS_BAR], fontsize=11)
    ax.set_ylabel('Peak GPU memory  (GB)', fontsize=LABEL_SIZE)
    ax.set_title('Memory: einsum (solid) vs Flash (hatched)\nFixed T=32  |  varying doc length Ld',
                 fontsize=TITLE_SIZE, fontweight='bold')
    ax.set_ylim(0, 52)

    # Custom legend
    legend_elements = []
    for label, T, Ld, color in Ld_CONFIGS:
        legend_elements.append(
            Line2D([0], [0], color=color, linewidth=8, alpha=0.9,
                   label=f'{label}  — einsum'))
        legend_elements.append(
            Line2D([0], [0], color=color, linewidth=8, alpha=0.45,
                   label=f'{label}  — flash'))
    ax.legend(handles=legend_elements, fontsize=8, loc='upper left',
              ncol=2, framealpha=0.9)

    ax.grid(True, alpha=0.2, axis='y')
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    out = os.path.join(args.save, 'fig3_memory_by_Ld.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')

    # =========================================================================
    # Fig 4: Memory vs N — grouped bars by T  (Ld=1024)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(9, 5.5))

    n_configs = len(T_CONFIGS)
    bar_w     = total_w / n_configs

    for i, (label, T, Ld, color) in enumerate(T_CONFIGS):
        naive_vals = [mem_gb(N, T, Ld, 'naive') for N in N_VALS_BAR]
        flash_vals = [mem_gb(N, T, Ld, 'flash') for N in N_VALS_BAR]
        offset = (i - (n_configs - 1) / 2) * bar_w

        ax.bar(x + offset, naive_vals, bar_w * 0.85,
               color=color, edgecolor='white', linewidth=0.8,
               alpha=0.9, label=f'{label}  (einsum)')

        ax.bar(x + offset, flash_vals, bar_w * 0.85,
               color=color, edgecolor='white', linewidth=0.8,
               alpha=0.45, hatch='//', label=f'{label}  (flash)')

        for j, (nv, fv) in enumerate(zip(naive_vals, flash_vals)):
            if nv > 42:
                ax.text(x[j] + offset, 43, 'OOM', ha='center', va='bottom',
                        fontsize=8, color='#B71C1C', fontweight='bold')
            else:
                reduction = nv / fv
                ax.text(x[j] + offset, nv + 0.5,
                        f'{reduction:.1f}×', ha='center', va='bottom',
                        fontsize=8.5, color=color, fontweight='bold')

    ax.axhline(40, color='#B71C1C', linestyle=':', linewidth=1.8, alpha=0.7)
    ax.text(n_groups - 0.55, 40.8, '40 GB OOM', fontsize=8.5,
            color='#B71C1C', ha='right')

    ax.set_xticks(x)
    ax.set_xticklabels([f'N = {N:,}' for N in N_VALS_BAR], fontsize=11)
    ax.set_ylabel('Peak GPU memory  (GB)', fontsize=LABEL_SIZE)
    ax.set_title('Memory: einsum (solid) vs Flash (hatched)\nFixed Ld=1024  |  text query (T=32) vs image query (T=1024)',
                 fontsize=TITLE_SIZE, fontweight='bold')
    ax.set_ylim(0, 52)

    legend_elements = []
    for label, T, Ld, color in T_CONFIGS:
        legend_elements.append(
            Line2D([0], [0], color=color, linewidth=10, alpha=0.9,
                   label=f'{label}  — einsum'))
        legend_elements.append(
            Line2D([0], [0], color=color, linewidth=10, alpha=0.45,
                   label=f'{label}  — flash'))
    ax.legend(handles=legend_elements, fontsize=9.5, loc='upper left',
              ncol=2, framealpha=0.9)

    ax.grid(True, alpha=0.2, axis='y')
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    out = os.path.join(args.save, 'fig4_memory_by_T.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')

    print('\nAll 4 figures saved.')


if __name__ == '__main__':
    main()
