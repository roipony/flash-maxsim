"""The Flash-MaxSim Regime Map: one figure that explains everything.

2x2 grid:
  Top-left:  Use-case map (T vs Ld) with bubbles for real workloads
  Top-right: Measured speedup vs 2T/d ratio — the master curve
  Bot-left:  Memory profile: D vs sim matrix vs total, per regime
  Bot-right: What flash eliminates per regime (stacked area)

Usage:
    python benchmarks/make_regime_plot.py [--save path.png]
"""
import numpy as np
import argparse

# Use cases with real dimensions
USE_CASES = [
    # (name, T, Ld, d, color, marker)
    ("ColBERT\n(text→text)", 32, 300, 128, '#4CAF50', 's'),
    ("ColBERT\nlong-doc", 32, 1024, 128, '#8BC34A', 's'),
    ("GVE\n(text→image)", 60, 729, 128, '#FF9800', 'D'),
    ("Jina ColBERT\nv2", 128, 512, 128, '#2196F3', '^'),
    ("ColPali\n(image→image)", 1024, 1024, 128, '#E53935', 'o'),
    ("Visual QA\n(image→image)", 512, 1024, 128, '#C62828', 'o'),
]

# Measured speedups (from bench_release_v020.py, A100)
MEASURED = [
    # (T, Ld, d, B, speedup)
    (32, 128, 128, 1000, 2.4),
    (32, 300, 128, 1000, 4.3),
    (32, 512, 128, 1000, 6.6),
    (32, 1024, 128, 1000, 7.6),
    (128, 128, 128, 1000, 4.6),
    (128, 300, 128, 1000, 9.0),
    (128, 512, 128, 1000, 10.8),
    (128, 1024, 128, 1000, 11.2),
    (512, 128, 128, 1000, 9.2),
    (512, 300, 128, 1000, 9.7),
    (512, 512, 128, 1000, 9.8),
    (512, 1024, 128, 1000, 10.2),
    (1024, 128, 128, 1000, 9.5),
    (1024, 300, 128, 1000, 9.5),
    (1024, 512, 128, 1000, 9.7),
    (1024, 1024, 128, 1000, 10.0),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='flash_maxsim_regimes.png')
    args = parser.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Flash-MaxSim: The Three Regimes\nSpeedup is governed by 2T/d — the sim-matrix-to-D ratio',
                 fontsize=14, fontweight='bold')

    # =========================================================================
    # Plot 1 (top-left): Use-case map — T vs Ld with regime zones
    # =========================================================================
    ax = axes[0, 0]

    # Background regime zones
    ax.axhspan(0, 64, alpha=0.08, color='green')    # text query zone
    ax.axhspan(64, 256, alpha=0.08, color='orange')  # transition zone
    ax.axhspan(256, 2048, alpha=0.08, color='red')   # visual query zone

    ax.text(1500, 20, '2T/d < 1\nFusion regime', fontsize=9, color='green',
            fontweight='bold', ha='center', style='italic')
    ax.text(1500, 128, '2T/d ≈ 1\nTransition', fontsize=9, color='#E65100',
            fontweight='bold', ha='center', style='italic')
    ax.text(1500, 700, '2T/d >> 1\nIO-tiling regime', fontsize=9, color='#B71C1C',
            fontweight='bold', ha='center', style='italic')

    for name, T, Ld, d, color, marker in USE_CASES:
        ratio = 2 * T / d
        size = 150 + ratio * 20  # bigger bubble for higher ratio
        ax.scatter(Ld, T, s=size, c=color, marker=marker, edgecolor='black',
                   linewidth=1, zorder=3)
        # Label offset
        ax.annotate(name, xy=(Ld, T), xytext=(12, 8), textcoords='offset points',
                    fontsize=7, color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor=color, alpha=0.85, lw=0.5))

    # Iso-ratio lines
    for ratio_val, ls in [(0.5, ':'), (1.0, '--'), (8.0, '-')]:
        T_line = ratio_val * 128 / 2  # T = ratio * d / 2
        ax.axhline(y=T_line, color='gray', linestyle=ls, alpha=0.3, linewidth=1)
        ax.text(50, T_line * 1.05, f'2T/d={ratio_val}', fontsize=7, color='gray')

    ax.set_xlabel('Document Length (Ld)', fontsize=10)
    ax.set_ylabel('Query Length (T)', fontsize=10)
    ax.set_title('Use-Case Map', fontsize=11)
    ax.set_xlim(0, 1800)
    ax.set_ylim(0, 1400)
    ax.set_yscale('log')
    ax.set_ylim(10, 2000)
    ax.grid(True, alpha=0.15)

    # =========================================================================
    # Plot 2 (top-right): Speedup vs 2T/d — the master curve
    # =========================================================================
    ax = axes[0, 1]

    ratios = [2 * T / d for T, Ld, d, B, sp in MEASURED]
    speedups = [sp for T, Ld, d, B, sp in MEASURED]

    # Color by T (query length = determines the regime)
    colors_meas = []
    for T, Ld, d, B, sp in MEASURED:
        if T <= 32:
            colors_meas.append('#4CAF50')   # green = text query
        elif T <= 128:
            colors_meas.append('#FF9800')   # orange = medium
        elif T <= 512:
            colors_meas.append('#E53935')   # red = visual
        else:
            colors_meas.append('#B71C1C')   # dark red = full ColPali

    ax.scatter(ratios, speedups, c=colors_meas, s=80, edgecolor='black', linewidth=0.5, zorder=3)

    # Theoretical curve: speedup_max = 1 + 4T/d (but observed is sublinear)
    r_theory = np.linspace(0.1, 20, 100)
    # Empirical fit: observed ≈ min(1 + 4*r, ~11) with diminishing returns
    sp_theory = np.minimum(1 + 4 * r_theory, 12) * (1 - np.exp(-r_theory * 2))
    ax.plot(r_theory, sp_theory, '--', color='gray', alpha=0.4, linewidth=1, label='Trend')

    # Regime zones
    ax.axvspan(0, 0.75, alpha=0.06, color='green')
    ax.axvspan(0.75, 2, alpha=0.06, color='orange')
    ax.axvspan(2, 20, alpha=0.06, color='red')

    ax.text(0.35, 10, 'Fusion\nregime', fontsize=8, color='green', fontweight='bold', ha='center')
    ax.text(1.2, 10, 'Transition', fontsize=8, color='#E65100', fontweight='bold', ha='center')
    ax.text(8, 10, 'IO-tiling\nregime', fontsize=8, color='#B71C1C', fontweight='bold', ha='center')

    # Use case annotations
    for name, T, Ld, d, color, marker in USE_CASES:
        ratio = 2 * T / d
        # Find closest measured speedup
        closest = min(MEASURED, key=lambda m: abs(2*m[0]/m[2] - ratio))
        sp = closest[4]
        label = name.replace('\n', ' ')
        ax.annotate(label, xy=(ratio, sp), xytext=(5, -10), textcoords='offset points',
                    fontsize=6, color=color)

    ax.set_xlabel('2T/d (sim-matrix-to-D ratio)', fontsize=10)
    ax.set_ylabel('Measured Speedup (vs einsum)', fontsize=10)
    ax.set_title('The Master Curve: Speedup = f(2T/d)', fontsize=11)
    ax.set_xscale('log')
    ax.set_xlim(0.3, 20)
    ax.set_ylim(0, 13)
    ax.grid(True, alpha=0.2)

    # Legend for T colors
    ax.scatter([], [], c='#4CAF50', s=40, label='T=32 (text query)')
    ax.scatter([], [], c='#FF9800', s=40, label='T=128 (medium)')
    ax.scatter([], [], c='#E53935', s=40, label='T=512 (visual)')
    ax.scatter([], [], c='#B71C1C', s=40, label='T=1024 (ColPali)')
    ax.legend(fontsize=7, loc='lower right')

    # =========================================================================
    # Plot 3 (bot-left): Memory breakdown per regime
    # =========================================================================
    ax = axes[1, 0]

    regimes = ['ColBERT\nT=32, Ld=300\n2T/d=0.5', 'Text→Image\nT=60, Ld=1024\n2T/d=0.94',
               'ColPali\nT=1024, Ld=1024\n2T/d=16']
    configs = [(32, 300, 128), (60, 1024, 128), (1024, 1024, 128)]
    N = 10000

    d_sizes = [N * Ld * d * 2 / 1e9 for T, Ld, d in configs]
    sim_sizes = [N * T * Ld * 4 / 1e9 for T, Ld, d in configs]
    ratios_mem = [s / d for s, d in zip(sim_sizes, d_sizes)]

    x = np.arange(len(regimes))
    w = 0.5

    # Stacked: D (blue) + sim (red)
    bars_d = ax.bar(x, d_sizes, w, label='D embeddings (FP16)', color='#42A5F5',
                     edgecolor='black', lw=0.5)
    bars_s = ax.bar(x, sim_sizes, w, bottom=d_sizes, label='Sim matrix (FP32)',
                     color='#EF5350', edgecolor='black', lw=0.5, alpha=0.85)

    # Flash line (= D only)
    ax.plot(x, d_sizes, 'o-', color='#1565C0', linewidth=2, markersize=8, zorder=3,
            label='Flash total (= D only)')

    # Labels
    for i in range(len(regimes)):
        total = d_sizes[i] + sim_sizes[i]
        ax.text(i, total + 0.5, f'Total: {total:.0f}GB\nSim/D = {ratios_mem[i]:.1f}x',
                ha='center', fontsize=7, fontweight='bold', color='#B71C1C')
        ax.text(i, d_sizes[i] / 2, f'D={d_sizes[i]:.1f}GB', ha='center', va='center',
                fontsize=7, color='white', fontweight='bold')

    # OOM line
    ax.axhline(y=40, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.text(2.3, 41, '40GB OOM', fontsize=7, color='red')

    ax.set_xticks(x)
    ax.set_xticklabels(regimes, fontsize=7)
    ax.set_ylabel('GPU Memory (GB), N=10K', fontsize=10)
    ax.set_title('Memory Profile: Sim Matrix vs D per Regime', fontsize=11)
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.15, axis='y')

    # =========================================================================
    # Plot 4 (bot-right): What flash saves per regime — percentage breakdown
    # =========================================================================
    ax = axes[1, 1]

    # For each regime: percentage of HBM traffic that is "wasted" (sim matrix write+read)
    # vs "necessary" (D read + output write)
    regime_labels = ['ColBERT\nT=32, d=128', 'Text→Image\nT=60, d=128',
                     'Medium\nT=128, d=128', 'ColPali\nT=1024, d=128']
    T_vals = [32, 60, 128, 1024]
    d_val = 128

    # Traffic breakdown (bytes per doc):
    # D read: Ld * d * 2 (same for both)
    # Sim write: T * Ld * 4 (naive only)
    # Sim read: T * Ld * 4 (naive only, for max reduction)
    # Output: 4 bytes (negligible)
    Ld_ref = 1024  # use Ld=1024 for all

    d_read_pct = []
    sim_wasted_pct = []
    for T in T_vals:
        d_traffic = Ld_ref * d_val * 2
        sim_traffic = 2 * T * Ld_ref * 4  # write + read
        total = d_traffic + sim_traffic
        d_read_pct.append(d_traffic / total * 100)
        sim_wasted_pct.append(sim_traffic / total * 100)

    x = np.arange(len(regime_labels))
    w = 0.6

    ax.bar(x, d_read_pct, w, label='D read (necessary)', color='#42A5F5',
           edgecolor='black', lw=0.5)
    ax.bar(x, sim_wasted_pct, w, bottom=d_read_pct, label='Sim matrix write+read (wasted)',
           color='#EF5350', edgecolor='black', lw=0.5, alpha=0.85)

    # Labels
    for i in range(len(regime_labels)):
        ratio = 2 * T_vals[i] / d_val
        ax.text(i, 50, f'2T/d = {ratio:.1f}\n\nFlash saves\n{sim_wasted_pct[i]:.0f}% of\nHBM traffic',
                ha='center', va='center', fontsize=7, fontweight='bold',
                color='white' if sim_wasted_pct[i] > 30 else '#B71C1C')

    ax.set_xticks(x)
    ax.set_xticklabels(regime_labels, fontsize=7)
    ax.set_ylabel('HBM Traffic Breakdown (%)', fontsize=10)
    ax.set_title('What Flash Eliminates: % of HBM Traffic Saved', fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.15, axis='y')

    plt.tight_layout()
    plt.savefig(args.save, dpi=150, bbox_inches='tight')
    print(f"Saved to {args.save}")


if __name__ == '__main__':
    main()
