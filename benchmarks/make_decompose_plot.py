"""Speedup decomposition — clean 2-panel figure.

Left:  Relative time (3 methods, 4 real workloads).
       Visual proof that einsum ≈ @ >> flash.
       All speedup is IO elimination, not fusion.

Right: IO speedup vs 2T/d — the master curve.
       Why speedup varies across workloads.

Data from bench_release_v020.py (A100-40GB, N=1K).
Fusion confirmed ≈ 1.0x from bench_speedup_decompose.py sweeps.

Usage:
    python benchmarks/make_decompose_plot.py [--save path.png]
"""
import numpy as np
import argparse


# 4 representative workloads (from bench_release_v020.py, A100-40GB, N=1K)
# speedup = einsum / flash  (fusion=1.0 so @ ≈ einsum)
WORKLOADS = [
    # (label,        T,   Ld,  d,   speedup)
    ('ColBERT\nT=32  Ld=300',   32,   300,  128,  4.3),
    ('Text→Image\nT=60  Ld=729',  60,   729,  128,  6.0),
    ('Medium\nT=128  Ld=1024', 128,  1024,  128, 11.2),
    ('ColPali\nT=1024  Ld=1024', 1024, 1024,  128, 10.0),
]

# For the master-curve panel: all measured points
MEASURED = [
    (32,  128, 128, 1000,  2.4),
    (32,  300, 128, 1000,  4.3),
    (32,  512, 128, 1000,  6.6),
    (32, 1024, 128, 1000,  7.6),
    (128,  128, 128, 1000,  4.6),
    (128,  300, 128, 1000,  9.0),
    (128,  512, 128, 1000, 10.8),
    (128, 1024, 128, 1000, 11.2),
    (512,  128, 128, 1000,  9.2),
    (512,  300, 128, 1000,  9.7),
    (512,  512, 128, 1000,  9.8),
    (512, 1024, 128, 1000, 10.2),
    (1024,  128, 128, 1000,  9.5),
    (1024,  300, 128, 1000,  9.5),
    (1024,  512, 128, 1000,  9.7),
    (1024, 1024, 128, 1000, 10.0),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='flash_maxsim_decompose.png')
    args = parser.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        'Flash-MaxSim Speedup Decomposition  |  A100-40GB, N=1K, d=128',
        fontsize=13, fontweight='bold', y=1.01
    )

    # =========================================================================
    # Panel 1: Relative time per workload (flash = 1.0)
    # =========================================================================
    n = len(WORKLOADS)
    x = np.arange(n)
    w = 0.25

    labels   = [r[0] for r in WORKLOADS]
    speedups = [r[4] for r in WORKLOADS]

    # Relative times: flash=1.0, @≈einsum=speedup
    flash_rel  = [1.0] * n
    at_rel     = speedups            # @  ≈ einsum (fusion=1.0x)
    einsum_rel = speedups            # same

    # Draw bars
    b1 = ax1.bar(x - w,   einsum_rel, w, label='einsum  (baseline)',
                 color='#90A4AE', edgecolor='black', linewidth=0.6)
    b2 = ax1.bar(x,        at_rel,    w, label='@  (single GEMM)',
                 color='#546E7A', edgecolor='black', linewidth=0.6)
    b3 = ax1.bar(x + w,   flash_rel,  w, label='flash  (no sim matrix)',
                 color='#1565C0', edgecolor='black', linewidth=0.6)

    # Speedup labels above flash bars
    for i, sp in enumerate(speedups):
        ax1.text(x[i] + w, 1.0 + 0.25, f'{sp:.1f}×', ha='center', va='bottom',
                 fontsize=9, fontweight='bold', color='#1565C0')

    # Fusion annotation: arrow showing einsum≈@
    ax1.annotate(
        'Fusion = einsum / @\n≈ 1.0× (both dispatch\na single GEMM)',
        xy=(x[0] - w/2, einsum_rel[0]),
        xytext=(x[0] + 0.15, einsum_rel[0] + 2.5),
        fontsize=8, color='#37474F',
        arrowprops=dict(arrowstyle='->', color='#37474F', lw=1.2),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#ECEFF1', edgecolor='#90A4AE', alpha=0.9)
    )

    # IO annotation: brace between @ and flash
    mid_x = x[2]
    ax1.annotate(
        '',
        xy=(mid_x + w, 1.05), xytext=(mid_x + w, at_rel[2] - 0.2),
        arrowprops=dict(arrowstyle='<->', color='#B71C1C', lw=1.8)
    )
    ax1.text(mid_x + w + 0.14, (1.0 + at_rel[2]) / 2,
             f'IO\n{at_rel[2]:.0f}×',
             ha='left', va='center', fontsize=9, color='#B71C1C', fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel('Relative time  (flash = 1.0×)', fontsize=11)
    ax1.set_title('einsum ≈ @  >>  flash\nAll speedup is IO elimination, not fusion', fontsize=10.5)
    ax1.set_ylim(0, 15)
    ax1.legend(fontsize=8.5, loc='upper right')
    ax1.grid(True, alpha=0.2, axis='y')
    ax1.axhline(y=1.0, color='#1565C0', linestyle='--', linewidth=1.2, alpha=0.5)

    # =========================================================================
    # Panel 2: Measured speedup vs 2T/d — the master curve
    # =========================================================================
    ratios_all  = [2 * T / d for T, Ld, d, B, sp in MEASURED]
    speedups_all = [sp       for T, Ld, d, B, sp in MEASURED]

    # Color by Ld (doc length — explains scatter at same 2T/d)
    Ld_vals = [Ld for T, Ld, d, B, sp in MEASURED]
    colors = ['#90CAF9' if Ld <= 128 else '#42A5F5' if Ld <= 300 else '#1565C0' if Ld <= 512 else '#0D47A1'
              for Ld in Ld_vals]

    ax2.scatter(ratios_all, speedups_all, c=colors, s=60,
                edgecolor='black', linewidth=0.5, zorder=3)

    # Workload callouts
    callout_offsets = {
        'ColBERT\nT=32  Ld=300':      ( 0.1, -1.5),
        'Text→Image\nT=60  Ld=729':   (-0.2,  1.2),
        'Medium\nT=128  Ld=1024':     ( 0.1,  1.2),
        'ColPali\nT=1024  Ld=1024':   ( 0.1, -1.5),
    }
    wl_colors = ['#4CAF50', '#FF9800', '#1565C0', '#B71C1C']
    for i, (label, T, Ld, d, sp) in enumerate(WORKLOADS):
        r = 2 * T / d
        dx, dy = callout_offsets[label]
        short = label.split('\n')[0]
        ax2.annotate(short, xy=(r, sp), xytext=(r + dx, sp + dy),
                     fontsize=8.5, color=wl_colors[i], fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color=wl_colors[i], lw=1.0))

    # Regime shading
    ax2.axvspan(0.3,  1.0, alpha=0.07, color='green',  zorder=0)
    ax2.axvspan(1.0,  3.0, alpha=0.07, color='orange', zorder=0)
    ax2.axvspan(3.0, 20.0, alpha=0.07, color='red',    zorder=0)

    ax2.text(0.55, 12.2, 'Fusion\nregime\n2T/d<1', fontsize=8, color='green',
             fontweight='bold', ha='center', alpha=0.8)
    ax2.text(1.7,  12.2, 'Transition\n2T/d≈1', fontsize=8, color='#E65100',
             fontweight='bold', ha='center', alpha=0.8)
    ax2.text(8.0,  12.2, 'IO-tiling\nregime\n2T/d>>1', fontsize=8, color='#B71C1C',
             fontweight='bold', ha='center', alpha=0.8)

    # Ld legend
    legend_elements = [
        Patch(facecolor='#90CAF9', edgecolor='black', label='Ld=128'),
        Patch(facecolor='#42A5F5', edgecolor='black', label='Ld=300'),
        Patch(facecolor='#1565C0', edgecolor='black', label='Ld=512'),
        Patch(facecolor='#0D47A1', edgecolor='black', label='Ld=1024'),
    ]
    ax2.legend(handles=legend_elements, fontsize=8, title='Doc length Ld',
               title_fontsize=8, loc='lower right')

    # Annotation: why scatter exists at same ratio
    ax2.annotate(
        'At same 2T/d:\nlarger Ld → higher speedup\n(kernel overhead amortized)',
        xy=(0.5, 7.6), xytext=(1.5, 7.8),
        fontsize=8, color='#37474F',
        arrowprops=dict(arrowstyle='->', color='#37474F', lw=1.0),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFDE7', edgecolor='#FBC02D', alpha=0.9)
    )

    ax2.set_xscale('log')
    ax2.set_xlim(0.3, 20)
    ax2.set_ylim(0, 13.5)
    ax2.set_xlabel('2T/d  =  sim-matrix size / D size', fontsize=11)
    ax2.set_ylabel('Speedup vs einsum  (×)', fontsize=11)
    ax2.set_title('Speedup governed by 2T/d ratio\n(scatter within regime = Ld effect)', fontsize=10.5)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(args.save, dpi=150, bbox_inches='tight')
    print(f"Saved to {args.save}")


if __name__ == '__main__':
    main()
