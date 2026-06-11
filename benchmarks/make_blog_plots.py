"""Generate blog plots: latency + memory across all MaxSim implementations.

Produces 4 plots:
  1. Latency vs N (number of docs)
  2. Latency vs Ld (doc length)
  3. Latency vs T (query length) — the T/d ratio story
  4. Peak memory vs N — OOM visualization

5 methods compared:
  - einsum: torch.einsum('qd,bld->bql', Q, D) — batched mm, what everyone writes
  - bmm: torch.bmm(Q.expand(...), D.transpose()) — explicit batched mm
  - @ operator: (Q @ D_flat.T).view(...) — single cuBLAS GEMM (the fast trick)
  - chunked bmm: mini-batch with CHUNK_SIZE cap — what vLLM does
  - flash_maxsim: fused kernel, no sim matrix

Usage:
    python benchmarks/make_blog_plots.py [--save blog/plots/]
"""

import torch, time, sys, os, argparse
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '.')

from flash_maxsim import flash_maxsim

device = 'cuda'


def bench_fn(fn, warmup=3, n=10):
    """Benchmark a function, return median ms. Returns None on OOM."""
    try:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(n):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        return np.median(times)
    except (torch.OutOfMemoryError, RuntimeError):
        torch.cuda.empty_cache()
        return None


def measure_memory(fn):
    """Measure peak additional GPU memory of fn(). Returns MB, or None on OOM."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    try:
        fn()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        return (peak - base) / 1e6
    except (torch.OutOfMemoryError, RuntimeError):
        torch.cuda.empty_cache()
        return None


def make_data(N, T, Ld, d=128):
    Q = F.normalize(torch.randn(T, d, device=device, dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(N, Ld, d, device=device, dtype=torch.float16), dim=-1)
    dl = torch.full((N,), Ld, device=device, dtype=torch.int32)
    return Q, D, dl


# --- Method implementations ---

def method_einsum(Q, D):
    S = torch.einsum('qd,bld->bql', Q, D)
    return S.max(dim=2).values.sum(dim=1)

def method_bmm(Q, D):
    B = D.shape[0]
    S = torch.bmm(Q.unsqueeze(0).expand(B, -1, -1), D.transpose(1, 2))
    return S.max(dim=2).values.sum(dim=1)

def method_at_operator(Q, D):
    N, Ld, d = D.shape
    T = Q.shape[0]
    D_flat = D.reshape(N * Ld, d)
    sim = (Q @ D_flat.T).view(T, N, Ld)
    return sim.max(dim=2).values.sum(dim=0)

CHUNK_SIZE = 1000
def method_chunked_bmm(Q, D):
    B, Ld, d = D.shape
    T = Q.shape[0]
    scores = []
    Q_exp = Q.float().unsqueeze(0)
    for i in range(0, B, CHUNK_SIZE):
        j = min(i + CHUNK_SIZE, B)
        chunk = D[i:j].float()
        S = torch.bmm(Q_exp.expand(j - i, -1, -1), chunk.transpose(1, 2))
        scores.append(S.max(dim=2).values.sum(dim=1))
    return torch.cat(scores)

def method_flash(Q, D, dl):
    return flash_maxsim(Q, D, dl)

def method_flash_chunked(Q, D, dl, chunk=None):
    B, Ld, d = D.shape
    # Auto chunk to avoid int32 pointer overflow (N*Ld*d < 2^31)
    if chunk is None:
        max_chunk = max(1, (2**31 - 1) // (Ld * d))
        chunk = min(max_chunk, 50000)
    if B <= chunk:
        return flash_maxsim(Q, D, dl)
    parts = []
    for i in range(0, B, chunk):
        j = min(i + chunk, B)
        parts.append(flash_maxsim(Q, D[i:j], dl[i:j]))
    return torch.cat(parts)


def run_sweep(sweep_param, sweep_values, fixed, methods_dict):
    """Run a parameter sweep. Returns {method_name: [values_or_None]}."""
    results = {name: [] for name in methods_dict}

    for val in sweep_values:
        params = dict(fixed)
        params[sweep_param] = val
        Q, D, dl = make_data(**params)

        print(f"  {sweep_param}={val}: ", end="", flush=True)
        for name, fn_maker in methods_dict.items():
            fn = fn_maker(Q, D, dl)
            ms = bench_fn(fn, warmup=2, n=5)
            results[name].append(ms)
            if ms is not None:
                print(f"{name}={ms:.1f}ms ", end="", flush=True)
            else:
                print(f"{name}=OOM ", end="", flush=True)
        print()

        del Q, D, dl
        torch.cuda.empty_cache()

    return results


def run_memory_sweep(N_values, T, Ld, d=128):
    """Measure peak memory for each method across N values."""
    methods = {}
    results = {name: [] for name in ['einsum', '@ operator', 'chunked bmm', 'flash_maxsim']}

    for N in N_values:
        Q, D, dl = make_data(N, T, Ld, d)
        print(f"  N={N}: ", end="", flush=True)

        # einsum
        mem = measure_memory(lambda: method_einsum(Q, D))
        results['einsum'].append(mem)
        print(f"ein={mem:.0f}MB " if mem else "ein=OOM ", end="", flush=True)

        # @ operator
        mem = measure_memory(lambda: method_at_operator(Q, D))
        results['@ operator'].append(mem)
        print(f"@={mem:.0f}MB " if mem else "@=OOM ", end="", flush=True)

        # chunked bmm
        mem = measure_memory(lambda: method_chunked_bmm(Q, D))
        results['chunked bmm'].append(mem)
        print(f"chunk={mem:.0f}MB " if mem else "chunk=OOM ", end="", flush=True)

        # flash
        mem = measure_memory(lambda: method_flash_chunked(Q, D, dl))
        results['flash_maxsim'].append(mem)
        print(f"flash={mem:.0f}MB " if mem else "flash=OOM ", end="", flush=True)

        print()
        del Q, D, dl
        torch.cuda.empty_cache()

    return results


def plot_sweep(ax, sweep_values, results, xlabel, title, logx=False, logy=True):
    colors = {
        'einsum': '#E53935',
        'bmm': '#FB8C00',
        '@ operator': '#7CB342',
        'chunked bmm': '#8E24AA',
        'flash_maxsim': '#1E88E5',
    }
    markers = {'einsum': 's', 'bmm': '^', '@ operator': 'D', 'chunked bmm': 'v', 'flash_maxsim': 'o'}

    for name, values in results.items():
        xs = [sweep_values[i] for i, v in enumerate(values) if v is not None]
        ys = [v for v in values if v is not None]
        if xs:
            ax.plot(xs, ys, marker=markers.get(name, 'o'), label=name,
                    color=colors.get(name, 'gray'), linewidth=2, markersize=6)
            # Mark OOM points
            for i, v in enumerate(values):
                if v is None and i > 0 and values[i-1] is not None:
                    ax.annotate('OOM', xy=(sweep_values[i], values[i-1]),
                                fontsize=8, color=colors.get(name, 'gray'),
                                fontweight='bold', ha='center')

    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Latency (ms)')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default=None, help='Directory to save plots')
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}\n")

    # Define methods as lambda factories (take Q, D, dl → return callable)
    methods_all = {
        'einsum':       lambda Q, D, dl: lambda: method_einsum(Q, D),
        'bmm':          lambda Q, D, dl: lambda: method_bmm(Q, D),
        '@ operator':   lambda Q, D, dl: lambda: method_at_operator(Q, D),
        'chunked bmm':  lambda Q, D, dl: lambda: method_chunked_bmm(Q, D),
        'flash_maxsim': lambda Q, D, dl: lambda: method_flash_chunked(Q, D, dl),
    }

    # --- Plot 5: INT8 (run FIRST, before other plots pollute GPU state) ---
    print("=== Plot 5: INT8 — naive dequant vs flash fused (sweep N) ===")
    from flash_maxsim import flash_maxsim_int8, quantize_int8

    N_int8 = [100, 500, 1000, 5000, 10000, 25000]
    T_int8, Ld_int8, d_int8 = 32, 300, 128
    res_int8 = {'naive FP16': [], 'naive INT8\n(dequant to HBM)': [],
                'flash FP16': [], 'flash INT8\n(fused dequant)': []}

    for N in N_int8:
        Q, D, dl = make_data(N, T_int8, Ld_int8, d_int8)
        D_q, sc, mn = quantize_int8(D)

        ms = bench_fn(lambda: method_einsum(Q, D), warmup=5, n=20)
        res_int8['naive FP16'].append(ms)

        def naive_int8():
            Df = D_q.float() * sc.float() + mn.float()
            return torch.einsum('qd,bld->bql', Q.float(), Df).max(2).values.sum(1)
        ms = bench_fn(naive_int8, warmup=5, n=20)
        res_int8['naive INT8\n(dequant to HBM)'].append(ms)

        ms = bench_fn(lambda: flash_maxsim(Q, D, dl), warmup=5, n=20)
        res_int8['flash FP16'].append(ms)

        ms = bench_fn(lambda: flash_maxsim_int8(Q, D_q, sc, mn, dl), warmup=5, n=20)
        res_int8['flash INT8\n(fused dequant)'].append(ms)

        k_ni = 'naive INT8\n(dequant to HBM)'
        k_fi = 'flash INT8\n(fused dequant)'
        print(f"  N={N}: naive_fp16={res_int8['naive FP16'][-1]:.1f}ms "
              f"naive_int8={res_int8[k_ni][-1]:.1f}ms "
              f"flash_fp16={res_int8['flash FP16'][-1]:.1f}ms "
              f"flash_int8={res_int8[k_fi][-1]:.1f}ms")

        del Q, D, dl, D_q, sc, mn
        torch.cuda.empty_cache()

    # INT8 memory data
    N_int8_mem = [1000, 5000, 10000, 25000, 50000]
    res_int8_mem = {'D FP16': [], 'D INT8': []}
    for N in N_int8_mem:
        fp16_mb = N * Ld_int8 * d_int8 * 2 / 1e6
        int8_mb = N * Ld_int8 * d_int8 / 1e6
        res_int8_mem['D FP16'].append(fp16_mb)
        res_int8_mem['D INT8'].append(int8_mb)

    # --- Plot 1: Latency vs N ---
    print("\n=== Plot 1: Latency vs N (T=32, Ld=300) ===")
    N_values = [100, 500, 1000, 5000, 10000, 25000, 50000, 100000, 200000]
    res_N = run_sweep('N', N_values, {'T': 32, 'Ld': 300}, methods_all)

    # --- Plot 2: Latency vs Ld ---
    print("\n=== Plot 2: Latency vs Ld (T=32, N=10000) ===")
    Ld_values = [32, 64, 128, 256, 512, 1024]
    res_Ld = run_sweep('Ld', Ld_values, {'T': 32, 'N': 10000}, methods_all)

    # --- Plot 3: Latency vs T ---
    print("\n=== Plot 3: Latency vs T (N=1000, Ld=1024) ===")
    T_values = [32, 64, 128, 256, 512, 1024]
    res_T = run_sweep('T', T_values, {'N': 1000, 'Ld': 1024}, methods_all)

    # --- Plot 4a: Peak memory vs N (ColBERT short docs) ---
    print("\n=== Plot 4a: Memory vs N (T=32, Ld=300) — short docs ===")
    N_mem_short = [100, 500, 1000, 5000, 10000, 25000, 50000]
    res_mem_short = run_memory_sweep(N_mem_short, T=32, Ld=300)

    # --- Plot 4b: Peak memory vs N (ColPali long docs — OOM drama!) ---
    print("\n=== Plot 4b: Memory vs N (T=1024, Ld=1024) — ColPali OOM ===")
    N_mem_long = [100, 250, 500, 1000, 2000, 5000, 10000]
    res_mem_long = run_memory_sweep(N_mem_long, T=1024, Ld=1024)

    # --- Plot 7: Crossover — sweep Ld at small N (N=4, shows @ vs mm vs einsum crossing) ---
    print("\n=== Plot 5: Crossover — Ld sweep at N=4 (small batch) ===")
    Ld_cross = [8, 16, 32, 64, 128, 256, 512]
    methods_cross = {
        'einsum':       lambda Q, D, dl: lambda: method_einsum(Q, D),
        '@ operator':   lambda Q, D, dl: lambda: method_at_operator(Q, D),
        'flash_maxsim': lambda Q, D, dl: lambda: method_flash(Q, D, dl),
    }
    res_cross_Ld = run_sweep('Ld', Ld_cross, {'T': 32, 'N': 4}, methods_cross)

    # --- Plot 6: Crossover — sweep N at fixed Ld=32 (shows batch overhead) ---
    print("\n=== Plot 6: Crossover — N sweep at Ld=32 (short docs) ===")
    N_cross = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000]
    res_cross_N = run_sweep('N', N_cross, {'T': 32, 'Ld': 32}, methods_cross)

    # --- Generate plots ---
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, 2, figsize=(14, 20))
        fig.suptitle(f'Flash-MaxSim: All MaxSim Implementations Compared\n{torch.cuda.get_device_name()}',
                     fontsize=14, fontweight='bold')

        # Plot 1
        plot_sweep(axes[0, 0], N_values, res_N, 'Number of Documents (N)',
                   'Latency vs N (T=32, Ld=300, d=128)', logx=True)

        # Plot 2
        plot_sweep(axes[0, 1], Ld_values, res_Ld, 'Document Length (Ld)',
                   'Latency vs Ld (T=32, N=10K, d=128)', logx=True)

        # Plot 3
        plot_sweep(axes[1, 0], T_values, res_T, 'Query Length (T)',
                   'Latency vs T (N=1K, Ld=1024, d=128)', logx=True)

        # Plot 4: Single stacked bar per N — D (blue) + sim matrix (red) on top
        ax = axes[1, 1]
        T_mem, Ld_mem, d_mem = 1024, 1024, 128
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9

        N_bars = [500, 1000, 2000, 5000, 10000, 50000, 100000]
        d_gb = [N * Ld_mem * d_mem * 2 / 1e9 for N in N_bars]
        sim_gb = [N * T_mem * Ld_mem * 4 / 1e9 for N in N_bars]

        print(f"\n  Memory plot (ColPali T=Ld=1024), GPU={gpu_total:.0f}GB")
        for i, N in enumerate(N_bars):
            oom = " OOM!" if d_gb[i] + sim_gb[i] > gpu_total else ""
            print(f"    N={N:>7}: D={d_gb[i]:.1f}GB + sim={sim_gb[i]:.0f}GB = {d_gb[i]+sim_gb[i]:.0f}GB{oom}")

        x = np.arange(len(N_bars))
        w = 0.6

        # Cap sim bars at display height
        max_display = gpu_total * 1.15
        sim_display = [min(s, max_display - d) for s, d in zip(sim_gb, d_gb)]

        # Stacked bars: D (blue bottom) + sim matrix (red top)
        ax.bar(x, d_gb, w, label='D embeddings (FP16)', color='#42A5F5',
               edgecolor='black', lw=0.5)
        ax.bar(x, sim_display, w, bottom=d_gb, label='Sim matrix (FP32) — naive only',
               color='#EF5350', edgecolor='black', lw=0.5, alpha=0.85)

        # OOM line
        ax.axhline(y=gpu_total, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.axhspan(gpu_total, max_display, alpha=0.08, color='red')
        ax.text(len(N_bars) - 0.3, gpu_total + 0.3,
                f'A100 {gpu_total:.0f}GB', fontsize=8, color='red', fontweight='bold', ha='right')

        # Labels on bars
        for i in range(len(N_bars)):
            total = d_gb[i] + sim_gb[i]
            bar_top = min(d_gb[i] + sim_display[i], max_display - 1)

            # Total label (dark red, white bg)
            if total > gpu_total:
                ax.text(i, bar_top + 0.3, f'{total:.0f}GB\nOOM!', ha='center',
                        fontsize=6, fontweight='bold', color='#B71C1C',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='#B71C1C', alpha=0.9, lw=0.5))
            else:
                ax.text(i, bar_top + 0.3, f'{total:.0f}GB', ha='center',
                        fontsize=6, color='#B71C1C',
                        bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=0.8))

            # D label inside blue bar (white text)
            if d_gb[i] > 1:
                ax.text(i, d_gb[i] / 2, f'D={d_gb[i]:.0f}GB', ha='center', va='center',
                        fontsize=6, color='white', fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels([f'{N//1000}K' if N >= 1000 else str(N) for N in N_bars], fontsize=7)
        ax.set_xlabel('Number of Documents (N)')
        ax.set_ylabel('GPU Memory (GiB)')
        ax.set_title('ColPali Naive Memory (T=Ld=1024)\nFlash-MaxSim = blue only (no sim matrix)')
        ax.legend(fontsize=7, loc='upper left')
        ax.set_ylim(0, max_display)
        ax.grid(True, alpha=0.2, axis='y')

        # Save a separate before/after figure
        fig_ba, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(12, 5),
                                                       sharey=True)
        fig_ba.suptitle('ColPali Memory (T=Ld=1024, d=128, N documents)',
                        fontsize=13, fontweight='bold')

        for ax_ba, show_red, title in [(ax_before, True, 'Before: Naive MaxSim'),
                                        (ax_after, False, 'After: Flash-MaxSim')]:
            ax_ba.bar(x, d_gb, w, label='D embeddings (FP16)', color='#42A5F5',
                      edgecolor='black', lw=0.5)
            if show_red:
                ax_ba.bar(x, sim_display, w, bottom=d_gb, label='Sim matrix (FP32)',
                          color='#EF5350', edgecolor='black', lw=0.5, alpha=0.85)

            ax_ba.axhline(y=gpu_total, color='red', linestyle='--', linewidth=2, alpha=0.7)
            ax_ba.axhspan(gpu_total, max_display, alpha=0.08, color='red')
            ax_ba.text(len(N_bars) - 0.3, gpu_total + 0.3,
                       f'A100 {gpu_total:.0f}GB', fontsize=8, color='red',
                       fontweight='bold', ha='right')

            for i in range(len(N_bars)):
                total = d_gb[i] + sim_gb[i] if show_red else d_gb[i]
                bar_top = min(d_gb[i] + sim_display[i], max_display - 1) if show_red else d_gb[i]

                if show_red:
                    if total > gpu_total:
                        ax_ba.text(i, bar_top + 0.3, f'{total:.0f}GB\nOOM!', ha='center',
                                   fontsize=6, fontweight='bold', color='#B71C1C',
                                   bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                             edgecolor='#B71C1C', alpha=0.9, lw=0.5))
                    else:
                        ax_ba.text(i, bar_top + 0.3, f'{total:.0f}GB', ha='center',
                                   fontsize=6, color='#B71C1C',
                                   bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                                             edgecolor='none', alpha=0.8))
                else:
                    if d_gb[i] > 0.5:
                        ax_ba.text(i, d_gb[i] + 0.3, f'{d_gb[i]:.1f}GB', ha='center',
                                   fontsize=6, color='#1565C0', fontweight='bold')

                if d_gb[i] > 1:
                    ax_ba.text(i, d_gb[i] / 2, f'D={d_gb[i]:.0f}GB', ha='center', va='center',
                               fontsize=6, color='white', fontweight='bold')

            ax_ba.set_xticks(x)
            ax_ba.set_xticklabels([f'{N//1000}K' if N >= 1000 else str(N) for N in N_bars], fontsize=7)
            ax_ba.set_xlabel('Number of Documents (N)')
            ax_ba.set_title(title, fontsize=11, fontweight='bold',
                            color='#B71C1C' if show_red else '#1565C0')
            ax_ba.legend(fontsize=7, loc='upper left')
            ax_ba.set_ylim(0, max_display)
            ax_ba.grid(True, alpha=0.2, axis='y')

        ax_before.set_ylabel('GPU Memory (GiB)')
        fig_ba.tight_layout()
        ba_path = args.save.replace('.png', '_before_after.png') if args.save else 'before_after.png'
        if args.save:
            ba_path = os.path.join(os.path.dirname(args.save) or '.', 'memory_before_after.png')
        fig_ba.savefig(ba_path, dpi=150, bbox_inches='tight')
        print(f"Saved before/after to {ba_path}")

        # Plot 5: INT8 latency comparison
        ax5 = axes[2, 0]
        colors_int8 = {'naive FP16': '#9E9E9E', 'naive INT8\n(dequant to HBM)': '#CE93D8',
                        'flash FP16': '#1E88E5', 'flash INT8\n(fused dequant)': '#2E7D32'}
        for name, values in res_int8.items():
            xs = [N_int8[i] for i, v in enumerate(values) if v is not None]
            ys = [v for v in values if v is not None]
            if xs:
                ax5.plot(xs, ys, marker='o', label=name, color=colors_int8[name],
                         linewidth=2, markersize=6)
        ax5.set_xscale('log'); ax5.set_yscale('log')
        ax5.set_xlabel('Number of Documents (N)')
        ax5.set_ylabel('Latency (ms)')
        ax5.set_title('INT8: Naive Dequant SLOWER, Flash Fused Same Speed\n(T=32, Ld=300, d=128)')
        ax5.legend(fontsize=7); ax5.grid(True, alpha=0.3)

        # Plot 6: INT8 storage
        ax6 = axes[2, 1]
        x = np.arange(len(N_int8_mem)); w = 0.3
        ax6.bar(x - w/2, res_int8_mem['D FP16'], w, label='FP16 (2 bytes/dim)',
                color='#FF7043', edgecolor='black', lw=0.5)
        ax6.bar(x + w/2, res_int8_mem['D INT8'], w, label='INT8 (1 byte/dim)',
                color='#2E7D32', edgecolor='black', lw=0.5)
        for i in range(len(N_int8_mem)):
            ax6.text(i, res_int8_mem['D FP16'][i] + 50, f"{res_int8_mem['D FP16'][i]:.0f}",
                     ha='center', fontsize=7, color='#FF7043')
            ax6.text(i, res_int8_mem['D INT8'][i] + 50, f"{res_int8_mem['D INT8'][i]:.0f}",
                     ha='center', fontsize=7, color='#2E7D32')
        ax6.set_xticks(x); ax6.set_xticklabels([f'N={n//1000}K' for n in N_int8_mem])
        ax6.set_ylabel('D Tensor Size (MB)')
        ax6.set_title('INT8: 2x Storage Compression\nDequant fused in SRAM — no HBM penalty')
        ax6.legend(fontsize=8); ax6.grid(True, alpha=0.3)

        # Plot 7: Crossover — Ld sweep at small N
        plot_sweep(axes[3, 0], Ld_cross, res_cross_Ld, 'Document Length (Ld)',
                   'Crossover: Ld sweep (N=4, T=32)\nSmall batch — @ vs einsum', logx=True)

        # Plot 8: Crossover — N sweep at short Ld
        plot_sweep(axes[3, 1], N_cross, res_cross_N, 'Number of Documents (N)',
                   'Crossover: N sweep (Ld=32, T=32)\nShort docs — batch overhead', logx=True)

        plt.tight_layout()

        if args.save:
            os.makedirs(args.save, exist_ok=True)
            path = os.path.join(args.save, 'flash_maxsim_comparison.png')
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"\nSaved to {path}")
        else:
            plt.savefig('flash_maxsim_comparison.png', dpi=150, bbox_inches='tight')
            print("\nSaved to flash_maxsim_comparison.png")

    except ImportError:
        print("\nmatplotlib not available — skipping plots. Data printed above.")

    print("\nDone.")


if __name__ == '__main__':
    main()
