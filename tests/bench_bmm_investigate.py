"""Investigate why bmm gets faster at N=750-1000 for ColPali shapes.
Separate bmm from max/sum, test different approaches."""
import sys, time, torch
import torch.nn.functional as F
sys.path.insert(0, '.')

device = 'cuda'
D_DIM = 128
DTYPE = torch.float16

def bench(fn, warmup=10, runs=30):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]

gpu = torch.cuda.get_device_name(0)
print(f"GPU: {gpu}\n")

# Fixed shapes to isolate the batch-size effect (no randomness in dims)
Lq = 1024
Ld = 1024

print("="*90)
print(f" BMM investigation: fixed Lq={Lq}, Ld={Ld}, d={D_DIM}")
print("="*90)
print(f"  {'N':>6}  {'bmm_ms':>8}  {'max_ms':>8}  {'sum_ms':>8}  {'total_ms':>9}  "
      f"{'einsum_ms':>10}  {'matmul_ms':>10}  {'sim_GB':>7}  {'tflops':>7}")
print(f"  {'-'*90}")

for N in [50, 100, 200, 300, 400, 500, 600, 700, 750, 800, 900, 1000, 1200, 1500, 2000, 3000, 5000]:
    Q = F.normalize(torch.randn(N, Lq, D_DIM, dtype=DTYPE, device=device), dim=-1)
    D = F.normalize(torch.randn(N, Ld, D_DIM, dtype=DTYPE, device=device), dim=-1)

    sim_gb = N * Lq * Ld * 2 / (1024**3)
    # FLOPs for bmm: N * Lq * Ld * 2 * D_DIM (multiply-add)
    flops = N * Lq * Ld * 2 * D_DIM

    # Pre-allocate output to see if allocation matters
    S_pre = torch.empty(N, Lq, Ld, dtype=DTYPE, device=device)

    # 1. Just bmm
    DT = D.transpose(1, 2).contiguous()
    bmm_ms = bench(lambda: torch.bmm(Q, DT))

    # 2. Just max on pre-computed S
    S = torch.bmm(Q, DT)
    max_ms = bench(lambda: S.max(dim=2).values)

    # 3. Just sum
    M = S.max(dim=2).values
    sum_ms = bench(lambda: M.sum(dim=1))

    # 4. Full pipeline with bmm
    total_ms = bench(lambda: torch.bmm(Q, DT).max(dim=2).values.sum(dim=1))

    # 5. einsum
    einsum_ms = bench(lambda: torch.einsum('nqd,nld->nql', Q, D).max(dim=2).values.sum(dim=1))

    # 6. torch.matmul (batched)
    matmul_ms = bench(lambda: torch.matmul(Q, DT).max(dim=2).values.sum(dim=1))

    tflops = flops / (bmm_ms * 1e-3) / 1e12

    print(f"  {N:>6}  {bmm_ms:>8.2f}  {max_ms:>8.2f}  {sum_ms:>8.2f}  {total_ms:>9.2f}  "
          f"{einsum_ms:>10.2f}  {matmul_ms:>10.2f}  {sim_gb:>6.1f}G  {tflops:>6.1f}T")

    del Q, D, DT, S, M, S_pre
    torch.cuda.empty_cache()

# Part 2: Is it the GEMM algorithm? Test with different BLAS backends
print(f"\n{'='*90}")
print(" Part 2: GEMM algorithm test — same shape [N, 1024, 128] x [N, 128, 1024]")
print(f"{'='*90}")

# Test if cuBLAS uses different algorithms at different batch sizes
# by checking if transposing D (non-contiguous) changes behavior
for N in [500, 750, 1000]:
    Q = F.normalize(torch.randn(N, Lq, D_DIM, dtype=DTYPE, device=device), dim=-1)
    D = F.normalize(torch.randn(N, Ld, D_DIM, dtype=DTYPE, device=device), dim=-1)

    DT_contig = D.transpose(1, 2).contiguous()
    DT_noncontig = D.transpose(1, 2)  # non-contiguous

    bmm_contig = bench(lambda: torch.bmm(Q, DT_contig))
    bmm_noncontig = bench(lambda: torch.bmm(Q, DT_noncontig))

    # Also try: loop of individual matmuls (what cuBLAS might do internally for small N)
    def loop_matmul():
        results = []
        for i in range(N):
            results.append(Q[i] @ DT_contig[i])
        return torch.stack(results)

    # Chunked bmm — process in chunks
    def chunked_bmm(chunk_size=256):
        results = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            S_chunk = torch.bmm(Q[start:end], DT_contig[start:end])
            results.append(S_chunk.max(dim=2).values.sum(dim=1))
        return torch.cat(results)

    chunk_ms = bench(chunked_bmm)

    print(f"  N={N}: contig={bmm_contig:.2f}ms  noncontig={bmm_noncontig:.2f}ms  chunked256={chunk_ms:.2f}ms")

    del Q, D, DT_contig, DT_noncontig
    torch.cuda.empty_cache()

# Part 3: Is it memory bandwidth? The sim matrix at N=750 is 1.5GB
print(f"\n{'='*90}")
print(" Part 3: Memory bandwidth test — allocation vs compute")
print(f"{'='*90}")

for N in [500, 750, 1000, 2000]:
    Q = F.normalize(torch.randn(N, Lq, D_DIM, dtype=DTYPE, device=device), dim=-1)
    DT = F.normalize(torch.randn(N, D_DIM, Ld, dtype=DTYPE, device=device), dim=-1)

    # Pre-allocate output
    S_out = torch.empty(N, Lq, Ld, dtype=DTYPE, device=device)

    # bmm with pre-allocated output via addmm
    def bmm_prealloc():
        torch.bmm(Q, DT, out=S_out)
        return S_out.max(dim=2).values.sum(dim=1)

    # Standard
    def bmm_standard():
        S = torch.bmm(Q, DT)
        return S.max(dim=2).values.sum(dim=1)

    ms_prealloc = bench(bmm_prealloc)
    ms_standard = bench(bmm_standard)

    sim_gb = N * Lq * Ld * 2 / (1024**3)
    print(f"  N={N}: standard={ms_standard:.2f}ms  prealloc={ms_prealloc:.2f}ms  sim={sim_gb:.1f}GB")

    del Q, DT, S_out
    torch.cuda.empty_cache()

print("\nDone.")
