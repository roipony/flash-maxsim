"""CUDA graph capture + replay smoke test for Flash-MaxSim.

§5.9 of the paper claims the deterministic per-arch launch-config
table makes the kernels CUDA-graph friendly (Triton autotune-based
kernels typically aren't, because each call can pick a different
configuration). This bench verifies that claim empirically:

  1. Forward-only capture: run flash_maxsim_batched under
     `torch.cuda.graph()` context, replay it, verify output matches
     a non-graph reference call.
  2. Latency comparison: measure single-call latency vs. replay
     latency. Graph replay should be ~as fast or faster than direct
     launch (no kernel-launch overhead from the Python side).
  3. Same drill for the split-d path (d=512).

If autotune were in play, capture would either fail (Triton tries
multiple configs during capture) or produce non-deterministic replays
(picks a different config on each replay). Our heuristic-table
kernels should be clean.
"""
import json, os, sys, time, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from flash_maxsim import flash_maxsim_batched

torch.set_float32_matmul_precision('high')
torch.manual_seed(0)


def _bench_replay(g, runs=20):
    """Median-of-N replay latency in ms."""
    torch.cuda.synchronize()
    ts = []
    for _ in range(runs):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); g.replay(); e.record(); e.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def _bench_direct(fn, runs=20):
    torch.cuda.synchronize()
    ts = []
    for _ in range(runs):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); e.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


print(f"GPU: {torch.cuda.get_device_name()}", flush=True)
out = {"gpu": torch.cuda.get_device_name(), "shapes": []}

for d, label in [(128, "in-shot"), (512, "split-d")]:
    print(f"\n=== d={d} ({label}) ===", flush=True)
    B, Lq, Ld = 128, 1024, 1024
    row = {"d": d, "B": B, "Lq": Lq, "Ld": Ld, "dispatch": label}

    # Allocate input + output buffers ONCE (cuda graph captures pointers)
    Q = F.normalize(torch.randn(1, Lq, d, device='cuda', dtype=torch.float16), dim=-1)
    D = F.normalize(torch.randn(B,  Ld, d, device='cuda', dtype=torch.float16), dim=-1)

    # Reference: direct call, no graph
    ref = flash_maxsim_batched(Q, D)
    torch.cuda.synchronize()

    # Warmup so kernels are JIT-cached before capture (Triton compile inside
    # a graph-capture context is not supported).
    for _ in range(3):
        _ = flash_maxsim_batched(Q, D)
    torch.cuda.synchronize()

    # Capture
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            captured = flash_maxsim_batched(Q, D)
        row["capture"] = "ok"
    except Exception as e:
        row["capture"] = f"FAILED: {type(e).__name__}: {str(e)[:120]}"
        print(f"  CAPTURE FAILED: {row['capture']}", flush=True)
        out["shapes"].append(row)
        continue

    # Replay correctness
    g.replay()
    torch.cuda.synchronize()
    max_abs = float((ref - captured).abs().max())
    row["replay_max_abs_delta"] = round(max_abs, 6)
    correctness = "OK" if max_abs < 1e-3 else "MISMATCH"
    row["correctness"] = correctness
    print(f"  capture: ok  replay-vs-direct |Δ|max={max_abs:.3e}  {correctness}", flush=True)

    # Latency comparison
    replay_ms = _bench_replay(g)
    direct_ms = _bench_direct(lambda: flash_maxsim_batched(Q, D))
    row["replay_ms"] = round(replay_ms, 4)
    row["direct_ms"] = round(direct_ms, 4)
    row["graph_speedup_vs_direct"] = round(direct_ms / replay_ms, 3)
    print(f"  replay={replay_ms:.3f}ms  direct={direct_ms:.3f}ms  "
          f"graph/direct={replay_ms/direct_ms:.2f}x", flush=True)

    out["shapes"].append(row)
    del Q, D, ref, captured, g; torch.cuda.empty_cache()

p = os.path.join(HERE, f"bench_cuda_graph_capture_{torch.cuda.get_device_name().replace(' ', '_').replace('/', '_')}.json")
json.dump(out, open(p, "w"), indent=2)
print(f"\nwrote {p}", flush=True)
