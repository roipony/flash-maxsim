"""E2 — cross-device negatives: conventional DDP vs Flash-MaxSim sharded.

Reviewer EUF4 W1: "with two GPUs, a conventional implementation can construct
the same negative pool by exchanging embeddings across devices."

This measures exactly that, on W GPUs (torchrun), at ColBERT training shapes.
For each global batch B (= negative pool):

  conventional : all_gather docs WITH gradients (torch.distributed.nn),
                 each rank scores its local queries against the GLOBAL pool
                 via the reference einsum -> [B/W, B, Lq, Ld] block,
                 CE loss, backward. Per-device operator memory ~ (B^2/W)LqLd.
  fm-sharded   : flash_maxsim.sharded_ddp.ddp_contrastive_step — docs never
                 leave their rank, FM scores tile-by-tile, gradient-exact.

Reported per arm: per-device peak GB, step wall-clock, OK/OOM.
The claim under test: conventional reachable pool grows ~sqrt(W); FM is
linear in per-device embedding capacity, so the two compose rather than
substitute.

Launch:  torchrun --nproc-per-node=2 e2_ddp_negatives.py
Output:  analysis/e1_results/e2_ddp_W<world>.json (rank 0)
"""
from __future__ import annotations
import gc, json, os, time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn
import torch.nn.functional as F

HERE = Path(os.path.dirname(os.path.abspath(__file__)))

Lq, Ld, DIM = 32, 180, 128
TAU = 0.02
GLOBAL_BATCHES = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
RUNS = 5


def make_shard(B_local: int, dev: str, seed: int):
    g = torch.Generator(device=dev).manual_seed(seed)
    Q = F.normalize(torch.randn(B_local, Lq, DIM, device=dev,
                                dtype=torch.float16, generator=g), dim=-1)
    D = F.normalize(torch.randn(B_local, Ld, DIM, device=dev,
                                dtype=torch.float16, generator=g), dim=-1)
    Q.requires_grad_(True)
    D.requires_grad_(True)
    return Q, D


def step_conventional(Q_local, D_shard, rank, world):
    """Reference einsum against the gathered global doc pool."""
    D_global = torch.cat(dist_nn.all_gather(D_shard), dim=0)  # grads flow back
    # [B_local, B_global, Lq, Ld] — the tensor under discussion
    sim = torch.einsum("qsd,btd->qbst", Q_local.float(), D_global.float())
    scores = sim.max(dim=3).values.sum(dim=2)                 # [B_local, B_glob]
    B_local = Q_local.shape[0]
    labels = torch.arange(B_local, device=Q_local.device) + rank * B_local
    # divide by world so the summed per-rank gradients equal the gradient of
    # the GLOBAL-mean CE (same objective the fm-sharded step computes);
    # standard DDP achieves the same via its 1/W gradient averaging.
    loss = F.cross_entropy(scores / TAU, labels) / world
    loss.backward()
    return float(loss)


def step_fm(Q_local, D_shard, rank, world):
    from flash_maxsim.sharded_ddp import ddp_contrastive_step
    loss, _gq = ddp_contrastive_step(Q_local, D_shard, tau=TAU)
    return loss


def step_fm_tiled(Q_local, D_shard, rank, world):
    """flash-sharded: row-sharded online softmax + intra-shard tiling —
    no [B,B] replication; per-rank score working set O(B*tile)."""
    from flash_maxsim.sharded_ddp import flash_sharded_contrastive_loss
    loss = flash_sharded_contrastive_loss(Q_local, D_shard, tau=TAU, tile=2048)
    loss.backward()
    return float(loss)


def measure(fn, B_global, rank, world, dev):
    B_local = B_global // world
    ok_local = torch.ones(1, device=dev)
    try:
        Q, D = make_shard(B_local, dev, seed=1234 + rank)
        # warmup
        fn(Q, D, rank, world)
        Q.grad = None; D.grad = None
        torch.cuda.synchronize()
        gc.collect(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(RUNS):
            fn(Q, D, rank, world)
            Q.grad = None; D.grad = None
        torch.cuda.synchronize()
        wall = (time.time() - t0) / RUNS
        peak = torch.cuda.max_memory_allocated() / 1e9
    except torch.cuda.OutOfMemoryError:
        ok_local[0] = 0.0
        wall, peak = None, None
    finally:
        Q = D = None
        gc.collect(); torch.cuda.empty_cache()
    # an arm fails if ANY rank OOMs
    dist.all_reduce(ok_local, op=dist.ReduceOp.MIN)
    if ok_local.item() < 1:
        return {"ok": False}
    # report the max peak across ranks
    peak_t = torch.tensor([peak], device=dev)
    dist.all_reduce(peak_t, op=dist.ReduceOp.MAX)
    wall_t = torch.tensor([wall], device=dev)
    dist.all_reduce(wall_t, op=dist.ReduceOp.MAX)
    return {"ok": True, "peak_gb": round(peak_t.item(), 4),
            "step_ms": round(wall_t.item() * 1e3, 2)}


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = f"cuda:{rank}"

    # gradient-equivalence check: same data through both paths at a size
    # where both fit; compare grad_D on this rank and the loss value.
    B_v = 256
    Qv, Dv = make_shard(B_v // world, dev, seed=777 + rank)
    step_conventional(Qv, Dv, rank, world)
    gD_ref = Dv.grad.detach().float().clone()
    gQ_ref = Qv.grad.detach().float().clone()
    Qv.grad = None; Dv.grad = None
    from flash_maxsim.sharded_ddp import ddp_contrastive_step
    _, gQ_fm = ddp_contrastive_step(Qv, Dv, tau=TAU)
    gD_fm = Dv.grad.detach().float()
    cos_d = F.cosine_similarity(gD_ref.flatten(), gD_fm.flatten(), dim=0)
    cos_q = F.cosine_similarity(gQ_ref.flatten(),
                                gQ_fm.detach().float().flatten(), dim=0)
    rel_d = ((gD_ref - gD_fm).norm() / gD_ref.norm()).item()
    Qv.grad = None; Dv.grad = None
    step_fm_tiled(Qv, Dv, rank, world)
    gD_t = Dv.grad.detach().float()
    cos_dt = F.cosine_similarity(gD_ref.flatten(), gD_t.flatten(), dim=0)
    rel_dt = ((gD_ref - gD_t).norm() / gD_ref.norm()).item()
    verify = {"B_global": B_v, "cos_gradD": round(cos_d.item(), 8),
              "cos_gradQ": round(cos_q.item(), 8),
              "rel_err_gradD": round(rel_d, 8),
              "tiled_cos_gradD": round(cos_dt.item(), 8),
              "tiled_rel_err_gradD": round(rel_dt, 8)}
    if rank == 0:
        print("VERIFY:", json.dumps(verify), flush=True)
    Qv = Dv = None
    gc.collect(); torch.cuda.empty_cache()

    results = []
    dead = {"conventional": False, "fm-sharded": False, "fm-flash-tiled": False}
    for B in GLOBAL_BATCHES:
        row = {"B_global": B, "B_local": B // world,
               "negatives_per_query": B - 1}
        for name, fn in (("conventional", step_conventional),
                         ("fm-sharded", step_fm),
                         ("fm-flash-tiled", step_fm_tiled)):
            if dead[name]:
                row[name] = {"ok": False, "skipped": "OOM at smaller B"}
                continue
            r = measure(fn, B, rank, world, dev)
            row[name] = r
            if not r["ok"]:
                dead[name] = True
        if rank == 0:
            print(json.dumps(row), flush=True)
        results.append(row)
        if all(dead.values()):
            break

    if rank == 0:
        out = HERE / "e1_results" / f"e2_ddp_W{world}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps({
            "gpu": torch.cuda.get_device_name(0), "world": world,
            "Lq": Lq, "Ld": Ld, "dim": DIM, "tau": TAU, "runs": RUNS,
            "gradient_verify": verify,
            "rows": results}, indent=2))
        print("wrote", out)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
