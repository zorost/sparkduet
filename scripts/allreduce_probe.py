#!/usr/bin/env python3
"""2-node NCCL all-reduce probe. Run by nccl-check.sh inside the serving image.
Prints BUSBW_GBS=<x> on rank 0. Ring all-reduce moves ~2x the payload."""
import os, time
import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
t = torch.ones(64 * 1024 * 1024, device="cuda")  # 256 MiB fp32
for _ in range(3):
    dist.all_reduce(t)
torch.cuda.synchronize()
t0 = time.time()
N = 20
for _ in range(N):
    dist.all_reduce(t)
torch.cuda.synchronize()
dt = (time.time() - t0) / N
gbs = 2 * t.numel() * 4 / dt / 1e9
if rank == 0:
    print(f"BUSBW_GBS={gbs:.1f}", flush=True)
dist.destroy_process_group()
