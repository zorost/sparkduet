# FIELD-NOTES.md, deployment lessons from real 2× DGX Spark clusters

Sanitized operational findings from bringing SparkDuet to live clusters. No
cluster-specific identifiers are included; treat every item as a checklist entry
before you benchmark anything. Ladder labels apply as usual ([M-else] = observed
on a live deployment, generalized).

---

## 1. The management plane is your first failure point, not the GPU

On consumer-premise clusters the two Sparks are frequently reached over **WiFi**
(the DGX-OS default interface is wireless). Observed failure pattern
[M-else]: the WiFi link **flaps on a minute scale**, existing TCP sessions
survive, but new connections fail with `No route to host` (ARP starts resolving
to a proxy-ARPing mesh node). SSH, benchmarking, and weight sync all stall
behind what looks like a compute problem but is purely management-plane.

Rules we now enforce:

1. **Management traffic goes on wired Ethernet or a mesh VPN (tailnet), never
   WiFi.** The 200G QSFP RoCE link is for NCCL/KV, do not also depend on it for
   operator access unless you have a routed path onto it.
2. **Hold long-lived sessions** (tmux on the head node, one SSH session per work
   batch). Long-lived flows ride through WiFi flaps; new flows die in them.
3. **Batch per session.** Never open a new connection per command; stage the
   whole inventory/benchmark script and run it in one shot.
4. `sparkduetctl.sh doctor` now includes a management-plane stability probe:
   it opens and closes several short SSH sessions and warns if the failure rate
   is nonzero before you attribute benchmark noise to the serving stack.

## 2. Identical boxes are not identical

- Two same-model GB10 boxes can report ~2 GiB different `MemTotal` [M-else].
  Budget KV pools against `MemAvailable` measured **on each node**, not against a
  fleet-average or someone else's published figure.
- A node may be deliberately **clock-capped** (e.g. to keep mixed GPU workloads
  from hard-rebooting the chassis). A capped node benchmarks lower, that is
  policy, not a regression. Record clocks (`nvidia-smi -q -d CLOCK`) in every
  `results/` artifact; `bench.py` does this automatically.
- Desktop images spend 10–15 GB of unified memory on GNOME/snaps. On a serving
  node, strip the desktop or accept the smaller pool.

## 3. The "house model" reality (why Lane F and the Qwen profile exist)

Real small clusters converge on the same pattern [M-else, generalized]: one
**always-on house model** (a 27B-class dense vLLM serving chat/automation
clients) on node 1, and an **on-demand library** (GGUF swapper loading bigger
models one at a time) on node 2. This is exactly Lane F with heterogeneous
models, and it is why SparkDuet treats "which model, which lane" as a first
class decision (`docs/MODELS.md`) instead of hard-wiring one checkpoint.
When benchmarking against an incumbent house model, **capture its exact
container spec first** (`docker inspect`, full command line, env, mounts) -
that capture is your revert plan. Stop, never remove.

## 4. Benchmark hygiene the field forces on you

- Thinking-mode defaults differ per model (some ship reasoning ON with preserved
  chain-of-thought). A "thinking on vs off" confusion looks exactly like a
  1.3× speed regression [M-else]. The workload triple (prompt, concurrency,
  thinking) is mandatory on every number, see `docs/BENCHMARK-PROTOCOL.md`.
- First-token latency on a freshly booted engine includes Triton/CUDA JIT;
  warm the kernel keys before measuring (`scripts/warmup.sh`) or your p50 TTFT
  is a compile benchmark.
- Weight downloads are the long pole (~160 GB per node for the flagship
  profile). Pre-stage with `prepare-models.sh` and keep `HF_HUB_OFFLINE=1`
  afterwards so a hub retry can never fill a disk mid-benchmark.

## 5. One GPU, one budget: contention is silent and large

The GB10's unified memory means every container on a node shares one GPU and
one memory pool. Two first-party measurements of the same on-demand GGUF 27B
on the same node, minutes apart [M-here, artifacts in `results/`]:

| Condition | Single-stream |
|---|---:|
| Node otherwise idle | 10.1 tok/s |
| TP=2 worker rank loading + compiling on the node | 3.3 tok/s |

Same model, same server, 3× apart. Rules: never benchmark during a lane
bring-up; expect on-demand latency to degrade while a TP rank is busy on that
node; and when a number looks wrong, check `docker ps` on the node before
blaming the engine.

Related, worse, and sneakier: a CUDA-built llama.cpp server that loses its GPU
context (driver update, device reset) **keeps serving from CPU** at roughly
1/3 to 1/7 speed, logging only `ggml_cuda_init: failed` once at load. It looks
alive, answers correctly, and quietly benchmarks like a potato. `doctor` greps
recent container logs for exactly this string; restart the container to
recover the GPU.

**The worst case is not slow, it is a dead box.** Field incident, 2026-08-25
[M-here]: an on-demand load of a ~65 GiB GGUF started while the TP=2 worker
rank held its ~110 GiB residency on the same node. The llama-server process
took a kernel rw-semaphore inside the driver mmap path and never released it;
`nvidia-smi`, `dockerd`, `nvidia-modeset`, and `sshd` all queued behind it
(`journalctl -b -1`: "task nvidia-smi blocked for more than 245 seconds ...
likely owned by task llama-server"). The kernel kept answering pings, the
reverse proxy kept serving its 401s, and nothing that needed a fork or the GPU
driver worked, including SSH logins. The node watchdog-rebooted itself roughly
twenty minutes in. The engine on the healthy node degraded to 41 s for an
8-token reply while its peer rank thrashed.

The rule that falls out: **an on-demand library that shares a node with a
resident TP rank may only list models that fit the leftover headroom.** On a
121 GiB node with a 0.78-utilization rank resident, that is roughly 26 GiB:
a 27B Q5 fits, a 122B Q4 or a 284B IQ3 never will. Comment the oversized
entries out of the llama-swap config while the depth lane is up (weights stay
on disk), and re-enable them only when the lane is stopped. If you need the
big GGUFs served, that is what Lane F on a dedicated node is for.

## 6. Dual-rail fabrics: pin the rendezvous, list the rails

The QSFP cable carries two independent 100/200G rails (two interfaces, two RDMA
devices). What we learned wiring TP=2 across them [M-here]:

- **NCCL** takes comma lists (`NCCL_IB_HCA=rocep1s0f0,rocep1s0f1`) and stripes
  across both rails. Give it both.
- **Gloo** (PyTorch's CPU rendezvous, used at engine boot) does not reliably
  handle interface lists. Pin `GLOO_SOCKET_IFNAME` to **one** rail whose subnet
  matches `MASTER_ADDR`, or rank 1 dies at startup with
  `Connection closed by peer` while NCCL itself is perfectly healthy.
- The container needs `--device /dev/infiniband` and `IPC_LOCK`, or NCCL
  silently falls back to TCP sockets: 4.0 GB/s bus bandwidth instead of
  42.0 GB/s measured on the same cable [M-here]. `nccl-check.sh` fails the gate
  below 8 GB/s precisely to catch this class of misconfiguration.

## 7. Reversibility checklist (print before any cluster change)

- [ ] Incumbent container specs captured (`docker inspect` → file, off-box copy)
- [ ] Our additions live only in: our containers, our directory, tmux sessions
      named `sparkduet-*`
- [ ] Stop = reversible; remove = forbidden without explicit owner approval
- [ ] Shared infrastructure (gateways, proxies, dashboards) untouched
- [ ] Background download/sync sessions belonging to the owner never killed
