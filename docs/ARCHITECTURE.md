# ARCHITECTURE.md, the design and the arithmetic behind it

This document explains *why* the system is shaped the way it is. Every
quantitative claim carries a ladder label ([M-here] / [M-else] / [P]) and a
derivation or citation. The evidence base with links is `docs/RESEARCH.md`.

---

## 1. Hardware truth table (GB10, per node)

| Quantity | Value | Source |
|---|---|---|
| Unified memory | 128 GB (~119–121 GiB usable; same-model boxes differ by ~2 GiB, measure `MemAvailable` on your own pair) | [M-else: entrpi README; M-here: both our nodes] |
| Memory bandwidth | ~273 GB/s | [M-else: Keys] |
| GPU arch | sm_121a (Blackwell, consumer line), **not** covered by several datacenter-only kernels (DeepGEMM, FLASHMLA_SPARSE ship no sm12x paths) | [M-else: Keys nine-walls] |
| Interconnect | 200G QSFP RoCEv2 direct link. Line rate ~25 GB/s; NCCL all-reduce measured **~9–10 GB/s bus bandwidth** on a live pair | [M-here: `nccl-check.sh --full` on our pair] |
| Bulk file sync over the link | ~460 MB/s via rsync/ssh (cipher + disk bound, not the link), a 156 GiB checkpoint moves head→worker in ~6 min | [M-here] |
| CPU | Grace aarch64; host OS + desktop can eat 10–15 GB if not stripped | [M-else: entrpi] |

Two consequences dominate everything else:

- **Decode is weight-bandwidth bound.** DSV4-Flash-0731 activates 13B
  params/token. On the 2-node TP=2 stack the no-spec decode is 26.6–26.7 tok/s
  regardless of whether KV is fp8 or NVFP4 [M-else: botAGI control]. With
  DSpark k=5 at acceptance a≈0.6, mean accepted length ≈ (1−a⁶)/(1−a) ≈ 2.38 →
  ~63 tok/s [P, matches the measured 62–83 band]. **Levers: kernel bandwidth
  efficiency × speculative acceptance. KV dtype is a capacity lever, never a
  speed lever.**
- **Prefill is compute bound** and parallelizes across independent engines:
  ~875–2,563 tok/s on the TP=2 pair depending on prompt size [M-else: MiaAI
  RESULTS]; ~960–1,010 tok/s on a single node with entrpi's ds4 [M-else].

## 2. Why lanes at all

One serving topology cannot be optimal for the three canonical workloads on
this hardware:

| Workload | Binding constraint | Optimal shape |
|---|---|---|
| One deep agent session (100K–1M ctx) on a >121 GiB model | single-stream decode bandwidth, huge KV pool, *model does not fit one node* | TP=2: halved weight-read per node, one shared pool → **Lane D** |
| Many agents/users, mixed prompt sizes, on a one-node-fit model | aggregate throughput, tail latency, fault isolation | independent replicas → **Lane F** |
| Cold long-prompt ingestion under live decode load | prefill/decode contention on shared engines | separate prefill and decode silicon → **Lane P** |

The reference recipe is Lane D only, with scheduler hotfixes to survive the
third workload. SparkDuet makes the topology a per-model, per-workload choice.

## 3. The fit rule, and the 79 GiB misreading

The most consequential correction in this repo. An ecosystem claim held that
the official FP8 checkpoint "loads in 79.17 GiB", implying one Spark could
serve it alone and two Sparks could run **two replicas** (DP=2) of the
flagship. The arithmetic says otherwise:

```text
DeepSeek-V4-Flash-0731 FP8 weights on disk:        ~156 GiB      [M-here: du -s]
GPU-usable unified memory per Spark:               ~121 GiB      [M-here]
79.17 GiB (the quoted figure):                     per-RANK load under TP=2,
                                                   i.e. HALF the model
                                                   [M-else: MiaAI issue #119 boot log]
156 GiB > 121 GiB  →  the official checkpoint CANNOT serve on one node.
```

Consequences, enforced by the configs rather than discovered at OOM time:

- **Lane D is the only lane for the official FP8 flagship.** TP=2, both nodes,
  ~78 GiB weights per node + KV pool (18.08 GiB ↔ 2,493,464 tokens at
  `nvfp4_ds_mla`, ≈7.2 KiB/token/node [M-else: boot log]).
- **Lane F and Lane P apply to one-node-fit checkpoints**: Qwen3.8-27B
  (~29 GiB), quantized DeepSeek builds under ~90 GiB (GGUF Q2/Q3 via their own
  servers), merged fine-tunes, anything ≤ ~90 GiB with headroom for KV.
- A *quantized* DeepSeek fleet (two independent GGUF replicas) is a legitimate
  configuration, at quantization-level quality, behind its own server. The
  fit rule is about the official FP8 checkpoint, not the model family.

## 4. Lane D, depth (TP=2 across the pair)

- vLLM `mp` executor, `nnodes 2`, worker rank launched first (headless), head
  rank second; NCCL over the RoCE link with explicit `NCCL_IB_HCA` /
  `NCCL_SOCKET_IFNAME` pins so traffic can never fall back to the management
  LAN silently.
- `nccl-check.sh` is the go/no-go gate: link up, RDMA port ACTIVE, then a real
  2-node all-reduce inside the serving image that must clear
  `MIN_BUSBW_GBS` (default 8 GB/s). A pair that fails the gate will "work" and
  decode at single-digit tok/s; the gate exists so you never debug that as a
  model problem.
- KV dtype `nvfp4_ds_mla`, block 256, DSpark speculation from the checkpoint's
  own draft module, `VLLM_USE_BREAKABLE_CUDAGRAPH=0` (regular graphs measured
  +13–28% [M-else]).
- Failure semantics: TP=2 couples both nodes into one failure domain, any
  rank error kills the server. That is the price of serving a model neither
  node can hold. Lane F exists so that *one-node-fit* traffic never has to pay
  that price.

## 5. Lane F, fleet (DP=2 replicas of one-node-fit models)

- Two independent vLLM engines, one per node, same model, no cross-node
  collective on the serving path. The router load-balances by
  least-outstanding-requests and drains a sick replica; a node failure
  degrades capacity by half instead of taking the service down.
- With Qwen3.8-27B NVFP4 (~29 GiB weights), each replica funds a large KV pool
  in the remaining ~80+ GiB at sane utilization; the boot line prints the
  exact pool, treat any pool number you did not read from your own boot log
  as [P].
- Cost: per-request decode speed of a 27B dense is bandwidth-bound at ~2× the
  bytes/token of the 13B-active MoE. Fleet is a throughput/reliability lane,
  not a latency lane. Measured on our pair, one node, NVFP4, vLLM resident
  serving alongside daily load: 12.8 tok/s c=1 → 46.9 tok/s aggregate c=4
  (256-token prompts), degrading to 8.5 tok/s aggregate at 32K×c=4 [M-here:
  `results/`].

## 6. Lane P, split (prefill/decode disaggregation), experimental

Classic PD-disagg objection: shipping KV to the decode node costs more than it
saves. That objection assumes fat multi-head KV. Sparse-MLA KV is ~7
KiB/token on this stack [M-else, derived from the boot log]. Over the measured
~9–10 GB/s fabric:

```text
128K-token conversation KV ≈ 128,000 × 7 KiB ≈ 0.9 GB  →  ~0.1 s handoff  [P]
1M-token conversation KV   ≈ 7 GB                       →  ~0.8 s handoff  [P]
```

Against a 128K prefill that itself takes 50–80 s, the handoff is noise, and
the decode node never sees prefill, so live streams hold their decode rate
instead of collapsing to the measured ~8 tok/s mixed-prefill floor [M-else:
MiaAI RESULTS, 6× cold 32–128K rows].

Status and honesty:

1. The KV handoff uses vLLM's **upstream** P2P disaggregation connector, no
   custom connector ships in this repo, and configs name the upstream one.
   Validate it on your image line with `bench.py --suite mixed-long` before
   trusting the lane; it is gated behind `LANE_SPLIT_ENABLE=1`.
2. The fit rule applies (each node holds the full model), so Lane P serves
   one-node-fit checkpoints. For the FP8 flagship on two nodes, chunked
   prefill + `long_prefill_token_threshold` inside Lane D is the only
   available mitigation, set honest expectations there.
3. Failure semantics: connector error → the decode node re-runs prefill
   locally. Degraded, never incorrect.
4. Expected effect, when it applies: worst-case per-stream decode under 6×
   cold 32–128K arrivals improves from the ~8 tok/s floor toward the node's
   clean decode rate [P, bounded by decode-node bandwidth; measure on your
   pair].

## 7. The router (scripts/router.py)

One OpenAI-compatible front door; per-request lane selection:

```text
explicit  model@lane suffix or X-SparkDuet-Lane header  → pinned lane
prompt_tokens ≥ P_PROMPT_THRESHOLD and LANE_SPLIT_ENABLE=1 → split
depth inflight > D_HIGH_WATER_SEQS and LANE_FLEET_ENABLE=1 → short requests spill to fleet
otherwise                                                → depth
```

Properties that are policy, not accident:

- Disabled lanes **never** receive implicit traffic, spill and split routing
  activate only behind their enable flags, so a single-lane deployment cannot
  ghost-route to a backend that is not running.
- Overload answers are typed: `413 prompt_too_large_for_lane` with the
  estimated shortfall, `429 lane_saturated` with `retry_after_s`, `503
  draining` during a switch. Never silent queueing.
- Streaming (SSE) passes through chunk-by-chunk; token accounting stays with
  the engine (`usage` frames), the router never re-counts.
- ~300 lines, stdlib only, auditable in one sitting. If you run one lane, you
  do not need it, point clients at the lane port.

## 8. SpecAdvisor, acceptance-adaptive draft depth, honestly scoped

Facts [M-else: entrpi 9-workload suite; classmethod; Keys]:

- DSpark acceptance spans 0.58 (creative prose) → 0.91 (factual QA); accepted
  length at k=5 ranges ~2.2 → ~4.7 tokens/step;
- speculation below break-even acceptance is a net *loss* (0.96× on prose);
- DeepSeek's own datacenter recipe ships k=7;
- cuda-graph capture is seqs×(k+1), so k must come from a captured set.

What the advisor does:

1. Scrapes the engine's real accepted/drafted token counters over
   `SPEC_WINDOW_S` windows (the same counters `bench.py` uses, one source of
   truth).
2. Computes the throughput-optimal k from the acceptance curve:
   maximize `accepted_len(a,k) / (1 + step_cost·k)` over the captured set,
   bounded by `seqs×(k+1) ≤ 48`.
3. Logs every recommendation to `results/specadvisor-log.jsonl` and POSTs it
   to the router's `/admin/spec-k`, where dashboards and operators read it.

What it deliberately does **not** do: hot-swap k on a live engine. Draft depth
is an engine-boot parameter in stock vLLM; *applying* a recommendation means
restarting the lane with the new `D_MTP_NUM_TOKENS` at a quiet moment
(`sparkduetctl.sh restart`). Any tool claiming live retuning on this stack is
overselling, per-class live acceptance metrics do not exist server-side
either, which is why per-class numbers come from controlled runs
(`bench.py --suite spec`, counter deltas per class) and the advisor works on
the live traffic mix instead. Fail-static: no counters → no recommendation.

Expected effect when applied: +8–17% decode on code/math-heavy mixes, ~0% on
prose (it will correctly recommend keeping or shallowing k) [P, from the
acceptance table; k=7 at a=0.85 yields mean accepted 4.85 vs 4.15 at k=5].

## 9. Fine-tuning on the same pair

The serving lanes deliberately leave the second box idle in single-lane
deployments, that box is the training node. Unified memory changes what is
finetunable at all: QLoRA to ~70B on one node, LoRA to ~27B, full fine-tune to
~7B [M-else: Unsloth DGX Spark guides; gate everything through the smoke test
before believing it on your boxes]. Two-node `torchrun` rides the same NCCL
fabric Lane D uses, with the same env pins. The full recipe, scheduling
patterns against serving, and the adapter→Lane F serving path live in
`finetune/README.md`.

## 10. Warm-up, validation, and other ops truths we adopted

From the reference repo's PR history (#118–#124) and ds4's governor:

- Triton/dflash kernels have six live BLOCK keys {8,16,32,64,128,256}; a cold
  cache JIT-compiles mid-request → `warmup.sh` covers the keys at boot.
- All numeric env knobs are validated (`^[0-9]+$`) before any arithmetic
  (their PR #124: `MAX_NUM_SEQS=010` silently parsed as octal and
  crash-looped).
- `doctor` refuses to start on image-digest mismatch between nodes, missing
  weights, dead fabric, or low disk, the failure modes that otherwise cost a
  20-minute model load each to discover.
- `VLLM_USE_BREAKABLE_CUDAGRAPH=0` stays [M-else: +13–28%].
- Management plane ≠ data plane: operator SSH rides wired LAN or a tailnet,
  never WiFi; NCCL rides the QSFP link, pinned by interface name. See
  `docs/FIELD-NOTES.md` for the failure pattern that taught us this.

## 11. Known limits (honest list)

- Lane D couples both nodes; a rank failure kills the lane. That is inherent
  to TP over two boxes.
- Lane F halves per-request speed vs a hypothetical same-model Lane D; it is
  a throughput/reliability lane.
- Lane P is experimental: upstream connector, enable flag, measure first.
- SpecAdvisor recommendations apply via restart, not live; its gains are
  workload-dependent and can be legitimately zero.
- The fabric's ~9–10 GB/s NCCL reality (not the 25 GB/s line rate) bounds
  gradient-sync-heavy distributed training; compute-bound fine-tunes scale
  near-linearly, sync-bound ones do not.
- We cannot upstream the laws of physics: ceilings are ceilings, and every
  table in this repo keeps them separate from speeds.
