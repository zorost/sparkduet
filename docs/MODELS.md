# MODELS.md, choosing the brain for your two Sparks

Ladder labels as in COMPARISON.md. Vendor scores are **[reported]**; community
measurements are **[M-else]**.

---

## Primary: DeepSeek-V4-Flash-0731 (all lanes)

- 284B MoE, **13B active**, MIT weights, 1M-native context, in-checkpoint DSpark
  speculative module, `reasoning_effort` low/high/max.
- Vendor agentic scores [reported]: Terminal Bench 2.1 **82.7**, DeepSWE **54.4**,
  Cybergym 76.7, Toolathlon 70.3, above V4-Pro (Preview) on agentic tasks.
- On 2× Spark: 62–83 tok/s single-stream, 162–191 aggregate at c=6 short
  [M-else: MiaAI RESULTS]; community quality check on the 0731 quant: MMLU 79.5,
  HumanEval 89, needle 70/70 through 130K [M-else: entrpi harness].
- On our pair [M-here, artifacts in `results/`], per workload class at c=1
  (k=5, util 0.72): math 72.2 tok/s (acceptance 0.78), code 68.1 (0.70),
  tool-calling 51.2 (0.50), prose 33.6 (0.23). The 2.1× spread between math
  and prose is the acceptance dependence in the flesh: budget from YOUR
  workload class, not from someone's best-case row.
- **Why it is the bandwidth-optimal choice here:** decode speed ≈ (memory bandwidth /
  active-params-per-token) × speculative accepted length. 13B active is the smallest
  frontier-agentic active footprint available with open weights; every dense
  alternative reads 2–9× more bytes per token.

### KV pool arithmetic from a live boot [M-here]

Third-party 1M-context claims come from KV-lean configs; our first boot shows
what the knobs cost. At `D_GPU_UTIL=0.72`, `max_model_len=131072`, nvfp4 KV,
and DSpark speculation `k=5`, the engine reports **3.96 GiB free for KV =
148,310 tokens** per node after weights, activations, CUDA graphs, and the
draft/indexer state take their cut. That is 1.13 concurrent full-length
requests: fine for one deep agent, thin for a team. The 2.49M-token pools you
see quoted [M-else] assume higher `gpu_memory_utilization` and no competing
draft state. Raise `D_GPU_MEM_UTIL` toward 0.85 if the node runs nothing else;
every 0.01 of utilization is ~1.2 GiB of KV on a 121 GiB node. Verified by
rebooting the same lane at 0.82 [M-here]: 16.39 GiB free for KV = 590,567
tokens = 4.51 concurrent full-context requests, i.e. 12.4 GiB of KV bought
with 0.10 of utilization, matching the arithmetic. Utilization changes
capacity, not speed: single-stream tok/s at a given context length is
unchanged within noise. But 0.82 left the head node 2 GiB of host memory
with its gateway stack running; the pair settled on **0.78** for daily
serving [M-here]: 395,259-token pool, 3.02x full-context concurrency,
15 GiB host headroom. That three-way trade (KV pool, host headroom, what
else the node runs) is the whole tuning story on unified memory.

### Operational gotchas we handle for you

- `max_tokens` counts **think + answer**; with `reasoning_effort=max` the model can
  spend ~12.5K tokens reasoning before answering [M-else]. Size budgets accordingly or
  set per-request effort; our client examples default to sane caps.
- Tool-call truncation semantics and stop-strings-inside-thinking are patched in the
  pinned image line; if you see garbling, validate direct `:8888` before blaming your
  harness [M-else: MiaAI troubleshooting flow].

## Secondary: Qwen/Qwen3.8-27B (Fleet lane profile)

- 27B **dense** (28B BF16), Apache-2.0, native image+video input, 262K native context
  (1M via YaRN), MTP support, thinking on by default, `preserve_thinking`.
- Vendor scores [reported]: DeepSWE 42.2, Terminal Bench 2.1 73.0, beats Claude Opus
  4.6 Max on SWE-bench Pro / LiveCodeBench v6 / OSWorld / AndroidWorld. Early hosted
  traffic is dominated by agentic-coding clients [M-else: OpenRouter app stats].
- **Where it fits SparkDuet:** dense 27B ≈ 27 GB at FP8 (≈14 GB at FP4) per replica -
  trivially fits one node. Its per-token decode is bandwidth-expensive vs 13B-active
  MoE (~2× the bytes/token at FP8), so single-stream speed is lower; but in **Lane F**
  you get two full replicas → 2× concurrency and 2× prefill, plus native vision with no
  sidecar [P]. For mixed teams that want one OpenAI endpoint serving both a
  DeepSeek lane and a Qwen lane, the router exposes them as separate model IDs.
- Predecessor Qwen3.6-27B already proved the 27B-dense category on Spark: 33 tok/s
  NVFP4 @256k on one node [M-else: MiaAI's Aug-2026 list]. Qwen3.8-27B profile ships
  as `configs/` + `prepare-models.sh --model qwen`; treat first-boot numbers as
  [P] until you run `bench.py`.
- Measured on our pair [M-here, artifacts in `results/`]: 12.8 tok/s single-stream,
  46.9 tok/s aggregate at c=4 (256-token prompts, thinking off, vLLM NVFP4,
  **no speculative decoding configured**). That no-spec figure sits exactly on the
  bandwidth ceiling (~273 GB/s ÷ ~20 GB effective weight read/token). The gap to
  the 40–60 tok/s numbers quoted around the ecosystem [reported] is MTP speculation, if your
  deployment does not enable Qwen's MTP draft, you are leaving a 2.5–4× decode
  multiple on the table. The same lesson generalizes: on bandwidth-bound silicon,
  speculation status is the single most important line in any tok/s claim.
- **Variant worth knowing: `orcarouter/Qwen3.8-27B-Uncensored-NVFP4`** (23 GiB,
  abliterated, mixed NVFP4+FP8, FP8 KV, 262K ctx, vision + MTP preserved,
  Apache-2.0). It is a vLLM-only compressed-tensors artifact: llama.cpp cannot
  load it, so it does not belong in a llama-swap on-demand library, and the
  same family's GGUF sibling covers that slot at identical single-stream speed
  (11.5 vLLM NVFP4 vs 11.6 llama.cpp on a GB10 [reported: Kubesimplify
  day-zero grid]). Where it earns a slot is as a **Lane F resident engine**:
  84 tok/s aggregate at c=10, 106 with MTP on, and near-flat decode to 100K
  context [reported: same grid]. One-line swap: point `F_MODEL` at it. Do not
  cold-load it beside a resident TP rank; the fit rule applies to vLLM engines
  exactly as it does to GGUF loads.
- **This fleet runs the FP8 sibling** (`orcarouter/Qwen3.8-27B-Uncensored-FP8`,
  29 GiB, same abliteration, same vision tower and 262K window, plain FP8
  compressed-tensors). Two reasons: the NVFP4 repo is gated on Hugging Face and
  needs a browser-side access grant, and FP8 keeps a little more numerical
  headroom at identical fit. The lane boots it with
  `--tool-call-parser hermes --reasoning-parser qwen3` so tool calls and
  separated thinking work through the gateway the same way they do on Lane D.
  If you later accept the NVFP4 gate, the swap is the same one-line `F_MODEL`
  change; expect a slightly smaller footprint and NVFP4 GEMM throughput in
  exchange for the precision margin.

## When the answer is neither

| Situation | Better answer |
|---|---|
| 3–4 Sparks, want max open intelligence | GLM-5.2 full (753B) via Keys' 4-node recipe: 49.2 tok/s @C12, 100K ctx, HumanEval 96.3 [M-else] |
| 1 Spark, want the best single-box engine | entrpi/ds4-on-spark (Q2 GGUF): ~28 tok/s, 3M tokens resident [M-else] |
| Want a different MoE flavor at 1M on 2× | MiMo-V2.5 NVFP4: ~30 tok/s, 1.97M pool [M-else: tonyd2wild]; Step-3.7-Flash @256k [M-else] |
| Need 2.4T-parameter-class quality | Qwen3.8-Max / Kimi K3 class, off-cluster (API) today; not a 2× Spark model [reported] |

## Decision rule (the short version)

- Bandwidth-bound decode + 1M agentic sessions → **DSV4-Flash-0731, Lane D**
  (the official FP8 checkpoint fits ONLY there; see the fit rule in README).
- Many concurrent agents, mixed prompt sizes → **Qwen3.8-27B or a quantized
  DeepSeek build, Lane F**, one-node-fit models only.
- Vision-native or Apache-2.0-purist fleet, moderate context → **Qwen3.8-27B, Lane F**.
- Your own fine-tune → train it in `finetune/`, merge, serve on **Lane F**.
- Intelligence-per-node above all, ≥3 nodes → **GLM-5.2** (not this repo).
