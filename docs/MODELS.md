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

## Swap lane: Qwen3.8-Flash-Next NVFP4 (Lane N)

Recipe on [`lane-n-flash-next`](https://github.com/zorost/sparkduet/tree/lane-n-flash-next).
Not the default flagship until a measured A/B on this pair says otherwise.

- 125B MoE, **6B active**, plus a 51B n-gram table. RadixArk NVFP4 on disk is
  ~135 GiB. Official FP8 is ~173 GiB and is not this recipe.
- Does **not** fit one 121 GiB node. Lane N is TP=2 only, same fabric as
  Lane D. Never concurrent with DeepSeek: `switch next` stops depth first.
- Vendor card [reported]: ahead of DeepSeek-V4-Flash-0731 on SWE-bench Pro
  (62.5 vs 56.0) and CoWorkBench (73.9 vs 45.1); behind on NL2Repo (48.1 vs
  54.2). No [M-here] artifacts yet. Independent benches have not landed.
- Engines, picked by `N_ENGINE`. `vllm` is
  `vllm/vllm-openai:qwen38-flash-next` plus `patches/next-ple-fp8.py` so the
  ModelOpt hybrid PLE loads. No speculation on this checkpoint. `sglang` is
  the SM121-patched image from `patches/next-sglang-sm121/` and runs NEXTN
  from the in-checkpoint MTP layer (steps 3, topk 1, draft 4). Stock
  `lmsysorg/sglang:qwen38flashnext` either fails to compile FA4 CuTe or
  silently decodes token id 0. Same `:30000`, same served id. The Lab pair
  runs `N_ENGINE=sglang`. First patched vLLM boot on this pair: 12 min to
  tokens. NEXTN measured [M-here]: math 49.2, code 38.4, tool 42.1, prose
  34.3 tok/s at c=1, thinking off
  (`results/2026-08-28-flash-next-laneN-sglang-nextn-spec.*`).
- Safety: no `--load-format dummy`, chunked prefill ≤1024, memory fraction
  ≤0.82, n-gram table left on auto offload. Same gateway.
- Stage: `./scripts/prepare-models.sh --model flash-next`
- Serve: `./scripts/sparkduetctl.sh switch next`
- Back: `./scripts/sparkduetctl.sh switch depth`

## Swap lane: GLM-5.3-Flash NVFP4 (Lane G)

Recipe on [`lane-g-glm-flash`](https://github.com/zorost/sparkduet/tree/lane-g-glm-flash).
Not the default flagship until a measured A/B on this pair says otherwise.

- Z.ai GLM-5.3-Flash: 320B MoE, **18B active**, hybrid sparse + linear
  attention, vision tower, MIT. Native context 1M (`max_position_embeddings`
  1048576). Released 26 Aug 2026.
- Checkpoint: `LibertAIDAI/GLM-5.3-Flash-NVFP4`, rev `11d73216`, ~181 GiB
  (`usedStorage` 194,692,661,135 bytes). Weight-only NVFP4 on routed experts
  (311.65B params); attention, vision, routers, shared experts, MTP, embeddings
  stay BF16. Cosine vs BF16 source ≈ 0.9967 [reported].
- Does **not** fit one 121 GiB node. Lane G is TP=2 only. Never concurrent
  with DeepSeek or Flash-Next: `switch glm` stops the incumbent first.
- Unsloth `GLM-5.3-Flash-FP8` exists but is larger than this NVFP4 and is not
  this recipe. `unsloth/GLM-5.3-Flash-GGUF` is a WIP stub: llama.cpp has no
  `glm5_next` yet, so there is no house sleeper until that lands.
- Engine: dedicated image `vllm/vllm-openai:glm53-flash-arm64-cu130`.
  `glm5_next` is not in vLLM main (PR #53906). This checkpoint is NoPE
  (`qk_rope_head_dim=0`); stock SM120 packed MLA dies with pe_dim must be 64.
  `glm-entry.sh` installs FlashInfer 0.6.18 (`ckv_scale_arr`) and
  `glm53-sm90.py` selects SM90 sparse-MLA. Stock 0.6.17 has no prefill
  backend and completions collapse to token 1023. Working house flags:
  `G_KV_DTYPE=fp8_e4m3`, `G_MAX_NUM_SEQS=8` (aggregate, not one stream).
  `start_glm` drops page cache and runs `cache_flusher.sh` through load.
  First honest boot is 20–60 min; later 12–20. TP4 1M-pool recipes need
  four boxes; this pair stays TP=2.
- Parsers: `--tool-call-parser glm47`, `--reasoning-parser glm45`.
  `VLLM_ENGINE_READY_TIMEOUT_S=3600`.
- Stage: `./scripts/prepare-models.sh --model glm-flash`
- Serve: `./scripts/sparkduetctl.sh switch glm` with `G_ENGINE=vllm`
- Back: `./scripts/sparkduetctl.sh switch depth`

## Swap lane: GLM-5.3-Flash EXL3/TR3 4bpw (Lane G, `G_ENGINE=exl3`)

Same hop and container names as the NVFP4 lane. Never beside it: one
`G_ENGINE` at a time.

- Checkpoint: `brandonmusic/GLM-5.3-Flash-tr3-4bpw`, pinned at
  `5ab363a8dcf6405955fd5f99671e01a1c9fb124b`, ~164 GiB, 120 shards.
  ShapleyMcg License v1.0. Attribution is a condition of that grant; the
  required notice lives in `patches/glm-exl3-sm121/README.md`. Base model
  `zai-org/GLM-5.3-Flash` is MIT.
- Why this engine exists: quality per byte. An independent teacher-logit
  KLD panel puts EXL3/TR3 4bpw at 0.024555 nats against NVFP4's 0.060535
  at the same footprint, level with official FP8 at 54% of the bytes
  [M-else]. Speed is not the claim. MiaAI-Lab's 62.9 tok/s headline
  [M-else] needs DFlash2 (`incoai/GLM-5.3-Flash-DFlash2`, CC BY-NC-ND
  4.0), which this repo will not ship. The lane runs the license-clean
  MTP rollback (`MTP_TOKENS=2`).
- Engine: `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3`. The
  published tag predates four patches; `glm-exl3-entry.sh` bind-mounts
  `patches/glm-exl3-sm121/` and refuses to serve if any file is missing.
  `G_EXL3_GPU_MEM_UTIL=0.80` is the house ceiling: 0.82 misses by ~0.7 GiB
  on Spark 2 after reboot with Comfy and the desktop session.
- Parsers: `--tool-call-parser glm47`, `--reasoning-parser glm45`.
- Stage: `./scripts/prepare-models.sh --model glm-exl3`
- Serve: set `G_ENGINE=exl3`, then `./scripts/sparkduetctl.sh switch glm`
- Back: `./scripts/sparkduetctl.sh switch depth`

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
- Want the Flash-Next preview on the same pair → **Lane N**,
  `switch next`, then A/B against depth before changing the default.
- Want GLM-5.3-Flash (vision, MIT, 18B active) on the same pair → **Lane G**,
  `switch glm`. `G_ENGINE=vllm` is NVFP4; `G_ENGINE=exl3` is the quality-per-byte
  checkpoint the Lab pair runs. Then A/B against depth.
- Many concurrent agents, mixed prompt sizes → **Qwen3.8-27B or a quantized
  DeepSeek build, Lane F**, one-node-fit models only.
- Vision-native or Apache-2.0-purist fleet, moderate context → **Qwen3.8-27B, Lane F**.
- Your own fine-tune → train it in `finetune/`, merge, serve on **Lane F**.
- Intelligence-per-node above all, ≥3 nodes → **GLM-5.2** (not this repo).
