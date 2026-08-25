# RESEARCH.md, How the DeepSeek-V4-Flash-on-DGX-Spark ecosystem actually works

> Date of research: 2026-08-23. Every claim below is tagged **[measured]** (a number
> someone published with a method), **[reported]** (a vendor/author claim without a
> reproducible method attached), or **[inference]** (our derivation from cited numbers).
> Sources are linked inline. This document is the evidence base for the SparkDuet design;
> the short version lives in `README.md`, the head-to-head in `docs/COMPARISON.md`.

<!-- ladder: M-else -->

---

## 1. The repo under study: `MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark`

**What it is.** A two-node serving *recipe* (not an engine) for
`deepseek-ai/DeepSeek-V4-Flash-0731` on 2× NVIDIA DGX Spark (GB10). It packages a
prebuilt vLLM image (`ghcr.io/anemll/dspark-vllm-gx10:0.1.1`), a docker-compose stack,
worker-first launch/stop/status scripts, a model-cache preparer, a benchmark script, a
Responses-API verifier, and a set of Python hotfixes applied at container start.
[Source: repo README, fetched 2026-08-23]

**Repo vitals** [measured, GitHub API 2026-08-23]:

| Field | Value |
|---|---|
| Created | 2026-06-28 |
| Stars / forks | 917 / 128 |
| Open issues | 13 |
| License | MIT |
| Homepage | https://x.com/MiaAI_lab |
| Default branch activity | Commits and merged PRs as recent as 2026-08-23 (PR #123) |

**Technical core of the recipe:**

- vLLM, TP=2 across the two Sparks (`mp` executor, `nnodes 2`), RoCE/NCCL over the
  200G QSFP link.
- Checkpoint pinned: official `deepseek-ai/DeepSeek-V4-Flash-0731` @ `9e165c30…`.
- KV cache dtype `nvfp4_ds_mla` (4-bit NVFP4 sparse-MLA envelope), block size 256.
- DSpark speculative decoding, `num_speculative_tokens=5` (MTP-5), probabilistic draft
  sampling.
- `max_model_len=1,048,576` (1M ceiling), `max_num_seqs=6`,
  `max_num_batched_tokens=8192`, `long_prefill_token_threshold=1024`,
  `gpu_memory_utilization=0.835` → measured KV pool **2,493,464 tokens** (18.08 GiB).
- MoE backend `flashinfer_b12x`; `VLLM_USE_BREAKABLE_CUDAGRAPH=0` (they measured
  regular CUDA graphs +13–28% faster than Anemll's breakable graphs).
- Six-plus issue hotfixes (#21/#22/#26/#27/#43/#52/#55-class) shipped as `patches/`
  applied at container start: long-context NVFP4 decode fix, prefill chunk cap,
  suppress-stop-in-reasoning, tool-call truncation semantics, encoder continuation
  fixes, spin-wait fix, etc.
- Optional Stage-C overlay image for the 200K/16-slot high-aggregate profile;
  optional Qwen3-VL-4B vision sidecar on `:8889`.

**Their measured performance** (`results/RESULTS-2026-08-14.md`, dated, methodical,
and unusually honest about what each number is):

| Workload | Result [measured] |
|---|---|
| Single chat decode, 256→128K prompt | 62–83 tok/s |
| 6 short chats (256 tok) | 162–191 tok/s aggregate (~30–37/stream) |
| 6× cold 32K–128K prompts at once | prefill **serializes** (issue #27); ~8 tok/s decode floor; 128K×6 median TTFT 282 s |
| 128K single prompt | ~75 tok/s decode, ~80 s TTFT |
| 900K single prompt | ~874.8 prefill tok/s, TTFT 1,028.85 s |
| Stage-C + Keys mask, 200K ctx, 16 slots | 315 tok/s static batch / 205 staggered |
| Regular vs breakable CUDA graphs | +28.6% c1 decode, +13.1% c2 aggregate |

**Commit/issue culture.** Reading the live commit log and PR list (Aug 22–23, 2026)
shows a maintenance-heavy repo: PR #121 removes an obsolete adaptive-top-k backport
after upstream vLLM traced silent output corruption to it (#51318 revert, #52492
capture guard); PR #122 fixes an encoder edge case where a trailing `latest_reminder`
after an assistant turn leaves the prompt without a generation header; PR #123 persists
the Triton JIT cache and warms all six live `_prepare_dflash_inputs_kernel` BLOCK keys
so cold boots don't JIT mid-request; PR #124 adds integer validation because
`MAX_NUM_SEQS=010` was silently read as **octal** and crash-looped the cluster. Open
issue #119 shows a user hitting a DeepGEMM `runtime != nullptr` assertion on FP8 init.
Takeaway: the recipe works, but a large share of its surface area is *compensating for
a moving vLLM/DSpark stack on an off-label GPU arch (sm_121a)*. That maintenance burden
is itself a design input for us (see §7).

**Why it went viral.** Mia (@MiaAI_lab on X) posted "What are the best models you can
run on your NVIDIA DGX Spark? Aug 2026 Edition" on 2026-08-05, ~90,000 views in a day
[reported by noze.it, 2026-08-06]. The 2× Spark "sweet spot" framing, a pinned
one-command recipe, and dated benchmark tables made it the reference deployment.
The noze.it analysis also documented the weak point of the viral table: it mixes
*maximum configurable context* with *short-prompt decode speed* from different
experiments, e.g. "1M context @ 82 tok/s" combines the c=1 short-prompt decode row
with a context ceiling that no published run exercised end-to-end. We adopt the
opposite rule (§7, claims ladder).

---

## 2. The lineage: who actually built what

The MiaAI repo is a *packaging* achievement on top of a deep public stack. The real
intellectual lineage (from the repos' own CREDITS/ATTRIBUTION files and forum threads):

| Who | Contribution |
|---|---|
| **DeepSeek-AI** | The model: DeepSeek-V4-Flash-0731, 284B MoE / 13B active, MIT weights, DSpark speculative module in-checkpoint, `reasoning_effort` low/high/max, official vLLM/SGLang recipes (vLLM: `--speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}'`). |
| **antirez** | `ds4` (DwarfStar), MIT-licensed C/CUDA inference engine for DSV4-Flash; the ~81 GiB IQ2_XXS/Q2_K/Q8 asymmetric GGUF quant recipe. |
| **entrpi** | `entrpi/ds4-on-spark` fork (v0.6.3): single-Spark CUDA serving engine, continuous batching, prefix caching with disk-persisted KV banks, DSpark lossless speculation with yield-quench, OpenAI+Anthropic+Responses APIs, and a memory governor that refuses cleanly instead of crashing. |
| **Rafael Caricio** | First DSpark↔vLLM integration (`rafaelcaricio/vllm#1`) and the DSpark docker runbook. |
| **Fraser Price** | DSV4-Flash-DSpark model/runtime work (`fraserprice/dspark-vllm`, HF checkpoint). |
| **drowzeys ("Keys")** | Three foundational pieces: (1) the **concurrency patch** fixing DSpark's request-stable main-KV slot mapping + ragged `query_start_loc` so continuous batching works under independent arrivals; (2) wiring `nvfp4_ds_mla` into DGX Spark launch recipes; (3) the full-GLM-5.2-on-4×Spark line (rebuilt sm_121a vLLM, NVFP4 KV, Marlin MoE). |
| **tonyd2wild** | The 1M NVFP4-KV recipe lineage and Stage A/B/C runtime packaging; the earlier 60 tok/s / 900K fp8 recipe; MiMo-V2.5 and GLM-5.2 NVFP4 fleet work. |
| **Anemll** | The prebuilt `dspark-vllm-gx10` images that the current recipe defaults to. |
| **aidendle94 / Aiden Le** | `sparkrun-vllm-ds4-gb10` b12x-optimized images; the fp8 fallback/diagnostic build. |
| **botAGI** | Independent reproduction + the sharpest measurement insight in the ecosystem (§3). |
| **Wpnx330, 0rand, paulbrav, Roady001, Fable, DaveCharland, CosmicRaisins, eugr, jasl, lukealonso** | CUDA-graph capture-size fix (capture size must be `seqs×(k+1)`), early MTP=3 call, long-context crash fix, cold-start garble root-cause, portable Triton sparse-MLA kernels, base image builders, b12x CuTe kernels. |
| **MiaAI-Lab** | Two-node packaging, worker-first launch runbook, env validation, CI gates, hotfix integration, docs. |

Also adjacent: `imanmostafavi` (Qwen-Vision sidecar fork), `HeNryous/mimo-spark-optimized`,
`0xdfi` (first GLM-5.2+NVFP4 deployment), `danielwoz/vllm-dspark-nvfp4` (the public
nvfp4_ds_mla patch series), `local-inference-lab/b12x` (current sparkinfer/b12x kernel
maintenance), and NVIDIA-forums regulars (0rand, CosmicRaisins, tonyd615, eb.spark,
11_p) who produced the 4×Spark TP=4 numbers (~70 tok/s single stream, 4,000 tok/s
prefill at 500K) [measured, forums.developer.nvidia.com 370309].

---

## 3. The single most important measurement in the ecosystem

botAGI's reproduction of tonyd2wild's NVFP4 build ran the decisive control experiment
[measured, 2026-06-30]:

- NVFP4-KV, no speculation: **26.6 tok/s**
- fp8-KV, no speculation: **26.7 tok/s**, *identical*.
- With DSpark: 58–63 tok/s single-stream.

**Conclusion: the 4-bit KV cache buys nothing on the raw forward pass.** Decode speed
on GB10 is set by (a) weight-read bandwidth and (b) speculative acceptance length -
not by KV read bandwidth, because MLA's compressed latent KV is already tiny
(~584-byte sparse envelope per token per the Stage-C layout; the MiaAI boot log implies
≈7.2 KiB/token/node at their pool size). NVFP4-KV's real value is **capacity**
(2.49M-token pool → the 1M ceiling + concurrency), not speed.

Second decisive measurement, also botAGI/Keys: **SSE chunk counting undercounts
spec-decode throughput by ~2.5×** because one streamed chunk carries ~2–3 accepted
tokens. Any benchmark that counts SSE events instead of `usage.completion_tokens` is
garbage. Several viral numbers in this ecosystem were produced exactly that way and
later corrected (Keys' GLM-5.2 "13 tok/s" → 15.3 tok/s true-token).

Third: **acceptance is workload-bound.** entrpi's 9-workload DSpark suite on one Spark
[measured, v0.1.1 stamp]: stepwise math 89% accept / 1.71× speedup; qa_factual 91% /
1.63×; code_python 72% / 1.29×; creative prose 58% / **0.96×** (slower than plain!).
classmethod's dual-Spark run independently found thinking-OFF code generation 32%
faster than thinking-ON, tied to MTP acceptance. Speculative depth that is optimal for
code is wasteful for prose. *A static `num_speculative_tokens=5` for all traffic is
provably suboptimal.* This is the opening for our adaptive-draft-depth sidecar (§7).

---

## 4. The alternative school: entrpi's ds4-on-spark (single node, C engine)

A genuinely different architecture, measured on one GB10 (0731 Q2 GGUF weights) [measured,
v0.5.0–v0.6.3 stamps]:

| Metric | ds4-on-spark (1 node) |
|---|---|
| Prefill | ~960 tok/s @2k, ~1,010 @12k, 933 @64k; 776 tok/s sustained over 518K-token ingestion |
| Chat decode (DSpark) | ~28 tok/s @12k, ~22 @240k |
| Aggregate serving | 59 tok/s @ 12 concurrent |
| Resident context | 2.26M tokens warm by default; **3,019,176 tokens** measured with the floor lowered |
| Deepest single conversation | 1,029,340-token prompt, needle at 99.9% depth retrieved exactly |
| vs upstream antirez engine | 2.43× prefill @2k, 3.30× @64k |

Design lessons we adopt:

1. **Measure admissions against live free memory; refuse with a typed error, never
   crash.** (vLLM's model is the opposite: pre-reserve a fraction, preempt/recompute
   under pressure. MiaAI operators hit this as earlyoom kills and engine deaths.)
2. **Demand-mapped context:** a deep `-c` ceiling costs nothing until used.
3. **Disk-persisted KV banks** survive restarts; warm-start shared prefixes (~7× TTFT).
4. **Per-layer CUDA-graph capture at every context depth** removed the deep-context
   decode cliff.
5. DSpark is *the only* speculation for 0731 (the checkpoint has no MTP head; the
   legacy MTP pairing measures ~52% accept ≈ break-even).

And its limits, for honesty: Q2-weight quality is not the official checkpoint (GaelicThndr:
"no perplexity, no eval suite" on the 2-bit build); users report memory-exhaustion
slowdowns on long sessions (emX0r, jbourny on the forum thread); decode tops out around
28 tok/s because one node reads all 13B active params per step.

---

## 5. The wider model field (what else people run on Spark fleets)

From MiaAI's X list (90K views), the NVIDIA forums, and vendor cards [mix of measured
and reported, tagged in docs/MODELS.md]:

| Fleet | Model | Headline |
|---|---|---|
| 1× Spark | DSV4-Flash (Q2 GGUF, ds4) | 1M ceiling, ~26–28 tok/s |
| 1× Spark | Qwen3.6-27B / 35B NVFP4 | 33 / 81 tok/s @256k |
| 2× Spark | DSV4-Flash-0731 (this lineage) | 62–83 tok/s c1, 1M ceiling |
| 2× Spark | MiMo-V2.5 NVFP4 (tonyd2wild) | ~30 tok/s, 1M, 1.97M-token pool |
| 2× Spark | Step-3.7-Flash | ~30 tok/s @256k |
| 3× Spark | GLM-5.2 NVFP4+AQLM (MiaAI) | ~21–26 tok/s, 348–380k |
| 4× Spark | GLM-5.2 QuantTrio full (Keys) | 49.2 tok/s @C12, 100K, first full non-pruned |
| 4× Spark | DSV4-Flash TP=4 (11_p) | ~70 tok/s c1, 4,000 tok/s prefill @500K |

**Qwen3.8-27B** (released open-weights 2026-08-14, Apache-2.0) is the most important
new entrant and the one the user asked about: 27B **dense** (28B BF16 weights), native
vision+video, 262K native context (1M via YaRN), MTP support, thinking on by default
with `reasoning_effort` xhigh/medium/low and `preserve_thinking`. Vendor-reported
scores: DeepSWE 42.2 (vs 13.3 for Qwen3.6-27B), Terminal Bench 2.1 73.0, beats Claude
Opus 4.6 Max on SWE-bench Pro / QwenSWEBench / LiveCodeBench v6 / OSWorld /
AndroidWorld [reported]. OpenRouter lists $0.45/$3.20 per M tokens; early production
traffic is dominated by agentic coding tools (Kilo Code, Zed, pi, Hermes Agent)
[measured, OpenRouter stats via a2aprotocol/cnblogs]. Mia herself noted "Qwen3.8-27B
should be released soon and this could change my recommendation" days before it
dropped.

**DSV4-Flash-0731 quality anchors** [reported by DeepSeek]: Terminal Bench 2.1 82.7,
DeepSWE 54.4, Cybergym 76.7, Toolathlon-Verified 70.3, above V4-Pro (Preview) on
agentic tasks at 13B active params. entrpi's frozen harness on the 0731 Q2 quant:
MMLU 63.5→79.5 vs preview, HumanEval 89, needle 70/70 through 130K [measured].

---

## 6. What the physics says (GB10 roofline, verified against measurements)

Per node: 128 GB LPDDR5X unified memory, **~273 GB/s** [measured, Keys], sm_121a,
"1 PFLOP FP4" class compute, aarch64 Grace host. Link: 200G QSFP RoCEv2 (~20–22 GB/s
effective).

**Decode** is weight-bandwidth bound: 13B active params ≈ 13 GB read per token-step at
FP8 (dense side; MoE experts are MXFP4 so the true number is lower, ~8–10 GB, the
measured no-spec 26.6 tok/s on TP=2 implies ≈10.3 GB/step/node-pair at 273 GB/s and
~65% bandwidth efficiency [inference]). DSpark multiplies by mean accepted length:
a=0.6, k=5 → (1−0.6⁶)/(1−0.6) ≈ 2.38 → ~63 tok/s, matching the measured 62–83 band.
**Therefore the only honest decode levers are: kernel bandwidth efficiency and
speculative acceptance.** KV dtype is not a speed lever (§3). This is why our design
spends its complexity budget on acceptance (adaptive depth) and topology (keeping
prefill off the decode path), not on KV exotica.

**Prefill** is compute bound: measured 875–2,563 tok/s on the 2-node vLLM stack,
~960–1,010 tok/s on one node with ds4. Two nodes as *independent replicas* therefore
roughly double fleet prefill, while TP=2 concentrates it on one request at a time.

**KV transfer is cheap.** MLA's latent KV (~7–14 KiB/token/node-pair [inference from
the 2.49M-token/18.08 GiB boot log]) over ~20 GB/s fabric moves a 128K-token
conversation in well under a second. *That is what makes prefill/decode disaggregation
essentially free on this hardware*, the classic objection to PD-disagg (KV transport
cost) does not apply to sparse-MLA models on 200G RoCE.

**TP=2's hidden tax:** every layer's forward does a cross-node collective; at 200G
this is workable (they measure it daily) but it couples both nodes' failure domains,
halves per-node kernel autonomy, and serializes big cold prefills against live decode
(the issue-#27 collapse: 6× cold 32K–128K → 8 tok/s decode floor, 282 s median TTFT
[measured]).

---

## 7. Design openings this research exposes (→ our architecture)

1. **The #27 collapse is architectural, not a scheduler bug.** Monolithic TP=2 mixes
   prefill and decode on the same engines; the hotfix serializes big prefills to
   protect decode, trading collapse for queueing. Disaggregating prefill and decode
   onto different nodes removes the contention entirely, and §6 shows KV handoff is
   ~free. Nobody in this lineage ships PD-disagg on 2× Spark. → **Lane P.**
2. **Concurrency scaling past 6 slots is proven but not productized.** Keys: c=16
   static 290 / staggered 191 tok/s (fp8, 200K) [measured]; 2 stacks scale 1.96×.
   MiaAI keeps Stage-C as a scary "experiment". → we make the fleet lane (DP=2
   replicas behind a router) a first-class, default-on configuration, and unlike
   Stage-C it needs no forked image, because full FP8 weights fit on one node
   (79.17 GiB model load measured in MiaAI issue #119's log; ~119 GiB usable per box).
3. **Static MTP-5 wastes the acceptance signal.** Workload acceptance ranges 58–91%
   [measured, entrpi suite]. Official DeepSeek recipe itself uses k=7 greedy on GB300.
   → **acceptance-adaptive draft depth** (specadvisor sidecar), bounded and
   fail-static.
4. **Viral tables mix incompatible measurements** (noze.it critique). → our
   **claims ladder**: every number in this repo is labeled measured-here /
   measured-elsewhere / projected-with-derivation, and `bench.py` counts
   `usage.completion_tokens`, never SSE chunks.
5. **Maintenance burden is the hidden cost** (§1: hotfixes for encoder edges, Triton
   warm keys, octal parsing, graph capture). → we ship warm-up + validation + watchdog
   as first-class ops scripts, and we pin images by digest with a tested rollback.
6. **Model choice is workload choice.** DSV4-Flash-0731 (MoE, 13B active) is the
   bandwidth-optimal decode model; Qwen3.8-27B (dense, Apache-2.0, native vision, MTP)
   is the per-replica fleet alternative where 2× DP gives it effective concurrency a
   single 27B could never host. We ship profiles for both and the reasoning to choose
   (docs/MODELS.md), including when *not* to use either (GLM-5.2 on 3–4 nodes,
   Qwen3.8-Max-class 2.4T off-cluster).

---

## 8. What "viral and praised" actually consisted of (for the record)

- X: @MiaAI_lab's Aug-2026 "best models on DGX Spark" list (~90K views/day) drove the
  star curve; the repo homepage is the X account. Qwen's own Qwen3.8 announcement
  tweet: 4.96M views / 20.1K likes (per latent.space AINews), the ecosystem is
  hungry for exactly this class of content.
- GitHub: 917★ / 128 forks in ~8 weeks; sister repos (1-Spark EXL3, GLM-5.2 3×Spark)
  show the same pattern.
- NVIDIA forums: 100+ post engineering threads (entrpi's ds4 thread; the FP8 2×Spark
  thread) functioned as the peer-review layer.
- Independent press/analysis: flowtivity.ai (dual-Spark + Hermes deployment,
  41 tok/s, cost payback math: ~$9.4K hardware vs API pricing), classmethod.jp
  (MTP acceptance analysis), kingy.ai (1-vs-2 Spark guide), noze.it (measurement
  critique).

The pattern to copy is not the numbers, it is **dated, methodical, honest tables plus
a one-command install**. The pattern to avoid is **mixing ceilings with speeds and
shipping hotfix-on-hotfix without a stability architecture**.

---

## 9. Source list (primary)

1. MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark, README, results/RESULTS-2026-08-14.md, GitHub API (repo vitals, commits, PRs #116–#124, issue #119).
2. tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark (+ CREDITS.md) and tonyd2wild/DeepSeek-v4-Flash-DSpark-60-tok-s-900K-ctx-2x-DGX-Spark.
3. drowzeys/Keys-Concurrency-Patch-for-DSpark-DeepSeek-V4-Flash (RESULTS: c16 290/191 tok/s, 2-stack 1.96×).
4. drowzeys/Keys---Full-GLM-5.2-Quantrio…4-x-DGX-Spark (+ ATTRIBUTION.md, TUNING-2026-07).
5. entrpi/ds4-on-spark + entrpi/ds4 fork README (v0.6.3) and NVIDIA forums thread 378855.
6. botAGI/DeepSeek-V4-Flash-DSpark-GB10-2x-DGX-Spark-1m-fp4-fp8 (benchmarks/20260630 checkpoint: the 26.6 vs 26.7 control).
7. huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731 (model card, benchmarks, official vLLM/SGLang recipes, MIT license).
8. MiaAI-Lab sister repos: DeepSeek-V4-Flash-Dual-DGX-Spark-1M-Context (old recipe), DeepSeek-v4-Flash-One-DGX-Spark (EXL3), GLM-5.2-NVFP4-AQLM-Triple-DGX-Sparks.
9. tonyd2wild/MiMo-V2.5-TP2-1M-NVFP4-KV-2xDGX-Spark; tonyd2wild/GLM-5.2-NVFP4-KV-4x-DGX-Spark-300kctx-42tok-s.
10. NVIDIA forums thread 370309 (FP8 2×/4× Spark; 11_p TP=4 numbers; 0rand/CosmicRaisins/eb.spark debate).
11. flowtivity.ai blog (2026-08-02), dev.classmethod.jp (2026-06-30), kingy.ai (2026-08-01), noze.it (2026-08-06).
12. latent.space AINews (Qwen3.8-Max + 27B launch, tweet metrics), a2aprotocol.ai & cnblogs Qwen3.8-27B guides (benchmarks, OpenRouter stats), buildfastwithai Qwen3.6-27B review.

---

## 10. Corrections from first-party deployment (2026-08-24, [M-here])

Bringing SparkDuet up on a live pair falsified three beliefs this document's
earlier sections inherited from the ecosystem. Recorded here because the
corrections are more valuable than the original claims.

**10.1 The 79.17 GiB figure is a per-rank number.** MiaAI issue #119's boot log
line ("model loads in 79.17 GiB") circulated as proof the FP8 flagship fits one
Spark, enabling single-node replicas (an earlier draft of our own Lane F was
designed on it). The log line comes from a TP=2 boot: each rank loads *half*
the 156 GiB checkpoint. 156 GiB > ~121 GiB usable per node; the official FP8
checkpoint cannot serve on one Spark, and any DP=2 fleet of it on two Sparks is
arithmetic fiction. Consequence: Lane F/P are scoped to one-node-fit
checkpoints; docs/ARCHITECTURE.md §3 carries the full derivation.

**10.2 Speculation status explains most of the ecosystem's tok/s spread.** A
dense 27B NVFP4 measured 12.8 tok/s single-stream on one Spark [M-here] -
exactly the ~273 GB/s ÷ ~20 GB/token bandwidth ceiling, and 3–4× below the
"40–60 tok/s" figures quoted for the same model class. The quoted figures
assume MTP/DSpark speculation; the measured deployment ran without it. Neither
number is wrong, they are different benchmarks, which is why this repo's
protocol makes speculation status a mandatory field (BENCHMARK-PROTOCOL rule 2).

**10.3 A GGUF library can silently fall back to CPU and keep serving.** A
llama.cpp on-demand server measured 1.4–2.2 tok/s (vs ~10 tok/s healthy) with
GPU utilization at 0% during generation; its log carried
`ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected`
while the host GPU was fine, a stale container GPU binding after a
driver/runtime update. llama.cpp treats CUDA-init failure as a warning and
serves from CPU. Detection is cheap (probe `utilization.gpu` during a
generation; grep the load log for `ggml_cuda_init`); the fix is a container
restart. `sparkduetctl.sh doctor` now checks for this class of silent
degradation on Lane F/GGUF backends.
