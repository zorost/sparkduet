# Container-start hotfixes

Lane D patches run from `entry.sh`. Lane N and Lane G have their own
entrypoints. Every patch is idempotent and no-ops if the file or symbol it
targets is gone.

## Lane D (DeepSeek)

Applied by `entry.sh` before `vllm serve` starts, on both ranks. Both are
idempotent and gated to the pinned engine image's exact source; on a new
image they no-op with a log line instead of breaking the boot.

| File | What it fixes | Origin |
|---|---|---|
| `hotfix-dsv4-issue55-tool-truncation.py` | A tool call truncated by `max_tokens` used to report `finish_reason: "tool_calls"` with unparseable JSON `arguments`, poisoning agent transcripts (HTTP 400 on replay). Truncated calls now report `"length"` and invalid arguments are dropped. | [MiaAI-Lab's recipe](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark), issue #55, MIT. Carried verbatim with the license notice. |
| `hotfix-dsv4-suppress-stops-in-reasoning.py` | Client `stop` strings could fire inside the reasoning block, ending a request mid-think with `content: null`. Stops now stay dormant until reasoning closes. | Same recipe, porting tonyd2wild's Stage-C patch 5 to the Anemll image path. MIT, carried verbatim. |

Status check inside a running head container:

```bash
docker exec sparkduet-depth-head \
  python3 /sparkduet-patches/hotfix-dsv4-issue55-tool-truncation.py --status
```

Opt-outs are the originals' env switches (`DSPARK_SUPPRESS_STOPS_IN_REASONING=0`
to let stops fire inside reasoning). Full credit in `../CREDITS.md`.

## Lane N (Flash-Next)

Applied by `next-entry.sh` before `vllm serve`.

| File | What it fixes | Origin |
|---|---|---|
| `next-ple-fp8.py` | RadixArk ModelOpt NVFP4 + FP8 PLE. Stock loader only accepts `Fp8Config` for the n-gram table, then dies on missing `ngram_embedding.weight_scale`. The patch detects `ple_embedding_dtype=float8_e4m3fn` and keeps the scale as a buffer. | House, 27 Aug 2026. First honest boot on this pair. |

## Lane G (GLM-5.3-Flash)

Applied by `glm-entry.sh` before `vllm serve`.

| File | What it fixes | Origin |
|---|---|---|
| `glm-entry.sh` | Official `glm53-flash-arm64-cu130` ships FlashInfer 0.6.17. Without 0.6.18 (`ckv_scale_arr`), sparse MLA has no prefill backend and completions collapse to token 1023 (`lock`). Installs 0.6.18, drops `flashinfer-jit-cache` (SM120 cubins fight the SM90 path). | House, after the 0.6.17 lock. |
| `glm53-sm90.py` | Checkpoint is NoPE (`qk_rope_head_dim=0`). Stock SM120 packed MLA dies with `pe_dim` must be 64. Forces the SM90 sparse-MLA path. | House, same boot. |

`G_KV_DTYPE=fp8_e4m3` and `G_MAX_NUM_SEQS=8` live in `sparkduet.env`. Eight-way
aggregate is not one-stream speed. First honest GLM boot is the long one
(20–60 min); later boots 12–20 min.

## Lane N on SGLang (N_ENGINE=sglang)

Build context in `next-sglang-sm121/`. Stock `lmsysorg/sglang:qwen38flashnext`
either fails to compile FA4 CuTe on GB10 or silently decodes token id 0.
The derivative ports sglang#36845 and #36806, adds NVFP4 KV for the QSA
pools, and installs the leftover token-id-0 abort from MiaAI-Lab `0f95001`.
See that directory's README.

## Lane G on EXL3 (G_ENGINE=exl3)

Applied by `glm-exl3-entry.sh` before `vllm serve`. The published overlay
predates these files, so they are mounted over `/opt/glm53/`. A missing
file is a hard failure.

| File | What it fixes | Origin |
|---|---|---|
| `glm-exl3-sm121/patch_hybrid_prefix_hit.py` | Prefix-cache block accounting on the hybrid mamba path | MiaAI-Lab, MIT |
| `glm-exl3-sm121/patch_scheduler_decode_floor.py` | Mixed prefill stealing a running decode step | MiaAI-Lab, MIT |
| `glm-exl3-sm121/patch_suppress_stops_in_reasoning.py` | Client stop strings firing inside `<think>` | MiaAI-Lab, MIT |
| `glm-exl3-sm121/patch_clamp_max_tokens.py` | OpenCode `max_tokens=9999999` 400s against `max_model_len` | House |

Weights are ShapleyMcg License v1.0. Attribution is required; the notice
is in `glm-exl3-sm121/README.md`.
