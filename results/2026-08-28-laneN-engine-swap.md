# Lane N engine swap: vLLM to SGLang NEXTN, 2026-08-28

Same weights (`RadixArk/Qwen3.8-Flash-Next-NVFP4`, ~135 GiB NVFP4), same TP=2
topology, same `:30000` hop, same served id. Only the engine changed, so no
picker, no catalog row, and no 9Router entry moved.

Recipe and the SM121 kernel patches come from MiaAI-Lab's
Qwen3.8-Flash-Next-Dual-DGX-Sparks (MIT, commit `344f9d0d`). The launch contract
is expressed as `configs/lane-next-sglang.compose.yml` and selected by
`N_ENGINE=sglang`; the upstream `start.sh` is deliberately not run, because a
second orchestrator competing for :30000 is what took this lane down earlier the
same day.

## Decode, same harness, same protocol

`scripts/bench.py --suite spec --trials 2`, concurrency 1, thinking off. Rule 1
throughput: `usage.completion_tokens / wall_time`, so prefill and TTFT are inside
the number.

| Workload | vLLM (no speculation) | SGLang NEXTN 3/1/4 | Change |
|---|---:|---:|---:|
| code | 20.9 tok/s | 38.4 tok/s | 1.84x |
| math | 21.8 tok/s | 49.2 tok/s | 2.26x |
| prose | 22.5 tok/s | 34.3 tok/s | 1.52x |
| tool | 22.4 tok/s | 42.1 tok/s | 1.88x |

TTFT p50 fell to 0.20 to 0.27 s. The engine's own decode counter reports 55 to
74 tok/s, which is the number comparable to the published 64.4: the gap to the
table above is prefill and scheduling inside a 384-token generation, not a
disagreement about speed.

Acceptance at the end of the run: `spec_accept_length` 3.325 of a 4-token draft,
`spec_accept_rate` 0.775. Prose gains least and math most, which is the expected
shape for draft acceptance and a sign the speculation is real rather than a
mismeasurement.

## What the boot actually allocated

Nine minutes to ready (383 s of weight load), then:

- KV cache NVFP4 (`float4_e2m1fn_x2`), **1,305,152 tokens** across the QSA and
  indexer pools, 5.13 GB. That is 5x the 262144 context, so KV is not the
  binding constraint.
- Mamba cache 30 slots. At 5 state slots per request the engine **capped
  `max_running_requests` to 6**, below the 16 asked for. Still above the 4 the
  vLLM lane served. Raising `N_SGLANG_MAMBA_RATIO` above 0.3 is the lever if
  more concurrency is wanted; it costs KV and free headroom.
- Decode CUDA graphs captured for verify, draft-decode, and draft-extend at
  batch 1 to 6. Prefill graphs stay off on GB10.
- `available_gpu_mem` 20.96 GB free per node.

## Correctness, because this engine can fail silently

On SM121 the stock FlashInfer TRT-LLM sparse-decode path returns token id 0 for
every token while still answering HTTP 200. The patched image keeps that path off
and substitutes a Triton packed-varlen kernel, so every claim below is a check
that the substitution held.

| Check | Result |
|---|---|
| Thinking off, temp 0 | correct, no reasoning block |
| Thinking on | reasoning separated, 155 chars, answer correct |
| Tool call | `get_weather({"city": "Paris"})` in `message.tool_calls`, no XML leak |
| Exact copy of a 40-line block | byte-exact, no token-0 run |
| Image input | 128x128 split image read correctly (`top=red bottom=blue`) |
| Video input | 2-shot clip, correct order (red then blue) |
| NIAH 1k to 16k, 3 positions | 11/11, verdict RELIABLE |
| NIAH 1k to 64k, 3 positions | 17/17, verdict RELIABLE |
| 32k with a concurrent partner request, 6 trials | 6/6 clean |

One caveat, recorded rather than buried. The **first** 64k sweep returned 64
tokens of id 0 for `niah/32768@0%` (16/17, verdict DEGRADED). That run was
contended: a separate gateway request was in flight and five of the eval's
`/flush_cache` calls were failing with HTTP 400 as a result. Re-running the same
sweep with nothing else touching the engine gave 17/17, a 32k-only repeat gave
3/3, and six deliberately concurrent 32k trials gave 6/6. So it did not
reproduce in 12 subsequent attempts, but it happened once, and the failure mode
is invisible to a health check. `scripts/test_lane_next_corruption.py` exists to
catch it; run it after any image or flag change on this lane.

## Rollback

One line and one swap, about ten minutes:

```bash
sed -i 's/^N_ENGINE=sglang/N_ENGINE=vllm/' sparkduet.env   # on both nodes
scripts/sparkduetctl.sh switch next
```

`zorost-lane-guard` restarts whatever `N_ENGINE` names, as a pair, so the
rollback survives a reboot without further action.
