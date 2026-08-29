# Flash-Next vs MiaAI-Lab Dual-DGX-Sparks, 2026-08-29

Upstream: [MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks)
commit `344f9d0`. Same commit the house vendored on 2026-08-28.

## Verdict

The live lane is already the Mia recipe: SGLang NEXTN 3/1/4, NVFP4 KV, SM121
Triton fallback, chunk 1024, mem-fraction 0.80, native 262k. Kernel patches
match the upstream `.patch/` files byte-for-byte. Do not run their `start.sh`
next to SparkDuet.

The 64.4 tok/s headline is `sglang bench_serving` decode. House Rule 1 is
`usage.completion_tokens / wall_time` and includes prefill. Those are different
numbers. Engine decode during warmup reached 91 to 105 tok/s at four streams.

## What was wrong in the house, and what changed

`scripts/bench.py` sent `chat_template_kwargs.thinking`. Flash-Next reads
`enable_thinking` and thinks on by default. The 28 Aug "thinking off" suite
was thinking on. Confirmed live: `thinking: false` still produced 88 chars of
reasoning; `enable_thinking: false` produced `2` in 0.20 s with zero reasoning
tokens.

Telegram `/think` had the same DeepSeek-only key. It now sends both switches
every time, so a Flash-Next chat can actually turn thinking off.

Host NCCL 2.30.7 is still not staged. The image uses NCCL 2.29.7. Compose and
`sparkduetctl.sh` now pin `libnccl.so.2.30.7` when the file exists on both
nodes. That is the remaining optional pin, not a missing kernel.

YaRN 1M is Mia's script default. House stays at native 262144 so a swap does
not change the picker contract. YaRN is not a decode-speed win.

## Live boot (this afternoon)

- Served: `RadixArk/Qwen3.8-Flash-Next-NVFP4` on SGLang, context 262144
- KV: nvfp4, pool 1,131,328 tokens, 19.53 GB free
- Engine capped `max_running_requests` to 5 (mamba ratio 0.3)
- Weight load 378 s, NCCL 2.29.7+cuda13.2
- 9Router lists `spark1/RadixArk/Qwen3.8-Flash-Next-NVFP4`

## Correctness

| Check | Result |
|---|---|
| Thinking off, 1+1 | `2`, 2 completion tokens, 0.20 s |
| Thinking on | reasoning 171 chars, answer `2` |
| Tool `get_weather(Paris)` | `message.tool_calls`, no XML leak |
| Old `thinking: false` key | still thinks (88 chars). The bug. |
| Short hello | `Hello`, no token-0 run |

## Decode, same harness as 28 Aug, thinking actually off

`scripts/bench.py --suite spec --trials 2`. Rule 1 wall-clock.

| Workload | 28 Aug (thinking still on) | 29 Aug (enable_thinking false) |
|---|---:|---:|
| code | 38.4 | 47.8 |
| math | 49.2 | 50.5 |
| prose | 34.3 | 36.3 |
| tool | 42.1 | 49.0 |

Acceptance this run: code 0.767, math 0.742, prose 0.225, tool 0.750.
Tokens per SSE chunk 2.3 to 3.5, so speculation is real.

Short-prompt C1/C2 (256 gen tokens, thinking off): 49.8 tok/s single stream,
70.7 aggregate at two streams. After the list prompt, engine
`spec_accept_length` was 3.825 of 4 (rate 0.94).

## Direct stream table vs Mia (same afternoon)

Mia published (`sglang bench_serving`, short structural decode):

| Streams | TTFT | Aggregate | Per stream |
|---|---:|---:|---:|
| ×1 | 117 ms | 64.4 | 64.4 |
| ×2 | 169 ms | 116.8 | 60.3 |
| ×4 | 517 ms | 114.1 | 33.2 |

House native `/generate`, repetitive English, 556 prompt tokens, 256 new,
ignore_eos:

| Streams | TTFT | Aggregate | Per stream |
|---|---:|---:|---:|
| ×1 | 349 ms | 44.0 | 44.0 |
| ×2 | 602 ms | 88.0 | 44.0 |
| ×4 | 1639 ms | 97.7 | 27.4 |

×2 is the honest gap: 88 vs 116.8, about 25 percent, on a longer prefill.
Engine `last_gen_throughput` after that run was 139.9. Random-ids and
generated-shared-prefix through their tool both fell to accept length 1.3
and 11 to 17 tok/s. That is the no-speculation floor, not their table.

## Do not change

- Checkpoint: keep `RadixArk/Qwen3.8-Flash-Next-NVFP4` rev `7b71922`
- Context: keep 262144 unless a real 500k job shows up
- Chunk: stay at 1024
- Mem-fraction: stay at 0.80
- Do not enable YaRN for speed
