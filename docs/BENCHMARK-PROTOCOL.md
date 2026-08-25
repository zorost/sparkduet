# BENCHMARK-PROTOCOL.md, how to measure without lying

This protocol is binding for every number published from this repo. It exists because
this ecosystem has already been burned twice by measurement errors (SSE chunk counting
undercounted speculative decode ~2.5×; viral tables mixed ceilings with speeds).

## Rules

1. **Token counting.** Throughput = `usage.completion_tokens / wall_time` from the
   non-streaming response (or the final usage frame in SSE). Never count SSE chunks,
   lines, or events, one chunk carries ~2–3 tokens under speculative decoding.
2. **Acceptance capture.** When speculative decoding is enabled, every decode
   number must be accompanied by the draft acceptance rate read from the server's
   SpecDecoding counters over the same window (`bench.py` scrapes the before/after
   delta automatically; `--require-acceptance` makes a missing counter fatal).
   When speculation is off, the artifact must say so, a no-spec number and a
   spec number are different benchmarks.
3. **Workload triple.** Every cell carries `(prompt_tokens, concurrency, thinking)`
  , no bare "tok/s". Prompts use unique cold prefixes unless labeled "warm".
4. **Ladder labels.** `[M-here]` measured on our pair with artifact in `results/`;
   `[M-else]` third-party with citation; `[P]` projection with derivation. A PR that
   changes a number must attach the artifact.
5. **TTFT percentiles**, not just mean: report p50 and p95.
6. **Thinking discipline.** `reasoning_effort` and `max_tokens` are part of the
   workload definition (max-effort thinking can consume ~12.5K tokens before the
   answer; an unbounded thinking run is a different benchmark, not a faster/slower one).
7. **Steady state.** Warm the kernel cache (`scripts/warmup.sh`) and discard the first
   run; report medians of ≥3 runs.
8. **Ceilings vs speeds.** A context ceiling is never printed in the same table cell
   as a speed measured at a smaller context.

## Suites

- `standard`, prompt ∈ {256, 2K, 8K, 32K, 128K} × concurrency ∈ {1,2,4,6},
  thinking off, `max_tokens=min_tokens=128`, `ignore_eos`. Mirrors the reference
  recipe's 2026-08-14 matrix so results are directly comparable.
- `fleet`, Lane F: c ∈ {2,4,8,12,16} staggered arrivals, mixed prompt sizes,
  router logging on.
- `mixed-long`, Lane P: 6× cold {32K…128K} arrivals while a steady 256-token chat
  stream holds on the decode node. The headline worst-case suite.
- `spec`, per-class acceptance measurement: code/math/prose/tool prompts run
  serially, acceptance captured as a counter delta per class. This is where
  per-class numbers come from (the live engine exposes only totals). Compare k
  values by rerunning the suite after a restart with a different
  `D_MTP_NUM_TOKENS`.

## Output contract

`bench.py` writes `results/<UTC-date>-<lane>-<suite>.json` (raw) plus a markdown table
with ladder labels pre-filled as `[M-here]`. Paste the markdown directly into
`results/README.md` or a PR description.

## Sanity checks built into the harness

- Refuses to publish a run shorter than 30 s (no steady state).
- Refuses any request whose final stream frame lacks `usage.completion_tokens`.
- Records `tokens_per_sse_chunk` in every cell: a ratio ≈ 1 with speculation
  active means somebody counted chunks, the exact mistake that produced the
  ecosystem's viral 2.5× undercount.
- `--require-acceptance` makes missing speculative counters fatal for engines
  that are supposed to be speculating.
- Captures GPU name and SM clocks (`nvidia-smi`) into the artifact when run on
  the node, so a clock-capped node can never masquerade as a regression.
- TTFT is measured from the first content chunk of a real SSE stream, per
  request, and reported as p50/p95.
