# Lane G engine A/B, 2026-11-20-1258 UTC

<!-- ladder: M-here -->
| Measure | Before: exl3 | After: tensorfold | Change |
|---|---:|---:|---:|
| Served window (max_model_len) | 1,000,000 | 262,144 | -73.8% |
| Switch to ready, incl. warmup (s) | 943 | 389 | -58.7% |
| Decode, code, 1 stream (tok/s) | 30.28 | 35.31 | +16.6% |
| Decode, prose, 1 stream (tok/s) | 19.04 | 27.05 | +42.1% |
| Decode, prose, 4 streams total (tok/s) | 29.74 | 46.47 | +56.3% |
| Cold prompt size (tokens) | 28,450 | 28,452 | +0.0% |
| Cold prompt, first token (s) | 19.99 | 21.19 | +6.0% |
| Cold prompt read (tok/s) | 1,423.2 | 1,342.8 | -5.6% |
| Next turn, first token (s) | 2.58 | 1.49 | -42.2% |
| Next turn, cached tokens | n/a | 26,624 |  |
| Tool call parsed | yes | yes |  |
| Reply stops on its own | yes | yes |  |
<!-- ladder: end -->

Model `brandonmusic/GLM-5.3-Flash-EXL3-TR3-4bpw`, greedy, thinking off, fresh salted prompts. Before 2026-11-20T13:21:30Z, after 2026-11-20T13:31:47Z.

```
GATE FAIL: served window shrank (1000000 -> 262144)
verdict: keep before
```
