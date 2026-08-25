# results/

Dated benchmark artifacts land here (`scripts/bench.py --output results/`).
Each artifact is a JSON + markdown pair carrying the workload triple
(prompt tokens, concurrency, thinking) and a claims-ladder label.

Nothing here is hand-edited (one exception: a header note added to an artifact
whose run conditions turned out to be contended; the numbers are untouched).
If a number in README/docs cites `[M-here]`, the artifact proving it must
exist in this directory.

## Index

| Artifact | What it measures |
|---|---|
| `2026-08-25-qwen27b-nvfp4-spark1-resident-*` | House-model baseline: Qwen3.8-27B NVFP4, vLLM, one node, while carrying its normal daily load |
| `2026-08-25-qwen27b-gguf-spark2-ondemand-standard-CLEAN.*` | On-demand GGUF Q5 27B via llama-swap, node otherwise idle |
| `2026-08-25-qwen27b-gguf-spark2-CONTENDED-laneD-boot-*` | Same server while a TP=2 rank loaded on the same node (kept as the contention reference) |
| `2026-08-25-deepseek-v4-flash-tp2-laneD-*` | Lane D flagship: DeepSeek-V4-Flash FP8, TP=2 across both Sparks, DSpark speculation |
