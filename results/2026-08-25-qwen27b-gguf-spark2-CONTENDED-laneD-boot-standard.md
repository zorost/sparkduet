### [M-here] qwen27b-gguf-spark2, CONTENDED / standard, 2026-08-25

Conditions: this run executed while the Lane D DeepSeek TP=2 worker rank was
loading weights and compiling on the same Spark 2 GPU. It is kept as the honest
"worst case under contention" reference. The clean isolated run is
`2026-08-25-qwen27b-gguf-spark2-ondemand-standard-CLEAN.md` (about 3x faster).

| Prompt tok | Concurrency | Thinking | Per-stream tok/s | Aggregate tok/s |
|---:|---:|:---:|---:|---:|
| 256 | 1 | off | 3.3 | 3.3 |
| 256 | 2 | off | 2.1 | 4.3 |
| 2048 | 1 | off | 3.5 | 3.5 |
| 2048 | 2 | off | 2.0 | 4.0 |
