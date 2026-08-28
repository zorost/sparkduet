# CREDITS

SparkDuet is an original architecture (three routed serving lanes, PD-disaggregation
for sparse-MLA on GB10, acceptance-adaptive draft depth, claims-ladder measurement),
built on a large body of public work. Full source list with links: `docs/RESEARCH.md`.

## Foundations we depend on

<!-- ladder: M-else -->

- **DeepSeek-AI**, DeepSeek-V4-Flash-0731 weights (MIT) and the in-checkpoint DSpark
  speculative module; official vLLM/SGLang recipes.
- **vLLM** (Apache-2.0), **FlashInfer**, **NVIDIA** Blackwell/CUDA/NCCL tooling,
  **b12x** CuTe kernels (lukealonso / local-inference-lab).
- **Rafael Caricio**, first DSpark↔vLLM integration and deployment runbook.
- **Fraser Price**, DeepSeek V4 Flash DSpark model/runtime work.
- **drowzeys ("Keys")**, DSpark in-server concurrency patch (request-stable main-KV
  slot mapping, ragged `query_start_loc`); the `nvfp4_ds_mla` KV wiring on DGX Spark;
  the GLM-5.2 multi-node line and its tuning ledgers (including the true-token
  benchmark correction and the negative-blocks override).
- **tonyd2wild**, the 1M NVFP4-KV recipe lineage and Stage A/B/C packaging;
  MiMo-V2.5 and GLM-5.2 NVFP4 fleet references.
- **Anemll**, the prebuilt `dspark-vllm-gx10` image line our Lane D pins.
- **MiaAI-Lab**, two-node packaging lineage, worker-first launch ordering, hotfix
  integration practice, and a results-file culture worth emulating. Their issue/PR
  history (numeric env validation, Triton warm-key coverage, graph-capture guards)
  directly shaped our `doctor`, `warmup.sh`, and env validator. The two
  container-start hotfixes in `patches/` (issue #55 truncated tool calls, stops
  dormant inside reasoning, the latter porting tonyd2wild's Stage-C patch 5) are
  carried verbatim from their MIT-licensed recipe with notices preserved. Lane N
  on SGLang is theirs outright: the SM121 QSA kernel work in
  `patches/next-sglang-sm121/` (sglang#36845's Triton packed-varlen fallback,
  sglang#36806 keeping TRT-LLM sparse decode off SM121, and NVFP4 KV for the QSA
  pools) is what makes NEXTN speculative decoding serve this checkpoint on GB10
  at all, together with the measured recipe our lane file mirrors.
- **entrpi**, ds4-on-spark: the memory-governor philosophy (typed refusals over
  crashes), demand-mapped context, disk-persisted KV banks, and the 9-workload
  DSpark acceptance suite that proves workload-bound acceptance.
- **antirez**, the ds4 engine and the 0731 GGUF quant recipe.
- **botAGI**, the decisive NVFP4-vs-fp8 no-spec control (26.6 vs 26.7 tok/s) that
  anchors our "KV dtype is capacity, not speed" design rule.
- **Aiden Le (aidendle94)**, **Wpnx330**, **0rand**, **paulbrav**, **Roady001**,
  **Fable**, **DaveCharland**, **CosmicRaisins**, **eugr**, **jasl**, **0xdfi**,
  **danielwoz**, **11_p**, and the NVIDIA DGX Spark forum community, images, fixes,
  kernel overlays, measurements, and multi-node results we cite.
- **noze.it**, **flowtivity.ai**, **classmethod.jp**, **kingy.ai**, **latent.space** -
  independent analysis and field reports that kept the numbers honest.
- **Qwen team (Alibaba)**, Qwen3.8-27B weights (Apache-2.0) for the Fleet profile.

If you believe your work is used here without correct attribution, open an issue -
we will fix it promptly.
