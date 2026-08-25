# RESEARCH.md, evidence base for SparkDuet

> Date of research: 2026-08-23, corrected on first-party boots through
> 2026-08-29. Every claim is tagged **[measured]** (a number published with a
> method), **[reported]** (a vendor claim without a reproducible method), or
> **[inference]** (our derivation from cited numbers). The short version lives
> in `README.md`. The operating-model comparison is `docs/COMPARISON.md`.

<!-- ladder: M-else -->

---

## 1. What this document is for

SparkDuet is a two-node operating layer, not a study of another lab's
repository. This file records the hardware arithmetic, the measurement rules,
and the first-party corrections that the lane files enforce. Sources at the
end are vendor cards, NVIDIA documentation, and house artifacts.

## 2. The pair

Two NVIDIA DGX Spark (GB10) nodes, 128 GB unified memory each, about 121 GiB
GPU-usable per node after the OS and display stack, joined by a 200 Gb/s QSFP
RoCE link. One OpenAI-compatible hop. One resident in RAM.

Technical core of the house flagship lane (Lane D):

- vLLM, TP=2 (`mp` executor, `nnodes 2`), NCCL over the fabric.
- Checkpoint: official `deepseek-ai/DeepSeek-V4-Flash-0731`.
- KV cache dtype `nvfp4_ds_mla`, block size 256.
- DSpark speculative decoding. Live k is measured, not copied from a headline.
- House default: `max_model_len=1,048,576`, `gpu_memory_utilization=0.78` on
  a head that also runs a gateway stack. Pickers advertise 85% of the live
  pool. A quoted 2.49M-token pool [M-else] is a higher utilization on a pair
  that runs nothing else.

## 3. Decode is bandwidth-bound; prefill is compute-bound

DSV4-Flash-0731 activates 13B params/token. On the 2-node TP=2 stack the
no-spec decode is 26.6–26.7 tok/s whether KV is fp8 or NVFP4 [M-else:
independent NVFP4-vs-fp8 control]. With DSpark k=5 at acceptance a≈0.6, mean
accepted length ≈ (1−a⁶)/(1−a) ≈ 2.38 → ~63 tok/s [P, matches the measured
62–83 band]. **Levers: kernel bandwidth × speculative acceptance. KV dtype is
a capacity lever, never a speed lever.**

Prefill parallelizes across independent engines: hundreds to low thousands of
tok/s on the TP=2 pair depending on prompt size [M-else]. Mixed long prefills
on a monolithic TP=2 pair serialize and can collapse decode to a single-digit
floor [M-else]. That is why Lane P exists as an experimental split, and why
Lane D ships a one-in-flight long-prefill cap.

## 4. The 79 GiB misreading

An ecosystem claim held that the official FP8 checkpoint "loads in 79.17 GiB",
implying one Spark could serve it alone and two Sparks could run two replicas.
The arithmetic says otherwise:

```text
DeepSeek-V4-Flash-0731 FP8 weights on disk:        ~156 GiB      [M-here: du -s]
GPU-usable unified memory per Spark:               ~121 GiB      [M-here]
79.17 GiB (the quoted figure):                     per-RANK load under TP=2,
                                                   i.e. HALF the model
156 GiB > 121 GiB  →  the official checkpoint CANNOT serve on one node.
```

Consequences, enforced by the configs:

- **Lane D is the only lane for the official FP8 flagship.**
- **Lane F and Lane P apply to one-node-fit checkpoints** (Qwen 27B, GGUF,
  merged fine-tunes).

## 5. Speculation status explains most tok/s spreads

A dense 27B NVFP4 measured 12.8 tok/s single-stream on one Spark [M-here],
exactly the ~273 GB/s ÷ ~20 GB/token bandwidth ceiling, and 3–4× below the
"40–60 tok/s" figures quoted for the same model class. The quoted figures
assume MTP speculation; the measured deployment ran without it. Neither
number is wrong. They are different benchmarks. This repo's protocol makes
speculation status a mandatory field (`docs/BENCHMARK-PROTOCOL.md` rule 2).

House Lane D at c=1 [M-here]: math 72.2 (acceptance 0.78), code 68.1 (0.70),
tool-calling 51.2 (0.50), prose 33.6 (0.23). Budget from your workload class.

## 6. Design openings that became lanes

1. **Prefill/decode contention is architectural.** A hotfix can serialize big
   prefills. Disaggregating them onto different nodes removes the contention.
   KV handoff over 200G is noise against a 50–80 s 128K prefill [P]. → Lane P.
2. **Concurrency past a handful of slots wants replicas, not a forked image.**
   One-node-fit models run two independent engines. → Lane F.
3. **Static draft depth wastes the acceptance signal.** Workload acceptance
   ranges widely [M-else]. → specadvisor, bounded and fail-static.
4. **Viral tables mix ceilings with speeds.** → the claims ladder and
   `bench.py` counting `usage.completion_tokens`.
5. **A moving engine on sm_121a needs ops, not a screenshot.** → doctor,
   warmup, digest pins, revert.
6. **Model choice is workload choice.** 13B-active MoE for bandwidth-optimal
   decode; 27B dense for fleet + vision. GLM-5.3-Flash when you want that
   quality-per-byte / vision resident on the same hop.

## 7. Silent CPU fallback

A llama.cpp on-demand server measured 1.4–2.2 tok/s (vs ~10 tok/s healthy)
with GPU utilization at 0% during generation. The log carried
`ggml_cuda_init: failed to initialize CUDA` after a stale container GPU
binding. llama.cpp treats that as a warning and serves from CPU.
`sparkduetctl.sh doctor` probes this class of degradation.

## 8. Source list (primary)

1. huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731 (model card, official
   vLLM/SGLang recipes, MIT).
2. zai-org/GLM-5.3-Flash and brandonmusic EXL3/TR3 4bpw (ShapleyMcg v1.0).
3. Qwen3.8-27B and Qwen3.8-Flash-Next model cards.
4. vLLM, SGLang, FlashInfer, NVIDIA DGX Spark / NCCL documentation.
5. Anemll `dspark-vllm-gx10` image pin used by Lane D.
6. House artifacts in `results/` (every [M-here] number).
7. Independent NVFP4-vs-fp8 no-spec control (26.6 vs 26.7 tok/s): KV dtype is
   capacity, not speed.
8. NVIDIA forums field reports on 2× and 4× Spark TP (cited as [M-else] where
   a number is used).

This file does not study another lab's GitHub, X account, or start.sh.
SparkDuet updates from lessons learned on this pair.
