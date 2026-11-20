# CREDITS

SparkDuet is a Zorost AI Lab project: four resident lanes on two NVIDIA DGX
Sparks, one OpenAI-compatible endpoint, an on-demand library, and a
smoke-gated fine-tune lane. The design, the swap, the measurement protocol,
and the house patches are ours.

## Foundations the stack depends on

<!-- ladder: M-else -->

- **DeepSeek-AI**, DeepSeek-V4-Flash-0731 weights (MIT) and the in-checkpoint
  DSpark speculative module.
- **Z.ai**, GLM-5.3-Flash base weights (MIT).
- **Qwen team (Alibaba)**, Qwen3.8-27B and Qwen3.8-Flash-Next weights
  (Apache-2.0 / Qwen license as published on each card).
- **vLLM** (Apache-2.0), **SGLang**, **FlashInfer**, **NVIDIA** Blackwell /
  CUDA / NCCL tooling, and the **Anemll** `dspark-vllm-gx10` image pin Lane D
  boots (`ghcr.io/anemll/dspark-vllm-gx10:0.1.1`).
- **TensorFold** (Apache-2.0, github.com/ashhart/TensorFold), the native
  server the opt-in `G_ENGINE=tensorfold` Lane G engine builds or unpacks
  unmodified. Its notices travel inside the image.
- **Brandon M. Music (brandonmusic)**, EXL3/TR3 4bpw of GLM-5.3-Flash
  (`brandonmusic/GLM-5.3-Flash-tr3-4bpw`) under the ShapleyMcg License v1.0.
  Attribution is a condition of that grant. The required notice is in
  `patches/glm-exl3-sm121/README.md`.

Third-party images, models, and kernels keep their own licenses. SparkDuet
does not claim those grants, and it does not publish another lab's recipe as
its own.

If a license notice in this tree is wrong, open an issue. We will correct it.
