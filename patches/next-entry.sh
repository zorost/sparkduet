#!/usr/bin/env bash
# Lane N entry: FP8 PLE on ModelOpt NVFP4, then vllm serve.
# Stock qwen38-flash-next builds an unquantized ngram table for hybrid
# checkpoints and dies on ngram_embedding.weight_scale.
set -u
mkdir -p /root/.cache/vllm/jit/triton /root/.cache/vllm/jit/tilelang \
         /root/.cache/vllm/nccl-flight
python3 /sparkduet-patches/next-ple-fp8.py || echo "[next-entry] PLE patch skipped" >&2
exec vllm serve "$@"
