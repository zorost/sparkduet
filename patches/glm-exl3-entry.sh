#!/usr/bin/env bash
# Lane G EXL3 entry: apply the MiaAI-Lab runtime patches, then vllm serve.
#
# The overlay image bakes and applies the EXL3 quantization method, the
# FLASHINFER_MLA_SPARSE_SM120 NoPE padding, patch_model_overrides and the
# aarch64 exllamav3 stubs at build time. Later patches are not in the
# published tag and are mounted from patches/glm-exl3-sm121 instead; see the
# README there. They are required, not optional: patch_hybrid_prefix_hit
# repairs the prefix-cache path this lane runs with --enable-prefix-caching,
# so a missing file is a hard failure rather than a silent unpatched boot.
set -u
MOUNT=/sparkduet-patches/glm-exl3-sm121

mkdir -p /root/.cache/vllm/jit/triton /root/.cache/vllm/jit/tilelang \
         /root/.cache/vllm/nccl-flight

# Baked in the image and already applied at build, except video placeholders,
# which MiaAI-Lab's Dockerfile copies but does not run. Re-running an applied
# patch is a no-op; each checks its own marker.
for p in patch_glm_video_placeholders patch_glm5_drafter_group; do
  if [ -f "/opt/glm53/$p.py" ]; then
    python3 "/opt/glm53/$p.py" \
      || echo "[glm-exl3-entry] WARN $p did not apply" >&2
  fi
done

for p in patch_suppress_stops_in_reasoning patch_scheduler_decode_floor \
         patch_hybrid_prefix_hit patch_clamp_max_tokens; do
  if [ ! -f "$MOUNT/$p.py" ]; then
    echo "[glm-exl3-entry] FATAL $MOUNT/$p.py missing; refusing to serve" >&2
    exit 1
  fi
  python3 "$MOUNT/$p.py" || {
    echo "[glm-exl3-entry] FATAL $p failed to apply" >&2
    exit 1
  }
done

echo "[glm-exl3-entry] patches applied; fused-moe=${EXL3_FUSED_MOE:-1} mixed-prefill=${GLM53_MIXED_PREFILL_CHUNK:-skip}"
exec vllm serve "$@"
