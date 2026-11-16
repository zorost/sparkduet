#!/usr/bin/env bash
# Lane D entrypoint: apply the container-start hotfix chain, then exec vllm serve.
#
# House Lane D chain on the pinned Anemll 0.1.1 image. Encoder install and
# issue21 run first, then issue55, the nvfp4 MLA and GB10 spin-wait scripts,
# issue117 SHM ring (after spin-wait so it sees the issue79 overlay), the
# six numbered shell backports, empty-encoder, issue27 r3, issue43, issue26,
# issue133, and suppress-stops. Extra opt-in scripts are not in this chain.
#
# Each step is fail-open: a missing anchor or a failed script logs
# "[patches] <name> did not apply; engine boots without it" and the rest of
# the chain continues. Fail-open is kept so a later engine image bump cannot
# strand the lane; check the container log for "[patches]" lines after an
# image change. SPARKDUET_SKIP_HOTFIX_CHAIN=1 rolls back to the two original
# house hotfixes (issue55 and suppress-stops). SPARKDUET_PATCH_ONLY=1 applies
# the chain and exits without starting vllm serve.
set -u
# JIT-cache and flight-recorder paths live on the persisted cache volume;
# create them up front (a missing dump dir turns the on-timeout dump into a
# silent no-op, which defeats the point of having it).
mkdir -p /root/.cache/vllm/jit/triton /root/.cache/vllm/jit/tilelang \
         /root/.cache/vllm/jit/b12x-cute /root/.cache/vllm/nccl-flight

PATCHES=/sparkduet-patches
VLLM_ROOT=/usr/local/lib/python3.12/dist-packages/vllm
APPLIED=0
SKIPPED=0
FAILED=0

note_applied() {
  APPLIED=$((APPLIED + 1))
  echo "[patches] $1 applied"
}

note_skipped() {
  SKIPPED=$((SKIPPED + 1))
  echo "[patches] $1 skipped"
}

note_failed() {
  FAILED=$((FAILED + 1))
  echo "[patches] $1 did not apply; engine boots without it" >&2
}

run_py() {
  local f="$1"
  if python3 "$PATCHES/$f"; then
    note_applied "$f"
  else
    note_failed "$f"
  fi
}

run_sh() {
  local f="$1"
  if bash "$PATCHES/$f"; then
    note_applied "$f"
  else
    note_failed "$f"
  fi
}

apply_encoder_steps() {
  local enc dest
  enc="${SPARKDUET_ENCODING_FILE:-/srv/ai/models/llm/DeepSeek-V4-Flash-0731/encoding/encoding_dsv4.py}"
  dest="$VLLM_ROOT/tokenizers/deepseek_v4_encoding.py"
  if [ ! -f "$enc" ]; then
    echo "[patches] encoder file not found at $enc; encoder step skipped"
    note_skipped "encoder-copy"
    note_skipped "reasoning-effort-map"
    note_skipped "hotfix-encoding-dsv4-issue21.py"
    return
  fi
  if cp "$enc" "$dest"; then
    note_applied "encoder-copy"
  else
    echo "[patches] ENCODER COPY FAILED" >&2
    note_failed "encoder-copy"
  fi
  # Reasoning-effort map: 12-space elif, 16-space assignment.
  if python3 - "$VLLM_ROOT/tokenizers/deepseek_v4.py" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text()
old = (
    "elif reasoning_effort in (\"max\", \"xhigh\"):\n"
    "                reasoning_effort = \"max\"\n"
    "            else:\n"
    "                reasoning_effort = \"high\""
)
new = (
    "elif reasoning_effort in (\"max\", \"xhigh\"):\n"
    "                reasoning_effort = \"max\"\n"
    "            elif reasoning_effort == \"high\":\n"
    "                reasoning_effort = \"high\"\n"
    "            else:\n"
    "                reasoning_effort = \"low\""
)
if new in s:
    raise SystemExit(0)
updated = s.replace(old, new)
if new not in updated:
    raise SystemExit(1)
p.write_text(updated)
raise SystemExit(0)
PY
  then
    note_applied "reasoning-effort-map"
  else
    note_failed "reasoning-effort-map"
  fi
  run_py hotfix-encoding-dsv4-issue21.py
}

apply_hotfix_chain() {
  apply_encoder_steps
  run_py hotfix-dsv4-issue55-tool-truncation.py
  run_sh hotfix-nvfp4-ds-mla-issue22.sh
  run_sh hotfix-gb10-spin-wait.sh
  run_py hotfix-vllm-issue117-shm-ring-buffer.py
  run_sh hotfix-dsv4-mtp-buffer-50312.sh
  run_sh hotfix-dsv4-skip-topk-49486.sh
  run_sh hotfix-dsv4-dense-prefill-indexer-48407.sh
  run_sh hotfix-dsv4-skip-empty-c128-48957.sh
  run_sh hotfix-dsv4-flashmla-workspace-50298.sh
  run_sh hotfix-dsv4-grammar-advance.sh
  run_py hotfix-vllm-empty-encoder-output.py
  run_py hotfix-dsv4-issue27-partial-prefill-concurrency.py
  run_py hotfix-dsv4-issue43-decode-fairness-and-diag.py
  run_py hotfix-dsv4-issue26-hybrid-swa-min.py
  run_py hotfix-dsv4-issue133-triton-specialization.py
  run_py hotfix-dsv4-suppress-stops-in-reasoning.py
  if [ "${DSPARK_ENABLE_C128A_PREFILL_CACHE:-0}" = "1" ]; then
    run_py hotfix-vllm-c128a-prefill-cache.py
  fi
}

apply_original_two() {
  run_py hotfix-dsv4-issue55-tool-truncation.py
  run_py hotfix-dsv4-suppress-stops-in-reasoning.py
}

compile_patched() {
  local rel path
  for rel in \
    tokenizers/deepseek_v4.py \
    tokenizers/deepseek_v4_encoding.py \
    v1/attention/backends/mla/flashmla_sparse.py \
    distributed/device_communicators/shm_broadcast.py \
    models/deepseek_v4/nvidia/model.py \
    v1/worker/gpu/model_runner.py \
    models/deepseek_v4/attention.py \
    model_executor/layers/sparse_attn_indexer.py \
    models/deepseek_v4/sparse_mla.py \
    models/deepseek_v32/nvidia/attention.py \
    models/deepseek_v4/compressor.py \
    models/deepseek_v4/nvidia/flashmla.py \
    models/deepseek_v4/common/ops/cache_utils.py \
    v1/structured_output/__init__.py \
    v1/core/sched/scheduler.py \
    v1/outputs.py \
    v1/core/kv_cache_coordinator.py \
    entrypoints/openai/chat_completion/serving.py \
    v1/engine/detokenizer.py
  do
    path="$VLLM_ROOT/$rel"
    if [ ! -f "$path" ]; then
      echo "[patches] py_compile $rel: fail (missing)"
      continue
    fi
    if python3 -m py_compile "$path"; then
      echo "[patches] py_compile $rel: pass"
    else
      echo "[patches] py_compile $rel: fail"
    fi
  done
}

if [ "${SPARKDUET_SKIP_HOTFIX_CHAIN:-0}" = "1" ]; then
  apply_original_two
else
  apply_hotfix_chain
fi

if [ "${SPARKDUET_PATCH_ONLY:-0}" = "1" ]; then
  echo "[patches] summary: applied=$APPLIED skipped=$SKIPPED failed=$FAILED"
  compile_patched
  exit 0
fi

exec vllm serve "$@"
