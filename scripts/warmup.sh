#!/usr/bin/env bash
# warmup.sh, post-ready warm-up so the first real request never pays JIT.
# Covers the live dflash/_prepare_dflash_inputs kernel BLOCK keys {8,16,32,64,128,256}
# with scheduled-token shapes {1,6,20,45,100,200}, at C=1..MAX_NUM_SEQS, using both
# bounded and ordinary client-default request shapes (a lesson from MiaAI PR #123:
# pinned-max_tokens warmups missed a kernel that only ordinary short chats compile).
set -euo pipefail
PORT="${D_PORT:-${ROUTER_PORT:-30000}}"
MODEL="${DS_SERVED_NAME:-deepseek-ai/DeepSeek-V4-Flash-0731}"
MAXC="${D_MAX_NUM_SEQS:-6}"

fire() { # max_tokens concurrency label
  local mt="$1" c="$2" label="$3" i
  for i in $(seq 1 "$c"); do
    curl -fsS "http://127.0.0.1:${PORT}/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"warmup $label $i\"}],\"max_tokens\":$mt}" \
      >/dev/null 2>&1 &
  done
  wait
}

echo "warmup: covering scheduled-token shapes at C=1..$MAXC"
for shape in 1 6 20 45 100 200; do
  for c in 1 2 4 "$MAXC"; do
    fire "$shape" "$c" "bounded-$shape" || true
    fire 32 "$c" "ordinary" || true   # client-default shape: no reasoning pins
  done
done

# Sampler kernels JIT once per sampling-mode family (greedy, top-p, top-k,
# combined), so a first real chat that brings its own sampling params can
# still pay a compile after the shape sweep above. Touch every family here,
# at odd and even batch sizes, so the kernels land in the persisted Triton
# cache during boot instead of mid-serve.
fire_sampled() { # sampler-json concurrency label
  local args="$1" c="$2" label="$3" i
  for i in $(seq 1 "$c"); do
    curl -fsS "http://127.0.0.1:${PORT}/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"warmup $label $i\"}],\"max_tokens\":16,$args}" \
      >/dev/null 2>&1 &
  done
  wait
}
echo "warmup: sampler families at C=1..3"
for c in 1 2 3; do
  fire_sampled '"temperature":0'                           "$c" greedy   || true
  fire_sampled '"temperature":0.7,"top_p":0.9'             "$c" top-p    || true
  fire_sampled '"temperature":0.7,"top_k":40'              "$c" top-k    || true
  fire_sampled '"temperature":0.8,"top_p":0.95,"top_k":50' "$c" combined || true
done
echo "warmup: done"
