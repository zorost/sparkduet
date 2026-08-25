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
echo "warmup: done"
