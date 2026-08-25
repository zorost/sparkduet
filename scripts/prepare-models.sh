#!/usr/bin/env bash
# prepare-models.sh, stage pinned weights on BOTH nodes, then serve offline.
#
# Two ways to get weights onto the worker:
#   --model deepseek|qwen|both   download from HF into the cache on each node
#   --sync-worker <dir-name>     rsync an already-staged local dir head → worker
#                                over the node-to-node link (no second download;
#                                a 156 GiB checkpoint moves in ~6 min on 200G RoCE)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${SPARKDUET_ENV:-$ROOT/sparkduet.env}"
set -a; source "$ENV_FILE"; set +a

die(){ echo "prepare-models: ERROR: $*" >&2; exit 1; }
ssh_worker(){ ssh -o BatchMode=yes -o ConnectTimeout=8 "${WORKER_USER}@${WORKER_HOST}" "$@"; }

fetch_one() { # repo revision cache_dir, forces HF online even if HF_HUB_OFFLINE=1
  local repo="$1" rev="$2" cache="$3"
  [[ -n "$repo" ]] || return 0
  echo ">> $repo @ ${rev:-main} -> $cache"
  HF_HUB_OFFLINE=0 HF_HOME="$cache" python3 - "$repo" "$rev" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, rev = sys.argv[1], sys.argv[2] or None
p = snapshot_download(repo_id=repo, revision=rev)
print("downloaded:", p)
PY
}

verify_encoder() { # DeepSeek 0731 ships a custom encoding/ folder; check both nodes
  local base="$1"
  [[ -e "$base/encoding/encoding_dsv4.py" || -e "$base/config.json" ]] \
    || die "checkpoint at $base looks incomplete"
}

sync_worker() { # dir name under MODEL_DIR
  local name="${1:?usage: --sync-worker <dir-name-under-MODEL_DIR>}"
  local src="$MODEL_DIR/$name"
  [[ -d "$src" ]] || die "no local dir $src"
  echo ">> rsync $src -> worker:$MODEL_DIR/ (over the fabric link)"
  rsync -aL --info=progress2 "$src" "${WORKER_USER}@${WORKER_HOST}:$MODEL_DIR/"
  local h w
  h=$(du -sb "$src" | cut -f1)
  w=$(ssh_worker "du -sb '$MODEL_DIR/$name'" | cut -f1)
  [[ "$h" == "$w" ]] || die "size mismatch after sync: head=$h worker=$w"
  echo "worker copy verified: $w bytes"
}

stage_deepseek() {
  if [[ "${DS_MODEL:0:1}" != "/" ]]; then
    fetch_one "$DS_MODEL" "$DS_REVISION" "$HF_CACHE"
    ssh_worker "HF_HUB_OFFLINE=0 HF_HOME='$WORKER_HF_CACHE' python3 -c \
      \"from huggingface_hub import snapshot_download; snapshot_download('$DS_MODEL', revision='$DS_REVISION')\""
  else
    verify_encoder "$DS_MODEL"
    ssh_worker "test -f '$DS_MODEL/config.json'" \
      || { echo "worker missing $DS_MODEL, syncing over the fabric"; sync_worker "$(basename "$DS_MODEL")"; }
  fi
}

stage_qwen() {
  fetch_one "$QWEN_MODEL" "${QWEN_REVISION:-}" "$HF_CACHE"
  ssh_worker "HF_HUB_OFFLINE=0 HF_HOME='$WORKER_HF_CACHE' python3 -c \
    \"from huggingface_hub import snapshot_download; snapshot_download('$QWEN_MODEL')\""
}

case "${1:-}" in
  --sync-worker) sync_worker "${2:-}";;
  --model)
    MODEL="${2:-both}"
    case "$MODEL" in
      deepseek) stage_deepseek;;
      qwen)     stage_qwen;;
      both)     stage_deepseek; stage_qwen;;
      *)        die "unknown --model $MODEL (deepseek|qwen|both)";;
    esac;;
  *) die "usage: prepare-models.sh --model deepseek|qwen|both | --sync-worker <dir>";;
esac

echo "weights staged. Keep HF_HUB_OFFLINE=1 in sparkduet.env from here on."
