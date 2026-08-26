#!/usr/bin/env bash
# prepare-models.sh, stage pinned weights on BOTH nodes, then serve offline.
#
# Two ways to get weights onto the worker:
#   --model deepseek|qwen|flash-next|both   download from HF into the cache on each node
#   --sync-worker <dir-name>     rsync an already-staged local dir head → worker
#                                over the node-to-node link (no second download;
#                                a 156 GiB checkpoint moves in ~6 min on 200G RoCE)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${SPARKDUET_ENV:-$ROOT/sparkduet.env}"
set -a; source "$ENV_FILE"; set +a

die(){ echo "prepare-models: ERROR: $*" >&2; exit 1; }
ssh_worker(){ ssh -o BatchMode=yes -o ConnectTimeout=8 "${WORKER_USER}@${WORKER_HOST}" "$@"; }

hf_download() { # repo rev dest_or_empty
  local repo="$1" rev="$2" dest="$3"
  if command -v hf >/dev/null; then
    if [[ -n "$dest" ]]; then
      HF_HUB_OFFLINE=0 hf download "$repo" --revision "${rev:-main}" --local-dir "$dest"
    else
      HF_HUB_OFFLINE=0 hf download "$repo" --revision "${rev:-main}"
    fi
    return 0
  fi
  return 1
}

fetch_one() { # repo revision cache_dir, forces HF online even if HF_HUB_OFFLINE=1
  local repo="$1" rev="$2" cache="$3"
  [[ -n "$repo" ]] || return 0
  echo ">> $repo @ ${rev:-main} -> $cache"
  if hf_download "$repo" "$rev" ""; then
    return 0
  fi
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

stage_flash_next() {
  local dest="${N_MODEL}"
  local repo="RadixArk/Qwen3.8-Flash-Next-NVFP4"
  local rev="${N_REVISION:-7b719225242aacd3dbd3f9407468c2ee9a9d2594}"
  if [[ "${dest:0:1}" == "/" ]]; then
    echo ">> $repo @ $rev -> $dest (then fabric sync)"
    if ! hf_download "$repo" "$rev" "$dest"; then
      HF_HUB_OFFLINE=0 python3 - "$repo" "$rev" "$dest" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, rev, dest = sys.argv[1], sys.argv[2], sys.argv[3]
snapshot_download(repo_id=repo, revision=rev, local_dir=dest)
print("downloaded:", dest)
PY
    fi
    [[ -f "$dest/config.json" ]] || die "Flash-Next download incomplete at $dest"
    sync_worker "$(basename "$dest")"
  else
    fetch_one "$dest" "$rev" "$HF_CACHE"
    ssh_worker "HF_HUB_OFFLINE=0 HF_HOME='$WORKER_HF_CACHE' python3 -c \
      \"from huggingface_hub import snapshot_download; snapshot_download('$dest', revision='$rev')\""
  fi
}

case "${1:-}" in
  --sync-worker) sync_worker "${2:-}";;
  --model)
    MODEL="${2:-both}"
    case "$MODEL" in
      deepseek)    stage_deepseek;;
      qwen)        stage_qwen;;
      flash-next|next) stage_flash_next;;
      both)        stage_deepseek; stage_qwen;;
      *)           die "unknown --model $MODEL (deepseek|qwen|flash-next|both)";;
    esac;;
  *) die "usage: prepare-models.sh --model deepseek|qwen|flash-next|both | --sync-worker <dir>";;
esac

echo "weights staged. Keep HF_HUB_OFFLINE=1 in sparkduet.env from here on."
