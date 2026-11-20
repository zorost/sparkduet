#!/usr/bin/env bash
# Lane G on TensorFold (G_ENGINE=tensorfold): check the rank, then serve.
#
# Both ranks get the same serve arguments; TensorFold compares them, with the
# model and the build, before the weights load and names any field that
# differs. The environment places each rank: TF_GLM_RANK of TF_GLM_WORLD, and
# NCCL_COMM_ID for rank 0's address on the fabric. Rank 0 serves HTTP; rank 1
# follows its steps and binds no port.
#
# The public lane drafts with the checkpoint's own MTP head and nothing else:
# exactly one `--drafter none`, no drafter directory, TF_GLM_MTP=1. Anything
# else is refused here before the engine starts.
set -euo pipefail
BIN=/opt/tensorfold/bin/tensorfold-native

die() { echo "[glm-tf-entry] FATAL $*" >&2; exit 1; }

case "${TF_GLM_RANK:-}" in 0|1) ;; *) die "TF_GLM_RANK must be 0 or 1 (got '${TF_GLM_RANK:-}')" ;; esac
[[ "${TF_GLM_WORLD:-}" == 2 ]] || die "TF_GLM_WORLD must be 2 (this lane is one rank per Spark)"
[[ "${NCCL_COMM_ID:-}" == *:* ]] || die "NCCL_COMM_ID must be rank 0's fabric address:port"
[[ "${TF_GLM_MTP:-1}" == 1 ]] || die "TF_GLM_MTP must be 1: the lane drafts with the checkpoint's MTP head"
[[ -x "$BIN" ]] || die "$BIN missing: build the image from patches/glm-tensorfold/Dockerfile"
[[ -f /model/config.json ]] || die "/model/config.json missing: mount the staged EXL3/TR3 checkpoint"
[[ -f /model/generation_config.json ]] || die "/model/generation_config.json missing"

drafter=0
prev=
for a in "$@"; do
  if [[ "$prev" == --drafter ]]; then
    [[ "$a" == none ]] || die "--drafter must be none on the public lane (got '$a')"
    drafter=$((drafter + 1))
  fi
  case "$a" in
    --drafter=*) die "use '--drafter none' as two words" ;;
    --drafter-bits*) die "--drafter-bits has no meaning without an external drafter" ;;
  esac
  prev="$a"
done
[[ "$drafter" -eq 1 ]] || die "exactly one '--drafter none' is required (found $drafter)"

mkdir -p "${HOME:?}"
echo "[glm-tf-entry] rank ${TF_GLM_RANK}/${TF_GLM_WORLD} via ${NCCL_COMM_ID}; $("$BIN" --version 2>&1 | head -n 1)"
echo "[glm-tf-entry] end tokens: $(python3 -c 'import json; print(json.load(open("/model/generation_config.json")).get("eos_token_id"))' 2>/dev/null || echo unknown)"
exec "$BIN" serve /model "$@"
