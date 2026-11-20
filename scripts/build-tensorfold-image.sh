#!/usr/bin/env bash
# build-tensorfold-image.sh, the Lane G TensorFold image on both Sparks.
#
# Builds G_TF_IMAGE on the head from patches/glm-tensorfold/Dockerfile, from
# the pinned upstream commit (G_TF_SOURCE=git) or the pinned v1.3.0 release
# archive (G_TF_SOURCE=release), then gives the worker the same image by
# content over ssh, so `sparkduetctl.sh switch glm` with G_ENGINE=tensorfold
# finds one image id on both ranks.
#
#   scripts/build-tensorfold-image.sh            build if missing, ship if the worker differs
#   scripts/build-tensorfold-image.sh --rebuild  build again (e.g. after a pin change)
#   scripts/build-tensorfold-image.sh --dry-run  print what would run
#
# Builds nothing on the worker and starts no lane.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${SPARKDUET_ENV:-$ROOT/sparkduet.env}"
REBUILD=0 DRY=0
for a in "$@"; do
  case "$a" in
    --rebuild) REBUILD=1 ;;
    --dry-run) DRY=1 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown argument: $a" >&2; exit 2 ;;
  esac
done

die() { echo "build-tensorfold-image: $*" >&2; exit 1; }
[[ -f "$ENV_FILE" ]] || die "missing $ENV_FILE (copy configs/sparkduet.env.example)"
# shellcheck disable=SC1090
set -a; source "$ENV_FILE"; set +a
IMAGE="${G_TF_IMAGE:-sparkduet-tensorfold:81de26f6}"
SOURCE="${G_TF_SOURCE:-git}"
case "$SOURCE" in git|release) ;; *) die "G_TF_SOURCE must be git or release (got '$SOURCE')" ;; esac
[[ -n "${WORKER_HOST:-}" && -n "${WORKER_USER:-}" ]] || die "WORKER_HOST and WORKER_USER must be set in $ENV_FILE"
[[ "$(uname -m)" == aarch64 ]] || die "run this on the head Spark (aarch64)"
ssh_worker() { ssh -o BatchMode=yes -o ConnectTimeout=8 "${WORKER_USER}@${WORKER_HOST}" "$@"; }
run() { echo "+ $*"; [[ "$DRY" == 1 ]] || "$@"; }

have=$(docker image inspect "$IMAGE" --format '{{.Id}}' 2>/dev/null || true)
if [[ -z "$have" || "$REBUILD" == 1 ]]; then
  run docker build --build-arg "TF_SOURCE=$SOURCE" -t "$IMAGE" "$ROOT/patches/glm-tensorfold"
  [[ "$DRY" == 1 ]] || have=$(docker image inspect "$IMAGE" --format '{{.Id}}')
fi
echo "head: $IMAGE (${SOURCE}) ${have:-<not built>}"

theirs=$(ssh_worker "docker image inspect '$IMAGE' --format '{{.Id}}'" 2>/dev/null || true)
if [[ -n "$have" && "$theirs" == "$have" ]]; then
  echo "worker: same image already present"
  exit 0
fi
echo "worker: ${theirs:-<none>}, shipping $IMAGE over ssh"
if [[ "$DRY" == 1 ]]; then
  echo "+ docker save $IMAGE | ssh ${WORKER_USER}@${WORKER_HOST} docker load"
  exit 0
fi
docker save "$IMAGE" | ssh_worker "docker load" >/dev/null
theirs=$(ssh_worker "docker image inspect '$IMAGE' --format '{{.Id}}'")
[[ "$theirs" == "$have" ]] || die "worker holds $theirs after load, head $have"
echo "worker: $IMAGE $theirs"
