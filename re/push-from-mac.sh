#!/usr/bin/env bash
# SparkDuet RE desk: ship re/ from the operator Mac to Spark 1, then deploy.
#
# Run on the operator Mac, from a checkout of this repo:
#
#   re/push-from-mac.sh [--apply-agents] [--restart-opencode]
#
# What it does:
#   1. Exports the committed re/ tree (git archive HEAD), so ignored payloads,
#      evidence and local-* projects on the Mac are never shipped.
#   2. ssh dspark1: checks the remote user is dspark1, creates
#      /home/dspark1/re-desk if needed.
#   3. rsync -a --delete of that tree to /home/dspark1/re-desk/re/ only.
#      re/projects/ on the box is excluded, so investigations there survive.
#   4. ssh -t dspark1: cd /home/dspark1/re-desk && re/deploy-spark1.sh, with
#      the flags above passed through. Without flags the agent registration
#      is a dry run.
#
# Uses your normal ssh config and agent for the host alias "dspark1". No
# sshpass, no key in this file. Override the alias with RE_SSH_HOST.
set -u
set -o pipefail

usage() { sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
die() { printf 'push-from-mac.sh: %s\n' "$1" >&2; exit "${2:-2}"; }

SSH_HOST="${RE_SSH_HOST:-dspark1}"
REMOTE_USER="dspark1"
REMOTE_DESK="/home/dspark1/re-desk"

PASS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --apply-agents|--restart-opencode) PASS="$PASS $1" ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown option: $1" ;;
  esac
  shift
done

[ "$(uname -s)" = "Darwin" ] || [ "${RE_PUSH_ANY_OS:-0}" = "1" ] \
  || die "meant for the operator Mac (uname -s is $(uname -s)). Set RE_PUSH_ANY_OS=1 to override."
command -v rsync >/dev/null 2>&1 || die "rsync is not on PATH"
command -v git   >/dev/null 2>&1 || die "git is not on PATH"

RE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_DIR="$(git -C "$RE_DIR" rev-parse --show-toplevel 2>/dev/null)" || die "not inside a git checkout"
[ "$RE_DIR" = "$REPO_DIR/re" ] || die "expected this script at <repo>/re/push-from-mac.sh"
[ -f "$REPO_DIR/re/deploy-spark1.sh" ] || die "re/deploy-spark1.sh is missing"

SHA="$(git -C "$REPO_DIR" rev-parse HEAD)" || die "cannot read HEAD"
if [ -n "$(git -C "$REPO_DIR" status --porcelain -- re)" ]; then
  printf 'note: re/ has uncommitted changes. Only the committed tree at %s is shipped.\n' "$SHA"
fi

STAGE="$(mktemp -d "${TMPDIR:-/tmp}/re-desk-push.XXXXXX")" || die "mktemp failed"
trap 'rm -rf "$STAGE"' EXIT
git -C "$REPO_DIR" archive --format=tar HEAD re | tar -x -C "$STAGE" || die "git archive of re/ failed"
[ -x "$STAGE/re/deploy-spark1.sh" ] || die "deploy-spark1.sh is not committed as executable"
printf 'commit %s\nshipped %s\n' "$SHA" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STAGE/re/DEPLOYED_FROM"

printf '== Remote preflight on %s\n' "$SSH_HOST"
ssh "$SSH_HOST" "test \"\$(id -un)\" = $REMOTE_USER && mkdir -p $REMOTE_DESK/re/projects" \
  || die "ssh $SSH_HOST failed, or the remote user is not $REMOTE_USER"

printf '== rsync re/ (%s) to %s:%s/re/\n' "$SHA" "$SSH_HOST" "$REMOTE_DESK"
# Source "$STAGE/re" without a trailing slash: the transfer root is $STAGE, so
# --delete only ever acts inside $REMOTE_DESK/re, and the anchored exclude
# keeps the box's re/projects/ out of the transfer and out of --delete.
rsync -a --delete --exclude=/re/projects/ -e ssh \
  "$STAGE/re" "$SSH_HOST:$REMOTE_DESK/" || die "rsync failed"

printf '== Deploy on %s:%s\n' "$SSH_HOST" "$REMOTE_DESK"
# shellcheck disable=SC2086  # PASS holds only the two known flags above
ssh -t "$SSH_HOST" "cd $REMOTE_DESK && re/deploy-spark1.sh$PASS"
