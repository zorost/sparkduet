#!/usr/bin/env bash
# ab_glm_engine.sh, Lane G engine A/B on the pair: before (G_ENGINE as it is,
# normally exl3) against after (tensorfold), one table at the end.
#
# Run on the head Spark from the repo checkout that sparkduetctl uses:
#   scripts/ab_glm_engine.sh --dry-run     print the plan, change nothing
#   scripts/ab_glm_engine.sh               build, measure both, restore the original engine
#   scripts/ab_glm_engine.sh --adopt       same, but keep tensorfold when it passes every gate
#   scripts/ab_glm_engine.sh --skip-build  reuse G_TF_IMAGE as it is on both Sparks
#
# What it does, in order:
#   1. checks the head (aarch64, docker, sparkduet.env) and that the worker's
#      sparkduet.env has the G_TF_* block, appending the head's block there
#      (after a timestamped backup) when it is missing
#   2. scripts/build-tensorfold-image.sh (build on the head, ship to the worker)
#   3. for each engine: sparkduetctl.sh switch glm with that G_ENGINE (a temp
#      copy of sparkduet.env), timed to ready, then scripts/ab_glm_probe.py run
#   4. writes results/<UTC stamp>-glm-engine-ab/{before,after}.json and README.md
#      (the table and the gate verdict) and prints the table
#   5. switches back to the original engine, or, with --adopt and a passing
#      verdict, sets G_ENGINE=tensorfold in sparkduet.env (backup kept) and stays
#
# The resident lane is down while each engine loads: plan for up to an hour.
# It never touches Lane D, the gateway, Hermes or OpenCode.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${SPARKDUET_ENV:-$ROOT/sparkduet.env}"
ADOPT=0 BUILD=1 DRY=0
for a in "$@"; do
  case "$a" in
    --adopt) ADOPT=1 ;;
    --skip-build) BUILD=0 ;;
    --dry-run) DRY=1 ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "unknown argument: $a" >&2; exit 2 ;;
  esac
done
die() { echo "ab_glm_engine: $*" >&2; exit 1; }
say() { printf '\n== %s\n' "$*"; }

[[ -f "$ENV_FILE" ]] || die "missing $ENV_FILE"
# shellcheck disable=SC1090
set -a; source "$ENV_FILE"; set +a
ORIG="${G_ENGINE:-vllm}"
BEFORE="${AB_BEFORE:-$ORIG}"
[[ "$BEFORE" == tensorfold ]] && BEFORE=exl3
AFTER=tensorfold
[[ "$(uname -m)" == aarch64 ]] || die "run this on the head Spark"
command -v docker >/dev/null || die "docker missing"
ssh_worker() { ssh -o BatchMode=yes -o ConnectTimeout=8 "${WORKER_USER}@${WORKER_HOST}" "$@"; }
WROOT="${SPARKDUET_DIR:-$ROOT}"
STAMP="$(date -u +%Y-%m-%d-%H%M)"
OUT="$ROOT/results/${STAMP}-glm-engine-ab"
MODEL="${G_EXL3_SERVED_NAME:?G_EXL3_SERVED_NAME unset}"
PORT="${G_PORT:-30000}"

say "plan"
echo "before: G_ENGINE=$BEFORE   after: G_ENGINE=$AFTER   restore: $([[ $ADOPT == 1 ]] && echo "tensorfold if it passes, else $ORIG" || echo "$ORIG")"
echo "model: $MODEL on :$PORT   results: $OUT"
[[ "$DRY" == 1 ]] && { echo "(dry run: nothing changed)"; exit 0; }

say "1/5 worker env"
if ! ssh_worker "grep -q '^G_TF_IMAGE=' '$WROOT/sparkduet.env'"; then
  ssh_worker "cp -p '$WROOT/sparkduet.env' '$WROOT/sparkduet.env.ab-$STAMP.bak'"
  grep -E '^(# ---- Lane G on TensorFold|G_TF_)' "$ENV_FILE" | ssh_worker "cat >> '$WROOT/sparkduet.env'"
  echo "appended the G_TF_* block to the worker's sparkduet.env (backup .ab-$STAMP.bak)"
else
  echo "worker env already has G_TF_*"
fi
ssh_worker "test -f '$WROOT/scripts/glm_eos_overlay.py'" \
  || die "worker checkout at $WROOT lacks scripts/glm_eos_overlay.py: git pull opencode-glm there first"

say "2/5 image"
if [[ "$BUILD" == 1 ]]; then "$ROOT/scripts/build-tensorfold-image.sh"; else echo "skipped (--skip-build)"; fi

mkdir -p "$OUT"
restore_on_failure() {
  echo "ab_glm_engine: a step failed; putting G_ENGINE=$ORIG back" >&2
  "$ROOT/scripts/sparkduetctl.sh" switch glm || echo "ab_glm_engine: restore failed too; run sparkduetctl.sh switch glm by hand" >&2
}
trap restore_on_failure ERR
leg() { # engine label
  local engine="$1" label="$2" tmp t0 t1
  tmp="$(mktemp)"
  sed -E "s/^G_ENGINE=.*/G_ENGINE=$engine/" "$ENV_FILE" > "$tmp"
  grep -q "^G_ENGINE=$engine" "$tmp" || echo "G_ENGINE=$engine" >> "$tmp"
  say "$label: switch glm with G_ENGINE=$engine"
  t0=$(date +%s)
  SPARKDUET_ENV="$tmp" "$ROOT/scripts/sparkduetctl.sh" switch glm
  t1=$(date +%s)
  rm -f "$tmp"
  python3 "$ROOT/scripts/ab_glm_probe.py" run --base-url "http://127.0.0.1:$PORT/v1" \
    --model "$MODEL" --label "$engine" --ready-s "$((t1 - t0))" --out "$OUT/$label.json"
  docker logs sparkduet-glm-head 2>&1 | tail -n 200 > "$OUT/$label-head.log" || true
}

say "3/5 measure before ($BEFORE)"
leg "$BEFORE" before
say "4/5 measure after ($AFTER)"
leg "$AFTER" after

say "5/5 table"
trap - ERR
verdict_rc=0
python3 "$ROOT/scripts/ab_glm_probe.py" verdict "$OUT/before.json" "$OUT/after.json" > "$OUT/verdict.txt" || verdict_rc=$?
{
  echo "# Lane G engine A/B, $STAMP UTC"
  echo
  python3 "$ROOT/scripts/ab_glm_probe.py" table "$OUT/before.json" "$OUT/after.json"
  echo
  echo '```'
  cat "$OUT/verdict.txt"
  echo '```'
} > "$OUT/README.md"
cat "$OUT/README.md"

if [[ "$ADOPT" == 1 && "$verdict_rc" == 0 ]]; then
  cp -p "$ENV_FILE" "$ENV_FILE.ab-$STAMP.bak"
  sed -i -E "s/^G_ENGINE=.*/G_ENGINE=$AFTER/" "$ENV_FILE"
  say "adopted: G_ENGINE=$AFTER in $ENV_FILE (backup .ab-$STAMP.bak); the lane stays on $AFTER"
else
  say "restoring G_ENGINE=$ORIG"
  "$ROOT/scripts/sparkduetctl.sh" switch glm
fi
echo "results: $OUT"
