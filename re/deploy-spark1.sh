#!/usr/bin/env bash
# SparkDuet RE desk: make it available on Spark 1.
#
# Run on Spark 1, as dspark1, from the side directory the Mac push creates:
#
#   cd /home/dspark1/re-desk && re/deploy-spark1.sh [options]
#
#   (no option)          check, install the pinned REA, check again, then
#                        print the OpenCode and Hermes registration plans
#                        (rea setup --dry-run). No agent config is written.
#   --apply-agents       also back up opencode.json and the Hermes
#                        config.yaml, run the official rea setup for both
#                        clients, and shape the rea rows (see re/README.md).
#   --restart-opencode   restart zorost-opencode.service, only after the
#                        OpenCode config parses and holds mcp.rea.
#   -h, --help           this text
#
# Safe to rerun. Each --apply-agents run takes fresh timestamped backups.
# Never touches /srv/ai/sparkduet, never calls sparkduetctl, never stops
# Caddy, never starts, stops or restarts a unit whose name contains
# "gateway", never restarts Hermes or its dashboard, writes no API key.
# Operator doc: re/README.md, section "Make it available".
set -u
set -o pipefail

usage() { sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

say()  { printf '%s\n' "$*"; }
step() { printf '\n== %s\n' "$*"; }
die()  { printf 'deploy-spark1.sh: %s\n' "$1" >&2; exit "${2:-2}"; }

APPLY_AGENTS=0
RESTART_OPENCODE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --apply-agents)     APPLY_AGENTS=1 ;;
    --restart-opencode) RESTART_OPENCODE=1 ;;
    -h|--help)          usage; exit 0 ;;
    *)                  usage >&2; die "unknown option: $1" ;;
  esac
  shift
done

# ---- Guard 1: architecture. First, before anything reads or writes. --------
ARCH="$(uname -m 2>/dev/null || echo unknown)"
if [ "$ARCH" != "aarch64" ]; then
  die "refusing to run: uname -m is '$ARCH', Spark 1 is aarch64. Nothing was installed or changed." 3
fi

# ---- Guard 2: who and where. ------------------------------------------------
DESK_USER="dspark1"
DESK_HOME="/home/$DESK_USER"
DESK_DIR="$DESK_HOME/re-desk"
LIVE_TREE="/srv/ai/sparkduet"

[ "$(id -u)" -ne 0 ] || die "do not run as root; run as $DESK_USER so every file stays owned by $DESK_USER"
[ "$(id -un)" = "$DESK_USER" ] || die "run as $DESK_USER (current user: $(id -un))"
[ "${HOME:-}" = "$DESK_HOME" ] || die "HOME is '${HOME:-}', expected $DESK_HOME"

RE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
case "$RE_DIR/" in
  "$LIVE_TREE"/*) die "this copy sits inside $LIVE_TREE, the live serving tree. The desk belongs in $DESK_DIR." ;;
esac
[ "$RE_DIR" = "$DESK_DIR/re" ] || die "expected this script at $DESK_DIR/re/deploy-spark1.sh, found it in $RE_DIR"
cd "$DESK_DIR" || die "cannot cd to $DESK_DIR"

export RE_PREFIX="$DESK_HOME/.local/share/sparkduet-re"
REA="$RE_PREFIX/bin/rea"
REA_PKG="$RE_PREFIX/lib/node_modules/rea-agents"
PIN="$(tr -d ' \t\r\n' < "$RE_DIR/REA_VERSION" 2>/dev/null || true)"
[ -n "$PIN" ] || die "re/REA_VERSION is empty"

OC_DIR="$DESK_HOME/.config/opencode"
OC_JSON="$OC_DIR/opencode.json"
OC_UNIT="zorost-opencode.service"
HERMES_YAML="$DESK_HOME/.hermes/config.yaml"
DASH_UNIT="zorost-hermes-dashboard.service"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

# Uses the system node. This script never installs or replaces Node.js.
NODE="$(command -v node || true)"
[ -n "$NODE" ] || die "node is not on PATH. Installing Node.js is an operator decision outside this desk."

# ---- Helpers -----------------------------------------------------------------
# systemctl scope of a unit: "--user", "system", or "" if not found. Read only.
unit_scope() {
  if systemctl --user cat "$1" >/dev/null 2>&1; then printf -- '--user'
  elif systemctl cat "$1" >/dev/null 2>&1; then printf 'system'
  fi
}

restart_cmd() { # unit
  case "$(unit_scope "$1")" in
    --user) printf 'systemctl --user restart %s' "$1" ;;
    system) printf 'sudo systemctl restart %s' "$1" ;;
    *)      printf 'systemctl restart %s   (unit not found by systemctl; check its name and scope first)' "$1" ;;
  esac
}

# Active units whose name contains "gateway", both scopes. Read only.
gateway_snapshot() {
  {
    systemctl list-units --all --plain --no-legend 2>/dev/null | awk '{print "system", $1, $3}'
    systemctl --user list-units --all --plain --no-legend 2>/dev/null | awk '{print "user", $1, $3}'
  } | awk 'tolower($2) ~ /gateway/ && $3 == "active"' | sort
}

opencode_ok() { # 0 when opencode.json parses and mcp.rea is a local list command
  "$NODE" -e '
    const fs = require("fs");
    const j = JSON.parse(fs.readFileSync(process.argv[1], "utf8"));
    const r = j && j.mcp && j.mcp.rea;
    if (!r || r.type !== "local" || !Array.isArray(r.command) || r.command.length === 0
        || !r.command.every(x => typeof x === "string") || r.enabled !== true) process.exit(1);
  ' "$OC_JSON" 2>/dev/null
}

# ---- Step 1: check ------------------------------------------------------------
step "1/4 re/re.sh check (read only)"
re/re.sh check
rc=$?
[ "$rc" -eq 0 ] || die "host check reported missing core requirements (exit $rc). Fix the MISSING rows, then rerun." 1

# ---- Step 2: install the pinned REA -----------------------------------------
step "2/4 re/re.sh install --apply (rea-agents@$PIN into $RE_PREFIX)"
have_v=""
[ -x "$REA" ] && have_v="$("$REA" --version 2>/dev/null | head -n 1)"
if [ "$have_v" = "$PIN" ]; then
  say "Already installed at the pin ($have_v). Skipping npm."
else
  re/re.sh install --apply || die "install failed" 1
fi

# ---- Step 3: check again ----------------------------------------------------
step "3/4 re/re.sh check (read only, after install)"
re/re.sh check
rc=$?
[ "$rc" -eq 0 ] || die "host check failed after install (exit $rc)" 1
have_v="$("$REA" --version 2>/dev/null | head -n 1)"
[ "$have_v" = "$PIN" ] || die "$REA reports '$have_v', pin is $PIN" 1

# ---- Step 4: agent registration ---------------------------------------------
step "4/4 Agent registration (official rea setup)"
GW_BEFORE="$(gateway_snapshot)"

if [ "$APPLY_AGENTS" -ne 1 ]; then
  for client in opencode hermes; do
    say ""
    say "-- rea setup --client $client --dry-run"
    "$REA" setup --client "$client" --dry-run --filter-output status,plannedActions \
      || die "rea setup --client $client --dry-run failed" 1
  done
  say ""
  say "Dry run only. Nothing in opencode.json or the Hermes config.yaml changed."
  say "Review the plans above, then rerun with --apply-agents."
else
  [ -f "$OC_JSON" ] || die "missing $OC_JSON"
  [ -f "$HERMES_YAML" ] || die "missing $HERMES_YAML"
  [ -f "$REA_PKG/node_modules/yaml/package.json" ] || die "REA's bundled yaml library is missing under $REA_PKG"
  "$NODE" -e 'JSON.parse(require("fs").readFileSync(process.argv[1], "utf8"))' "$OC_JSON" \
    || die "$OC_JSON does not parse as JSON before setup; not touching it"
  "$NODE" -e '
    const Y = require(process.argv[1]);
    const d = Y.parseDocument(require("fs").readFileSync(process.argv[2], "utf8"));
    if (d.errors.length) { console.error(String(d.errors[0])); process.exit(1); }
  ' "$REA_PKG/node_modules/yaml" "$HERMES_YAML" \
    || die "$HERMES_YAML does not parse as YAML before setup; not touching it"

  cp -p "$OC_JSON" "$OC_JSON.re-desk.$STAMP.bak" || die "backup of $OC_JSON failed"
  cp -p "$HERMES_YAML" "$HERMES_YAML.re-desk.$STAMP.bak" || die "backup of $HERMES_YAML failed"
  say "Backups:"
  say "  $OC_JSON.re-desk.$STAMP.bak"
  say "  $HERMES_YAML.re-desk.$STAMP.bak"

  # OpenCode: official setup, then shape only the rea row to the live form.
  say ""
  say "-- rea setup --client opencode --yes"
  "$REA" setup --client opencode --yes --filter-output status,appliedActions \
    || die "rea setup --client opencode failed. Restore from $OC_JSON.re-desk.$STAMP.bak if needed." 1
  "$NODE" -e '
    const fs = require("fs");
    const f = process.argv[1];
    const j = JSON.parse(fs.readFileSync(f, "utf8"));
    const r = j && j.mcp && j.mcp.rea;
    if (!r) { console.error("mcp.rea is missing after setup"); process.exit(1); }
    if (typeof r.command === "string") {
      r.command = [r.command].concat(Array.isArray(r.args) ? r.args : []);
      delete r.args;
    }
    r.type = "local";
    r.enabled = true;
    r.timeout = 15000;
    r.environment = Object.assign({}, r.environment, { REA_MCP_OUTPUT_SCHEMA_PROFILE: "compact" });
    const tmp = f + ".re-desk.tmp";
    fs.writeFileSync(tmp, JSON.stringify(j, null, 2) + "\n", { mode: fs.statSync(f).mode & 0o777 });
    fs.renameSync(tmp, f);
  ' "$OC_JSON" || die "could not shape mcp.rea in $OC_JSON. Restore from $OC_JSON.re-desk.$STAMP.bak" 1
  opencode_ok || die "$OC_JSON failed validation after setup. Restore from $OC_JSON.re-desk.$STAMP.bak" 1
  say "OpenCode: mcp.rea present (type local, command list, enabled, timeout 15000, compact output schemas)."

  # Hermes: official setup, then add the compact profile to the rea env only.
  say ""
  say "-- rea setup --client hermes --yes"
  "$REA" setup --client hermes --yes --filter-output status,appliedActions \
    || die "rea setup --client hermes failed. Restore from $HERMES_YAML.re-desk.$STAMP.bak if needed." 1
  "$NODE" -e '
    const fs = require("fs");
    const Y = require(process.argv[1]);
    const f = process.argv[2];
    const d = Y.parseDocument(fs.readFileSync(f, "utf8"));
    if (d.errors.length) { console.error(String(d.errors[0])); process.exit(1); }
    if (!d.hasIn(["mcp_servers", "rea"])) { console.error("mcp_servers.rea is missing after setup"); process.exit(1); }
    d.setIn(["mcp_servers", "rea", "env", "REA_MCP_OUTPUT_SCHEMA_PROFILE"], "compact");
    const tmp = f + ".re-desk.tmp";
    fs.writeFileSync(tmp, String(d), { mode: fs.statSync(f).mode & 0o777 });
    fs.renameSync(tmp, f);
  ' "$REA_PKG/node_modules/yaml" "$HERMES_YAML" \
    || die "could not shape mcp_servers.rea in $HERMES_YAML. Restore from $HERMES_YAML.re-desk.$STAMP.bak" 1
  say "Hermes: mcp_servers.rea present with compact output schemas."
  say "Hermes is not restarted. The gateway is not started. The dashboard is not restarted."
fi

GW_AFTER="$(gateway_snapshot)"
if [ "$GW_BEFORE" != "$GW_AFTER" ]; then
  say ""
  say "WARNING: the set of active *gateway* units changed during this run."
  say "Before:"; say "${GW_BEFORE:-  (none)}"
  say "After:";  say "${GW_AFTER:-  (none)}"
  say "This script did not touch them and will not. Check by hand."
  exit 4
fi

# ---- OpenCode restart: printed, or run only on request ---------------------
say ""
if [ "$RESTART_OPENCODE" -eq 1 ]; then
  opencode_ok || die "not restarting $OC_UNIT: $OC_JSON does not parse or has no valid mcp.rea row" 1
  scope="$(unit_scope "$OC_UNIT")"
  case "$scope" in
    --user) systemctl --user restart "$OC_UNIT" || die "restart of $OC_UNIT failed" 1
            sleep 3; systemctl --user is-active "$OC_UNIT" ;;
    system) sudo systemctl restart "$OC_UNIT" || die "restart of $OC_UNIT failed" 1
            sleep 3; systemctl is-active "$OC_UNIT" ;;
    *)      die "$OC_UNIT not found by systemctl; not restarting" 1 ;;
  esac
  rc=$?
  [ "$rc" -eq 0 ] || die "$OC_UNIT is not active after restart. Restore $OC_JSON from the newest .re-desk.*.bak and restart again." 1
  say "$OC_UNIT restarted and active."
else
  say "OpenCode picks up mcp.rea after a restart. When ready:"
  say "  $(restart_cmd "$OC_UNIT")"
  say "  or rerun: re/deploy-spark1.sh --restart-opencode"
fi
say ""
say "Hermes: open a new hermes CLI session and confirm rea is listed among its MCP tools."
say "If the dashboard still does not list rea after a page reload, its restart is:"
say "  $(restart_cmd "$DASH_UNIT")"
say "This script does not run it."
say ""
say "Done."
