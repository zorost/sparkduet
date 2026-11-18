#!/usr/bin/env bash
# SparkDuet reverse-engineering workbench: one entry point.
#
#   re/re.sh check                       read-only host report (run on Spark 1)
#   re/re.sh install [--apply]           pinned REA install into a user prefix
#   re/re.sh new <slug> --target <path>  create one investigation folder
#   re/re.sh list                        show investigations and their status
#   re/re.sh model <slug>                GGUF / safetensors header inspection
#   re/re.sh gpu <slug>                  CUDA / HIP code-object inventory
#
# Nothing here starts, stops, swaps or reconfigures a lane, Caddy, Hermes or
# any other service. Nothing here loads a model or touches the GPU.
# Operator doc: re/README.md
set -u

RE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$RE_DIR/.." && pwd)"
PROJECTS_DIR="$RE_DIR/projects"
TEMPLATE_DIR="$RE_DIR/template"
REA_PIN="$(tr -d ' \t\r\n' < "$RE_DIR/REA_VERSION" 2>/dev/null || true)"
RE_PREFIX="${RE_PREFIX:-${XDG_DATA_HOME:-$HOME/.local/share}/sparkduet-re}"
# House margin for GPU code-object inspection. REA hashes the whole file, and
# file reads grow the page cache, which the GB10 driver cannot allocate from.
RE_MAX_GPU_FILE_MIB="${RE_MAX_GPU_FILE_MIB:-2048}"

say()  { printf '%s\n' "$*"; }
row()  { printf '  %-8s %-22s %s\n' "$1" "$2" "$3"; }
die()  { printf 're.sh: %s\n' "$*" >&2; exit 2; }
have() { command -v "$1" >/dev/null 2>&1; }

usage() {
  sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

# ---- Node.js range from the upstream README (rea-agents engines field) ------
# 22.x >= 22.23.4, 24.x >= 24.22.1, or >= 26.12.1. 23, 25 and prereleases: no.
node_supported() {
  local v="${1#v}" maj min pat
  case "$v" in *-*) return 1 ;; esac
  IFS=. read -r maj min pat <<EOF
$v
EOF
  case "$maj$min$pat" in ''|*[!0-9]*) return 1 ;; esac
  if [ "$maj" -eq 22 ]; then
    [ "$min" -gt 23 ] || { [ "$min" -eq 23 ] && [ "$pat" -ge 4 ]; }
  elif [ "$maj" -eq 24 ]; then
    [ "$min" -gt 22 ] || { [ "$min" -eq 22 ] && [ "$pat" -ge 1 ]; }
  elif [ "$maj" -eq 26 ]; then
    [ "$min" -gt 12 ] || { [ "$min" -eq 12 ] && [ "$pat" -ge 1 ]; }
  elif [ "$maj" -gt 26 ] && [ $((maj % 2)) -eq 0 ]; then
    return 0
  else
    return 1
  fi
}

# ---- Locate the pinned REA executable --------------------------------------
rea_bin() {
  if [ -n "${RE_REA_BIN:-}" ]; then printf '%s\n' "$RE_REA_BIN"; return; fi
  if [ -x "$RE_PREFIX/bin/rea" ]; then printf '%s\n' "$RE_PREFIX/bin/rea"; return; fi
  if have rea; then command -v rea; return; fi
  return 1
}

meminfo_gib() {
  awk -v k="$1:" '$1==k {printf "%.1f", $2/1048576; f=1} END {if(!f) printf "?"}' \
    /proc/meminfo 2>/dev/null || printf '?'
}

# ============================================================================
# check: read-only. Reports, never changes anything.
# ============================================================================
cmd_check() {
  local fail=0 arch node_v npm_v ps_bin ps_ver rea rea_v

  say "SparkDuet RE workbench host check (read-only)"
  say "pinned package: rea-agents@${REA_PIN:-unknown}"
  say ""
  say "Core requirements"

  arch="$(uname -m 2>/dev/null || echo unknown)"
  if [ "$arch" = "aarch64" ] || [ "$arch" = "arm64" ]; then
    row OK arch "$arch"
  else
    row WARN arch "$arch (not a GB10; a pass here says nothing about the Sparks)"
  fi
  row INFO kernel "$(uname -sr 2>/dev/null || echo unknown)"

  if have node; then
    node_v="$(node --version 2>/dev/null || echo unknown)"
    if node_supported "$node_v"; then
      row OK node "$node_v"
    else
      row MISSING node "$node_v is outside 22.x>=22.23.4, 24.x>=24.22.1, >=26.12.1"
      fail=1
    fi
  else
    row MISSING node "not on PATH"
    fail=1
  fi

  if have npm; then
    npm_v="$(npm --version 2>/dev/null || echo unknown)"
    row OK npm "$npm_v"
  else
    row MISSING npm "not on PATH"
    fail=1
  fi

  # REA looks at /bin/ps, then /usr/bin/ps, before PATH (upstream install doc).
  ps_bin=""
  for p in /bin/ps /usr/bin/ps; do
    if [ -x "$p" ]; then ps_bin="$p"; break; fi
  done
  if [ -z "$ps_bin" ] && have ps; then ps_bin="$(command -v ps)"; fi
  if [ -n "$ps_bin" ]; then
    ps_ver="$("$ps_bin" --version 2>&1 | head -n 1)"
    case "$ps_ver" in
      *procps*) row OK ps "$ps_bin ($ps_ver)" ;;
      *) row WARN ps "$ps_bin does not report procps ($ps_ver)" ;;
    esac
  else
    row MISSING ps "no ps found; install procps before JavaScript analysis"
    fail=1
  fi

  say ""
  say "Unified memory (from /proc/meminfo, GiB)"
  if [ -r /proc/meminfo ]; then
    row INFO MemTotal "$(meminfo_gib MemTotal)"
    row INFO MemFree "$(meminfo_gib MemFree)"
    row INFO MemAvailable "$(meminfo_gib MemAvailable)"
    row INFO Cached "$(meminfo_gib Cached)"
    row INFO SwapFree "$(meminfo_gib SwapFree)"
    say "  On GB10 the GPU driver allocates from MemFree, not MemAvailable, so"
    say "  large file reads that grow Cached compete with the resident lane."
    say "  This desk reads headers and small objects only. See re/README.md."
  else
    row WARN meminfo "/proc/meminfo is not readable on this host"
  fi

  say ""
  say "REA"
  if rea="$(rea_bin)"; then
    rea_v="$("$rea" --version 2>/dev/null | head -n 1)"
    if [ "$rea_v" = "$REA_PIN" ]; then
      row OK rea "$rea ($rea_v, matches pin)"
    else
      row WARN rea "$rea reports '$rea_v', pin is $REA_PIN"
    fi
  else
    row ABSENT rea "not installed; see: re/re.sh install"
  fi

  say ""
  say "Optional providers (presence only; none is installed or started here)"
  say "  In the program once present:"
  provider objdump "GNU objdump, static disassembly and headers" objdump "${REA_OBJDUMP_COMMAND:-}"
  say "  Possible upstream, not yet tried on this pair:"
  provider_ghidra
  provider rizin "Rizin, static commands only" rizin "${REA_RIZIN_COMMAND:-}"
  provider kuna "Kuna decompiler" kuna "${REA_KUNA_COMMAND:-}"
  provider_wabt
  provider tshark "TShark, saved packet captures" tshark ""
  provider capinfos "capinfos, packet summaries" capinfos ""
  provider prlimit "util-linux prlimit, used by several owned tools" prlimit ""
  provider_browser
  say "  Not available on this pair (upstream is Linux x64 only, or commercial):"
  row N/A hopper "official Linux packages are x86_64 only"
  row N/A ida "commercial; no installation is planned on the Sparks"
  row N/A jadx "upstream requires an x64 host"
  row N/A pwntools-layout "upstream refuses Linux arm64 hosts; also not a house feature"
  row N/A recorded-crashes "upstream is Linux x64; also not a house feature"
  row N/A evm "upstream is Linux x64"

  say ""
  say "Resident lane: not probed and not touched by this check."
  say "Use scripts/sparkduetctl.sh status yourself if you want to see it."
  say ""
  if [ "$fail" -eq 0 ]; then
    say "RESULT: core requirements present."
  else
    say "RESULT: core requirements missing. Fix the MISSING rows first."
  fi
  return "$fail"
}

provider() { # id, description, command name, explicit override
  local id="$1" desc="$2" cmd="$3" override="$4" path=""
  if [ -n "$override" ] && { [ -x "$override" ] || have "$override"; }; then
    path="$override"
  elif have "$cmd"; then
    path="$(command -v "$cmd")"
  fi
  if [ -n "$path" ]; then row PRESENT "$id" "$path ($desc)"
  else row ABSENT "$id" "$desc"; fi
}

provider_ghidra() {
  local dir="${GHIDRA_INSTALL_DIR:-}" dec="" java="absent"
  if have java; then
    java="$(java -version 2>&1 | grep -m 1 ' version ' || echo 'present, version unknown')"
  fi
  if [ -n "$dir" ] && [ -d "$dir" ]; then
    for d in "$dir/Ghidra/Features/Decompiler/os/linux_arm_64" \
             "$dir/Ghidra/Features/Decompiler/build/os/linux_arm_64"; do
      [ -d "$d" ] && dec="$d"
    done
    if [ -n "$dec" ]; then
      row PRESENT ghidra "$dir, linux_arm_64 decompiler dir found, java: $java"
    else
      row PARTIAL ghidra "$dir has no linux_arm_64 decompiler dir, java: $java"
    fi
  else
    row ABSENT ghidra "GHIDRA_INSTALL_DIR not set (needs a JVM heap; operator decision), java: $java"
  fi
}

provider_wabt() {
  local dir="${REA_WABT_BIN_DIRECTORY:-}"
  if [ -n "$dir" ] && [ -x "$dir/wasm-validate" ] && [ -x "$dir/wasm-objdump" ] && [ -x "$dir/wasm2wat" ]; then
    row PRESENT wabt "$dir"
  elif have wasm-objdump; then
    row PARTIAL wabt "found on PATH; REA wants REA_WABT_BIN_DIRECTORY"
  else
    row ABSENT wabt "WABT 1.0.42, WASM inspection"
  fi
}

provider_browser() {
  local b
  for b in chromium chromium-browser google-chrome brave-browser; do
    if have "$b"; then row PRESENT browser "$(command -v "$b") (not part of the first slice)"; return; fi
  done
  row ABSENT browser "Chrome-family browser (not part of the first slice)"
}

# ============================================================================
# install: prints the plan. Only --apply runs npm.
# ============================================================================
cmd_install() {
  local apply=0 node_v
  [ "${1:-}" = "--apply" ] && apply=1
  [ -n "$REA_PIN" ] || die "re/REA_VERSION is empty"
  have node || die "node is not on PATH"
  have npm  || die "npm is not on PATH"
  node_v="$(node --version)"
  node_supported "$node_v" || die "node $node_v is outside the range REA supports"

  say "Plan: install rea-agents@$REA_PIN for this user only."
  say "  prefix : $RE_PREFIX"
  say "  command: npm install --global --prefix \"$RE_PREFIX\" --omit=optional --ignore-scripts rea-agents@$REA_PIN"
  say "  effect : writes files under the prefix. No sudo, no service, no agent config."
  say "  skipped: the optional Frida binding (runtime instrumentation is not part of this desk)."
  if [ "$apply" -ne 1 ]; then
    say ""
    say "Dry run. Re-run with --apply to install."
    return 0
  fi
  # Run npm from a neutral directory so it does not read this repo as a project.
  ( cd "${TMPDIR:-/tmp}" && \
    npm install --global --prefix "$RE_PREFIX" --omit=optional --ignore-scripts "rea-agents@$REA_PIN" ) \
    || die "npm install failed"
  say ""
  say "Installed: $("$RE_PREFIX/bin/rea" --version 2>/dev/null) at $RE_PREFIX/bin/rea"
  say "Next: re/re.sh check"
}

# ============================================================================
# new: one folder per investigation. Target stays outside the repo.
# ============================================================================
valid_slug() { [[ "$1" =~ ^[a-z0-9][a-z0-9-]{1,47}$ ]]; }

cmd_new() {
  local slug="" target="" question="" dir today f body abs
  while [ $# -gt 0 ]; do
    case "$1" in
      --target)   [ $# -ge 2 ] || die "--target needs a path"; target="$2"; shift 2 ;;
      --question) [ $# -ge 2 ] || die "--question needs text"; question="$2"; shift 2 ;;
      -*)         die "unknown option: $1" ;;
      *)          [ -z "$slug" ] || die "one slug only"; slug="$1"; shift ;;
    esac
  done
  [ -n "$slug" ] || die "usage: re/re.sh new <slug> --target /absolute/path [--question \"...\"]"
  valid_slug "$slug" || die "slug must be 2 to 48 chars of a-z, 0-9 and dash, starting with a letter or digit"
  [ -n "$target" ] || die "--target is required (absolute path, outside this repo)"
  case "$target" in /*) ;; *) die "target must be an absolute path" ;; esac
  case "$target" in *$'\n'*) die "target must be a single line" ;; esac
  abs="$(realpath -m -- "$target" 2>/dev/null || printf '%s' "$target")"
  case "$abs/" in
    "$REPO_DIR"/*) die "target is inside this repo. Keep targets outside it so they cannot be committed." ;;
  esac
  dir="$PROJECTS_DIR/$slug"
  [ -e "$dir" ] && die "project already exists: $dir"
  [ -e "$abs" ] || say "note: target does not exist on this host yet: $abs"
  [ -n "$question" ] || question="(write the one question this investigation answers)"

  today="$(date -u +%Y-%m-%d)"
  mkdir -p "$dir/evidence" || die "cannot create $dir"
  for f in QUESTION.md EVIDENCE.md STATUS.md; do
    body="$(cat "$TEMPLATE_DIR/$f")" || die "missing template $f"
    body="${body//@@SLUG@@/$slug}"
    body="${body//@@DATE@@/$today}"
    body="${body//@@TARGET@@/$abs}"
    body="${body//@@QUESTION@@/$question}"
    printf '%s\n' "$body" > "$dir/$f"
  done
  say "Created $dir"
  say "  QUESTION.md  the question and the target path"
  say "  EVIDENCE.md  what you found, with the evidence file it came from"
  say "  STATUS.md    one status line an operator can read"
  say "  evidence/    REA output for this project (ignored by git)"
  case "$slug" in local-*) say "  This is a local-* project: git ignores the whole folder." ;; esac
}

project_dir() {
  local slug="${1:-}"
  [ -n "$slug" ] || die "project slug required"
  valid_slug "$slug" || die "bad slug: $slug"
  [ -d "$PROJECTS_DIR/$slug" ] || die "no such project: $slug (see: re/re.sh list)"
  printf '%s\n' "$PROJECTS_DIR/$slug"
}

project_target() {
  local t
  t="$(sed -n 's/^- Target path: //p' "$1/QUESTION.md" 2>/dev/null | head -n 1)"
  [ -n "$t" ] || die "no '- Target path:' line in $1/QUESTION.md"
  printf '%s\n' "$t"
}

cmd_list() {
  local d any=0 st
  for d in "$PROJECTS_DIR"/*/; do
    [ -f "$d/STATUS.md" ] || continue
    any=1
    st="$(sed -n 's/^Status: //p' "$d/STATUS.md" | head -n 1)"
    printf '%-32s %s\n' "$(basename "$d")" "${st:-unknown}"
  done
  [ "$any" -eq 1 ] || say "No projects yet. Create one with: re/re.sh new <slug> --target /absolute/path"
}

# ============================================================================
# model / gpu: the two static first paths. CPU only, header-sized reads.
# The flags are the ones in upstream docs/model-artifacts.md and
# docs/gpu-code-objects.md, run here exactly as documented.
# ============================================================================
run_inspection() { # kind, slug
  local kind="$1" dir target rea stamp ev out rc size_mib
  dir="$(project_dir "${2:-}")" || exit 2
  target="$(project_target "$dir")" || exit 2
  rea="$(rea_bin)" || die "REA is not installed. Run: re/re.sh install"
  [ -f "$target" ] || die "target is not a regular file on this host: $target"
  mkdir -p "$dir/evidence"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  ev="$dir/evidence/$kind-$stamp.evidence.json"
  out="$dir/evidence/$kind-$stamp.summary.json"

  case "$kind" in
    model)
      # Default hash mode is "header": REA reads the metadata and tensor
      # directory only. This wrapper never asks for a full-file hash.
      "$rea" inspect-model-artifact "$target" \
        --detail summary --evidence_output "$ev" --json > "$out"
      rc=$?
      ;;
    gpu)
      size_mib=$(( $(stat -c %s -- "$target") / 1048576 ))
      if [ "$size_mib" -gt "$RE_MAX_GPU_FILE_MIB" ] && [ "${RE_ALLOW_LARGE:-0}" != "1" ]; then
        die "target is ${size_mib} MiB. REA hashes the whole file for this inspection. Limit is ${RE_MAX_GPU_FILE_MIB} MiB; set RE_ALLOW_LARGE=1 to override on purpose."
      fi
      "$rea" inspect-gpu-code-objects "$target" \
        --detail summary --evidence_output "$ev" --json > "$out"
      rc=$?
      ;;
    *) die "unknown inspection: $kind" ;;
  esac

  {
    printf '\n- %s `%s` on `%s`, exit %s, summary `evidence/%s`' \
      "$stamp" "$kind" "$target" "$rc" "$(basename "$out")"
    [ -f "$ev" ] && printf ', evidence `evidence/%s`' "$(basename "$ev")"
    printf '\n'
  } >> "$dir/EVIDENCE.md"

  if [ "$rc" -eq 0 ]; then
    say "Done. Summary:  $out"
    say "      Evidence: $ev"
    say "Write what you learned in $dir/EVIDENCE.md and update STATUS.md."
  else
    say "REA exited $rc. Its reply is in $out"
  fi
  return "$rc"
}

case "${1:-}" in
  check)   shift; cmd_check "$@" ;;
  install) shift; cmd_install "$@" ;;
  new)     shift; cmd_new "$@" ;;
  list)    shift; cmd_list "$@" ;;
  model)   shift; run_inspection model "$@" ;;
  gpu)     shift; run_inspection gpu "$@" ;;
  ''|-h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac
