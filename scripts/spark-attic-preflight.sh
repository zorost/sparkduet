#!/usr/bin/env bash
# spark-attic-preflight.sh: read-only spark-attic preflight for one Spark.
#
# Runs the official spark-attic v0.1.6 installer in check mode and nothing
# else:
#     install.sh --check [--from <release tarball or unpacked artifact>]
# The installer's --check "writes nothing anywhere". This script does not
# install, does not pass --with-sysctl, does not pass --drm-dir, never reads or
# prints /etc/spark-attic/key, and touches no container. Its exit code is the
# installer's (0 clean, 1 problems, 2 usage).
#
# Usage, on the node itself:
#   scripts/spark-attic-preflight.sh [--installer PATH] [--from TGZ|DIR] [--plain] [--format json]
#
#   --installer PATH  a local copy of the v0.1.6 install.sh. Without it the
#                     script downloads the v0.1.6 release install.sh into a
#                     temporary directory. Either way the file must match
#                     SPARK_ATTIC_INSTALL_SHA256 (default: the digest of the
#                     v0.1.6 release install.sh, read 2026-11-19).
#   --from TGZ|DIR    spark-attic-0.1.6-aarch64.tar.gz or an unpacked artifact.
#                     Without it the shim and display checks cannot run and the
#                     installer only warns about them.
#
# See docs/SPARK-ATTIC.md for the gates and the results of 2026-11-19.
set -euo pipefail

VERSION=0.1.6
URL="https://github.com/bertholomus/spark-attic/releases/download/v${VERSION}/install.sh"
SHA256="${SPARK_ATTIC_INSTALL_SHA256:-28a6fe8e4e028d4a9069bfb8a4036756e3ee2684cf0aa44e033dda8af628165b}"

die() { echo "spark-attic-preflight: ERROR: $*" >&2; exit 2; }

installer="" from="" passthru=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --installer) installer="${2:?--installer needs a path}"; shift 2 ;;
    --from) from="${2:?--from needs a tarball or directory}"; shift 2 ;;
    --plain) passthru+=(--plain); shift ;;
    --format) [[ "${2:-}" == json ]] || die "--format takes only json"
              passthru+=(--format json); shift 2 ;;
    --with-sysctl|--drm-dir|--root|--demo|--fail|--url|--version)
      die "$1 is not passed by this read-only preflight" ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

tmp=""
cleanup() { [[ -n "$tmp" ]] && rm -rf "$tmp"; }
trap cleanup EXIT

if [[ -z "$installer" ]]; then
  tmp="$(mktemp -d)"
  installer="$tmp/install.sh"
  curl -fsSL --proto '=https' -o "$installer" "$URL" || die "could not download $URL"
fi
[[ -f "$installer" ]] || die "no installer at $installer"
echo "${SHA256}  ${installer}" | sha256sum -c --status \
  || die "$installer does not match the pinned v${VERSION} install.sh digest"
grep -Fq "VERSION=\${INSTALL_VERSION:-${VERSION}}" "$installer" \
  || die "$installer is not stamped v${VERSION}"
if [[ -n "$from" ]]; then
  [[ -e "$from" ]] || die "no artifact at $from"
  from="$(cd "$(dirname "$from")" && pwd)/$(basename "$from")"
fi

args=(--check)
[[ -n "$from" ]] && args+=(--from "$from")
args+=(${passthru[@]+"${passthru[@]}"})

echo "spark-attic-preflight: $(hostname) running install.sh v${VERSION} ${args[*]} (read-only)" >&2
sudo_cmd=()
[[ "$(id -u)" -eq 0 ]] || sudo_cmd=(sudo)
set +e
# The installer's test seams are cleared so the check reads the real node.
# ATTIC_OTHER_SOCKETS is kept: it names a second lender's socket (INSTALL.md).
${sudo_cmd[@]+"${sudo_cmd[@]}"} env -u INSTALL_VERSION -u ATTIC_URL_BASE -u ATTIC_VERSION_FILE \
  -u ATTIC_PROBE_ARGS -u ATTIC_MODES -u ATTIC_LIBCUDA \
  bash "$installer" "${args[@]}"
rc=$?
set -e
echo "spark-attic-preflight: installer exit code $rc" >&2
exit "$rc"
