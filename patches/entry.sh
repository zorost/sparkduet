#!/usr/bin/env bash
# Lane D entrypoint: apply the container-start hotfixes, then exec vllm serve.
#
# Each hotfix is idempotent (safe on restart) and anchored to exact source
# strings in the pinned image. If an anchor is missing (a new engine image),
# the patch reports and the engine boots unpatched rather than failing the
# lane; check the container log for "[patches]" lines after an image bump.
set -u
for p in /sparkduet-patches/hotfix-*.py; do
  [ -e "$p" ] || continue
  python3 "$p" || echo "[patches] $p did not apply; engine boots unpatched" >&2
done
exec vllm serve "$@"
