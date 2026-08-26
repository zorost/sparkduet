#!/usr/bin/env bash
# Lane D entrypoint: apply the container-start hotfixes, then exec vllm serve.
#
# Each hotfix is idempotent (safe on restart) and anchored to exact source
# strings in the pinned image. If an anchor is missing (a new engine image),
# the patch reports and the engine boots unpatched rather than failing the
# lane; check the container log for "[patches]" lines after an image bump.
set -u
# JIT-cache and flight-recorder paths live on the persisted cache volume;
# create them up front (a missing dump dir turns the on-timeout dump into a
# silent no-op, which defeats the point of having it).
mkdir -p /root/.cache/vllm/jit/triton /root/.cache/vllm/jit/tilelang \
         /root/.cache/vllm/jit/b12x-cute /root/.cache/vllm/nccl-flight
for p in /sparkduet-patches/hotfix-*.py; do
  [ -e "$p" ] || continue
  python3 "$p" || echo "[patches] $p did not apply; engine boots unpatched" >&2
done
exec vllm serve "$@"
