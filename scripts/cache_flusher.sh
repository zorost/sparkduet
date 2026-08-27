#!/usr/bin/env bash
# GB10 NVRM allocates from MemFree, not MemAvailable. Weight load refills
# page cache and the KV slab then dies. Hold Cached under 40 GiB for 25 min.
set -u
end=$((SECONDS + 1500))
while [ "$SECONDS" -lt "$end" ]; do
  c=$(awk '/^Cached:/{print int($2/1048576)}' /proc/meminfo)
  if [ "${c:-0}" -gt 40 ]; then
    sync
    echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null || true
  fi
  sleep 5
done
