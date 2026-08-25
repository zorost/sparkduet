#!/usr/bin/env bash
# nccl-check.sh, the go/no-go gate before any TP=2 launch.
#
# Lane D is only as good as the fabric under it. This proves, in order:
#   1. the RoCE link is up at both ends and jumbo-framed
#   2. the RDMA device is visible (ACTIVE port)
#   3. a real 2-node NCCL all-reduce runs INSIDE the serving image and
#      clears a minimum bus bandwidth
#
# Run on the head:  scripts/nccl-check.sh          (steps 1-2, fast)
#                   scripts/nccl-check.sh --full   (adds step 3, ~2 min)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${SPARKDUET_ENV:-$ROOT/sparkduet.env}"
set -a; source "$ENV_FILE"; set +a

MIN_BUSBW_GBS="${MIN_BUSBW_GBS:-8}"
pass=true
say(){ printf '%-52s %s\n' "$1" "$2"; }
ssh_worker(){ ssh -o BatchMode=yes -o ConnectTimeout=8 "${WORKER_USER}@${WORKER_HOST}" "$@"; }

# 1. link state + MTU, both ends, NCCL_SOCKET_IFNAME may be a comma list (dual rail)
IFS=',' read -ra IFACES <<< "$NCCL_SOCKET_IFNAME"
for side in head worker; do
  for ifc in "${IFACES[@]}"; do
    if [[ $side == head ]]; then out=$(ip -o link show "$ifc" 2>/dev/null || true)
    else out=$(ssh_worker "ip -o link show '$ifc'" 2>/dev/null || true); fi
    if [[ "$out" == *"state UP"* ]]; then say "fabric $ifc up ($side)" OK
    else say "fabric $ifc up ($side)" FAIL; pass=false; fi
    mtu=$(sed -n 's/.*mtu \([0-9]*\).*/\1/p' <<<"$out")
    if [[ "${mtu:-0}" -ge 4000 ]]; then say "MTU >= 4000 ($side $ifc: ${mtu:-none})" OK
    else say "MTU >= 4000 ($side $ifc: ${mtu:-none})" WARN; fi
  done
done

# 2. RDMA devices active, NCCL_IB_HCA may be a comma list; strip :port suffixes
IFS=',' read -ra HCAS <<< "$NCCL_IB_HCA"
for side in head worker; do
  for hca in "${HCAS[@]}"; do
    hca="${hca%%:*}"
    if [[ $side == head ]]; then st=$(ibv_devinfo -d "$hca" 2>/dev/null | grep -c 'PORT_ACTIVE' || true)
    else st=$(ssh_worker "ibv_devinfo -d '$hca' 2>/dev/null | grep -c PORT_ACTIVE" || true); fi
    if [[ "${st:-0}" -ge 1 ]]; then say "RDMA $hca PORT_ACTIVE ($side)" OK
    else say "RDMA $hca PORT_ACTIVE ($side)" FAIL; pass=false; fi
  done
done

# 3. optional: real all-reduce inside the serving image. The probe script rides
# the repo (synced to SPARKDUET_DIR on both nodes) and is bind-mounted in, which
# avoids the quoting swamp of inlining python through ssh + docker.
if [[ "${1:-}" == "--full" ]]; then
  echo ">> 2-node NCCL all-reduce inside $VLLM_IMAGE (256 MiB tensor, 20 iters)"
  SDIR="${SPARKDUET_DIR:-$ROOT}"
  run_rank() { # side rank
    # --device /dev/infiniband is NOT optional: without it NCCL cannot see the
    # RDMA verbs devices and silently falls back to TCP sockets at ~40% of the
    # RDMA rate. Measured here: 4.0 GB/s (sockets) vs 9–10 GB/s (RDMA).
    local dockercmd="docker run --rm --network host --gpus all --ipc host \
      --device /dev/infiniband --cap-add IPC_LOCK \
      -v '$SDIR':/sparkduet:ro \
      -e NCCL_IB_HCA='$NCCL_IB_HCA' -e NCCL_SOCKET_IFNAME='$NCCL_SOCKET_IFNAME' \
      -e NCCL_IB_GID_INDEX='$NCCL_IB_GID_INDEX' -e NCCL_DEBUG=WARN \
      -e MASTER_ADDR='$MASTER_ADDR' -e MASTER_PORT=29799 \
      -e RANK=$2 -e WORLD_SIZE=2 -e LOCAL_RANK=0 \
      --entrypoint python3 '$VLLM_IMAGE' /sparkduet/scripts/allreduce_probe.py"
    if [[ $1 == head ]]; then bash -c "$dockercmd"; else ssh_worker "$dockercmd"; fi
  }
  run_rank worker 1 >/tmp/nccl-worker.log 2>&1 &
  wpid=$!
  bw=$(run_rank head 0 2>/tmp/nccl-head.log | sed -n 's/^BUSBW_GBS=//p' || true)
  wait "$wpid" || true
  if [[ -n "$bw" ]] && awk "BEGIN{exit !($bw >= $MIN_BUSBW_GBS)}"; then
    say "all-reduce bus bandwidth ${bw} GB/s >= ${MIN_BUSBW_GBS}" OK
  else
    say "all-reduce bus bandwidth (${bw:-none}) >= ${MIN_BUSBW_GBS}" FAIL; pass=false
    echo "   (rank logs: /tmp/nccl-head.log here, /tmp/nccl-worker.log on the worker)"
  fi
fi

$pass && echo "NCCL-CHECK: PASS" || { echo "NCCL-CHECK: FAIL, do not launch Lane D"; exit 1; }
