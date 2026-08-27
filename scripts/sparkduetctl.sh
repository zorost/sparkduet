#!/usr/bin/env bash
# sparkduetctl.sh, SparkDuet control plane.
#   start|stop|restart|switch|status|logs|doctor|bench|router|capture-incumbent
# Worker-first launch ordering (avoids the multi-node mp init race).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${SPARKDUET_ENV:-$ROOT/sparkduet.env}"
COMPOSE_DIR="$ROOT/configs"

die() { echo "sparkduetctl: ERROR: $*" >&2; exit 1; }

load_env() {
  [[ -f "$ENV_FILE" ]] || die "missing $ENV_FILE (cp configs/sparkduet.env.example sparkduet.env)"
  set -a; source "$ENV_FILE"; set +a
}

# --- numeric validation: decimal integers only (no octal, no floats, no CRLF).
# A malformed value here used to crash-loop a cluster (octal parse of "010").
validate_int() { # name value
  local name="$1" val="$2"
  [[ -z "$val" ]] && return 0
  [[ "$val" =~ ^[0-9]+$ ]] || die "$name must be a decimal integer, got '$val'"
  printf '%d' "$((10#$val))"   # 10# strips leading zeros (octal trap)
}

validate_config() {
  local v
  for knob in MASTER_PORT D_PORT D_MAX_MODEL_LEN D_MAX_NUM_SEQS D_MAX_NUM_BATCHED_TOKENS \
              D_LONG_PREFILL_TOKEN_THRESHOLD D_MTP_NUM_TOKENS \
              F_MAX_MODEL_LEN F_MAX_NUM_SEQS F_MTP_NUM_TOKENS F_PORT_A F_PORT_B \
              P_PROMPT_THRESHOLD P_MAX_MODEL_LEN P_DECODE_MAX_NUM_SEQS P_MTP_NUM_TOKENS \
              P_DECODE_PORT P_PREFILL_PORT ROUTER_PORT D_HIGH_WATER_SEQS \
              N_PORT N_MAX_MODEL_LEN N_MAX_NUM_SEQS N_MAX_NUM_BATCHED_TOKENS \
              N_LONG_PREFILL_TOKEN_THRESHOLD N_MTP_NUM_TOKENS \
              G_PORT G_MAX_MODEL_LEN G_MAX_NUM_SEQS G_MAX_NUM_BATCHED_TOKENS \
              G_LONG_PREFILL_TOKEN_THRESHOLD G_BLOCK_SIZE G_KV_CACHE_MEMORY \
              G_MTP_NUM_TOKENS \
              LANE_MAX_INFLIGHT SPEC_WINDOW_S SPEC_MIN_DRAFT_TOKENS NCCL_IB_GID_INDEX; do
    v="$(validate_int "$knob" "${!knob:-}")" || exit 1
    [[ -n "$v" ]] && printf -v "$knob" '%s' "$v"
  done
  case "${LANE_DEFAULT:-depth}" in depth|fleet|next|glm|split) ;; *) die "LANE_DEFAULT invalid";; esac
}

ssh_worker() { ssh -o BatchMode=yes -o ConnectTimeout=8 "${WORKER_USER}@${WORKER_HOST}" "$@"; }

compose_head()   { docker compose --env-file "$ENV_FILE" -f "$COMPOSE_DIR/$1" "${@:2}"; }
compose_worker() { ssh_worker "cd '${SPARKDUET_DIR:-$ROOT}' && docker compose --env-file sparkduet.env -f 'configs/$1' ${*:2}"; }

doctor() {
  load_env; validate_config
  echo "== sparkduet doctor =="
  command -v docker >/dev/null || die "docker missing"
  docker compose version >/dev/null 2>&1 || die "docker compose plugin missing"
  # management-plane stability probe: WiFi-reached clusters flap; catch it here,
  # not mid-benchmark (docs/FIELD-NOTES.md §1)
  local fails=0 i
  for i in 1 2 3; do ssh_worker true 2>/dev/null || fails=$((fails+1)); done
  [[ $fails -eq 0 ]] || echo "WARN: ssh to worker failed $fails/3 probes, fix the management path first"
  [[ $fails -lt 3 ]] || die "worker unreachable at ${WORKER_HOST}"
  # image presence + digest equality across nodes
  local lh lw
  lh=$(docker image inspect "$VLLM_IMAGE" --format '{{.Id}}' 2>/dev/null || true)
  lw=$(ssh_worker "docker image inspect '$VLLM_IMAGE' --format '{{.Id}}'" 2>/dev/null || true)
  [[ -n "$lh" ]] || die "head missing image $VLLM_IMAGE (docker pull it)"
  [[ -n "$lw" ]] || die "worker missing image $VLLM_IMAGE (docker pull it there)"
  [[ "$lh" == "$lw" ]] || die "image digest mismatch head=$lh worker=$lw"
  echo "image digest identical on both nodes: ${lh:0:19}…"
  # weights present where DS_MODEL points (local-path mode)
  if [[ "${DS_MODEL:0:1}" == "/" ]]; then
    [[ -f "$DS_MODEL/config.json" ]] || die "head missing weights at $DS_MODEL"
    ssh_worker "test -f '$DS_MODEL/config.json'" || die "worker missing weights at $DS_MODEL (run prepare-models.sh --sync-worker)"
    echo "weights present on both nodes: $DS_MODEL"
  fi
  if [[ "${N_MODEL:0:1}" == "/" ]]; then
    if [[ -f "$N_MODEL/config.json" ]]; then
      ssh_worker "test -f '$N_MODEL/config.json'" \
        || echo "WARN: worker missing Flash-Next weights at $N_MODEL (prepare-models.sh --model flash-next)"
      echo "Flash-Next weights present on head: $N_MODEL"
    else
      echo "WARN: Flash-Next weights not staged at $N_MODEL (prepare-models.sh --model flash-next)"
    fi
    local nh nw
    nh=$(docker image inspect "${N_VLLM_IMAGE:-}" --format '{{.Id}}' 2>/dev/null || true)
    nw=$(ssh_worker "docker image inspect '${N_VLLM_IMAGE:-}' --format '{{.Id}}'" 2>/dev/null || true)
    [[ -n "$nh" ]] || echo "WARN: head missing Flash-Next image ${N_VLLM_IMAGE:-unset}"
    [[ -n "$nw" ]] || echo "WARN: worker missing Flash-Next image ${N_VLLM_IMAGE:-unset}"
  fi
  if [[ "${G_MODEL:0:1}" == "/" ]]; then
    if [[ -f "$G_MODEL/config.json" ]]; then
      ssh_worker "test -f '$G_MODEL/config.json'" \
        || echo "WARN: worker missing GLM-5.3-Flash weights at $G_MODEL (prepare-models.sh --model glm-flash)"
      echo "GLM-5.3-Flash weights present on head: $G_MODEL"
    else
      echo "WARN: GLM-5.3-Flash weights not staged at $G_MODEL (prepare-models.sh --model glm-flash)"
    fi
    local gh gw
    gh=$(docker image inspect "${G_VLLM_IMAGE:-}" --format '{{.Id}}' 2>/dev/null || true)
    gw=$(ssh_worker "docker image inspect '${G_VLLM_IMAGE:-}' --format '{{.Id}}'" 2>/dev/null || true)
    [[ -n "$gh" ]] || echo "WARN: head missing GLM image ${G_VLLM_IMAGE:-unset}"
    [[ -n "$gw" ]] || echo "WARN: worker missing GLM image ${G_VLLM_IMAGE:-unset}"
  fi
  # fabric: sysfs is always present on the host; ibstat usually is not
  if compgen -G "/sys/class/infiniband/*/ports/*/state" >/dev/null; then
    grep -l ACTIVE /sys/class/infiniband/*/ports/*/state >/dev/null 2>&1 \
      || echo "WARN: no RDMA port is ACTIVE (RoCE fabric down?)"
  else
    ibstat 2>/dev/null | grep -q "LinkUp" || echo "WARN: no RDMA devices visible (RoCE fabric down?)"
  fi
  # memory truth per node (identical boxes are not identical, measure, don't assume)
  echo "head   MemAvailable: $(awk '/MemAvailable/{print int($2/1048576)" GiB"}' /proc/meminfo)"
  echo "worker MemAvailable: $(ssh_worker "awk '/MemAvailable/{print int(\$2/1048576)\" GiB\"}' /proc/meminfo")"
  df -h --output=avail "$MODEL_DIR" 2>/dev/null | tail -1 | xargs -I{} echo "head model disk avail: {}"
  # silent GPU-fallback probe: a CUDA-built llama.cpp/ggml backend that lost its
  # device KEEPS SERVING from CPU at ~1/7th speed (docs/RESEARCH.md §10.3).
  local c
  for c in $(docker ps --format '{{.Names}}'); do
    if docker logs "$c" --since 24h 2>&1 | grep -q "ggml_cuda_init: failed"; then
      echo "WARN: container $c logged a CUDA init failure, it is likely serving on CPU. Restart it."
    fi
  done
  echo "doctor: OK"
}

lane_port() { case "$1" in depth) echo "$D_PORT";; fleet) echo "$F_PORT_A";; next) echo "${N_PORT:-$D_PORT}";; glm) echo "${G_PORT:-$D_PORT}";; split) echo "$P_DECODE_PORT";; esac; }

capture_running_state() { # automatic before every start: the revert plan
  local out="$ROOT/results/incumbent-$(date -u +%Y%m%d-%H%M)"
  mkdir -p "$out"
  docker ps --format '{{.Names}}' > "$out/head-running.txt"
  while read -r name; do
    [[ "$name" == sparkduet-* ]] && continue
    docker inspect "$name" > "$out/head-$name.json" 2>/dev/null || true
  done < "$out/head-running.txt"
  ssh_worker "docker ps --format '{{.Names}}'" > "$out/worker-running.txt" 2>/dev/null || true
  echo "incumbent state captured → $out"
}

revert() { # stop ours, docker-start anything from the newest capture that is down
  load_env
  local cap
  cap=$(ls -d "$ROOT"/results/incumbent-* 2>/dev/null | sort | tail -1) \
    || die "no incumbent capture found under results/"
  echo "reverting to $cap"
  stop_all
  while read -r name; do
    [[ -z "$name" || "$name" == sparkduet-* ]] && continue
    docker ps --format '{{.Names}}' | grep -qx "$name" \
      || { echo "starting displaced container: $name"; docker start "$name" || true; }
  done < "$cap/head-running.txt"
  if [[ -f "$cap/worker-running.txt" ]]; then
    while read -r name; do
      [[ -z "$name" || "$name" == sparkduet-* ]] && continue
      ssh_worker "docker ps --format '{{.Names}}' | grep -qx '$name' || docker start '$name'" || true
    done < "$cap/worker-running.txt"
  fi
  echo "revert complete, verify with: sparkduetctl.sh status"
}

capture_incumbent() { # container-name → revert spec under results/
  load_env
  local name="${1:?usage: capture-incumbent <container>}"
  local out="$ROOT/results/incumbent-$name-$(date -u +%Y%m%d).json"
  docker inspect "$name" > "$out" || die "no container named $name"
  echo "captured $name spec → $out (this is your revert plan; stop, never remove)"
}

start_depth() { # worker rank first, then head
  compose_worker lane-depth.compose.yml up -d depth-worker
  sleep 5
  compose_head   lane-depth.compose.yml up -d depth-head
}
start_next() { # worker rank first, then head. Same fabric as depth.
  [[ -f "${N_MODEL}/config.json" ]] || die "Flash-Next weights missing at $N_MODEL (prepare-models.sh --model flash-next)"
  local nh nw
  nh=$(docker image inspect "${N_VLLM_IMAGE}" --format '{{.Id}}' 2>/dev/null || true)
  nw=$(ssh_worker "docker image inspect '${N_VLLM_IMAGE}' --format '{{.Id}}'" 2>/dev/null || true)
  [[ -n "$nh" ]] || die "head missing image $N_VLLM_IMAGE (docker pull it)"
  [[ -n "$nw" ]] || die "worker missing image $N_VLLM_IMAGE (docker pull it there)"
  compose_worker lane-next.compose.yml up -d next-worker
  sleep 5
  compose_head   lane-next.compose.yml up -d next-head
}
start_glm() { # worker rank first, then head. Same fabric as depth.
  [[ -f "${G_MODEL}/config.json" ]] || die "GLM-5.3-Flash weights missing at $G_MODEL (prepare-models.sh --model glm-flash)"
  local gh gw
  gh=$(docker image inspect "${G_VLLM_IMAGE}" --format '{{.Id}}' 2>/dev/null || true)
  gw=$(ssh_worker "docker image inspect '${G_VLLM_IMAGE}' --format '{{.Id}}'" 2>/dev/null || true)
  [[ -n "$gh" ]] || die "head missing image $G_VLLM_IMAGE (docker pull it)"
  [[ -n "$gw" ]] || die "worker missing image $G_VLLM_IMAGE (docker pull it there)"
  # GB10: NVRM needs MemFree for the KV slab. Weight load refills page cache.
  sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' || true
  ssh_worker "sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'" || true
  mkdir -p "$ROOT/results"
  nohup "$ROOT/scripts/cache_flusher.sh" >>"$ROOT/results/cache-flusher.log" 2>&1 &
  ssh_worker "mkdir -p '${SPARKDUET_DIR:-$ROOT}/results'; nohup '${SPARKDUET_DIR:-$ROOT}/scripts/cache_flusher.sh' >>'${SPARKDUET_DIR:-$ROOT}/results/cache-flusher.log' 2>&1 &" || true
  compose_worker lane-glm.compose.yml up -d glm-worker
  sleep 20
  compose_head   lane-glm.compose.yml up -d glm-head
}
start_fleet() {
  # Lane F profile resolution. Fit rule: one-node models only (see the compose header).
  local model="${QWEN_MODEL}" name="${QWEN_SERVED_NAME}"
  if [[ "${F_MODEL:-qwen}" != qwen ]]; then model="$F_MODEL"; name="${F_SERVED_NAME:-sparkduet-fleet-custom}"; fi
  # Env prefixes do not cross SSH: the worker compose must get the computed
  # values inline or it boots `vllm serve` with a blank model and crash-loops.
  ssh_worker "cd '${SPARKDUET_DIR:-$ROOT}' && FLEET_MODEL_ID='$model' FLEET_SERVED_NAME='$name' FLEET_PORT='$F_PORT_B' \
    docker compose --env-file sparkduet.env -f 'configs/lane-fleet.compose.yml' up -d fleet-replica"
  FLEET_MODEL_ID="$model" FLEET_SERVED_NAME="$name" FLEET_PORT="$F_PORT_A" \
    compose_head   lane-fleet.compose.yml up -d fleet-replica
}
start_split() {
  compose_worker lane-pd.compose.yml up -d sparkduet-prefill
  compose_head   lane-pd.compose.yml up -d sparkduet-decode
}

start_router() {
  pgrep -f "scripts/router.py" >/dev/null && { echo "router already up"; return 0; }
  mkdir -p "$ROOT/results"
  nohup python3 "$ROOT/scripts/router.py" >>"$ROOT/results/router.log" 2>&1 &
  echo "router on :${ROUTER_PORT}"
}

refresh_pickers() {
  # 9Router lists whatever :30000 is serving. This pushes that name into
  # OpenCode, Chat, Hermes, and dsh so a swap is the house model everywhere.
  if [[ -x /usr/local/bin/zorost-apply-house-catalog.py ]]; then
    sudo -n python3 /usr/local/bin/zorost-apply-house-catalog.py \
      || python3 /usr/local/bin/zorost-apply-house-catalog.py \
      || echo "WARN: pickers not refreshed; the 10-minute timer will follow"
  fi
}

wait_ready() { # port [timeout-steps]
  local port="$1" n=0 max="${2:-240}" looping
  echo "waiting for :$port (multi-node load can take 10–20 min on first boot)"
  until curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; do
    looping=$(docker ps --filter "name=sparkduet-" --format '{{.Names}} {{.Status}}' | grep Restarting || true)
    if [[ -n "$looping" ]]; then
      echo "lane is restart-looping:" >&2
      echo "$looping" >&2
      docker logs --tail 30 sparkduet-next-head 2>&1 | tail -20 >&2 || true
      docker logs --tail 20 sparkduet-glm-head 2>&1 | tail -10 >&2 || true
      die "lane crash-looped while waiting for :$port"
    fi
    n=$((n+1)); [[ $n -gt $max ]] && die "timeout waiting for :$port"; sleep 15
  done
  curl -fsS "http://127.0.0.1:${port}/v1/models" && echo
}

stop_all() {
  pkill -f "scripts/router.py" 2>/dev/null || true
  pkill -f "scripts/specadvisor.py" 2>/dev/null || true
  for f in lane-depth lane-next lane-glm lane-fleet lane-pd; do
    compose_head "$f.compose.yml" down 2>/dev/null || true
    compose_worker "$f.compose.yml" down 2>/dev/null || true
  done
}

cmd="${1:-}"; arg="${2:-depth}"
case "$cmd" in
  start)  load_env; validate_config; doctor; capture_running_state
          "start_$arg"; port="$(lane_port "$arg")"; wait_ready "$port"
          [[ "${ROUTER_ENABLE:-0}" == 1 ]] && start_router
          if [[ "$arg" == depth ]]; then
            D_PORT="$D_PORT" DS_SERVED_NAME="$DS_SERVED_NAME" D_MAX_NUM_SEQS="$D_MAX_NUM_SEQS" \
              bash "$ROOT/scripts/warmup.sh"
          elif [[ "$arg" == next ]]; then
            D_PORT="${N_PORT:-$D_PORT}" DS_SERVED_NAME="$N_SERVED_NAME" D_MAX_NUM_SEQS="$N_MAX_NUM_SEQS" \
              bash "$ROOT/scripts/warmup.sh"
          elif [[ "$arg" == glm ]]; then
            D_PORT="${G_PORT:-$D_PORT}" DS_SERVED_NAME="$G_SERVED_NAME" D_MAX_NUM_SEQS="$G_MAX_NUM_SEQS" \
              bash "$ROOT/scripts/warmup.sh"
          else
            D_PORT="$port" DS_SERVED_NAME="${QWEN_SERVED_NAME}" D_MAX_NUM_SEQS="${F_MAX_NUM_SEQS}" \
              bash "$ROOT/scripts/warmup.sh"
          fi
          refresh_pickers;;
  stop)   load_env; stop_all;;
  restart) load_env; stop_all; sleep 3; "$0" start "$arg";;
  revert) revert;;
  switch) load_env; validate_config
          echo "switching to lane: $arg (drain-aware)"
          if curl -fsS -X POST "http://127.0.0.1:${ROUTER_PORT}/admin/drain" >/dev/null 2>&1; then
            for _ in $(seq 1 12); do  # wait up to 60 s for inflight to reach zero
              busy=$(curl -fsS "http://127.0.0.1:${ROUTER_PORT}/admin/health" 2>/dev/null \
                     | python3 -c 'import json,sys; print(sum(json.load(sys.stdin)["inflight"].values()))' 2>/dev/null || echo 0)
              [[ "${busy:-0}" == "0" ]] && break; sleep 5
            done
          fi
          stop_all; sleep 3; "$0" start "$arg"
          curl -fsS -X POST "http://127.0.0.1:${ROUTER_PORT}/admin/undrain" >/dev/null 2>&1 || true;;
  status) load_env
          curl -fsS "http://127.0.0.1:${D_PORT}/v1/models" 2>/dev/null \
            | python3 -c 'import json,sys
for m in json.load(sys.stdin)["data"]:
    print("lane D serving: %s (ctx %s)" % (m["id"], m.get("max_model_len", "?")))' \
            2>/dev/null || echo "lane D endpoint down"
          docker ps --format 'table {{.Names}}\t{{.Status}}' | grep -E 'sparkduet|NAMES' || true
          ssh_worker "docker ps --format 'table {{.Names}}\t{{.Status}}' | grep -E 'sparkduet|NAMES'" || true;;
  logs)   load_env; docker logs -f "sparkduet-${arg}-head" --tail 200 2>/dev/null \
            || docker logs -f "sparkduet-${arg}" --tail 200;;
  doctor) doctor;;
  router) load_env; validate_config; start_router;;
  bench)  load_env; python3 "$ROOT/scripts/bench.py" --lane "$arg" --suite "${3:-standard}" \
            --base-url "http://127.0.0.1:${D_PORT}/v1" --model "$DS_SERVED_NAME" \
            --output "$ROOT/results";;
  capture-incumbent) load_env
          if [[ -n "${2:-}" ]]; then capture_incumbent "$2"; else capture_running_state; fi;;
  *)      die "usage: sparkduetctl.sh start|stop|restart|revert|switch|status|logs|doctor|router|bench|capture-incumbent [depth|fleet|next|glm|split]";;
esac
