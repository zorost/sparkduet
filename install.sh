#!/usr/bin/env bash
# install.sh, SparkDuet bootstrap. Run ON THE HEAD NODE, from the repo root.
#
#   git clone <repo> /srv/ai/sparkduet && cd /srv/ai/sparkduet && ./install.sh
#
# What it does, in order (idempotent, rerun safely):
#   1. detects your RoCE fabric (interfaces, RDMA devices, peer IP) and confirms
#   2. writes sparkduet.env from the template with your values
#   3. syncs the repo to the worker over the fabric
#   4. runs the doctor and the NCCL gate
#   5. offers to stage model weights
# It never starts serving by itself: that is an explicit `sparkduetctl.sh start`.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$ROOT/sparkduet.env"

say()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
ask()  { local p="$1" d="${2:-}"; local a; read -r -p "$p${d:+ [$d]}: " a; echo "${a:-$d}"; }

say "SparkDuet installer, head node bootstrap"
command -v docker >/dev/null || { echo "docker is required"; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "docker compose plugin is required"; exit 1; }

# --- 1. fabric detection ------------------------------------------------------
say "1/5 Detecting the RoCE fabric"
mapfile -t CAND < <(ip -o -4 addr show 2>/dev/null \
  | awk '$2 !~ /^(lo|docker|veth|br-|tailscale|wl)/ {print $2" "$4}' | sort -u)
printf '   %s\n' "${CAND[@]:-none-found}"
DEFAULT_IF=""; DEFAULT_IP=""
for c in "${CAND[@]}"; do
  # prefer 10.x point-to-point nets on ethernet-style names (the QSFP ports)
  if [[ "$c" == en* && "$c" == *" 10."* ]]; then
    DEFAULT_IF="${c%% *}"; DEFAULT_IP="${c##* }"; DEFAULT_IP="${DEFAULT_IP%%/*}"; break
  fi
done
HEAD_IF=$(ask "Head fabric interface(s), comma-separated for dual rail" "${DEFAULT_IF}")
HEAD_IP=$(ask "Head fabric IP (MASTER_ADDR)" "${DEFAULT_IP}")
GLOO_IF=$(ask "Gloo rendezvous rail (ONE interface, subnet of MASTER_ADDR)" "${HEAD_IF%%,*}")
WORKER_IP=$(ask "Worker fabric IP" "")
[[ -n "$WORKER_IP" ]] || { echo "worker IP is required"; exit 1; }
WORKER_SSH=$(ask "SSH target for the worker (alias or user@host)" "$WORKER_IP")
WORKER_USER_DEFAULT="${WORKER_SSH%%@*}"; [[ "$WORKER_SSH" == *@* ]] || WORKER_USER_DEFAULT="$USER"
WORKER_USER=$(ask "SSH user on the worker" "$WORKER_USER_DEFAULT")

if command -v ibv_devinfo >/dev/null 2>&1; then
  HCAS=$(ibv_devinfo -l 2>/dev/null | awk 'NR>1{print $1}' | paste -sd, -)
else
  HCAS=""
fi
NCCL_HCA=$(ask "RDMA devices for NCCL (comma-separated)" "${HCAS:-rocep1s0f1}")

# --- 2. env file ---------------------------------------------------------------
say "2/5 Writing sparkduet.env"
if [[ -f "$ENV_FILE" ]]; then
  cp "$ENV_FILE" "$ENV_FILE.bak-$(date -u +%Y%m%d%H%M)"
  echo "   existing env backed up"
fi
MODEL_DIR=$(ask "Local model weights directory (on both nodes)" "/srv/ai/models/llm")
HF_CACHE=$(ask "HF cache directory (on both nodes)" "/srv/ai/cache/huggingface")
SDIR=$(ask "Repo path on both nodes" "$ROOT")

sed -e "s|^MASTER_ADDR=.*|MASTER_ADDR=$HEAD_IP|" \
    -e "s|^WORKER_FABRIC_IP=.*|WORKER_FABRIC_IP=$WORKER_IP|" \
    -e "s|^WORKER_HOST=.*|WORKER_HOST=${WORKER_SSH#*@}|" \
    -e "s|^WORKER_USER=.*|WORKER_USER=$WORKER_USER|" \
    -e "s|^NCCL_IB_HCA=.*|NCCL_IB_HCA=$NCCL_HCA|" \
    -e "s|^NCCL_SOCKET_IFNAME=.*|NCCL_SOCKET_IFNAME=$HEAD_IF|" \
    -e "s|^GLOO_SOCKET_IFNAME=.*|GLOO_SOCKET_IFNAME=$GLOO_IF|" \
    -e "s|^MODEL_DIR=.*|MODEL_DIR=$MODEL_DIR|" \
    -e "s|^HF_CACHE=.*|HF_CACHE=$HF_CACHE|" \
    -e "s|^WORKER_HF_CACHE=.*|WORKER_HF_CACHE=$HF_CACHE|" \
    -e "s|^SPARKDUET_DIR=.*|SPARKDUET_DIR=$SDIR|" \
    "$ROOT/configs/sparkduet.env.example" > "$ENV_FILE"
chmod 600 "$ENV_FILE"
echo "   wrote $ENV_FILE (never commit this file)"

# --- 3. worker sync -------------------------------------------------------------
say "3/5 Syncing the repo to the worker"
ssh -o BatchMode=yes -o ConnectTimeout=8 "$WORKER_USER@${WORKER_SSH#*@}" "mkdir -p '$SDIR'" \
  || { echo "   ssh to the worker failed, fix key auth first (ssh-copy-id)"; exit 1; }
rsync -a --exclude .git "$ROOT/" "$WORKER_USER@${WORKER_SSH#*@}:$SDIR/"
echo "   synced"

# --- 4. gates --------------------------------------------------------------------
say "4/5 Doctor + NCCL gate"
bash "$ROOT/scripts/sparkduetctl.sh" doctor
if ask "Run the full NCCL all-reduce gate now (needs both GPUs idle)? y/n" "y" | grep -qi '^y'; then
  bash "$ROOT/scripts/nccl-check.sh" --full
fi

# --- 5. weights -------------------------------------------------------------------
say "5/5 Model weights"
echo "   stage now:   scripts/prepare-models.sh --model deepseek|qwen|both"
echo "   or sync:     scripts/prepare-models.sh --sync-worker <dir-under-MODEL_DIR>"
echo
say "Done. Next: ./scripts/sparkduetctl.sh start depth   (then scripts/warmup.sh)"
