#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-/nas_data/xueyue.yang/ohora_runs/selection_component_24run_20260505}"
DEADLINE_LOCAL="${DEADLINE_LOCAL:-2026-05-14 23:50:00 Asia/Shanghai}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-60}"

SEED42_LOG="${RUN_ROOT}/l2_commonsense_seed42/progress.log"
SEED42_DONE_MARKER="run finished model=l2 seed=42 method=relative_scores "
SEED42_SESSION="sel24_l2cs_s42"
L2_SESSIONS=(sel24_l2cs_s42 sel24_l2cs_s43 sel24_l2cs_s44)

deadline_ts="$(TZ=Asia/Shanghai date -d "${DEADLINE_LOCAL}" +%s)"
seed42_stopped=0

log() {
  printf '[%s] %s\n' "$(TZ=Asia/Shanghai date '+%F %T %Z')" "$*"
}

kill_session_if_exists() {
  local session="$1"
  tmux has-session -t "${session}" 2>/dev/null || return 0
  tmux kill-session -t "${session}" 2>/dev/null || true
  log "killed session ${session}"
}

kill_all_l2() {
  local session
  for session in "${L2_SESSIONS[@]}"; do
    kill_session_if_exists "${session}"
  done
}

while true; do
  now_ts="$(TZ=Asia/Shanghai date +%s)"

  if [[ "${seed42_stopped}" -eq 0 ]] && [[ -f "${SEED42_LOG}" ]] && grep -Fq "${SEED42_DONE_MARKER}" "${SEED42_LOG}"; then
    kill_session_if_exists "${SEED42_SESSION}"
    seed42_stopped=1
    log "seed42 relative_scores finished; stopped remaining seed42 queue"
  fi

  if [[ "${now_ts}" -ge "${deadline_ts}" ]]; then
    log "deadline reached; stopping remaining L2 sessions"
    kill_all_l2
    exit 0
  fi

  sleep "${CHECK_INTERVAL_SEC}"
done
