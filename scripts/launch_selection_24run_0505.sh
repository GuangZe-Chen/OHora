#!/usr/bin/env bash
set -euo pipefail

cd /data/xueyue.yang/OHORA/ohora

RUN_ROOT="${RUN_ROOT:-/nas_data/xueyue.yang/ohora_runs/selection_component_24run_20260505}"
mkdir -p "${RUN_ROOT}"

launch_session() {
  local session="$1"
  local cmd="$2"
  tmux new-session -d -s "${session}" "bash -lc '${cmd}'"
}

for seed in 42 43 44; do
  launch_session \
    "sel24_l2cs_s${seed}" \
    "cd /data/xueyue.yang/OHORA/ohora && RUN_ROOT=${RUN_ROOT} ./scripts/run_selection_commonsense_block_0505.sh l2 ${seed}"
  launch_session \
    "sel24_l3cs_s${seed}" \
    "cd /data/xueyue.yang/OHORA/ohora && RUN_ROOT=${RUN_ROOT} ./scripts/run_selection_commonsense_block_0505.sh l3 ${seed}"
  launch_session \
    "sel24_l2gsm_s${seed}" \
    "cd /data/xueyue.yang/OHORA/ohora && RUN_ROOT=${RUN_ROOT} ./scripts/run_selection_gsm8k_block_0505.sh ${seed}"
done

echo "RUN_ROOT=${RUN_ROOT}"
tmux ls
