#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ONE="${SCRIPT_DIR}/repro_l2_commonsense_one.sh"

SEEDS="${SEEDS:-42 43 44}"
METHODS="${METHODS:-classic relative_scores relative_scores_mix classic_mix_rerank}"
SESSION_PREFIX="${SESSION_PREFIX:-ohora_l2cs}"
USE_TMUX="${USE_TMUX:-1}"

if [[ -z "${MODEL_PATH:-}" ]]; then
  echo "Set MODEL_PATH=/path/to/Llama-2-7b-hf before launching." >&2
  exit 2
fi

if [[ "${USE_TMUX}" == "1" ]] && ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found. Set USE_TMUX=0 to run sequentially." >&2
  exit 2
fi

for seed in ${SEEDS}; do
  for method in ${METHODS}; do
    session="${SESSION_PREFIX}_s${seed}_${method}"
    if [[ "${USE_TMUX}" == "1" ]]; then
      if tmux has-session -t "${session}" 2>/dev/null; then
        echo "session exists, skipping: ${session}"
        continue
      fi
      tmux new-session -d -s "${session}" "${RUN_ONE} ${seed} ${method}"
      echo "started tmux session: ${session}"
    else
      "${RUN_ONE}" "${seed}" "${method}"
    fi
  done
done
