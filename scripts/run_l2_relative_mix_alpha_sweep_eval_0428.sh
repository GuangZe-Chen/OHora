#!/usr/bin/env bash
set -euo pipefail

cd /data/xueyue.yang/OHORA/ohora

PY="/data/xueyue.yang/OHORA/venvs/ohora_cs_pure/bin/python"
EVAL_SCRIPT="evaluate/run_commonsense_evaluate.py"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
GPU_CANDIDATES="${GPU_CANDIDATES:-2 4 5 6 0 1 3 7}"
MIN_FREE_MB="${MIN_FREE_MB:-42000}"
MAX_UTIL="${MAX_UTIL:-25}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-120}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EVAL_MODE="${EVAL_MODE:-generate}"
EVAL_ROOT="${RUN_ROOT}/eval_commonsense_final_$(date +%Y%m%d_%H%M%S)"
PROGRESS_LOG="${EVAL_ROOT}/progress.log"
LOCK_DIR="${RUN_ROOT}/gpu_locks"

DATASETS=(
  boolq
  piqa
  social_i_qa
  hellaswag
  winogrande
  ARC-Challenge
  ARC-Easy
  openbookqa
)

MODELS=(
  "llama2_relative_mix_alpha0p3:${RUN_ROOT}/llama2_7b_lr7e-4_bs2_ga32_relative_scores_mix_alpha0p3_wrr0.01_e3/sft_lora_model"
  "llama2_relative_mix_alpha0p7:${RUN_ROOT}/llama2_7b_lr7e-4_bs2_ga32_relative_scores_mix_alpha0p7_wrr0.01_e3/sft_lora_model"
)

mkdir -p "${EVAL_ROOT}" "${LOCK_DIR}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${PROGRESS_LOG}"
}

cleanup_lock() {
  if [[ -n "${GPU_LOCK_PATH:-}" && -d "${GPU_LOCK_PATH}" ]]; then
    rmdir "${GPU_LOCK_PATH}" 2>/dev/null || true
  fi
}
trap cleanup_lock EXIT

pick_gpu() {
  while true; do
    for gpu in ${GPU_CANDIDATES}; do
      local used total util free lock_path
      IFS=, read -r used total util < <(
        nvidia-smi --id="${gpu}" \
          --query-gpu=memory.used,memory.total,utilization.gpu \
          --format=csv,noheader,nounits
      )
      used="$(echo "${used}" | xargs)"
      total="$(echo "${total}" | xargs)"
      util="$(echo "${util}" | xargs)"
      free=$((total - used))
      lock_path="${LOCK_DIR}/gpu${gpu}"
      if [[ "${free}" -ge "${MIN_FREE_MB}" && "${util}" -le "${MAX_UTIL}" ]]; then
        if mkdir "${lock_path}" 2>/dev/null; then
          GPU_LOCK_PATH="${lock_path}"
          echo "${gpu}"
          return 0
        fi
        log "wait gpu=${gpu} free=${free}MB util=${util}% reserved=1" >&2
      else
        log "wait gpu=${gpu} free=${free}MB util=${util}%" >&2
      fi
    done
    sleep "${CHECK_INTERVAL_SEC}"
  done
}

wait_for_models() {
  for item in "${MODELS[@]}"; do
    local name="${item%%:*}"
    local model_dir="${item#*:}"
    while [[ ! -f "${model_dir}/model.safetensors.index.json" ]]; do
      log "wait model=${name} dir=${model_dir}"
      sleep "${CHECK_INTERVAL_SEC}"
    done
    log "model ready name=${name} dir=${model_dir}"
  done
}

summarize_model() {
  local out_dir="$1"
  "${PY}" - "${out_dir}" <<'PY' | tee "${out_dir}/summary.csv"
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
print("dataset,n,correct,accuracy")
for path in sorted(root.glob("GPT-j-6B-ohora-*.json")):
    data = json.load(open(path))
    n = len(data)
    c = sum(1 for item in data if item.get("flag"))
    ds = path.name.replace("GPT-j-6B-ohora-", "").replace(".json", "")
    print(f"{ds},{n},{c},{c / n if n else 0:.6f}")
PY
}

eval_one_model() {
  local name="$1"
  local model_dir="$2"
  local out_dir="${EVAL_ROOT}/${name}"
  local gpu="$3"

  mkdir -p "${out_dir}/logs"
  log "start model=${name} gpu=${gpu} batch_size=${BATCH_SIZE} eval_mode=${EVAL_MODE} out=${out_dir}"

  for ds in "${DATASETS[@]}"; do
    local result_file="${out_dir}/GPT-j-6B-ohora-${ds}.json"
    local log_file="${out_dir}/logs/${ds}.log"
    if [[ -s "${result_file}" ]]; then
      log "skip existing model=${name} dataset=${ds}"
      continue
    fi

    log "start model=${name} dataset=${ds}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONPATH=/data/xueyue.yang/OHORA/ohora \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "${PY}" "${EVAL_SCRIPT}" \
        --dataset "${ds}" \
        --model GPT-j-6B \
        --base_model "${model_dir}" \
        --batch_size "${BATCH_SIZE}" \
        --gpu_id 0 \
        --eval_mode "${EVAL_MODE}" \
        --output_dir "${out_dir}" 2>&1 | tee -a "${log_file}"
    log "done model=${name} dataset=${ds}"
  done

  summarize_model "${out_dir}"
  log "done model=${name}"
}

log "eval watcher start root=${EVAL_ROOT}"
wait_for_models
gpu="$(pick_gpu)"
for item in "${MODELS[@]}"; do
  eval_one_model "${item%%:*}" "${item#*:}" "${gpu}"
done
log "eval finished"
