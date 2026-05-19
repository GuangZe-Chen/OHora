#!/usr/bin/env bash
set -euo pipefail

cd /data/xueyue.yang/OHORA/ohora

PY="/data/xueyue.yang/OHORA/venvs/ohora_cs_pure/bin/python"
EVAL_SCRIPT="evaluate/run_commonsense_evaluate.py"
RUN_ROOT="/nas_data/xueyue.yang/ohora_runs/requested_l2_l3_commonsense_bs2_lr7e4_20260425_113527"
GPU_ID="${GPU_ID:-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EVAL_ROOT="${RUN_ROOT}/eval_commonsense_final_$(date +%Y%m%d_%H%M%S)"
PROGRESS_LOG="${EVAL_ROOT}/progress.log"

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
  "llama2_bs2_lr7e4_ga32:${RUN_ROOT}/llama2_7b_lr7e-4_bs2_wrr0.01_e3/sft_lora_model"
  "llama3_bs2_lr7e4_ga32:${RUN_ROOT}/llama3_8b_lr7e-4_bs2_ws30_e3/sft_lora_model"
)

mkdir -p "${EVAL_ROOT}/logs"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${PROGRESS_LOG}"
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

  if [[ ! -d "${model_dir}" ]]; then
    log "missing model=${name} dir=${model_dir}"
    return 1
  fi

  mkdir -p "${out_dir}/logs"
  log "start model=${name} gpu=${GPU_ID} batch_size=${BATCH_SIZE} out=${out_dir}"

  for ds in "${DATASETS[@]}"; do
    local result_file="${out_dir}/GPT-j-6B-ohora-${ds}.json"
    local log_file="${out_dir}/logs/${ds}.log"

    if [[ -s "${result_file}" ]]; then
      log "skip existing model=${name} dataset=${ds}"
      continue
    fi

    log "start model=${name} dataset=${ds}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    PYTHONPATH=/data/xueyue.yang/OHORA/ohora \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "${PY}" "${EVAL_SCRIPT}" \
        --dataset "${ds}" \
        --model GPT-j-6B \
        --base_model "${model_dir}" \
        --batch_size "${BATCH_SIZE}" \
        --gpu_id 0 \
        --output_dir "${out_dir}" 2>&1 | tee -a "${log_file}"
    log "done model=${name} dataset=${ds}"
  done

  summarize_model "${out_dir}"
  log "done model=${name}"
}

log "eval start root=${EVAL_ROOT}"
for item in "${MODELS[@]}"; do
  name="${item%%:*}"
  model_dir="${item#*:}"
  eval_one_model "${name}" "${model_dir}"
done
log "eval finished"
