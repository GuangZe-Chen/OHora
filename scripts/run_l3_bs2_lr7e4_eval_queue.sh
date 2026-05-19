#!/usr/bin/env bash
set -euo pipefail

cd /data/xueyue.yang/OHORA/ohora

PY="/data/xueyue.yang/OHORA/venvs/ohora_cs_pure/bin/python"
EVAL_SCRIPT="evaluate/run_commonsense_evaluate.py"

RUN_ROOT="/nas_data/xueyue.yang/ohora_runs/l3_bs2_lr7e4_dual/20260422_113751"
GPU_ID="${1:-2}"
BATCH_SIZE="${2:-4}"
EVAL_ROOT="${RUN_ROOT}/eval_commonsense_$(date +%Y%m%d_%H%M%S)"
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
  "classic:${RUN_ROOT}/gpu7_bs2_classic/sft_lora_model"
  "relative_scores:${RUN_ROOT}/gpu6_bs2_relative_scores/sft_lora_model"
  "relative_scores_mix:${RUN_ROOT}/gpu6_bs2_relative_scores_mix/sft_lora_model"
)

mkdir -p "${EVAL_ROOT}/logs"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${PROGRESS_LOG}"
}

wait_for_model() {
  local name="$1"
  local model_dir="$2"

  while [[ ! -f "${model_dir}/model.safetensors.index.json" ]]; do
    log "wait model=${name} dir=${model_dir}"
    sleep 300
  done
}

eval_one_model() {
  local name="$1"
  local model_dir="$2"
  local out_dir="${EVAL_ROOT}/${name}"

  wait_for_model "${name}" "${model_dir}"
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

  log "done model=${name}"
}

log "eval queue start root=${EVAL_ROOT}"
for item in "${MODELS[@]}"; do
  name="${item%%:*}"
  model_dir="${item#*:}"
  eval_one_model "${name}" "${model_dir}"
done
log "eval queue finished"
