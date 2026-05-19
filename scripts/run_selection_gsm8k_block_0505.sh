#!/usr/bin/env bash
set -euo pipefail

cd /data/xueyue.yang/OHORA/ohora

PY="/data/xueyue.yang/OHORA/venvs/ohora_cs_pure/bin/python"
TRAIN="/data/xueyue.yang/OHORA/ohora/fine-tuning.py"
EVAL_SCRIPT="/data/xueyue.yang/OHORA/ohora/evaluate/run_gsm8k_eval.py"
MODEL="/nas_data/xueyue.yang/hf_models/Llama-2-7b-hf"
DATASET_DIR="/data/xueyue.yang/OHORA/ohora/datasets/task_ft/gsm8k"
CACHE_DIR="${CACHE_DIR:-/nas_data/xueyue.yang/hf_cache}"
RUN_ROOT="${RUN_ROOT:-/nas_data/xueyue.yang/ohora_runs/selection_component_24run_20260505}"
LOCK_DIR="${RUN_ROOT}/gpu_locks"
SEED="${1:-}"

GPU_CANDIDATES="${GPU_CANDIDATES:-7 4 2 6 1 3 0 5}"
MIN_FREE_MB="${MIN_FREE_MB:-42000}"
MAX_UTIL="${MAX_UTIL:-100}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-120}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"

if [[ -z "${SEED}" ]]; then
  echo "Usage: $0 {seed}" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}" "${LOCK_DIR}"

BLOCK_NAME="l2_gsm8k_seed${SEED}"
BLOCK_ROOT="${RUN_ROOT}/${BLOCK_NAME}"
DATA_CACHE_DIR="${BLOCK_ROOT}/cache/datasets"
PROGRESS_LOG="${BLOCK_ROOT}/progress.log"
mkdir -p "${BLOCK_ROOT}" "${DATA_CACHE_DIR}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${PROGRESS_LOG}"
}

cleanup_lock() {
  if [[ -n "${GPU_LOCK_PATH:-}" && -d "${GPU_LOCK_PATH}" ]]; then
    rmdir "${GPU_LOCK_PATH}" 2>/dev/null || true
  fi
  GPU_LOCK_PATH=""
}

pick_gpu() {
  cleanup_lock
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
      fi
      log "wait gpu=${gpu} free=${free}MB util=${util}%" >&2
    done
    sleep "${CHECK_INTERVAL_SEC}"
  done
}

latest_eval_log() {
  local method_tag="$1"
  find "${BLOCK_ROOT}" -type f \
    -path "*/eval_gsm8k_${method_tag}_*/${method_tag}/eval.log" \
    -print 2>/dev/null | sort | tail -n 1
}

run_one() {
  local method="$1"
  local alpha="$2"
  local method_tag="$3"
  local out_dir="${BLOCK_ROOT}/${method_tag}"
  local eval_root eval_out gpu done_eval

  mkdir -p "${out_dir}"
  done_eval="$(latest_eval_log "${method_tag}")"
  if [[ -n "${done_eval}" ]]; then
    log "skip existing eval task=gsm8k seed=${SEED} method=${method} eval_log=${done_eval}"
    return 0
  fi

  gpu="$(pick_gpu)"
  if [[ -f "${out_dir}/sft_lora_model/model.safetensors.index.json" ]]; then
    log "train skip existing final task=gsm8k seed=${SEED} method=${method} gpu=${gpu} out=${out_dir}"
  else
    log "train start task=gsm8k seed=${SEED} method=${method} gpu=${gpu} out=${out_dir}"

    CUDA_VISIBLE_DEVICES="${gpu}" \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "${PY}" "${TRAIN}" \
        --model_name_or_path "${MODEL}" \
        --tokenizer_name_or_path "${MODEL}" \
        --dataset_dir "${DATASET_DIR}" \
        --cache_dir "${CACHE_DIR}" \
        --data_cache_dir "${DATA_CACHE_DIR}" \
        --output_dir "${out_dir}" \
        --do_train True \
        --do_eval False \
        --seed "${SEED}" \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 3 \
        --gradient_checkpointing True \
        --num_train_epochs 1 \
        --logging_steps 10 \
        --save_strategy epoch \
        --save_total_limit 1 \
        --learning_rate 3e-4 \
        --lr_scheduler_type cosine \
        --warmup_steps 0 \
        --warmup_ratio 0.01 \
        --max_seq_length 1024 \
        --bf16 True \
        --torch_dtype bfloat16 \
        --lora_rank 64 \
        --lora_alpha 128 \
        --lora_dropout 0.05 \
        --lora_nums 0 \
        --trainable gate_proj,down_proj,up_proj,q_proj,k_proj,v_proj,o_proj \
        --use_ohora True \
        --ohora_init_method "${method}" \
        --ohora_mix_alpha "${alpha}" \
        --load_in_kbits 16 \
        --evaluation_strategy no \
        --overwrite_output_dir \
        2>&1 | tee "${out_dir}/train.log"
    log "train done task=gsm8k seed=${SEED} method=${method} out=${out_dir}"
  fi

  if [[ ! -f "${out_dir}/sft_lora_model/model.safetensors.index.json" ]]; then
    log "missing final model task=gsm8k seed=${SEED} method=${method}"
    cleanup_lock
    return 1
  fi

  eval_root="${BLOCK_ROOT}/eval_gsm8k_${method_tag}_$(date +%Y%m%d_%H%M%S)"
  eval_out="${eval_root}/${method_tag}"
  mkdir -p "${eval_out}"
  log "eval start task=gsm8k seed=${SEED} method=${method} gpu=${gpu} out=${eval_out}"

  CUDA_VISIBLE_DEVICES="${gpu}" \
  "${PY}" "${EVAL_SCRIPT}" \
    --base_model "${out_dir}/sft_lora_model" \
    --output_dir "${eval_out}" \
    --batch_size "${EVAL_BATCH_SIZE}" 2>&1 | tee "${eval_out}/eval.log"

  log "run finished task=gsm8k seed=${SEED} method=${method} eval_out=${eval_out}"
  cleanup_lock
}

trap cleanup_lock EXIT

run_one "classic" "0.5" "l2_gsm8k_classic_seed${SEED}_lr3e4_bs3_e1"
run_one "classic_mix_rerank" "0.5" "l2_gsm8k_crr_alpha0p5_seed${SEED}_lr3e4_bs3_e1"

log "block finished ${BLOCK_NAME}"
