#!/usr/bin/env bash
set -euo pipefail

cd /data/xueyue.yang/OHORA/ohora

PY="/data/xueyue.yang/OHORA/venvs/ohora_cs_pure/bin/python"
TRAIN="/data/xueyue.yang/OHORA/ohora/fine-tuning_commonse.py"
EVAL_SCRIPT="evaluate/run_commonsense_evaluate.py"
MODEL="/nas_data/xueyue.yang/hf_models/Llama-2-7b-hf"
DATASET_DIR="/data/xueyue.yang/OHORA/ohora/datasets/commonsense_reasoning"
CACHE_DIR="${CACHE_DIR:-/nas_data/xueyue.yang/hf_cache}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-${CACHE_DIR}/datasets}"

RUN_ROOT="${RUN_ROOT:-/nas_data/xueyue.yang/ohora_runs/l2_ohora_classic_vs_crr_20260504}"
PROGRESS_LOG="${RUN_ROOT}/progress.log"
LOCK_DIR="${RUN_ROOT}/gpu_locks"

GPU_CANDIDATES="${GPU_CANDIDATES:-7 2 4 6 1 3 0 5}"
MIN_FREE_MB="${MIN_FREE_MB:-42000}"
MAX_UTIL="${MAX_UTIL:-25}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-120}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EVAL_MODE="${EVAL_MODE:-generate}"

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

mkdir -p "${RUN_ROOT}" "${DATA_CACHE_DIR}" "${LOCK_DIR}"

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

method="${1:-}"
if [[ "${method}" != "classic" && "${method}" != "classic_mix_rerank" ]]; then
  echo "Usage: $0 {classic|classic_mix_rerank}" >&2
  exit 2
fi

if [[ "${method}" == "classic" ]]; then
  exp_name="llama2_7b_lr5e-4_bs4_ga8_ohora_classic_e3"
  mix_alpha="0.5"
else
  exp_name="llama2_7b_lr5e-4_bs4_ga8_ohora_classic_mix_rerank_alpha0p5_e3"
  mix_alpha="0.5"
fi

OUT_DIR="${RUN_ROOT}/${exp_name}"
mkdir -p "${OUT_DIR}"

gpu="$(pick_gpu)"
log "run start method=${method} gpu=${gpu} out=${OUT_DIR}"

if [[ ! -f "${OUT_DIR}/sft_lora_model/model.safetensors.index.json" ]]; then
  log "train start method=${method}"
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
      --per_device_train_batch_size 4 \
      --per_device_eval_batch_size 1 \
      --do_train True \
      --do_eval True \
      --seed 42 \
      --bf16 True \
      --num_train_epochs 3 \
      --lr_scheduler_type linear \
      --learning_rate 5e-4 \
      --warmup_ratio 0.01 \
      --weight_decay 0 \
      --logging_strategy steps \
      --logging_steps 10 \
      --save_strategy steps \
      --save_total_limit 1 \
      --evaluation_strategy no \
      --eval_steps 5000 \
      --save_steps 5000 \
      --gradient_accumulation_steps 8 \
      --max_seq_length 1024 \
      --output_dir "${OUT_DIR}" \
      --logging_first_step True \
      --lora_rank 16 \
      --lora_alpha 32 \
      --lora_nums 0 \
      --trainable gate_proj,down_proj,up_proj,q_proj,k_proj,v_proj,o_proj \
      --lora_dropout 0.05 \
      --torch_dtype bfloat16 \
      --use_ohora True \
      --ohora_init_method "${method}" \
      --ohora_mix_alpha "${mix_alpha}" \
      --load_in_kbits 16 \
      --overwrite_output_dir \
      2>&1 | tee "${OUT_DIR}/train.log"
  log "train done method=${method} out=${OUT_DIR}"
else
  log "skip train existing model=${OUT_DIR}/sft_lora_model"
fi

if [[ ! -f "${OUT_DIR}/sft_lora_model/model.safetensors.index.json" ]]; then
  log "missing final model after training method=${method} out=${OUT_DIR}"
  exit 1
fi

EVAL_ROOT="${RUN_ROOT}/eval_commonsense_final_$(date +%Y%m%d_%H%M%S)"
EVAL_OUT="${EVAL_ROOT}/${method}"
mkdir -p "${EVAL_OUT}/logs"
log "eval start method=${method} gpu=${gpu} out=${EVAL_OUT}"

for ds in "${DATASETS[@]}"; do
  result_file="${EVAL_OUT}/GPT-j-6B-ohora-${ds}.json"
  log_file="${EVAL_OUT}/logs/${ds}.log"
  if [[ -s "${result_file}" ]]; then
    log "skip existing dataset=${ds}"
    continue
  fi
  log "eval dataset=${ds}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  PYTHONPATH=/data/xueyue.yang/OHORA/ohora \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PY}" "${EVAL_SCRIPT}" \
      --dataset "${ds}" \
      --model GPT-j-6B \
      --base_model "${OUT_DIR}/sft_lora_model" \
      --batch_size "${BATCH_SIZE}" \
      --gpu_id 0 \
      --eval_mode "${EVAL_MODE}" \
      --output_dir "${EVAL_OUT}" 2>&1 | tee -a "${log_file}"
  log "done dataset=${ds}"
done

summarize_model "${EVAL_OUT}"
log "run finished method=${method} eval_out=${EVAL_OUT}"
