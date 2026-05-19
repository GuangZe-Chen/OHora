#!/usr/bin/env bash
set -euo pipefail

cd /data/xueyue.yang/OHORA/ohora

PY="/data/xueyue.yang/OHORA/venvs/ohora_cs_pure/bin/python"
TRAIN="/data/xueyue.yang/OHORA/ohora/fine-tuning_commonse.py"
EVAL_SCRIPT="evaluate/run_commonsense_evaluate.py"
DATASET_DIR="/data/xueyue.yang/OHORA/ohora/datasets/commonsense_reasoning"
CACHE_DIR="${CACHE_DIR:-/nas_data/xueyue.yang/hf_cache}"
RUN_ROOT="${RUN_ROOT:-/nas_data/xueyue.yang/ohora_runs/selection_component_24run_20260505}"
LOCK_DIR="${RUN_ROOT}/gpu_locks"
MODEL_KEY="${1:-}"
SEED="${2:-}"

GPU_CANDIDATES="${GPU_CANDIDATES:-7 4 2 6 1 3 0 5}"
MIN_FREE_MB="${MIN_FREE_MB:-42000}"
MAX_UTIL="${MAX_UTIL:-100}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-120}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
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

if [[ -z "${MODEL_KEY}" || -z "${SEED}" ]]; then
  echo "Usage: $0 {l2|l3} {seed}" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}" "${LOCK_DIR}"

if [[ "${MODEL_KEY}" == "l2" ]]; then
  MODEL="/nas_data/xueyue.yang/hf_models/Llama-2-7b-hf"
  WARMUP_STEPS=0
  WARMUP_RATIO=0.01
  PAPER_REPRO_MODE=True
  TRAIN_BS=2
  GA=4
  METHODS=(classic relative_scores relative_scores_mix classic_mix_rerank)
  BLOCK_NAME="l2_commonsense_seed${SEED}"
elif [[ "${MODEL_KEY}" == "l3" ]]; then
  MODEL="/nas_data/xueyue.yang/hf_models/Meta-Llama-3-8B"
  WARMUP_STEPS=30
  WARMUP_RATIO=0
  PAPER_REPRO_MODE=False
  TRAIN_BS=2
  GA=4
  METHODS=(classic classic_mix_rerank)
  BLOCK_NAME="l3_commonsense_seed${SEED}"
else
  echo "MODEL_KEY must be l2 or l3" >&2
  exit 2
fi

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

summarize_eval() {
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

latest_summary() {
  local method_tag="$1"
  find "${BLOCK_ROOT}" -type f \
    -path "*/eval_commonsense_${method_tag}_*/${method_tag}/summary.csv" \
    -print 2>/dev/null | sort | tail -n 1
}

run_one() {
  local method="$1"
  local alpha="$2"
  local method_tag="$3"
  local out_dir="${BLOCK_ROOT}/${method_tag}"
  local eval_root eval_out gpu done_summary

  mkdir -p "${out_dir}"
  done_summary="$(latest_summary "${method_tag}")"
  if [[ -n "${done_summary}" ]]; then
    log "skip existing eval model=${MODEL_KEY} seed=${SEED} method=${method} summary=${done_summary}"
    return 0
  fi

  gpu="$(pick_gpu)"
  if [[ -f "${out_dir}/sft_lora_model/model.safetensors.index.json" ]]; then
    log "train skip existing final model=${MODEL_KEY} seed=${SEED} method=${method} gpu=${gpu} out=${out_dir}"
  else
    log "train start model=${MODEL_KEY} seed=${SEED} method=${method} gpu=${gpu} out=${out_dir}"

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
        --per_device_train_batch_size "${TRAIN_BS}" \
        --gradient_accumulation_steps "${GA}" \
        --gradient_checkpointing True \
        --num_train_epochs 3 \
        --logging_steps 10 \
        --save_strategy steps \
        --save_steps 500 \
        --save_total_limit 2 \
        --learning_rate 7e-4 \
        --lr_scheduler_type cosine \
        --warmup_steps "${WARMUP_STEPS}" \
        --warmup_ratio "${WARMUP_RATIO}" \
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
        --paper_repro_mode "${PAPER_REPRO_MODE}" \
        --load_in_kbits 16 \
        --evaluation_strategy no \
        --overwrite_output_dir \
        2>&1 | tee "${out_dir}/train.log"
    log "train done model=${MODEL_KEY} seed=${SEED} method=${method} out=${out_dir}"
  fi

  if [[ ! -f "${out_dir}/sft_lora_model/model.safetensors.index.json" ]]; then
    log "missing final model model=${MODEL_KEY} seed=${SEED} method=${method}"
    cleanup_lock
    return 1
  fi

  eval_root="${BLOCK_ROOT}/eval_commonsense_${method_tag}_$(date +%Y%m%d_%H%M%S)"
  eval_out="${eval_root}/${method_tag}"
  mkdir -p "${eval_out}/logs"
  log "eval start model=${MODEL_KEY} seed=${SEED} method=${method} gpu=${gpu} out=${eval_out}"

  for ds in "${DATASETS[@]}"; do
    local result_file="${eval_out}/GPT-j-6B-ohora-${ds}.json"
    local log_file="${eval_out}/logs/${ds}.log"
    log "eval dataset=${ds}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONPATH=/data/xueyue.yang/OHORA/ohora \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "${PY}" "${EVAL_SCRIPT}" \
        --dataset "${ds}" \
        --model GPT-j-6B \
        --base_model "${out_dir}/sft_lora_model" \
        --batch_size "${EVAL_BATCH_SIZE}" \
        --gpu_id 0 \
        --eval_mode "${EVAL_MODE}" \
        --output_dir "${eval_out}" 2>&1 | tee -a "${log_file}"
    log "done dataset=${ds}"
  done

  summarize_eval "${eval_out}"
  log "run finished model=${MODEL_KEY} seed=${SEED} method=${method} eval_out=${eval_out}"
  cleanup_lock
}

trap cleanup_lock EXIT

for method in "${METHODS[@]}"; do
  alpha="0.5"
  case "${method}" in
    classic)
      method_tag="${MODEL_KEY}_classic_seed${SEED}_lr7e4_bs8eff_e3"
      ;;
    relative_scores)
      method_tag="${MODEL_KEY}_relative_seed${SEED}_lr7e4_bs8eff_e3"
      ;;
    relative_scores_mix)
      method_tag="${MODEL_KEY}_mix_alpha0p5_seed${SEED}_lr7e4_bs8eff_e3"
      ;;
    classic_mix_rerank)
      method_tag="${MODEL_KEY}_crr_alpha0p5_seed${SEED}_lr7e4_bs8eff_e3"
      ;;
    *)
      echo "Unsupported method ${method}" >&2
      exit 2
      ;;
  esac
  run_one "${method}" "${alpha}" "${method_tag}"
done

log "block finished ${BLOCK_NAME}"
