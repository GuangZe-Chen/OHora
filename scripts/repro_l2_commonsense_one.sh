#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${REPO_ROOT}/fine-tuning_commonse.py}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${REPO_ROOT}/evaluate/run_commonsense_evaluate.py}"
MODEL_PATH="${MODEL_PATH:-}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/datasets/commonsense_reasoning}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/l2_commonsense_selection}"
CACHE_DIR="${CACHE_DIR:-${RUN_ROOT}/cache/hf}"
LOCK_DIR="${RUN_ROOT}/gpu_locks"

SEED="${1:-}"
METHOD="${2:-}"

GPU_CANDIDATES="${GPU_CANDIDATES:-0}"
MIN_FREE_MB="${MIN_FREE_MB:-42000}"
MAX_UTIL="${MAX_UTIL:-100}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-120}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
EVAL_MODE="${EVAL_MODE:-generate}"
FORCE_OVERWRITE="${FORCE_OVERWRITE:-0}"

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

usage() {
  cat >&2 <<USAGE
Usage: MODEL_PATH=/path/to/Llama-2-7b-hf $0 {seed} {classic|relative_scores|relative_scores_mix|classic_mix_rerank}

Environment:
  RUN_ROOT           Output root. Default: ${REPO_ROOT}/runs/l2_commonsense_selection
  DATASET_DIR        Training data dir. Default: ${REPO_ROOT}/datasets/commonsense_reasoning
  GPU_CANDIDATES     Space-separated GPU ids. Default: 0
  FORCE_OVERWRITE=1  Delete existing train/eval dirs for this run before starting.
USAGE
}

if [[ -z "${SEED}" || -z "${METHOD}" || -z "${MODEL_PATH}" ]]; then
  usage
  exit 2
fi

case "${METHOD}" in
  classic)
    METHOD_TAG="l2_classic_seed${SEED}_lr7e4_bs8eff_e3"
    ;;
  relative_scores)
    METHOD_TAG="l2_relative_seed${SEED}_lr7e4_bs8eff_e3"
    ;;
  relative_scores_mix)
    METHOD_TAG="l2_mix_alpha0p5_seed${SEED}_lr7e4_bs8eff_e3"
    ;;
  classic_mix_rerank)
    METHOD_TAG="l2_crr_alpha0p5_seed${SEED}_lr7e4_bs8eff_e3"
    ;;
  *)
    usage
    exit 2
    ;;
esac

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH does not exist: ${MODEL_PATH}" >&2
  exit 2
fi

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "DATASET_DIR does not exist: ${DATASET_DIR}" >&2
  exit 2
fi

BLOCK_NAME="l2_commonsense_seed${SEED}"
BLOCK_ROOT="${RUN_ROOT}/${BLOCK_NAME}"
OUT_DIR="${BLOCK_ROOT}/${METHOD_TAG}"
DATA_CACHE_DIR="${BLOCK_ROOT}/cache/datasets"
PROGRESS_LOG="${BLOCK_ROOT}/progress.log"

mkdir -p "${RUN_ROOT}" "${LOCK_DIR}" "${BLOCK_ROOT}" "${DATA_CACHE_DIR}"

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
  local eval_out="$1"
  "${PYTHON_BIN}" - "${eval_out}" <<'PY' | tee "${eval_out}/summary.csv"
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

trap cleanup_lock EXIT

if [[ "${FORCE_OVERWRITE}" == "1" ]]; then
  rm -rf "${OUT_DIR}"
  find "${BLOCK_ROOT}" -maxdepth 1 -type d -name "eval_commonsense_${METHOD_TAG}_*" -exec rm -rf {} +
fi

if [[ -e "${OUT_DIR}" ]]; then
  echo "Output already exists: ${OUT_DIR}" >&2
  echo "Use FORCE_OVERWRITE=1 for a clean rerun, or choose a different RUN_ROOT." >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"
GPU="$(pick_gpu)"

log "train start seed=${SEED} method=${METHOD} gpu=${GPU} out=${OUT_DIR} resume=none"
CUDA_VISIBLE_DEVICES="${GPU}" \
OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
    --model_name_or_path "${MODEL_PATH}" \
    --tokenizer_name_or_path "${MODEL_PATH}" \
    --dataset_dir "${DATASET_DIR}" \
    --cache_dir "${CACHE_DIR}" \
    --data_cache_dir "${DATA_CACHE_DIR}" \
    --output_dir "${OUT_DIR}" \
    --do_train True \
    --do_eval False \
    --seed "${SEED}" \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing True \
    --num_train_epochs 3 \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 7e-4 \
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
    --ohora_init_method "${METHOD}" \
    --ohora_mix_alpha 0.5 \
    --paper_repro_mode True \
    --load_in_kbits 16 \
    --evaluation_strategy no \
    --overwrite_output_dir \
    2>&1 | tee "${OUT_DIR}/train.log"

log "train done seed=${SEED} method=${METHOD} out=${OUT_DIR}"

if [[ ! -f "${OUT_DIR}/sft_lora_model/model.safetensors.index.json" ]]; then
  log "missing final model seed=${SEED} method=${METHOD} out=${OUT_DIR}"
  exit 1
fi

EVAL_ROOT="${BLOCK_ROOT}/eval_commonsense_${METHOD_TAG}_$(date +%Y%m%d_%H%M%S)"
EVAL_OUT="${EVAL_ROOT}/${METHOD_TAG}"
mkdir -p "${EVAL_OUT}/logs"
log "eval start seed=${SEED} method=${METHOD} gpu=${GPU} out=${EVAL_OUT}"

for ds in "${DATASETS[@]}"; do
  log "eval dataset=${ds}"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  PYTHONPATH="${REPO_ROOT}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PYTHON_BIN}" "${EVAL_SCRIPT}" \
      --dataset "${ds}" \
      --model GPT-j-6B \
      --base_model "${OUT_DIR}/sft_lora_model" \
      --batch_size "${EVAL_BATCH_SIZE}" \
      --gpu_id 0 \
      --eval_mode "${EVAL_MODE}" \
      --output_dir "${EVAL_OUT}" 2>&1 | tee -a "${EVAL_OUT}/logs/${ds}.log"
  log "done dataset=${ds}"
done

summarize_eval "${EVAL_OUT}"
log "run finished seed=${SEED} method=${METHOD} eval_out=${EVAL_OUT}"
