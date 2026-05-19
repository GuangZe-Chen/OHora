#!/usr/bin/env bash
set -euo pipefail

# Auto scheduler for lr=5e-4 with exactly 5 experiments:
# 1) warmup_ratio=0.01
# 2) warmup_ratio=0.1
# 3) warmup_steps=30
# 4) warmup_steps=60
# 5) warmup_steps=100
# It waits until a GPU is sufficiently idle before launching each run.

BASE_OUT="${1:-/nas_data/xueyue.yang/ohora_runs/lr5e4_wr_steps_sweep}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-90}"
MIN_FREE_MB="${MIN_FREE_MB:-60000}"
MAX_UTIL="${MAX_UTIL:-25}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"

PY="/data/xueyue.yang/OHORA/venvs/ohora_cs_pure/bin/python"
TRAIN_PY="/data/xueyue.yang/OHORA/ohora/fine-tuning_commonse.py"
MODEL="/nas_data/xueyue.yang/hf_models/Llama-2-7b-hf"
DATASET="/data/xueyue.yang/OHORA/ohora/datasets/commonsense_reasoning"
CACHE_DIR="/nas_data/xueyue.yang/hf_cache"
DATA_CACHE_DIR="${BASE_OUT}/cache/datasets"

mkdir -p "${BASE_OUT}" "${DATA_CACHE_DIR}"
MASTER_LOG="${BASE_OUT}/autoschedule.log"

echo "===== AUTOSCHED START $(date '+%F %T') =====" | tee -a "${MASTER_LOG}"
echo "CHECK_INTERVAL_SEC=${CHECK_INTERVAL_SEC} MIN_FREE_MB=${MIN_FREE_MB} MAX_UTIL=${MAX_UTIL} NUM_EPOCHS=${NUM_EPOCHS}" | tee -a "${MASTER_LOG}"

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits \
    | awk -F',' -v min_free="${MIN_FREE_MB}" -v max_util="${MAX_UTIL}" '
      {
        gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3);
        idx=$1; free=$2+0; util=$3+0;
        if (free>=min_free && util<=max_util) {
          if (free>best_free) { best_free=free; best_idx=idx; found=1; }
        }
      }
      END {
        if (found==1) print best_idx; else print "";
      }'
}

wait_for_gpu() {
  local gpu=""
  while true; do
    gpu="$(pick_gpu)"
    if [[ -n "${gpu}" ]]; then
      echo "${gpu}"
      return 0
    fi
    echo "[$(date '+%F %T')] no idle gpu yet, sleep ${CHECK_INTERVAL_SEC}s" | tee -a "${MASTER_LOG}" >&2
    sleep "${CHECK_INTERVAL_SEC}"
  done
}

run_one() {
  local tag="$1"
  local warmup_steps="$2"
  local warmup_ratio="$3"
  local out_dir="${BASE_OUT}/${tag}"
  local log_file="${out_dir}/train.log"
  local rc=0
  local gpu=""

  if [[ -f "${out_dir}/train_results.json" ]]; then
    echo "===== SKIP ${tag} (train_results.json exists) $(date '+%F %T') =====" | tee -a "${MASTER_LOG}"
    return 0
  fi

  mkdir -p "${out_dir}"
  gpu="$(wait_for_gpu)"
  echo "===== START ${tag} gpu=${gpu} lr=5e-4 warmup_steps=${warmup_steps} warmup_ratio=${warmup_ratio} num_epochs=${NUM_EPOCHS} $(date '+%F %T') =====" | tee -a "${MASTER_LOG}"

  set +e
  env CUDA_VISIBLE_DEVICES="${gpu}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${PY}" "${TRAIN_PY}" \
    --model_name_or_path "${MODEL}" \
    --tokenizer_name_or_path "${MODEL}" \
    --dataset_dir "${DATASET}" \
    --cache_dir "${CACHE_DIR}" \
    --data_cache_dir "${DATA_CACHE_DIR}" \
    --output_dir "${out_dir}" \
    --do_train True \
    --do_eval False \
    --seed 42 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs "${NUM_EPOCHS}" \
    --logging_steps 5 \
    --save_strategy epoch \
    --save_total_limit 1 \
    --learning_rate 5e-4 \
    --lr_scheduler_type cosine \
    --warmup_steps "${warmup_steps}" \
    --warmup_ratio "${warmup_ratio}" \
    --max_seq_length 1024 \
    --bf16 True \
    --torch_dtype bfloat16 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_nums 0 \
    --trainable gate_proj,down_proj,up_proj,q_proj,k_proj,v_proj,o_proj \
    --use_ohora True \
    --paper_repro_mode True \
    --load_in_kbits 16 \
    --evaluation_strategy no \
    --overwrite_output_dir \
    2>&1 | tee "${log_file}"
  rc=${PIPESTATUS[0]}
  set -e

  if [[ ${rc} -ne 0 ]]; then
    echo "===== FAIL ${tag} rc=${rc} $(date '+%F %T') =====" | tee -a "${MASTER_LOG}"
    return 0
  fi

  echo "===== END ${tag} $(date '+%F %T') =====" | tee -a "${MASTER_LOG}"
}

# Ratio-based experiments (warmup_steps=0)
run_one M_lr5e-4_wrr0.01 0 0.01
run_one N_lr5e-4_wrr0.1 0 0.1

# Warmup-steps-based experiments (warmup_ratio=0)
run_one O_lr5e-4_wrs30 30 0
run_one P_lr5e-4_wrs60 60 0
run_one Q_lr5e-4_wrs100 100 0

echo "===== AUTOSCHED ALL DONE $(date '+%F %T') =====" | tee -a "${MASTER_LOG}"
