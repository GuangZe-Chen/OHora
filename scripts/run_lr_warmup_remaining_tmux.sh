#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-1}"
BASE_OUT="${2:-/nas_data/xueyue.yang/ohora_runs/lr_warmup_sweep_50_noguangze}"

PY="/data/xueyue.yang/OHORA/venvs/ohora_cs_pure/bin/python"
TRAIN_PY="/data/xueyue.yang/OHORA/ohora/fine-tuning_commonse.py"
MODEL="/nas_data/xueyue.yang/hf_models/Llama-2-7b-hf"
DATASET="/data/xueyue.yang/OHORA/ohora/datasets/commonsense_reasoning"
CACHE_DIR="/nas_data/xueyue.yang/hf_cache"
DATA_CACHE_DIR="${BASE_OUT}/cache/datasets"

mkdir -p "${BASE_OUT}" "${DATA_CACHE_DIR}"

run_one () {
  local tag="$1"
  local lr="$2"
  local wr="$3"
  local out_dir="${BASE_OUT}/${tag}"
  local log_file="${out_dir}/train.log"
  local rc=0

  if [[ -d "${out_dir}/checkpoint-50" ]]; then
    echo "===== SKIP ${tag} (checkpoint-50 exists) $(date '+%F %T') ====="
    return 0
  fi

  mkdir -p "${out_dir}"
  echo "===== START ${tag} lr=${lr} warmup_ratio=${wr} $(date '+%F %T') ====="

  set +e
  env CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${PY}" "${TRAIN_PY}" \
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
    --max_steps 50 \
    --logging_steps 5 \
    --save_steps 50 \
    --save_total_limit 1 \
    --learning_rate "${lr}" \
    --lr_scheduler_type cosine \
    --warmup_steps 0 \
    --warmup_ratio "${wr}" \
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

  if [[ ${rc} -ne 0 ]] && grep -qi "out of memory\|cuda out of memory" "${log_file}"; then
    echo "===== RETRY ${tag} after OOM $(date '+%F %T') =====" | tee -a "${log_file}"
    env CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${PY}" "${TRAIN_PY}" \
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
      --max_steps 50 \
      --logging_steps 5 \
      --save_steps 50 \
      --save_total_limit 1 \
      --learning_rate "${lr}" \
      --lr_scheduler_type cosine \
      --warmup_steps 0 \
      --warmup_ratio "${wr}" \
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
      2>&1 | tee -a "${log_file}"
    rc=${PIPESTATUS[0]}
  fi
  set -e

  if [[ ${rc} -ne 0 ]]; then
    echo "===== FAIL ${tag} rc=${rc} $(date '+%F %T') ====="
    return 0
  fi

  echo "===== END ${tag} lr=${lr} warmup_ratio=${wr} $(date '+%F %T') ====="
}

# Eight groups after A. Completed ones are auto-skipped.
run_one B_lr2e-4_wr0.01 2e-4 0.01
run_one C_lr2e-4_wr0.1  2e-4 0.1
run_one D_lr1e-4_wr0.0  1e-4 0.0
run_one E_lr1e-4_wr0.01 1e-4 0.01
run_one F_lr1e-4_wr0.1  1e-4 0.1
run_one G_lr5e-5_wr0.0  5e-5 0.0
run_one H_lr5e-5_wr0.01 5e-5 0.01
run_one I_lr5e-5_wr0.1  5e-5 0.1

echo "===== ALL REQUESTED RUNS FINISHED $(date '+%F %T') ====="
