#!/usr/bin/env bash
set -euo pipefail

cd /data/xueyue.yang/guangze/ohora

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

PYTHON_BIN="/data/xueyue.yang/guangze/venvs/ohora_cs_pure/bin/python"
MODEL_PATH="/nas_data/xueyue.yang/guangze/hf_models/Llama-2-7b-hf"
DATASET_DIR="/data/xueyue.yang/guangze/ohora/datasets/commonsense_reasoning"
RUN_ROOT="/nas_data/xueyue.yang/guangze/ohora_runs/target_modules_stage1"
CACHE_ROOT="/nas_data/xueyue.yang/guangze/hf_cache"
DATA_CACHE_ROOT="${RUN_ROOT}/cache/datasets"

mkdir -p "${RUN_ROOT}" "${DATA_CACHE_ROOT}"

DATASETS=(boolq piqa social_i_qa ARC-Challenge ARC-Easy openbookqa hellaswag winogrande)

run_eval_suite() {
  local model_path="$1"
  local eval_dir="$2"

  mkdir -p "${eval_dir}"
  : > "${eval_dir}/run_evalfix.log"

  for ds in "${DATASETS[@]}"; do
    echo "===== START ${ds} $(date '+%F %T') =====" | tee -a "${eval_dir}/run_evalfix.log"
    "${PYTHON_BIN}" evaluate/run_commonsense_evaluate.py \
      --model LLaMA-7B \
      --base_model "${model_path}" \
      --dataset "${ds}" \
      --batch_size 4 \
      --output_dir "${eval_dir}" 2>&1 | tee -a "${eval_dir}/run_evalfix.log"
    echo "===== END ${ds} $(date '+%F %T') =====" | tee -a "${eval_dir}/run_evalfix.log"
  done
}

run_experiment() {
  local exp_name="$1"
  local trainable="$2"
  local output_dir="${RUN_ROOT}/${exp_name}"
  local train_log="${output_dir}/train.log"
  local eval_dir="${output_dir}/eval"

  mkdir -p "${output_dir}" "${eval_dir}"

  echo "===== TRAIN ${exp_name} $(date '+%F %T') =====" | tee -a "${RUN_ROOT}/orchestrator.log"
  "${PYTHON_BIN}" fine-tuning_commonse.py \
    --model_name_or_path "${MODEL_PATH}" \
    --tokenizer_name_or_path "${MODEL_PATH}" \
    --dataset_dir "${DATASET_DIR}" \
    --cache_dir "${CACHE_ROOT}" \
    --data_cache_dir "${DATA_CACHE_ROOT}" \
    --output_dir "${output_dir}" \
    --do_train True \
    --do_eval False \
    --seed 42 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_steps 1000 \
    --logging_steps 20 \
    --save_steps 250 \
    --save_total_limit 2 \
    --learning_rate 2e-4 \
    --lr_scheduler_type linear \
    --warmup_steps 0 \
    --max_seq_length 1024 \
    --bf16 True \
    --torch_dtype bfloat16 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_nums 0 \
    --trainable "${trainable}" \
    --use_ohora True \
    --paper_repro_mode True \
    --load_in_kbits 16 \
    --overwrite_output_dir 2>&1 | tee "${train_log}"

  echo "===== EVAL ${exp_name} $(date '+%F %T') =====" | tee -a "${RUN_ROOT}/orchestrator.log"
  run_eval_suite "${output_dir}/sft_lora_model" "${eval_dir}"
  echo "===== DONE ${exp_name} $(date '+%F %T') =====" | tee -a "${RUN_ROOT}/orchestrator.log"
}

: > "${RUN_ROOT}/orchestrator.log"

run_experiment "A_qv" "q_proj,v_proj"
run_experiment "B_qkvo" "q_proj,k_proj,v_proj,o_proj"
run_experiment "C_mlp_attn" "gate_proj,down_proj,up_proj,q_proj,k_proj,v_proj,o_proj"