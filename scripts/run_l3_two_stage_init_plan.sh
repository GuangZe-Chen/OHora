#!/usr/bin/env bash
set -euo pipefail

# Two-stage plan for LLaMA-3:
# Stage A: 10-step recipe scan
#   1) bs=2, lr=5e-4, ga=8
#   2) bs=2, lr=7e-4, ga=8
#   3) bs=8, lr=5e-4, ga=8
# Stage B: use best Stage A recipe to compare init methods
#   classic / relative_scores / relative_scores_mix
#
# Usage:
#   bash scripts/run_l3_two_stage_init_plan.sh [GPU_ID]

GPU_ID="${1:-4}"
TS="$(date +%Y%m%d_%H%M%S)"
ROOT="/nas_data/xueyue.yang/ohora_runs/l3_two_stage_${TS}"
mkdir -p "${ROOT}"

PY="/data/xueyue.yang/OHORA/venvs/ohora_cs_pure/bin/python"
TRAIN="/data/xueyue.yang/OHORA/ohora/fine-tuning_commonse.py"
MODEL="/nas_data/xueyue.yang/hf_models/Meta-Llama-3-8B"
DATASET_DIR="/data/xueyue.yang/OHORA/ohora/datasets/commonsense_reasoning"
CACHE_DIR="/nas_data/xueyue.yang/hf_cache"
DATA_CACHE_DIR="/nas_data/xueyue.yang/ohora_runs/lr5e4_wr_steps_sweep/cache/datasets"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[INFO] Root: ${ROOT}"
echo "[INFO] GPU: ${GPU_ID}"

run_stage_a_one () {
  local name="$1"
  local bs="$2"
  local lr="$3"
  local ga="$4"
  local out_dir="${ROOT}/stageA_${name}"
  local log_file="${ROOT}/stageA_${name}.log"
  mkdir -p "${out_dir}"

  echo "[StageA] ${name}: bs=${bs}, lr=${lr}, ga=${ga}" | tee -a "${ROOT}/stageA_progress.log"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PY}" "${TRAIN}" \
    --model_name_or_path "${MODEL}" \
    --tokenizer_name_or_path "${MODEL}" \
    --dataset_dir "${DATASET_DIR}" \
    --cache_dir "${CACHE_DIR}" \
    --data_cache_dir "${DATA_CACHE_DIR}" \
    --output_dir "${out_dir}" \
    --do_train True --do_eval False --seed 42 \
    --per_device_train_batch_size "${bs}" --gradient_accumulation_steps "${ga}" \
    --max_steps 10 --num_train_epochs 3 --logging_steps 1 \
    --save_strategy no \
    --learning_rate "${lr}" --lr_scheduler_type cosine --warmup_steps 30 --warmup_ratio 0 \
    --max_seq_length 1024 --bf16 True --torch_dtype bfloat16 \
    --lora_rank 64 --lora_alpha 128 --lora_dropout 0.05 --lora_nums 0 \
    --trainable gate_proj,down_proj,up_proj,q_proj,k_proj,v_proj,o_proj \
    --use_ohora True --ohora_init_method classic --paper_repro_mode False \
    --load_in_kbits 16 --evaluation_strategy no --overwrite_output_dir \
    2>&1 | tee "${log_file}"
}

extract_last_loss () {
  local log_file="$1"
  "${PY}" - "$log_file" <<'PY'
import re, sys
p=sys.argv[1]
text=open(p,'r',errors='ignore').read()
vals=re.findall(r"\{'loss':\s*([0-9.]+)", text)
print(vals[-1] if vals else "N/A")
PY
}

run_stage_b_one () {
  local method="$1"
  local bs="$2"
  local lr="$3"
  local ga="$4"
  local extra="$5"
  local out_dir="${ROOT}/stageB_l3_${method}"
  local log_file="${ROOT}/stageB_l3_${method}.log"
  mkdir -p "${out_dir}"

  echo "[StageB] ${method}: bs=${bs}, lr=${lr}, ga=${ga} ${extra}" | tee -a "${ROOT}/stageB_progress.log"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PY}" "${TRAIN}" \
    --model_name_or_path "${MODEL}" \
    --tokenizer_name_or_path "${MODEL}" \
    --dataset_dir "${DATASET_DIR}" \
    --cache_dir "${CACHE_DIR}" \
    --data_cache_dir "${DATA_CACHE_DIR}" \
    --output_dir "${out_dir}" \
    --do_train True --do_eval False --seed 42 \
    --per_device_train_batch_size "${bs}" --gradient_accumulation_steps "${ga}" \
    --num_train_epochs 3 --logging_steps 10 --save_strategy epoch --save_total_limit 2 \
    --learning_rate "${lr}" --lr_scheduler_type cosine --warmup_steps 30 --warmup_ratio 0 \
    --max_seq_length 1024 --bf16 True --torch_dtype bfloat16 \
    --lora_rank 64 --lora_alpha 128 --lora_dropout 0.05 --lora_nums 0 \
    --trainable gate_proj,down_proj,up_proj,q_proj,k_proj,v_proj,o_proj \
    --use_ohora True --ohora_init_method "${method}" --paper_repro_mode False \
    --load_in_kbits 16 --evaluation_strategy no --overwrite_output_dir ${extra} \
    2>&1 | tee "${log_file}"
}

# ---------------- Stage A ----------------
run_stage_a_one bs2_lr5e4_ga8 2 5e-4 8 || true
run_stage_a_one bs2_lr7e4_ga8 2 7e-4 8 || true
run_stage_a_one bs8_lr5e4_ga8 8 5e-4 8 || true

L1="$(extract_last_loss "${ROOT}/stageA_bs2_lr5e4_ga8.log")"
L2="$(extract_last_loss "${ROOT}/stageA_bs2_lr7e4_ga8.log")"
L3="$(extract_last_loss "${ROOT}/stageA_bs8_lr5e4_ga8.log")"

echo "[StageA] last_loss bs2_lr5e4_ga8=${L1}" | tee -a "${ROOT}/stageA_summary.txt"
echo "[StageA] last_loss bs2_lr7e4_ga8=${L2}" | tee -a "${ROOT}/stageA_summary.txt"
echo "[StageA] last_loss bs8_lr5e4_ga8=${L3}" | tee -a "${ROOT}/stageA_summary.txt"

BEST_NAME=""
BEST_LOSS=""
BEST_BS=""
BEST_LR=""
BEST_GA=""

pick_best () {
  local name="$1" loss="$2" bs="$3" lr="$4" ga="$5"
  if [[ "$loss" == "N/A" || -z "$loss" ]]; then
    return
  fi
  if [[ -z "$BEST_LOSS" ]]; then
    BEST_NAME="$name"; BEST_LOSS="$loss"; BEST_BS="$bs"; BEST_LR="$lr"; BEST_GA="$ga"; return
  fi
  awk -v a="$loss" -v b="$BEST_LOSS" 'BEGIN{exit (a<b)?0:1}' && {
    BEST_NAME="$name"; BEST_LOSS="$loss"; BEST_BS="$bs"; BEST_LR="$lr"; BEST_GA="$ga"
  }
}

pick_best bs2_lr5e4_ga8 "$L1" 2 5e-4 8
pick_best bs2_lr7e4_ga8 "$L2" 2 7e-4 8
pick_best bs8_lr5e4_ga8 "$L3" 8 5e-4 8

if [[ -z "$BEST_NAME" ]]; then
  echo "[ERROR] StageA did not produce valid losses. Stop." | tee -a "${ROOT}/stageA_summary.txt"
  exit 1
fi

echo "[StageA] BEST=${BEST_NAME}, loss=${BEST_LOSS}, bs=${BEST_BS}, lr=${BEST_LR}, ga=${BEST_GA}" | tee -a "${ROOT}/stageA_summary.txt"

# ---------------- Stage B ----------------
run_stage_b_one classic "$BEST_BS" "$BEST_LR" "$BEST_GA" ""
run_stage_b_one relative_scores "$BEST_BS" "$BEST_LR" "$BEST_GA" ""
run_stage_b_one relative_scores_mix "$BEST_BS" "$BEST_LR" "$BEST_GA" "--ohora_mix_alpha 0.5"

echo "[DONE] Two-stage run finished. Root=${ROOT}"
