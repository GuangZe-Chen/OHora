#!/usr/bin/env bash
set -euo pipefail

# One-click 10-step loss comparison for LLaMA-3 with different batch sizes.
# Default: run bs=2 then bs=8 on GPU 4.
# Usage:
#   bash scripts/run_l3_bs_compare_10steps.sh
#   bash scripts/run_l3_bs_compare_10steps.sh 4

GPU_ID="${1:-6}"
TS="$(date +%Y%m%d_%H%M%S)"

PY="/data/xueyue.yang/OHORA/venvs/ohora_cs_pure/bin/python"
TRAIN="/data/xueyue.yang/OHORA/ohora/fine-tuning_commonse.py"

MODEL="/nas_data/xueyue.yang/hf_models/Meta-Llama-3-8B"
DATASET_DIR="/data/xueyue.yang/OHORA/ohora/datasets/commonsense_reasoning"
CACHE_DIR="/nas_data/xueyue.yang/hf_cache"
DATA_CACHE_DIR="/nas_data/xueyue.yang/ohora_runs/lr5e4_wr_steps_sweep/cache/datasets"

OUT_ROOT="/nas_data/xueyue.yang/ohora_runs/ohora_test_steps_${TS}"
mkdir -p "${OUT_ROOT}"

run_one () {
  local lr="$1"
  local out_dir="${OUT_ROOT}/lr${lr}"
  local log_file="${OUT_ROOT}/lr${lr}.log"

  mkdir -p "${out_dir}"
  echo "[INFO] Start lr=${lr}, GPU=${GPU_ID}, out=${out_dir}"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
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
    --seed 42 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --save_strategy epoch \
    --save_total_limit 1 \
    --learning_rate 9e-4 \
    --lr_scheduler_type cosine \
    --warmup_steps 30 \
    --warmup_ratio 0 \
    --max_seq_length 1024 \
    --bf16 True \
    --torch_dtype bfloat16 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_nums 0 \
    --trainable gate_proj,down_proj,up_proj,q_proj,k_proj,v_proj,o_proj \
    --use_olora True \
    --paper_repro_mode False \
    --load_in_kbits 16 \
    --evaluation_strategy no \
    --overwrite_output_dir \
    2>&1 | tee "${log_file}"

  echo "[INFO] Done lr=${lr}. Log: ${log_file}"
}

extract_last_loss () {
  local log_file="$1"
  python - "$log_file" <<'PY'
import re, sys
p = sys.argv[1]
text = open(p, 'r', errors='ignore').read()
# Match lines like: {'loss': 0.1234, ...}
m = re.findall(r"\{'loss':\s*([0-9.]+)", text)
if m:
    print(m[-1])
else:
    print("N/A")
PY
}

echo "[INFO] Output root: ${OUT_ROOT}"
run_one 9e-4

# bs=8 may OOM depending on current GPU usage; continue and report status

Lr1_LOG="${OUT_ROOT}/lr1.log"
echo "========== Loss Summary =========="
echo "lr=9e-4 last_loss: ${Lr1_LAST}"
echo "logs: ${OUT_ROOT}"
echo "=================================="
