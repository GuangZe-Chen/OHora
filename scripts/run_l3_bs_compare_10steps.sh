#!/usr/bin/env bash
set -euo pipefail

# One-click 10-step loss comparison for LLaMA-3 with different batch sizes.
# Default: run bs=2 then bs=8 on GPU 4.
# Usage:
#   bash scripts/run_l3_bs_compare_10steps.sh
#   bash scripts/run_l3_bs_compare_10steps.sh 4

GPU_ID="${1:-1}"
TS="$(date +%Y%m%d_%H%M%S)"

PY="/data/xueyue.yang/OHORA/venvs/ohora_cs_pure/bin/python"
TRAIN="/data/xueyue.yang/OHORA/ohora/fine-tuning_commonse.py"

MODEL="/nas_data/xueyue.yang/hf_models/Meta-Llama-3-8B"
DATASET_DIR="/data/xueyue.yang/OHORA/ohora/datasets/commonsense_reasoning"
CACHE_DIR="/nas_data/xueyue.yang/hf_cache"
DATA_CACHE_DIR="/nas_data/xueyue.yang/ohora_runs/lr5e4_wr_steps_sweep/cache/datasets"

OUT_ROOT="/nas_data/xueyue.yang/ohora_runs/bs2_ohora_test_l3_100steps_${TS}"
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
    --max_steps 100 \
    --num_train_epochs 3 \
    --logging_steps 10 \
    --save_strategy no \
    --learning_rate "${lr}" \
    --lr_scheduler_type cosine \
    --warmup_steps 10 \
    --warmup_ratio 0 \
    --max_seq_length 1024 \
    --bf16 True \
    --torch_dtype bfloat16 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_nums 0 \
    --trainable gate_proj,down_proj,up_proj,q_proj,k_proj,v_proj,o_proj \
    --use_ohora True \
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
run_one 1e-3
run_one 2e-3
run_one 3e-3
run_one 4e-3
run_one 5e-3
run_one 6e-3
run_one 7e-3
run_one 8e-3

# bs=8 may OOM depending on current GPU usage; continue and report status.
if run_one 9e-3; then
  true
else
  echo "[WARN] bs=8 run failed (likely OOM or resource conflict)."
fi

Lr1_LOG="${OUT_ROOT}/lr1.log"
Lr2_LOG="${OUT_ROOT}/lr2.log"
Lr3_LOG="${OUT_ROOT}/lr3.log"
Lr4_LOG="${OUT_ROOT}/lr4.log"
Lr5_LOG="${OUT_ROOT}/lr5.log"
Lr6_LOG="${OUT_ROOT}/lr6.log"
Lr7_LOG="${OUT_ROOT}/lr7.log"
Lr8_LOG="${OUT_ROOT}/lr8.log"
Lr9_LOG="${OUT_ROOT}/lr1.log"
Lr2_LAST="$(extract_last_loss "${Lr2_LOG}")"
Lr3_LAST="$(extract_last_loss "${Lr3_LOG}")"
Lr4_LAST="$(extract_last_loss "${Lr4_LOG}")"
Lr5_LAST="$(extract_last_loss "${Lr5_LOG}")"
Lr6_LAST="$(extract_last_loss "${Lr6_LOG}")"
Lr7_LAST="$(extract_last_loss "${Lr7_LOG}")"
Lr8_LAST="$(extract_last_loss "${Lr8_LOG}")"
Lr9_LAST="N/A"
if [[ -f "${Lr9_LOG}" ]]; then
  BS8_LAST="$(extract_last_loss "${Lr9_LOG}")"
fi

echo "========== Loss Summary =========="
echo "lr=1e-3 last_loss: ${Lr1_LAST}"
echo "lr=2e-3 last_loss: ${Lr2_LAST}"
echo "lr=3e-3 last_loss: ${Lr3_LAST}"
echo "lr=4e-3 last_loss: ${Lr4_LAST}"
echo "lr=5e-3 last_loss: ${Lr5_LAST}"
echo "lr=6e-3 last_loss: ${Lr6_LAST}"
echo "lr=7e-3 last_loss: ${Lr7_LAST}"
echo "lr=8e-3 last_loss: ${Lr8_LAST}"
echo "lr=9e-3 last_loss: ${Lr9_LAST}"
echo "logs: ${OUT_ROOT}"
echo "=================================="
