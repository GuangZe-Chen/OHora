#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

PYTHON_BIN="/data/xueyue.yang/OHORA/venvs/ohora_cs_pure/bin/python"
MODEL_PATH="/nas_data/xueyue.yang/guangze/hf_models/Llama-2-7b-hf"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/datasets/commonsense_reasoning}"
RUN_ROOT="/nas_data/xueyue.yang/guangze/ohora_runs/target_modules_stage1"
CACHE_ROOT="/nas_data/xueyue.yang/guangze/hf_cache"
DATA_CACHE_ROOT="${RUN_ROOT}/cache/datasets"

QUICK_EVAL_MODE="${QUICK_EVAL_MODE:-true}"
if [[ "${QUICK_EVAL_MODE}" == "true" ]]; then
  DATASETS=(boolq piqa ARC-Challenge openbookqa)
else
  DATASETS=(boolq piqa social_i_qa ARC-Challenge ARC-Easy openbookqa hellaswag winogrande)
fi

mkdir -p "${RUN_ROOT}" "${DATA_CACHE_ROOT}"

calc_quick_summary() {
  local eval_dir="$1"
  local model_name="$2"
  "${PYTHON_BIN}" - <<PY
import json, os
eval_dir = "${eval_dir}"
model_name = "${model_name}"
name_map = [
    ("BoolQ", "LLaMA-7B-ohora-boolq.json"),
    ("PIQA", "LLaMA-7B-ohora-piqa.json"),
    ("ARC-C", "LLaMA-7B-ohora-ARC-Challenge.json"),
    ("OBQA", "LLaMA-7B-ohora-openbookqa.json"),
]
vals = []
for label, fn in name_map:
    p = os.path.join(eval_dir, fn)
    if not os.path.exists(p):
        vals.append((label, None))
        continue
    data = json.load(open(p, "r"))
    n = len(data)
    c = sum(1 for x in data if bool(x.get("flag", False)))
    vals.append((label, (100.0 * c / n) if n else 0.0))
avail = [v for _, v in vals if v is not None]
avg_q = (sum(avail) / len(avail)) if avail else 0.0
parts = [model_name] + [f"{v:.2f}" if v is not None else "NA" for _, v in vals] + [f"{avg_q:.2f}"]
line = " | ".join(parts)
with open(os.path.join(eval_dir, "quick_summary.txt"), "w") as f:
    f.write("model | BoolQ | PIQA | ARC-C | OBQA | AVG_quick\n")
    f.write(line + "\n")
print(line)
PY
}

run_eval_suite() {
  local model_path="$1"
  local eval_dir="$2"
  local exp_name="$3"

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

  if [[ "${QUICK_EVAL_MODE}" == "true" ]]; then
    calc_quick_summary "${eval_dir}" "${exp_name}"
  fi
}

run_experiment() {
  local exp_name="$1"
  local trainable="$2"
  local output_dir="${RUN_ROOT}/${exp_name}"
  local train_log="${output_dir}/train.log"
  local eval_dir="${output_dir}/eval"

  mkdir -p "${output_dir}" "${eval_dir}"

  echo "===== TRAIN ${exp_name} $(date '+%F %T') =====" | tee -a "${RUN_ROOT}/orchestrator_quick.log"
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

  echo "===== EVAL ${exp_name} $(date '+%F %T') =====" | tee -a "${RUN_ROOT}/orchestrator_quick.log"
  run_eval_suite "${output_dir}/sft_lora_model" "${eval_dir}" "${exp_name}"
  echo "===== DONE ${exp_name} $(date '+%F %T') =====" | tee -a "${RUN_ROOT}/orchestrator_quick.log"
}

: > "${RUN_ROOT}/orchestrator_quick.log"
run_experiment "B_qkvo" "q_proj,k_proj,v_proj,o_proj"
run_experiment "C_mlp_attn" "gate_proj,down_proj,up_proj,q_proj,k_proj,v_proj,o_proj"
