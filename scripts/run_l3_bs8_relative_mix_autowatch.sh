#!/usr/bin/env bash
set -euo pipefail

# Auto-watch GPUs and launch two queued LLaMA-3 OHoRA runs with bs=8:
# 1) relative_scores
# 2) relative_scores_mix (alpha=0.5)
#
# Usage:
#   bash scripts/run_l3_bs8_relative_mix_autowatch.sh
#
# Notes:
# - It waits for any GPU with enough free memory.
# - It runs jobs one by one to reduce OOM risk.

PY="/data/xueyue.yang/OHORA/venvs/ohora_cs_pure/bin/python"
TRAIN="/data/xueyue.yang/OHORA/ohora/fine-tuning_commonse.py"
MODEL="/nas_data/xueyue.yang/hf_models/Meta-Llama-3-8B"
DATASET_DIR="/data/xueyue.yang/OHORA/ohora/datasets/commonsense_reasoning"
CACHE_DIR="/nas_data/xueyue.yang/hf_cache"
DATA_CACHE_DIR="/nas_data/xueyue.yang/ohora_runs/lr5e4_wr_steps_sweep/cache/datasets"

TS="$(date +%Y%m%d_%H%M%S)"
ROOT="/nas_data/xueyue.yang/ohora_runs/l3_bs8_init_compare/${TS}"
mkdir -p "${ROOT}"

# Conservative threshold for A100-80G multi-tenant env.
MIN_FREE_MIB=52000
CHECK_INTERVAL=120

pick_free_gpu() {
  python - <<'PY'
import subprocess
MIN_FREE=52000
gpus=subprocess.check_output(
    "nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits",
    shell=True,text=True
).strip().splitlines()
# Pick the GPU with largest free memory above threshold.
cands=[]
for l in gpus:
    i,free,util=[x.strip() for x in l.split(',')]
    i=int(i); free=int(free); util=int(util)
    if free >= MIN_FREE:
        cands.append((free, -util, i))
if not cands:
    print("")
else:
    cands.sort(reverse=True)
    print(cands[0][2])
PY
}

wait_for_gpu() {
  local gpu=""
  while true; do
    gpu="$(pick_free_gpu)"
    if [[ -n "${gpu}" ]]; then
      echo "${gpu}"
      return 0
    fi
    echo "[$(date '+%F %T')] no GPU with free_mem>=${MIN_FREE_MIB} MiB, sleep ${CHECK_INTERVAL}s"
    sleep "${CHECK_INTERVAL}"
  done
}

run_one() {
  local method="$1"
  local extra="$2"
  local gpu
  gpu="$(wait_for_gpu)"

  local out_dir="${ROOT}/l3_bs8_${method}"
  local log_file="${ROOT}/l3_bs8_${method}.log"
  mkdir -p "${out_dir}"

  echo "[$(date '+%F %T')] start method=${method} on gpu=${gpu}" | tee -a "${ROOT}/progress.log"

  CUDA_VISIBLE_DEVICES="${gpu}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PY}" "${TRAIN}" \
    --model_name_or_path "${MODEL}" \
    --tokenizer_name_or_path "${MODEL}" \
    --dataset_dir "${DATASET_DIR}" \
    --cache_dir "${CACHE_DIR}" \
    --data_cache_dir "${DATA_CACHE_DIR}" \
    --output_dir "${out_dir}" \
    --do_train True --do_eval False --seed 42 \
    --per_device_train_batch_size 8 --gradient_accumulation_steps 8 \
    --num_train_epochs 3 --logging_steps 10 --save_strategy epoch --save_total_limit 2 \
    --learning_rate 5e-4 --lr_scheduler_type cosine --warmup_steps 30 --warmup_ratio 0 \
    --max_seq_length 1024 --bf16 True --torch_dtype bfloat16 \
    --lora_rank 64 --lora_alpha 128 --lora_dropout 0.05 --lora_nums 0 \
    --trainable gate_proj,down_proj,up_proj,q_proj,k_proj,v_proj,o_proj \
    --use_ohora True --ohora_init_method "${method}" --paper_repro_mode False \
    --load_in_kbits 16 --evaluation_strategy no --overwrite_output_dir ${extra} \
    2>&1 | tee "${log_file}"

  echo "[$(date '+%F %T')] finish method=${method}" | tee -a "${ROOT}/progress.log"
}

echo "[INFO] ROOT=${ROOT}" | tee -a "${ROOT}/progress.log"
run_one relative_scores ""
run_one relative_scores_mix "--ohora_mix_alpha 0.5"
echo "[DONE] all queued bs=8 jobs finished" | tee -a "${ROOT}/progress.log"
