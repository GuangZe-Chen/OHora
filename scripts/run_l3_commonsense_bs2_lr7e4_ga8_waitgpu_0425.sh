#!/usr/bin/env bash
set -euo pipefail

cd /data/xueyue.yang/OHORA/ohora

PY="/data/xueyue.yang/OHORA/venvs/ohora_cs_pure/bin/python"
TRAIN="/data/xueyue.yang/OHORA/ohora/fine-tuning_commonse.py"
MODEL="/nas_data/xueyue.yang/hf_models/Meta-Llama-3-8B"
DATASET_DIR="/data/xueyue.yang/OHORA/ohora/datasets/commonsense_reasoning"
CACHE_DIR="/nas_data/xueyue.yang/hf_cache"

RUN_ROOT="${RUN_ROOT:-/nas_data/xueyue.yang/ohora_runs/requested_l3_commonsense_bs2_lr7e4_ga8_$(date +%Y%m%d_%H%M%S)}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-${RUN_ROOT}/cache/datasets}"
OUTDIR="${RUN_ROOT}/llama3_8b_lr7e-4_bs2_ga8_ws30_e3"
PROGRESS_LOG="${RUN_ROOT}/progress.log"
GPU_CANDIDATES="${GPU_CANDIDATES:-1 0 3 6 5 7 2 4}"
MIN_FREE_MB="${MIN_FREE_MB:-42000}"
MAX_UTIL="${MAX_UTIL:-25}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-120}"

mkdir -p "${RUN_ROOT}" "${DATA_CACHE_DIR}" "${OUTDIR}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${PROGRESS_LOG}"
}

pick_gpu() {
  while true; do
    for gpu in ${GPU_CANDIDATES}; do
      local used total util free
      IFS=, read -r used total util < <(
        nvidia-smi --id="${gpu}" \
          --query-gpu=memory.used,memory.total,utilization.gpu \
          --format=csv,noheader,nounits
      )
      used="$(echo "${used}" | xargs)"
      total="$(echo "${total}" | xargs)"
      util="$(echo "${util}" | xargs)"
      free=$((total - used))
      if [[ "${free}" -ge "${MIN_FREE_MB}" && "${util}" -le "${MAX_UTIL}" ]]; then
        echo "${gpu}"
        return 0
      fi
      log "wait gpu=${gpu} free=${free}MB util=${util}%" >&2
    done
    sleep "${CHECK_INTERVAL_SEC}"
  done
}

gpu="$(pick_gpu)"
log "train start name=LLaMA-3-8B_lr7e-4_bs2_ga8_ws30_e3 gpu=${gpu} out=${OUTDIR}"

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
    --output_dir "${OUTDIR}" \
    --do_train True \
    --do_eval False \
    --seed 42 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --num_train_epochs 3 \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 7e-4 \
    --lr_scheduler_type cosine \
    --warmup_steps 30 \
    --warmup_ratio 0 \
    --max_seq_length 1024 \
    --bf16 True \
    --torch_dtype bfloat16 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_nums 0 \
    --trainable gate_proj,down_proj,up_proj,q_proj,k_proj,v_proj,o_proj \
    --use_ohora True \
    --ohora_init_method classic \
    --paper_repro_mode False \
    --load_in_kbits 16 \
    --evaluation_strategy no \
    --overwrite_output_dir \
    2>&1 | tee "${OUTDIR}/train.log"

code=${PIPESTATUS[0]}
if [[ "${code}" -eq 0 ]]; then
  log "train done name=LLaMA-3-8B_lr7e-4_bs2_ga8_ws30_e3 out=${OUTDIR}"
else
  log "train failed name=LLaMA-3-8B_lr7e-4_bs2_ga8_ws30_e3 exit=${code} out=${OUTDIR}"
fi
exit "${code}"
