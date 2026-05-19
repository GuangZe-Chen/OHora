#!/usr/bin/env bash
set -euo pipefail

cd /data/xueyue.yang/OHORA/ohora

PY="/data/xueyue.yang/OHORA/venvs/ohora_cs_pure/bin/python"
TRAIN="/data/xueyue.yang/OHORA/ohora/fine-tuning_commonse.py"
DATASET_DIR="/data/xueyue.yang/OHORA/ohora/datasets/commonsense_reasoning"
CACHE_DIR="/nas_data/xueyue.yang/hf_cache"

RUN_ROOT="${RUN_ROOT:-/nas_data/xueyue.yang/ohora_runs/init_ablation_bs2_lr7e4_$(date +%Y%m%d_%H%M%S)}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-/nas_data/xueyue.yang/ohora_runs/requested_l2_l3_commonsense_bs2_lr7e4_20260425_113527/cache/datasets}"
PROGRESS_LOG="${RUN_ROOT}/progress.log"
LOCK_DIR="${RUN_ROOT}/gpu_locks"
GPU_CANDIDATES="${GPU_CANDIDATES:-0 1 2 3 4 5 6 7}"
MIN_FREE_MB="${MIN_FREE_MB:-42000}"
MAX_UTIL="${MAX_UTIL:-25}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-120}"
OHORA_MIX_ALPHA="${OHORA_MIX_ALPHA:-0.5}"

mkdir -p "${RUN_ROOT}" "${DATA_CACHE_DIR}" "${LOCK_DIR}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${PROGRESS_LOG}"
}

cleanup_lock() {
  if [[ -n "${GPU_LOCK_PATH:-}" && -d "${GPU_LOCK_PATH}" ]]; then
    rmdir "${GPU_LOCK_PATH}" 2>/dev/null || true
  fi
}
trap cleanup_lock EXIT

pick_gpu() {
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
        log "wait gpu=${gpu} free=${free}MB util=${util}% reserved=1" >&2
      else
        log "wait gpu=${gpu} free=${free}MB util=${util}%" >&2
      fi
    done
    sleep "${CHECK_INTERVAL_SEC}"
  done
}

usage() {
  echo "Usage: $0 {l2|l3} {relative_scores|relative_scores_mix}" >&2
}

model_key="${1:-}"
method="${2:-}"
if [[ "${model_key}" != "l2" && "${model_key}" != "l3" ]]; then
  usage
  exit 2
fi
if [[ "${method}" != "relative_scores" && "${method}" != "relative_scores_mix" ]]; then
  usage
  exit 2
fi

if [[ "${model_key}" == "l2" ]]; then
  model="/nas_data/xueyue.yang/hf_models/Llama-2-7b-hf"
  warmup_steps=0
  warmup_ratio=0.01
  paper_repro_mode=True
  model_name="LLaMA-2-7B"
  outdir="${RUN_ROOT}/llama2_7b_lr7e-4_bs2_ga32_${method}_wrr0.01_e3"
else
  model="/nas_data/xueyue.yang/hf_models/Meta-Llama-3-8B"
  warmup_steps=30
  warmup_ratio=0
  paper_repro_mode=False
  model_name="LLaMA-3-8B"
  outdir="${RUN_ROOT}/llama3_8b_lr7e-4_bs2_ga32_${method}_ws30_e3"
fi

mkdir -p "${outdir}"
gpu="$(pick_gpu)"
log "train start name=${model_name}_lr7e-4_bs2_ga32_${method} gpu=${gpu} out=${outdir}"

CUDA_VISIBLE_DEVICES="${gpu}" \
OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PY}" "${TRAIN}" \
    --model_name_or_path "${model}" \
    --tokenizer_name_or_path "${model}" \
    --dataset_dir "${DATASET_DIR}" \
    --cache_dir "${CACHE_DIR}" \
    --data_cache_dir "${DATA_CACHE_DIR}" \
    --output_dir "${outdir}" \
    --do_train True \
    --do_eval False \
    --seed 42 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --gradient_checkpointing True \
    --num_train_epochs 3 \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 7e-4 \
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
    --ohora_init_method "${method}" \
    --ohora_mix_alpha "${OHORA_MIX_ALPHA}" \
    --paper_repro_mode "${paper_repro_mode}" \
    --load_in_kbits 16 \
    --evaluation_strategy no \
    --overwrite_output_dir \
    2>&1 | tee "${outdir}/train.log"

code=${PIPESTATUS[0]}
if [[ "${code}" -eq 0 ]]; then
  log "train done name=${model_name}_lr7e-4_bs2_ga32_${method} out=${outdir}"
else
  log "train failed name=${model_name}_lr7e-4_bs2_ga32_${method} exit=${code} out=${outdir}"
fi
exit "${code}"
