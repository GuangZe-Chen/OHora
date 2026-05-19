#!/usr/bin/env bash
set -euo pipefail

cd /data/xueyue.yang/OHORA/ohora

PY="/data/xueyue.yang/OHORA/venvs/ohora_cs_pure/bin/python"
TRAIN="/data/xueyue.yang/OHORA/ohora/fine-tuning_commonse.py"
DATASET_DIR="/data/xueyue.yang/OHORA/ohora/datasets/commonsense_reasoning"
CACHE_DIR="/nas_data/xueyue.yang/hf_cache"

RUN_ROOT="${RUN_ROOT:-/nas_data/xueyue.yang/ohora_runs/requested_l2_l3_commonsense_$(date +%Y%m%d_%H%M%S)}"
DATA_CACHE_DIR="${RUN_ROOT}/cache/datasets"
PROGRESS_LOG="${RUN_ROOT}/progress.log"
MIN_FREE_MB="${MIN_FREE_MB:-55000}"
MAX_UTIL="${MAX_UTIL:-30}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-120}"

mkdir -p "${RUN_ROOT}" "${DATA_CACHE_DIR}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${PROGRESS_LOG}"
}

pick_gpu() {
  local candidates="${1:-0 1 2 3 4 5 6 7}"
  while true; do
    for gpu in ${candidates}; do
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

run_one() {
  local name="$1"
  local model="$2"
  local outdir="$3"
  local gpu_candidates="$4"
  local warmup_steps="$5"
  local warmup_ratio="$6"
  local paper_repro_mode="$7"
  local ohora_method="${8:-classic}"
  local gpu

  mkdir -p "${outdir}"
  gpu="$(pick_gpu "${gpu_candidates}")"
  log "train start name=${name} gpu=${gpu} out=${outdir}"

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
      --per_device_train_batch_size 8 \
      --gradient_accumulation_steps 8 \
      --num_train_epochs 3 \
      --logging_steps 10 \
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
      --ohora_init_method "${ohora_method}" \
      --paper_repro_mode "${paper_repro_mode}" \
      --load_in_kbits 16 \
      --evaluation_strategy no \
      --overwrite_output_dir \
      2>&1 | tee "${outdir}/train.log"

  local code=${PIPESTATUS[0]}
  if [[ "${code}" -eq 0 ]]; then
    log "train done name=${name} out=${outdir}"
  else
    log "train failed name=${name} exit=${code} out=${outdir}"
  fi
  return "${code}"
}

case "${1:-both}" in
  l2)
    run_one \
      "LLaMA-2-7B_lr5e-4_bs8_wr0.01_e3" \
      "/nas_data/xueyue.yang/hf_models/Llama-2-7b-hf" \
      "${RUN_ROOT}/llama2_7b_lr5e-4_bs8_wrr0.01_e3" \
      "${GPU_CANDIDATES:-${2:-0 1 2 3 4 5 6 7}}" \
      0 \
      0.01 \
      True
    ;;
  l3)
    run_one \
      "LLaMA-3-8B_lr5e-4_bs8_ws30_e3" \
      "/nas_data/xueyue.yang/hf_models/Meta-Llama-3-8B" \
      "${RUN_ROOT}/llama3_8b_lr5e-4_bs8_ws30_e3" \
      "${GPU_CANDIDATES:-${2:-0 1 2 3 4 5 6 7}}" \
      30 \
      0 \
      False
    ;;
  both)
    run_one \
      "LLaMA-2-7B_lr5e-4_bs8_wr0.01_e3" \
      "/nas_data/xueyue.yang/hf_models/Llama-2-7b-hf" \
      "${RUN_ROOT}/llama2_7b_lr5e-4_bs8_wrr0.01_e3" \
      "${GPU_CANDIDATES:-${2:-0 1 2 3 4 5 6 7}}" \
      0 \
      0.01 \
      True
    run_one \
      "LLaMA-3-8B_lr5e-4_bs8_ws30_e3" \
      "/nas_data/xueyue.yang/hf_models/Meta-Llama-3-8B" \
      "${RUN_ROOT}/llama3_8b_lr5e-4_bs8_ws30_e3" \
      "${GPU_CANDIDATES:-${2:-0 1 2 3 4 5 6 7}}" \
      30 \
      0 \
      False
    ;;
  *)
    echo "Usage: $0 {l2|l3|both} [gpu_candidates]" >&2
    exit 2
    ;;
esac

log "all requested jobs finished root=${RUN_ROOT}"
