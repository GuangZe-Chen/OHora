#!/usr/bin/env bash
set -u

ROOT_TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="/nas_data/xueyue.yang/ohora_runs/l3_bs8_gpu6_queue/${ROOT_TS}"
MODEL="/nas_data/xueyue.yang/hf_models/Meta-Llama-3-8B"
DATASET_DIR="/data/xueyue.yang/OHORA/ohora/datasets/commonsense_reasoning"
CACHE_DIR="/nas_data/xueyue.yang/hf_cache"
DATA_CACHE_DIR="/nas_data/xueyue.yang/ohora_runs/lr5e4_wr_steps_sweep/cache/datasets"
PYTHON_BIN="/data/xueyue.yang/OHORA/venvs/ohora_cs_pure/bin/python"
TRAIN_SCRIPT="/data/xueyue.yang/OHORA/ohora/fine-tuning_commonse.py"

mkdir -p "${RUN_ROOT}"
PROGRESS_LOG="${RUN_ROOT}/progress.log"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${PROGRESS_LOG}"
}

run_one() {
  local method="$1"
  local extra_args="$2"
  local outdir="${RUN_ROOT}/bs8_${method}"
  mkdir -p "${outdir}"

  log "start method=${method} gpu=6 outdir=${outdir}"

  CUDA_VISIBLE_DEVICES=6 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
    --model_name_or_path "${MODEL}" \
    --tokenizer_name_or_path "${MODEL}" \
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
    --save_total_limit 2 \
    --learning_rate 5e-4 \
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
    --ohora_init_method "${method}" \
    --paper_repro_mode False \
    --load_in_kbits 16 \
    --evaluation_strategy no \
    --overwrite_output_dir \
    ${extra_args} \
    2>&1 | tee "${outdir}/train.log"

  local code=${PIPESTATUS[0]}
  if [[ ${code} -eq 0 ]]; then
    log "done method=${method}"
  else
    log "failed method=${method} exit=${code}"
  fi

  return ${code}
}

log "queue start root=${RUN_ROOT}"
run_one "relative_scores" "" || true
run_one "classic" "" || true
run_one "relative_scores_mix" "--ohora_mix_alpha 0.5" || true
log "queue finished"
