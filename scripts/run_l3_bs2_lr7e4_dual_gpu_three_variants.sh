#!/usr/bin/env bash
set -u

ROOT_TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="/nas_data/xueyue.yang/ohora_runs/l3_bs2_lr7e4_dual/${ROOT_TS}"
MODEL="/nas_data/xueyue.yang/hf_models/Meta-Llama-3-8B"
DATASET_DIR="/data/xueyue.yang/OHORA/ohora/datasets/commonsense_reasoning"
CACHE_DIR="/nas_data/xueyue.yang/hf_cache"
DATA_CACHE_DIR="/nas_data/xueyue.yang/ohora_runs/lr5e4_wr_steps_sweep/cache/datasets"
PYTHON_BIN="/data/xueyue.yang/OHORA/venvs/ohora_cs_pure/bin/python"
TRAIN_SCRIPT="/data/xueyue.yang/OHORA/ohora/fine-tuning_commonse.py"

mkdir -p "${RUN_ROOT}"
PROGRESS_LOG="${RUN_ROOT}/progress.log"
CPU_LOG="${RUN_ROOT}/cpu_monitor.log"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${PROGRESS_LOG}"
}

run_one() {
  local gpu="$1"
  local method="$2"
  local extra_args="$3"
  local outdir="${RUN_ROOT}/gpu${gpu}_bs2_${method}"
  mkdir -p "${outdir}"

  log "start gpu=${gpu} method=${method} outdir=${outdir}"

  CUDA_VISIBLE_DEVICES="${gpu}" \
  OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 \
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
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 3 \
    --logging_steps 10 \
    --save_strategy epoch \
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
    --ohora_init_method "${method}" \
    --paper_repro_mode False \
    --load_in_kbits 16 \
    --evaluation_strategy no \
    --overwrite_output_dir \
    ${extra_args} \
    2>&1 | tee "${outdir}/train.log"

  local code=${PIPESTATUS[0]}
  if [[ ${code} -eq 0 ]]; then
    log "done gpu=${gpu} method=${method}"
  else
    log "failed gpu=${gpu} method=${method} exit=${code}"
  fi
  return ${code}
}

monitor_cpu() {
  log "cpu monitor start"
  while true; do
    if [[ ! -f "${RUN_ROOT}/.running" ]]; then
      break
    fi
    # vmstat columns: ... us sy id wa ... ; use second sample to avoid warmup noise
    local line
    line="$(vmstat 1 2 | tail -n 1)"
    local us sy id
    us="$(echo "${line}" | awk '{print $(NF-4)}')"
    sy="$(echo "${line}" | awk '{print $(NF-3)}')"
    id="$(echo "${line}" | awk '{print $(NF-2)}')"
    local cpu
    cpu=$((100-id))
    echo "[$(date '+%F %T')] cpu_used=${cpu}% us=${us}% sy=${sy}% id=${id}%" >> "${CPU_LOG}"
    if [[ ${cpu} -ge 85 ]]; then
      log "warn cpu_high=${cpu}%"
    fi
    sleep 60
  done
  log "cpu monitor stop"
}

log "queue start root=${RUN_ROOT}"
: > "${CPU_LOG}"
touch "${RUN_ROOT}/.running"
monitor_cpu &
MON_PID=$!

# Parallel stage: run two variants on two GPUs simultaneously.
(
  run_one 6 "relative_scores" ""
  run_one 6 "relative_scores_mix" "--ohora_mix_alpha 0.5"
) &
PID6=$!

(
  run_one 7 "classic" ""
) &
PID7=$!

wait ${PID6}
wait ${PID7}
rm -f "${RUN_ROOT}/.running"
wait ${MON_PID} 2>/dev/null || true
log "queue finished"
