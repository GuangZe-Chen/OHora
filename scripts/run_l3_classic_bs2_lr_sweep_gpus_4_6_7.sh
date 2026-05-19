#!/usr/bin/env bash
set -euo pipefail

cd /data/xueyue.yang/OHORA/ohora

ROOT_TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="/nas_data/xueyue.yang/ohora_runs/l3_classic_bs2_lr_sweep/${ROOT_TS}"
MODEL="/nas_data/xueyue.yang/hf_models/Meta-Llama-3-8B"
DATASET_DIR="/data/xueyue.yang/OHORA/ohora/datasets/commonsense_reasoning"
CACHE_DIR="/nas_data/xueyue.yang/hf_cache"
DATA_CACHE_DIR="/nas_data/xueyue.yang/ohora_runs/lr5e4_wr_steps_sweep/cache/datasets"
PY="/data/xueyue.yang/OHORA/venvs/ohora_cs_pure/bin/python"
TRAIN_SCRIPT="/data/xueyue.yang/OHORA/ohora/fine-tuning_commonse.py"
EVAL_SCRIPT="/data/xueyue.yang/OHORA/ohora/evaluate/run_commonsense_evaluate.py"
PROGRESS_LOG="${RUN_ROOT}/progress.log"
CPU_LOG="${RUN_ROOT}/cpu_monitor.log"

mkdir -p "${RUN_ROOT}" "${DATA_CACHE_DIR}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${PROGRESS_LOG}"
}

wait_gpu_free() {
  local gpu="$1"
  local min_free_mb="${2:-42000}"

  while true; do
    local used total free
    IFS=, read -r used total < <(
      nvidia-smi --id="${gpu}" \
        --query-gpu=memory.used,memory.total \
        --format=csv,noheader,nounits
    )
    used="$(echo "${used}" | xargs)"
    total="$(echo "${total}" | xargs)"
    free=$((total - used))
    if [[ "${free}" -ge "${min_free_mb}" ]]; then
      log "gpu=${gpu} free=${free}MB ready"
      return 0
    fi
    log "gpu=${gpu} free=${free}MB wait"
    sleep 300
  done
}

run_quick_eval() {
  local gpu="$1"
  local outdir="$2"
  local eval_dir="${outdir}/eval_quick_200"
  mkdir -p "${eval_dir}/logs"

  for ds in boolq piqa social_i_qa; do
    log "quick_eval start gpu=${gpu} dataset=${ds} out=${eval_dir}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "${PY}" "${EVAL_SCRIPT}" \
        --dataset "${ds}" \
        --model GPT-j-6B \
        --base_model "${outdir}/sft_lora_model" \
        --batch_size 4 \
        --gpu_id 0 \
        --output_dir "${eval_dir}" \
        --max_samples 200 2>&1 | tee "${eval_dir}/logs/${ds}.log"
    log "quick_eval done gpu=${gpu} dataset=${ds}"
  done

  "${PY}" - "${eval_dir}" <<'PY' | tee "${eval_dir}/summary.csv"
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
print("dataset,n,correct,accuracy")
for path in sorted(root.glob("GPT-j-6B-ohora-*.json")):
    data = json.load(open(path))
    n = len(data)
    c = sum(1 for item in data if item.get("flag"))
    ds = path.name.replace("GPT-j-6B-ohora-", "").replace(".json", "")
    print(f"{ds},{n},{c},{c / n if n else 0:.6f}")
PY
}

run_one() {
  local gpu="$1"
  local tag="$2"
  local lr="$3"
  local warmup_ratio="$4"
  local outdir="${RUN_ROOT}/${tag}_classic_lr${lr}_wr${warmup_ratio}"
  mkdir -p "${outdir}"

  wait_gpu_free "${gpu}" 42000
  log "train start gpu=${gpu} tag=${tag} lr=${lr} wr=${warmup_ratio} out=${outdir}"

  set +e
  CUDA_VISIBLE_DEVICES="${gpu}" \
  OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PY}" "${TRAIN_SCRIPT}" \
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
      --save_total_limit 1 \
      --learning_rate "${lr}" \
      --lr_scheduler_type cosine \
      --warmup_steps 0 \
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
      --ohora_init_method classic \
      --paper_repro_mode False \
      --load_in_kbits 16 \
      --evaluation_strategy no \
      --overwrite_output_dir 2>&1 | tee "${outdir}/train.log"
  local rc=${PIPESTATUS[0]}
  set -e

  if [[ "${rc}" -ne 0 ]]; then
    log "train failed gpu=${gpu} tag=${tag} exit=${rc} out=${outdir}"
    return "${rc}"
  fi

  log "train done gpu=${gpu} tag=${tag} out=${outdir}"
  if [[ -f "${outdir}/sft_lora_model/model.safetensors.index.json" ]]; then
    run_quick_eval "${gpu}" "${outdir}" || log "quick_eval failed gpu=${gpu} tag=${tag}"
  else
    log "quick_eval skip missing_model gpu=${gpu} tag=${tag}"
  fi
}

monitor_cpu() {
  log "cpu monitor start"
  while [[ -f "${RUN_ROOT}/.running" ]]; do
    local line us sy id cpu
    line="$(vmstat 1 2 | tail -n 1)"
    us="$(echo "${line}" | awk '{print $(NF-4)}')"
    sy="$(echo "${line}" | awk '{print $(NF-3)}')"
    id="$(echo "${line}" | awk '{print $(NF-2)}')"
    cpu=$((100 - id))
    echo "[$(date '+%F %T')] cpu_used=${cpu}% us=${us}% sy=${sy}% id=${id}%" >> "${CPU_LOG}"
    sleep 60
  done
  log "cpu monitor stop"
}

log "classic lr sweep start root=${RUN_ROOT}"
touch "${RUN_ROOT}/.running"
: > "${CPU_LOG}"
monitor_cpu &
MON_PID=$!

(
  run_one 4 A 1e-4 0.03
  run_one 4 D 5e-4 0.03
) &
PID4=$!

(
  run_one 6 B 2e-4 0.03
  run_one 6 C 3e-4 0.03
) &
PID6=$!

(
  run_one 7 E 4e-4 0.03
  run_one 7 F 6e-4 0.03
) &
PID7=$!

status=0
wait "${PID4}" || status=1
wait "${PID6}" || status=1
wait "${PID7}" || status=1
rm -f "${RUN_ROOT}/.running"
wait "${MON_PID}" 2>/dev/null || true
log "classic lr sweep finished status=${status} root=${RUN_ROOT}"
exit "${status}"
