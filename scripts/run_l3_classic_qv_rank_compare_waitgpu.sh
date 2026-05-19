#!/usr/bin/env bash
set -euo pipefail

cd /data/xueyue.yang/OHORA/ohora

TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="/nas_data/xueyue.yang/ohora_runs/l3_classic_qv_rank_compare/${TS}"
MODEL="/nas_data/xueyue.yang/hf_models/Meta-Llama-3-8B"
DATASET_DIR="/data/xueyue.yang/OHORA/ohora/datasets/commonsense_reasoning"
CACHE_DIR="/nas_data/xueyue.yang/hf_cache"
DATA_CACHE_DIR="/nas_data/xueyue.yang/ohora_runs/lr5e4_wr_steps_sweep/cache/datasets"
PY="/data/xueyue.yang/OHORA/venvs/ohora_cs_pure/bin/python"
TRAIN_SCRIPT="/data/xueyue.yang/OHORA/ohora/fine-tuning_commonse.py"
EVAL_SCRIPT="/data/xueyue.yang/OHORA/ohora/evaluate/run_commonsense_evaluate.py"
PROGRESS_LOG="${RUN_ROOT}/progress.log"

mkdir -p "${RUN_ROOT}" "${DATA_CACHE_DIR}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${PROGRESS_LOG}"
}

pick_gpu() {
  local candidates="${1:-0 5 2 4 6 7 1 3}"
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
      if [[ "${free}" -ge 42000 && "${util}" -le 35 ]]; then
        echo "${gpu}"
        return 0
      fi
      log "wait tag=${TAG:-unknown} gpu=${gpu} free=${free}MB util=${util}%"
    done
    sleep 300
  done
}

run_quick_eval() {
  local gpu="$1"
  local outdir="$2"
  local eval_dir="${outdir}/eval_quick_200"
  mkdir -p "${eval_dir}/logs"

  for ds in boolq piqa social_i_qa; do
    log "quick_eval start out=${outdir} gpu=${gpu} dataset=${ds}"
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
    log "quick_eval done out=${outdir} gpu=${gpu} dataset=${ds}"
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
  TAG="$1"
  local rank="$2"
  local outdir="${RUN_ROOT}/${TAG}_classic_qv_rank${rank}"
  mkdir -p "${outdir}"

  local gpu
  gpu="$(pick_gpu)"
  log "train start tag=${TAG} gpu=${gpu} rank=${rank} out=${outdir}"

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
      --learning_rate 2e-4 \
      --lr_scheduler_type cosine \
      --warmup_steps 0 \
      --warmup_ratio 0.03 \
      --max_seq_length 1024 \
      --bf16 True \
      --torch_dtype bfloat16 \
      --lora_rank "${rank}" \
      --lora_alpha 128 \
      --lora_dropout 0.05 \
      --lora_nums 0 \
      --trainable q_proj,v_proj \
      --use_ohora True \
      --ohora_init_method classic \
      --paper_repro_mode False \
      --load_in_kbits 16 \
      --evaluation_strategy no \
      --overwrite_output_dir 2>&1 | tee "${outdir}/train.log"

  log "train done tag=${TAG} gpu=${gpu} out=${outdir}"
  if [[ -f "${outdir}/sft_lora_model/model.safetensors.index.json" ]]; then
    run_quick_eval "${gpu}" "${outdir}" || log "quick_eval failed tag=${TAG}"
  fi
}

log "classic qv-rank compare start root=${RUN_ROOT}"

(
  run_one A 8
) &
P1=$!

(
  run_one B 64
) &
P2=$!

status=0
wait "${P1}" || status=1
wait "${P2}" || status=1
log "classic qv-rank compare finished status=${status} root=${RUN_ROOT}"
exit "${status}"
