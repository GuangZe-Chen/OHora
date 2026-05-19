#!/usr/bin/env bash
set -euo pipefail

cd /data/xueyue.yang/OHORA/ohora

PY="/data/xueyue.yang/OHORA/venvs/ohora_cs_pure/bin/python"
EVAL_SCRIPT="evaluate/run_commonsense_evaluate.py"
RUN_ROOT="/nas_data/xueyue.yang/ohora_runs/l3_bs2_lr7e4_dual/20260422_113751"
GPU_ID="${1:-2}"
BATCH_SIZE="${2:-4}"
EVAL_ROOT="${RUN_ROOT}/eval_full_two_finished_$(date +%Y%m%d_%H%M%S)"
PROGRESS_LOG="${EVAL_ROOT}/progress.log"

DATASETS=(
  boolq
  piqa
  social_i_qa
  hellaswag
  winogrande
  ARC-Challenge
  ARC-Easy
  openbookqa
)

MODELS=(
  "classic:${RUN_ROOT}/gpu7_bs2_classic/sft_lora_model"
  "relative_scores:${RUN_ROOT}/gpu6_bs2_relative_scores/sft_lora_model"
)

mkdir -p "${EVAL_ROOT}/logs"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${PROGRESS_LOG}"
}

summarize() {
  "${PY}" - "${EVAL_ROOT}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for model_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "logs"):
    for path in sorted(model_dir.glob("GPT-j-6B-ohora-*.json")):
        data = json.load(open(path))
        n = len(data)
        correct = sum(1 for item in data if item.get("flag"))
        dataset = path.name.replace("GPT-j-6B-ohora-", "").replace(".json", "")
        rows.append({
            "model": model_dir.name,
            "dataset": dataset,
            "n": n,
            "correct": correct,
            "accuracy": correct / n if n else None,
        })

with open(root / "summary.json", "w") as f:
    json.dump(rows, f, indent=2)

print("model,dataset,n,correct,accuracy")
for row in rows:
    print(f"{row['model']},{row['dataset']},{row['n']},{row['correct']},{row['accuracy']:.6f}")
PY
}

log "eval full two finished start root=${EVAL_ROOT} gpu=${GPU_ID} batch_size=${BATCH_SIZE}"
for item in "${MODELS[@]}"; do
  name="${item%%:*}"
  model_dir="${item#*:}"
  out_dir="${EVAL_ROOT}/${name}"
  mkdir -p "${out_dir}/logs"

  log "start model=${name} model_dir=${model_dir}"
  for ds in "${DATASETS[@]}"; do
    result_file="${out_dir}/GPT-j-6B-ohora-${ds}.json"
    log_file="${out_dir}/logs/${ds}.log"

    if [[ -s "${result_file}" ]]; then
      log "skip existing model=${name} dataset=${ds}"
      continue
    fi

    log "start model=${name} dataset=${ds}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "${PY}" "${EVAL_SCRIPT}" \
        --dataset "${ds}" \
        --model GPT-j-6B \
        --base_model "${model_dir}" \
        --batch_size "${BATCH_SIZE}" \
        --gpu_id 0 \
        --output_dir "${out_dir}" 2>&1 | tee -a "${log_file}"
    log "done model=${name} dataset=${ds}"
    summarize | tee "${EVAL_ROOT}/summary.csv"
  done
  log "done model=${name}"
done

summarize | tee "${EVAL_ROOT}/summary.csv"
log "eval full two finished done root=${EVAL_ROOT}"
