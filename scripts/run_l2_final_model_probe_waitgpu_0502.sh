#!/usr/bin/env bash
set -euo pipefail

cd /data/xueyue.yang/OHORA/ohora

PY="/data/xueyue.yang/OHORA/venvs/ohora_cs_pure/bin/python"
OUT_DIR="${OUT_DIR:-/nas_data/xueyue.yang/ohora_runs/ohora_numeric_analysis/llama2_final_model_probe_200_20260502}"
GPU_CANDIDATES="${GPU_CANDIDATES:-4 6 2 5 0 1 3 7}"
MIN_FREE_MB="${MIN_FREE_MB:-22000}"
MAX_UTIL="${MAX_UTIL:-45}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-120}"
MAX_SAMPLES="${MAX_SAMPLES:-200}"
MODELS="${MODELS:-alpha0_old_classic,alpha0p3_mix,alpha0p5_mix,alpha0p7_mix,alpha1_relative}"
LOCK_DIR="/nas_data/xueyue.yang/ohora_runs/ohora_numeric_analysis/gpu_locks"
PROGRESS_LOG="${OUT_DIR}/progress.log"

mkdir -p "${OUT_DIR}" "${LOCK_DIR}"

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

gpu="$(pick_gpu)"
log "start final model probe gpu=${gpu} max_samples=${MAX_SAMPLES} models=${MODELS} out=${OUT_DIR}"

CUDA_VISIBLE_DEVICES="${gpu}" \
OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"${PY}" scripts/run_ohora_final_model_probe.py \
  --device cuda:0 \
  --out_dir "${OUT_DIR}" \
  --max_samples "${MAX_SAMPLES}" \
  --models "${MODELS}" \
  --resume \
  2>&1 | tee -a "${PROGRESS_LOG}"

log "done final model probe out=${OUT_DIR}"
