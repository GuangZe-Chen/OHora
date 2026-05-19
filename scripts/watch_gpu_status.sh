#!/usr/bin/env bash
set -euo pipefail
OUT=/nas_data/xueyue.yang/ohora_runs/l2_evaltasks/gpu_watch_$(date +%Y%m%d_%H%M%S).log
mkdir -p /nas_data/xueyue.yang/ohora_runs/l2_evaltasks
while true; do
  echo "===== $(date '+%F %T') =====" | tee -a "$OUT"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | tee -a "$OUT"
  echo | tee -a "$OUT"
  sleep 60
done