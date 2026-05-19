#!/usr/bin/env bash
set -euo pipefail

BASE="${1:-/nas_data/xueyue.yang/ohora_runs/lr5e4_wr_steps_sweep}"
INTERVAL="${INTERVAL:-180}"

TAGS=(
  M_lr5e-4_wrr0.01
  N_lr5e-4_wrr0.1
  O_lr5e-4_wrs30
  P_lr5e-4_wrs60
  Q_lr5e-4_wrs100
)

LOG="${BASE}/compare_watch.log"
OUT_RAW="${BASE}/compare_5runs_raw.tsv"
OUT_RANK="${BASE}/compare_5runs_ranked.tsv"

mkdir -p "${BASE}"
echo "===== WATCH START $(date '+%F %T') =====" | tee -a "${LOG}"

all_done() {
  local t
  for t in "${TAGS[@]}"; do
    if [[ ! -f "${BASE}/${t}/train_results.json" ]]; then
      return 1
    fi
  done
  return 0
}

while ! all_done; do
  echo "[$(date '+%F %T')] waiting for all 5 runs to finish..." | tee -a "${LOG}"
  sleep "${INTERVAL}"
done

python3 - <<'PY' "$BASE" "$OUT_RAW" "$OUT_RANK"
import json, os, sys
base, out_raw, out_rank = sys.argv[1:4]
tags = [
  'M_lr5e-4_wrr0.01',
  'N_lr5e-4_wrr0.1',
  'O_lr5e-4_wrs30',
  'P_lr5e-4_wrs60',
  'Q_lr5e-4_wrs100',
]
rows = []
for t in tags:
    d = os.path.join(base, t)
    tr = os.path.join(d, 'train_results.json')
    ts = os.path.join(d, 'trainer_state.json')
    train_loss = None
    train_runtime = None
    last_loss = None
    if os.path.exists(tr):
        j = json.load(open(tr))
        train_loss = j.get('train_loss')
        train_runtime = j.get('train_runtime')
    if os.path.exists(ts):
        j = json.load(open(ts))
        for it in j.get('log_history', []):
            if 'loss' in it:
                last_loss = it['loss']
    rows.append((t, train_loss, last_loss, train_runtime))

with open(out_raw, 'w') as f:
    f.write('tag\ttrain_loss\tlast_loss\ttrain_runtime\n')
    for r in rows:
        f.write('\t'.join('' if v is None else str(v) for v in r) + '\n')

ranked = [r for r in rows if r[1] is not None]
ranked.sort(key=lambda x: x[1])
with open(out_rank, 'w') as f:
    f.write('rank\ttag\ttrain_loss\tlast_loss\ttrain_runtime\n')
    for i, r in enumerate(ranked, 1):
        f.write(f"{i}\t{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\n")

print('WROTE', out_raw)
print('WROTE', out_rank)
PY

echo "===== WATCH DONE $(date '+%F %T') =====" | tee -a "${LOG}"
