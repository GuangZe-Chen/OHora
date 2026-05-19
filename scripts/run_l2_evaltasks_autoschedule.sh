#!/usr/bin/env bash
set -euo pipefail

PY=/data/xueyue.yang/OHORA/venvs/ohora_cs_pure/bin/python
TRAIN_SCRIPT=/data/xueyue.yang/OHORA/ohora/fine-tuning.py
MODEL=/nas_data/xueyue.yang/hf_models/Llama-2-7b-hf
CACHE=/nas_data/xueyue.yang/hf_cache
DATA_CACHE=/nas_data/xueyue.yang/ohora_runs/l2_evaltasks/cache/datasets
RUN_ROOT=/nas_data/xueyue.yang/ohora_runs/l2_evaltasks
TS=$(date +%Y%m%d_%H%M%S)
RUN_BASE="$RUN_ROOT/$TS"
mkdir -p "$RUN_BASE" "$DATA_CACHE"

# task_name dataset_dir lr warmup_ratio epoch bs
TASKS=(
  "humaneval /data/xueyue.yang/OHORA/ohora/datasets/task_ft/humaneval 6e-4 0.01 1 3"
  "mmlu /data/xueyue.yang/OHORA/ohora/datasets/task_ft/mmlu 2e-4 0.01 1 3"
  "mtbench /data/xueyue.yang/OHORA/ohora/datasets/task_ft/mt_bench 2e-4 0.01 1 3"
  "gsm8k /data/xueyue.yang/OHORA/ohora/datasets/task_ft/gsm8k 3e-4 0.01 1 3"
)

pick_gpu() {
  while true; do
    while IFS=, read -r idx util used total; do
      idx=$(echo "$idx" | xargs)
      util=$(echo "$util" | xargs)
      used=$(echo "$used" | xargs)
      total=$(echo "$total" | xargs)
      free=$((total-used))
      # Keep a larger safety margin so jobs are less likely to OOM on busy GPUs.
      if [[ "$free" -ge 42000 ]]; then
        echo "$idx"
        return 0
      fi
    done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits)
    sleep 30
  done
}

run_train_once() {
  local task=$1
  local dataset_dir=$2
  local lr=$3
  local wr=$4
  local epoch=$5
  local bs=$6
  local out_dir=$7
  local log=$8

  local gpu
  gpu=$(pick_gpu)
  echo "[$(date '+%F %T')] START task=$task gpu=$gpu bs=$bs lr=$lr wr=$wr e=$epoch out=$out_dir"

  set +e
  CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" "$TRAIN_SCRIPT" \
      --model_name_or_path "$MODEL" \
      --tokenizer_name_or_path "$MODEL" \
      --dataset_dir "$dataset_dir" \
      --cache_dir "$CACHE" \
      --data_cache_dir "$DATA_CACHE" \
      --output_dir "$out_dir" \
      --do_train True \
      --do_eval False \
      --seed 42 \
      --per_device_train_batch_size "$bs" \
      --gradient_accumulation_steps 8 \
      --num_train_epochs "$epoch" \
      --logging_steps 10 \
      --save_strategy epoch \
      --save_total_limit 2 \
      --learning_rate "$lr" \
      --lr_scheduler_type cosine \
      --warmup_steps 0 \
      --warmup_ratio "$wr" \
      --max_seq_length 1024 \
      --bf16 True \
      --torch_dtype bfloat16 \
      --lora_rank 64 \
      --lora_alpha 128 \
      --lora_dropout 0.05 \
      --lora_nums 0 \
      --trainable gate_proj,down_proj,up_proj,q_proj,k_proj,v_proj,o_proj \
      --use_ohora True \
      --load_in_kbits 16 \
      --evaluation_strategy no \
      --overwrite_output_dir 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  set -e

  return "$rc"
}

run_one() {
  local task=$1
  local dataset_dir=$2
  local lr=$3
  local wr=$4
  local epoch=$5
  local bs=$6
  local out_dir="$RUN_BASE/${task}_lr${lr}_bs${bs}_wr${wr}_e${epoch}"
  mkdir -p "$out_dir"
  local log="$out_dir/train.log"

  if run_train_once "$task" "$dataset_dir" "$lr" "$wr" "$epoch" "$bs" "$out_dir" "$log"; then
    return 0
  fi

  if grep -qiE 'outofmemoryerror|cuda out of memory' "$log"; then
    for retry_bs in 2 1; do
      local retry_out="$out_dir/retry_bs${retry_bs}"
      local retry_log="$retry_out/train.log"
      mkdir -p "$retry_out"
      echo "[$(date '+%F %T')] OOM_RETRY task=$task bs=$retry_bs"
      if run_train_once "$task" "$dataset_dir" "$lr" "$wr" "$epoch" "$retry_bs" "$retry_out" "$retry_log"; then
        return 0
      fi
      if ! grep -qiE 'outofmemoryerror|cuda out of memory' "$retry_log"; then
        echo "[$(date '+%F %T')] FAIL_NON_OOM task=$task see_log=$retry_log"
        return 1
      fi
    done
  fi

  echo "[$(date '+%F %T')] FAIL task=$task see_log=$log"
  return 1
}

for item in "${TASKS[@]}"; do
  read -r task dataset_dir lr wr epoch bs <<< "$item"
  run_one "$task" "$dataset_dir" "$lr" "$wr" "$epoch" "$bs"
  echo "[$(date '+%F %T')] DONE task=$task"
  sleep 10
done

echo "ALL_TASKS_DONE run_base=$RUN_BASE"