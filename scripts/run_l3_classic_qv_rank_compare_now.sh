#!/usr/bin/env bash
set -euo pipefail

cd /data/xueyue.yang/OHORA/ohora

ROOT="/nas_data/xueyue.yang/ohora_runs/l3_classic_qv_rank_compare/20260424_104004"
PY="/data/xueyue.yang/OHORA/venvs/ohora_cs_pure/bin/python"
TRAIN="/data/xueyue.yang/OHORA/ohora/fine-tuning_commonse.py"
MODEL="/nas_data/xueyue.yang/hf_models/Meta-Llama-3-8B"
DATASET_DIR="/data/xueyue.yang/OHORA/ohora/datasets/commonsense_reasoning"
CACHE_DIR="/nas_data/xueyue.yang/hf_cache"
DATA_CACHE_DIR="/nas_data/xueyue.yang/ohora_runs/lr5e4_wr_steps_sweep/cache/datasets"

mkdir -p "${ROOT}/A_classic_qv_rank8" "${ROOT}/B_classic_qv_rank64"

run_one() {
  local gpu="$1"
  local outdir="$2"
  local rank="$3"

  CUDA_VISIBLE_DEVICES="${gpu}" \
  OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PY}" "${TRAIN}" \
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
}

run_one 4 "${ROOT}/A_classic_qv_rank8" 8 &
P1=$!

run_one 6 "${ROOT}/B_classic_qv_rank64" 64 &
P2=$!

wait "${P1}"
wait "${P2}"
