#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}

lr=0.0005
lora_rank=16
lora_alpha=32
lora_trainable="gate_proj,down_proj,up_proj,q_proj,k_proj,v_proj,o_proj"
lora_dropout=0.05
dataset_dir=${DATASET_DIR:-"${SCRIPT_DIR}/datasets/train/commonsense_reasoning"}
per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}
per_device_eval_batch_size=${PER_DEVICE_EVAL_BATCH_SIZE:-1}
gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-8}
max_seq_length=${MAX_SEQ_LENGTH:-1024}
num_train_epochs=${NUM_TRAIN_EPOCHS:-3}
seed=${SEED:-42}
output_root=${OHORA_OUTPUT_ROOT:-/nas_data/xueyue.yang/guangze/ohora_runs}
cache_dir=${CACHE_DIR:-${HF_HOME:-/nas_data/xueyue.yang/guangze/hf_cache}}
exp_name=${EXP_NAME:-commonsense_ohora}
gpu_selection=${CUDA_VISIBLE_DEVICES:-}
print_only=0
model_name_or_path=
tokenizer_name_or_path=

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-name-or-path)
            model_name_or_path="$2"
            shift 2
            ;;
        --tokenizer-name-or-path)
            tokenizer_name_or_path="$2"
            shift 2
            ;;
        --dataset-dir)
            dataset_dir="$2"
            shift 2
            ;;
        --output-dir)
            output_root="$2"
            shift 2
            ;;
        --cache-dir)
            cache_dir="$2"
            shift 2
            ;;
        --gpu)
            gpu_selection="$2"
            shift 2
            ;;
        --print-only)
            print_only=1
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$model_name_or_path" ]]; then
    echo "Missing required argument: --model-name-or-path" >&2
    exit 1
fi

if [[ -z "$tokenizer_name_or_path" ]]; then
    tokenizer_name_or_path="$model_name_or_path"
fi

output_dir="${output_root%/}/${exp_name}"

cmd=(
    "$PYTHON_BIN" "$SCRIPT_DIR/fine-tuning_commonse.py"
    --model_name_or_path "$model_name_or_path"
    --tokenizer_name_or_path "$tokenizer_name_or_path"
    --dataset_dir "$dataset_dir"
    --cache_dir "$cache_dir"
    --data_cache_dir "$cache_dir/datasets"
    --per_device_train_batch_size "$per_device_train_batch_size"
    --per_device_eval_batch_size "$per_device_eval_batch_size"
    --do_train
    --do_eval
    --seed "$seed"
    --bf16
    --num_train_epochs "$num_train_epochs"
    --lr_scheduler_type linear
    --learning_rate "$lr"
    --warmup_ratio 0.01
    --weight_decay 0
    --logging_strategy steps
    --logging_steps 10
    --save_strategy steps
    --save_total_limit 1
    --evaluation_strategy no
    --eval_steps 5000
    --save_steps 5000
    --gradient_accumulation_steps "$gradient_accumulation_steps"
    --max_seq_length "$max_seq_length"
    --output_dir "$output_dir"
    --logging_first_step True
    --lora_rank "$lora_rank"
    --lora_alpha "$lora_alpha"
    --lora_nums 0
    --trainable "$lora_trainable"
    --lora_dropout "$lora_dropout"
    --torch_dtype bfloat16
    --use_ohora True
    --load_in_kbits 16
    --overwrite_output_dir
)

if [[ -n "$gpu_selection" ]]; then
    cmd=(env CUDA_VISIBLE_DEVICES="$gpu_selection" "${cmd[@]}")
fi

if [[ "$print_only" -eq 1 ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
    exit 0
fi

exec "${cmd[@]}"