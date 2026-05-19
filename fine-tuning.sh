# User-defined parameters
lr=0.0002
lora_rank=1
lora_alpha=2
#lora_trainable="gate_proj,down_proj,up_proj,q_proj,k_proj,v_proj,o_proj"
lora_trainable="gate_proj,down_proj,up_proj,q_proj,k_proj,v_proj,o_proj"
lora_dropout=0.05                 
pretrained_model='hf_models/Llama-2-7B'
tokenizer_path='hf_models/Llama-2-7B'
dataset_dir='Dolly-15k/'
validation_file='mmlu_eval/five_shot_mmlu_val.json'
per_device_train_batch_size=1
per_device_eval_batch_size=1
gradient_accumulation_steps=8
max_seq_length=1024
output_dir='outputs/llama2-7B/ohora_test'
exp_name=lora_model

lora_b_nums=0  # Developer-specific, k-means, or DBSCAN et al.


python projects/HydraLoRA-main/HydraLoRA/fine-tuning.py \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --do_eval \
    --seed 42 \
    --bf16 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.01 \
    --weight_decay 0 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 1 \
    --evaluation_strategy steps \
    --eval_steps 5000 \
    --save_steps 5000 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --max_seq_length ${max_seq_length} \
    --output_dir ${output_dir}/${exp_name} \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --lora_nums ${lora_b_nums} \
    --trainable ${lora_trainable} \
    --lora_dropout ${lora_dropout} \
    --torch_dtype bfloat16 \
    --use_ohora True \
    --validation_file ${validation_file} \
    --load_in_kbits 16 \
    --overwrite_output_dir \