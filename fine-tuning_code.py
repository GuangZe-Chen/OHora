#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import datasets
import torch
import time
from build_dataset_code import DataCollatorForSupervisedDataset,build_code_instruction_dataset
import transformers
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    AutoModelForCausalLM,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version

from peft import LoraConfig, TaskType, get_peft_model, PeftModel, get_peft_model_state_dict
from peft.tuners.lora import LoraLayer

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from typing import List

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

def getLayerNum(name,layer_allow=False):
    if "block.10" in name or ("layers.10" in name and layer_allow) or ("h.10" in name and layer_allow):
        return 10
    if "block.11" in name or ("layers.11" in name and layer_allow) or ("h.11" in name and layer_allow):
        return 11  
    if "block.12" in name or ("layers.12" in name and layer_allow) or ("h.12" in name and layer_allow):
        return 12
    if "block.13" in name or ("layers.13" in name and layer_allow) or ("h.13" in name and layer_allow):
        return 13   
    if "block.14" in name or ("layers.14" in name and layer_allow) or ("h.14" in name and layer_allow):
        return 14
    if "block.15" in name or ("layers.15" in name and layer_allow) or ("h.15" in name and layer_allow):
        return 15   
    if "block.16" in name or ("layers.16" in name and layer_allow) or ("h.16" in name and layer_allow):
        return 16
    if "block.17" in name or ("layers.17" in name and layer_allow) or ("h.17" in name and layer_allow):
        return 17    
    if "block.18" in name or ("layers.18" in name and layer_allow) or ("h.18" in name and layer_allow):
        return 18
    if "block.19" in name or ("layers.19" in name and layer_allow) or ("h.19" in name and layer_allow):
        return 19  
    if "block.20" in name or ("layers.20" in name and layer_allow) or ("h.20" in name and layer_allow):
        return 20
    if "block.21" in name or ("layers.21" in name and layer_allow) or ("h.21" in name and layer_allow):
        return 21   
    if "block.22" in name or ("layers.22" in name and layer_allow) or ("h.22" in name and layer_allow):
        return 22
    if "block.23" in name or ("layers.23" in name and layer_allow) or ("h.23" in name and layer_allow):
        return 23   
    if "block.24" in name or ("layers.24" in name and layer_allow) or ("h.24" in name and layer_allow):
        return 24
    if "block.25" in name or ("layers.25" in name and layer_allow) or ("h.25" in name and layer_allow):
        return 25   
    if "block.26" in name or ("layers.26" in name and layer_allow) or ("h.26" in name and layer_allow):
        return 26
    if "block.27" in name or ("layers.27" in name and layer_allow) or ("h.27" in name and layer_allow):
        return 27    
    if "block.28" in name or ("layers.28" in name and layer_allow) or ("h.28" in name and layer_allow):
        return 28
    if "block.29" in name or ("layers.29" in name and layer_allow) or ("h.29" in name and layer_allow):
        return 29  
    if "block.30" in name or ("layers.30" in name and layer_allow) or ("h.30" in name and layer_allow):
        return 30
    if "block.31" in name or ("layers.31" in name and layer_allow) or ("h.31" in name and layer_allow):
        return 31   
    if "block.32" in name or ("layers.32" in name and layer_allow) or ("h.32" in name and layer_allow):
        return 32
    if "block.33" in name or ("layers.33" in name and layer_allow) or ("h.33" in name and layer_allow):
        return 33   
    if "block.34" in name or ("layers.34" in name and layer_allow) or ("h.34" in name and layer_allow):
        return 34
    if "block.35" in name or ("layers.35" in name and layer_allow) or ("h.35" in name and layer_allow):
        return 35
    if "block.0" in name or ("layers.0" in name and layer_allow) or ("h.0" in name and layer_allow):
        return 0
    if "block.1" in name or ("layers.1" in name and layer_allow) or ("h.1" in name and layer_allow):
        return 1
    if "block.2" in name or ("layers.2" in name and layer_allow) or ("h.2" in name and layer_allow):
        return 2
    if "block.3" in name or ("layers.3" in name and layer_allow) or ("h.3" in name and layer_allow):
        return 3
    if "block.4" in name or ("layers.4" in name and layer_allow) or ("h.4" in name and layer_allow):
        return 4
    if "block.5" in name or ("layers.5" in name and layer_allow) or ("h.5" in name and layer_allow):
        return 5
    if "block.6" in name or ("layers.6" in name and layer_allow) or ("h.6" in name and layer_allow):
        return 6
    if "block.7" in name or ("layers.7" in name and layer_allow) or ("h.7" in name and layer_allow):
        return 7
    if "block.8" in name or ("layers.8" in name and layer_allow) or ("h.8" in name and layer_allow):
        return 8
    if "block.9" in name or ("layers.9" in name and layer_allow) or ("h.9" in name and layer_allow):
        return 9

def constrained_gcd(m: int, n: int) -> int:
        """
        求 m, n 的最大公共约数，且结果 ≤ floor(sqrt(min(m,n)))。
        时间复杂度：O(sqrt(min(m,n)))，空间复杂度：O(1)。
        """
        t = min(m, n)
        k = math.isqrt(t)          # 等价于 floor(sqrt(t))
        for i in range(k, 0, -1):  # 从 k 递减到 1
            if m % i == 0 and n % i == 0:
                return i
        return 1


class SavePeftModelCallback(transformers.TrainerCallback):
    def __init__(self,interval_ratio=0.1):
        self.model = None
        self.params_history = {}
        self.losses = []
        self.steps = []
        self.interval_ratio = interval_ratio
        #self.param_names = ['q_proj.lora_A','q_proj.lora_B',"v_proj.lora_A","v_proj.lora_B"]
        self.param_names = ['q_proj.lora_A','q_proj.lora_B',"v_proj.lora_A","v_proj.lora_B","k_proj.lora_A","k_proj.lora_B","o_proj.lora_A","o_proj.lora_B"
                            "gate_proj.lora_A","gate_proj.lora_B","down_proj.lora_A","down_proj.lora_B","up_proj.lora_A","up_proj.lora_B"]
        for i in range(32):
            #self.params_history[i] = {'q_proj.lora_A':0, 'q_proj.lora_B':0,"v_proj.lora_A":0,"v_proj.lora_B":0}
            self.params_history[i] = {'q_proj.lora_A':0,'q_proj.lora_B':0,"v_proj.lora_A":0,"v_proj.lora_B":0,"k_proj.lora_A":0,"k_proj.lora_B":0,"o_proj.lora_A":0,"o_proj.lora_B":0,
                            "gate_proj.lora_A":0,"gate_proj.lora_B":0,"down_proj.lora_A":0,"down_proj.lora_B":0,"up_proj.lora_A":0,"up_proj.lora_B":0}
    
    def on_train_begin(self,args,state,control,**kwargs):
        self.model = kwargs['model']
        total_steps = state.max_steps
        self.interval = int(total_steps * self.interval_ratio)
        if isinstance(kwargs["model"],PeftModel):
            output_files = "weight_outputs/lora/ohora_alpha64_ori.pth"
            #self.record_weight_change_history(output_files)
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step%state.logging_steps==0:
            self.steps.append(state.global_step)
            
        step = state.global_step
        if step > 0 and step % self.interval == 0:
            print(f"[Callback] Step {step}: Performing QR-based OHoRA reinitialization.")
            self.perform_qr_update()
    
    def perform_qr_update(self):
        for name, module in self.model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                # 获取 LoRA 参数
                A = module.lora_A['default'].weight.data   
                B = module.lora_B['default'].weight.data 
                rank = A.shape[1]
                

                # 合并 LoRA 权重与原始权重
                W_base = module.weight.data     # 原始 full weight
                ohora_AB = torch.kron(A.contiguous(),B.contiguous())                # LoRA 的 delta W
                W_full = W_base + ohora_AB        # 汇合后的权重
                W_full_32 = W_full.to(torch.float32)
                # QR 分解
                Q, R = torch.linalg.qr(W_full_32.data,mode="reduced")
                R_diag = torch.diag(R)
                Q = Q.to(W_full.dtype)
                R = R.to(W_full.dtype) 
                _,indices = torch.topk(R_diag,2*rank)  #选择top_2r
                out_dim, in_dim = Q.shape[0],R.shape[1]
                b_index, a_index = indices[:rank], indices[rank:]

                A_row, B_col = int(out_dim/rank), int(in_dim/rank)
                Q_a = Q[:A_row,a_index]
                R_a = R[a_index,:rank]
                Q_b = Q[:B_col,b_index]
                R_b = R[b_index,:rank]
                lora_A = (Q_a@R_a)
                lora_B = (Q_b@R_b).T

                # 重新设置 A, B（注意维度变换）
                module.lora_A['default'].weight.data = lora_A
                module.lora_B['default'].weight.data = lora_B

                module.weight.data = W_full - torch.kron(lora_A.contiguous(),lora_B.contiguous())
                print("重置结束！")

    def save_model(self, args, state, kwargs):
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "sft_lora_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "sft_lora_model")
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["processing_class"].save_pretrained(peft_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control
    


    def on_train_end(self, args, state, control, **kwargs):
        self.model = kwargs["model"]
        if isinstance(kwargs["model"],PeftModel):
            output_path = "weight_outputs/lora/ohora_alpha64_trained.pth"
            #self.record_weight_change_history(output_path)
            if state.log_history:
                for log in state.log_history:
                     if "loss" in log:
                        self.losses.append(log["loss"])
            save_loss_path = 'training_loss/pissa/qkora_std_loss.pth' 
            #torch.save({"train_loss":self.losses,"log_steps":self.steps}, save_loss_path)
            kwargs["model"] = kwargs["model"].merge_and_unload()
        peft_model_path = os.path.join(args.output_dir, "sft_lora_model")
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["processing_class"].save_pretrained(peft_model_path)
    
    def record_weight_change_history(self,output_path=None):
        for n,p in self.model.named_parameters():
            for param in self.param_names:
                if param in n:
                    layer_num = getLayerNum(n,layer_allow=True)
                    param_value = p.detach().clone()
                    self.params_history[layer_num][param] = param_value.cpu()#np.mean(param_value.cpu().numpy())  # 转换为 NumPy 数组并移动到 CPU
        torch.save(self.params_history, output_path)
        print(f"Lora weight has been saved in {output_path}")
    



def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True):
    r"""
    This method wraps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)

    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    # cast all non INT8/INT4 parameters to fp32
    for param in model.parameters():
        if ((param.dtype == torch.float16) or (param.dtype == torch.bfloat16)) and loaded_in_kbit:
            param.data = param.data.to(torch.float32)

    for name, module in model.named_modules():
        if 'norm' in name:
            module = module.to(torch.float32)

    if loaded_in_kbit and use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, _input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

    return model


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The tokenizer for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )

    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_dir: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )

    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    data_cache_dir: Optional[str] = field(default=None, metadata={"help": "The datasets processed stored"})

    max_seq_length: Optional[int] = field(default=1024)


@dataclass
class MyTrainingArguments(TrainingArguments):
    trainable : Optional[str] = field(default="q_proj,v_proj")
    lora_rank : Optional[int] = field(default=8)
    lora_dropout : Optional[float] = field(default=0.1)
    lora_alpha : Optional[float] = field(default=32.)
    modules_to_save : Optional[str] = field(default=None)
    peft_path : Optional[str] = field(default=None)
    flash_attn : Optional[bool] = field(default=False)
    double_quant: Optional[bool] = field(default=True)
    quant_type: Optional[str] = field(default="nf4")
    load_in_kbits: Optional[int] = field(default=16)
    
    lora_nums: Optional[int] = field(default=0)
    lora_moe_use: Optional[bool] = field(default=False)

    use_pissa: Optional[bool] = field(default=False)
    use_ohora : Optional[bool] = field(default=False)
    use_olora : Optional[bool] = field(default=False)
    use_dora : Optional[bool] = field(default=False)
    use_hira : Optional[bool] = field(default=False)
    full_ft: Optional[bool] = field(default=False)

    subspace_mixed: Optional[bool] = field(default=False)



logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if training_args.flash_attn:
        from flash_attn_patch import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()

    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,  # if training_args.local_rank in [-1, 0] else logging.WARN,
        handlers=[logging.StreamHandler(sys.stdout)],)


    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # transformers.tokenization_utils.logging.set_verbosity_warning()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        print('last_checkpoint',last_checkpoint)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "bos_token": '<s>',
        "eos_token": '</s>',
        "unk_token": '<unk>',
        "pad_token": '<unk>'
    }
    
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.tokenizer_name_or_path:
        #tokenizer = LlamaTokenizer.from_pretrained(model_args.tokenizer_name_or_path, **tokenizer_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path)
        #gemma
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        '''
        #llama-3.1-8B
        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
        '''
        
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # if (len(tokenizer)) != 55296:
    #     raise ValueError(f"The vocab size of the tokenizer should be 55296, but found {len(tokenizer)}.\n"
    #                      "Please use Chinese-LLaMA-2 tokenizer.")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    eval_dataset=None
    train_dataset = None

    if training_args.do_train:
        with training_args.main_process_first(desc="loading and tokenization"):
            path = Path(data_args.dataset_dir)
            files = [os.path.join(path,file.name) for file in path.glob("*.json")]
            logger.info(f"Training files: {' '.join(files)}")
            train_dataset = build_code_instruction_dataset(
                data_path=files,
                tokenizer=tokenizer,
                max_seq_length=data_args.max_seq_length,
                data_cache_dir = None,
                preprocessing_num_workers = data_args.preprocessing_num_workers)
        logger.info(f"Num train_samples  {len(train_dataset)}")
        logger.info(f"Training example input: {tokenizer.decode(train_dataset[0]['input_ids'])}")
        logger.info(f"Training example: {train_dataset[0]}")
    print("/******Start to do evaluation******/")
    training_args.do_eval = False
    if training_args.do_eval:
        with training_args.main_process_first(desc="loading and tokenization"):
            files = [data_args.validation_file]
            logger.info(f"Evaluation files: {' '.join(files)}")
            eval_dataset = build_code_instruction_dataset(
                data_path=files,
                tokenizer=tokenizer,
                max_seq_length=data_args.max_seq_length,
                data_cache_dir = None,
                preprocessing_num_workers = data_args.preprocessing_num_workers)
        logger.info(f"Num eval_samples  {len(eval_dataset)}")
        logger.info(f"Evaluation example input: {tokenizer.decode(eval_dataset[0]['input_ids'])}")
        logger.info(f"Evaluation example: {eval_dataset[0]}")

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    if training_args.load_in_kbits in [4, 8]:
        load_in_4bit = training_args.load_in_kbits == 4
        load_in_8bit = training_args.load_in_kbits == 8
        if training_args.modules_to_save is not None:
            load_in_8bit_skip_modules = training_args.modules_to_save.split(',')
        else:
            load_in_8bit_skip_modules = None
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=training_args.load_in_kbits == 4,
            load_in_8bit=training_args.load_in_kbits == 8,
            llm_int8_threshold=6.0,
            load_in_8bit_skip_modules=load_in_8bit_skip_modules,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=training_args.double_quant,
            bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
        )
    else:
        load_in_4bit = False
        load_in_8bit = False
        quantization_config = None
    if quantization_config is not None:
        logger.info(f"quantization_config:{quantization_config.to_dict()}")
    device_map = {"":int(os.environ.get("LOCAL_RANK") or 0)}
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        torch_dtype=torch_dtype,
        # low_cpu_mem_usage=True,
        # device_map=device_map,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        quantization_config=quantization_config,
    )
    hash_r = {}
    for name,param in model.named_parameters():
        if param.dim()>1:
            d_in, d_out = param.data.shape[0], param.data.shape[1]
            r_str = str(min(d_in,d_out))+"-"+str(max(d_in,d_out))
            if r_str not in hash_r:
                hash_r[r_str] = constrained_gcd(d_in,d_out)
    model.enable_input_require_grads()
    if training_args.load_in_kbits in [4, 8]:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    model.config.use_cache = False

    model_vocab_size = model.get_input_embeddings().weight.shape[0]
    logger.info(f"Model vocab size: {model_vocab_size}")
    logger.info(f"len(tokenizer):{len(tokenizer)}")
    if model_vocab_size != len(tokenizer):
        logger.info(f"Resize model vocab size to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    if training_args.peft_path is not None: # --------------------------> train from the trained lora model
        logger.info("Peft from pre-trained model")

        model = PeftModel.from_pretrained(model, training_args.peft_path,
            # device_map=device_map
            )
    else: # --------------------------> train from the sketch
        logger.info("Init new peft model") 
        target_modules = training_args.trainable.split(',') # lora paras
        modules_to_save = training_args.modules_to_save # not lora paras, but is trainable, i.e., not freeze
        if modules_to_save is not None:
            modules_to_save = modules_to_save.split(',')
        lora_rank = training_args.lora_rank
        lora_dropout = training_args.lora_dropout
        lora_alpha = training_args.lora_alpha

        if training_args.use_pissa:
            init_lora_weights = "pissa"
        elif training_args.use_ohora:
            init_lora_weights = "ohora"
        elif training_args.use_olora:
            init_lora_weights = "olora"
        elif training_args.use_hira:
            init_lora_weights = "hira"
        else:
            init_lora_weights = True
        
        lora_nums = training_args.lora_nums
        lora_moe_use = training_args.lora_moe_use

        subspace_mixed = training_args.subspace_mixed
        
        
        logger.info(f"target_modules: {target_modules}")
        logger.info(f"lora_rank: {lora_rank}")
        logger.info(f"lora_nums: {lora_nums}") 
        logger.info(f"lora_moe_use: {lora_moe_use}")

        start_time = time.time()
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
            r=lora_rank, 
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_nums=lora_nums,
            init_lora_weights=init_lora_weights,
            lora_moe_use=lora_moe_use,
            subspace_mixed=subspace_mixed,
            modules_to_save=modules_to_save,
            use_dora = training_args.use_dora,
            hash_r=hash_r
            )
        
        model = get_peft_model(model, peft_config)
        end_time = time.time()
        print("Init_time: ",end_time-start_time)
    if training_args.gradient_checkpointing and \
        (not model.modules_to_save or 'embed_tokens' not in model.modules_to_save):
        # enable requires_grad to avoid exception during backward pass when using gradient_checkpoint without tuning embed.
        if hasattr(model.base_model, "enable_input_require_grads"):
            model.base_model.enable_input_require_grads()
        elif hasattr(model.base_model, "get_input_embeddings"):
            def make_inputs_require_grad(_module, _input, _output):
                _output.requires_grad_(True)
            model.base_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if training_args.bf16:
                module = module.to(torch.bfloat16)
            if training_args.fp16:
                module = module.to(torch.float16)
            '''
            if torch_dtype=="float32":
                module = module.to(torch.float32)
            '''
        if 'norm' in name:
            module = module.to(torch.float16)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if training_args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
                if training_args.fp16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.float16)
    
    if training_args.full_ft:
        for name,param in model.named_parameters():
            if "lora_A" in name  or "lora_B" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
                param.data = param.data.to(torch.bfloat16)

    model.print_trainable_parameters()
    logger.info(f"model.modules_to_save: {model.modules_to_save}")
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))
    
    for name, parameters in model.named_parameters():
        logger.info(f"{name}, :, {parameters.size()},{parameters.requires_grad}")
    
    training_args.remove_unused_columns = False
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    trainer.add_callback(SavePeftModelCallback)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics

        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


if __name__ == "__main__":
    main()
