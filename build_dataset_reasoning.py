import logging
import os
import sys
import importlib
import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Sequence, Union, List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
removed_paths = []
for path in ("", CURRENT_DIR):
    while path in sys.path:
        sys.path.remove(path)
        removed_paths.append(path)

try:
    datasets = importlib.import_module("datasets")
finally:
    for path in reversed(removed_paths):
        sys.path.insert(0, path)

load_dataset = datasets.load_dataset
concatenate_datasets = datasets.concatenate_datasets

import torch
import transformers


IGNORE_INDEX = -100

logger = logging.getLogger('__name__')

PROMPT_TEMPLATE = (
    "{instruction}{eos_token}"
)


def _tokenizer_cache_id(tokenizer: transformers.PreTrainedTokenizer) -> str:
    """Create a short stable cache id so tokenized datasets do not cross tokenizers."""
    payload = {
        "class": tokenizer.__class__.__name__,
        "name_or_path": getattr(tokenizer, "name_or_path", ""),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "len": len(tokenizer),
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token_id": tokenizer.unk_token_id,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    name = os.path.basename(str(payload["name_or_path"]).rstrip(os.sep)) or payload["class"]
    safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
    return f"{safe_name}_{digest}"


def build_commonsense_reasoning_instruction_dataset(data_path: Union[List[str],str],
                tokenizer: transformers.PreTrainedTokenizer,
                max_seq_length: int, data_cache_dir = None,
                preprocessing_num_workers = None,
                overwrite_cache: bool = False,
                ):

    def tokenization(examples):
        sources = []
        targets = []
        prompt = PROMPT_TEMPLATE
        for instruction, input, output,ans in zip(examples['instruction'], examples['input'], examples['output'],examples['answer']):
            instruct = "Below is an instruction that describes a task. Write a response that appropriately completes the request. \n### Instruction: \n" + instruction
            instruction = instruct+"\n ### Response:\n"
            source = prompt.format_map({'instruction':instruction,'eos_token':tokenizer.eos_token}) 
            target = f"{output}{tokenizer.eos_token}"

            sources.append(source)
            targets.append(target)

        tokenized_sources = tokenizer(sources,return_attention_mask=False)
        tokenized_targets = tokenizer(targets,return_attention_mask=False,add_special_tokens=False)

        all_input_ids = []
        all_labels = []
        for s,t in zip(tokenized_sources['input_ids'],tokenized_targets['input_ids']):
            input_ids = torch.LongTensor(s + t)[:max_seq_length]
            labels = torch.LongTensor([IGNORE_INDEX] * len(s) + t)[:max_seq_length]
            assert len(input_ids) == len(labels)
            all_input_ids.append(input_ids)
            all_labels.append(labels)

        results = {'input_ids':all_input_ids, 'labels': all_labels}
        return results


    logging.warning("building dataset...")
    all_datasets = []

    if not isinstance(data_path,(list,tuple)):
        data_path = [data_path]
    for file in data_path:

        if data_cache_dir is None:
            data_cache_dir = str(os.path.dirname(file))
        cache_name = (
            os.path.basename(file).split('.')[0]
            + f"_{max_seq_length}_{_tokenizer_cache_id(tokenizer)}"
        )
        cache_path = os.path.join(data_cache_dir, cache_name)
        os.makedirs(cache_path, exist_ok=True)
        if not overwrite_cache:
            try:
                processed_dataset = datasets.load_from_disk(cache_path)                #处理过的数据已提前放入缓存，直接从缓存加载数据
                logger.info(f'training datasets-{file} has been loaded from disk: {cache_path}')
            except Exception:
                processed_dataset = None
        else:
            processed_dataset = None

        if processed_dataset is None:
            print("cache_dir: ",cache_path)
            raw_dataset = load_dataset("json", data_files=file, cache_dir=cache_path)
            tokenization_func = tokenization
            tokenized_dataset = raw_dataset.map(
                tokenization_func,
                batched=True,
                num_proc=preprocessing_num_workers,
                remove_columns=["instruction","input","output","answer"],
                keep_in_memory=False,
                desc="preprocessing on dataset",
            )
            processed_dataset = tokenized_dataset
            processed_dataset.save_to_disk(cache_path)
        processed_dataset.set_format('torch')
        all_datasets.append(processed_dataset['train'])
    all_datasets = concatenate_datasets(all_datasets)
    return all_datasets

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
        )
