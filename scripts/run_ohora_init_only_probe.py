#!/usr/bin/env python3
import argparse
import csv
import gc
import json
import math
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/data/xueyue.yang/OHORA/ohora")
sys.path.insert(0, "/data/xueyue.yang/OHORA/ohora/peft/src")
from peft import LoraConfig, TaskType, get_peft_model  # noqa: E402


DATASETS = [
    "boolq",
    "piqa",
    "social_i_qa",
    "hellaswag",
    "winogrande",
    "ARC-Challenge",
    "ARC-Easy",
    "openbookqa",
]

TARGET_MODULES = [
    "gate_proj",
    "down_proj",
    "up_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]

STRATEGIES = {
    "classic": ("ohora", 0.5),
    "mix_alpha0p3": ("ohora_relative_scores_mix", 0.3),
    "mix_alpha0p5": ("ohora_relative_scores_mix", 0.5),
    "mix_alpha0p7": ("ohora_relative_scores_mix", 0.7),
    "relative": ("ohora_relative_scores", 0.5),
}


def constrained_gcd(m: int, n: int) -> int:
    t = min(m, n)
    k = math.isqrt(t)
    for i in range(k, 0, -1):
        if m % i == 0 and n % i == 0:
            return i
    return 1


def build_hash_r(model):
    hash_r = {}
    for _, param in model.named_parameters():
        if param.dim() > 1:
            d_in, d_out = param.data.shape[0], param.data.shape[1]
            key = f"{min(d_in, d_out)}-{max(d_in, d_out)}"
            hash_r.setdefault(key, constrained_gcd(d_in, d_out))
    return hash_r


def generate_prompt(instruction):
    return (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request. \n"
        "### Instruction: \n"
        + instruction
        + "\n ### Response:\n"
    )


def candidate_labels(dataset):
    return {
        "boolq": ["true", "false"],
        "piqa": ["solution1", "solution2"],
        "social_i_qa": ["answer1", "answer2", "answer3"],
        "hellaswag": ["ending1", "ending2", "ending3", "ending4"],
        "winogrande": ["option1", "option2"],
        "ARC-Challenge": ["answer1", "answer2", "answer3", "answer4"],
        "ARC-Easy": ["answer1", "answer2", "answer3", "answer4"],
        "openbookqa": ["answer1", "answer2", "answer3", "answer4"],
    }[dataset]


def answer_prefix_text(dataset):
    return "the correct answer is" if candidate_labels(dataset) else ""


def load_data(dataset, max_samples):
    path = Path("/data/xueyue.yang/OHORA/ohora/datasets/test") / dataset / "test.json"
    data = json.load(open(path))
    return data[:max_samples]


def score_batch(model, tokenizer, device, dataset, instructions):
    labels = candidate_labels(dataset)
    prefix = answer_prefix_text(dataset)
    predictions = []
    for instruction in instructions:
        prompt = generate_prompt(instruction)
        if prefix:
            prompt = prompt + (" " if not prompt.endswith((" ", "\n")) else "") + prefix
        prompt_inputs = tokenizer(prompt, return_tensors="pt")
        prompt_ids = prompt_inputs["input_ids"].to(device)
        prompt_mask = prompt_inputs["attention_mask"].to(device)
        best_label = None
        best_score = None
        for label in labels:
            candidate_text = label if label.startswith(" ") else f" {label}"
            label_ids = tokenizer(candidate_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
            input_ids = torch.cat([prompt_ids, label_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, torch.ones_like(label_ids, device=device)], dim=1)
            labels_masked = input_ids.clone()
            labels_masked[:, : prompt_ids.shape[1]] = -100
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_masked)
            score = -(outputs.loss.item() * label_ids.shape[1])
            if best_score is None or score > best_score:
                best_score = score
                best_label = label
        predictions.append(best_label or "")
    return predictions


def evaluate_strategy(args, strategy):
    init_method, mix_alpha = STRATEGIES[strategy]
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = 0

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    hash_r = build_hash_r(model)
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=TARGET_MODULES,
        inference_mode=False,
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        init_lora_weights=init_method,
        ohora_mix_alpha=mix_alpha,
        hash_r=hash_r,
    )
    model = get_peft_model(model, config)
    model.to(device)
    model.eval()

    strategy_dir = Path(args.out_dir) / strategy
    strategy_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in DATASETS:
        out_path = strategy_dir / f"{dataset}.json"
        if out_path.exists() and args.resume:
            data = json.load(open(out_path))
            n = len(data)
            c = sum(1 for item in data if item.get("flag"))
            rows.append((dataset, n, c, c / n if n else 0.0))
            continue

        data = load_data(dataset, args.max_samples)
        outputs = []
        correct = 0
        for i in range(0, len(data), args.batch_size):
            batch = data[i : i + args.batch_size]
            preds = score_batch(model, tokenizer, device, dataset, [x["instruction"] for x in batch])
            for item, pred in zip(batch, preds):
                flag = pred == item.get("answer")
                correct += int(flag)
                new_item = dict(item)
                new_item["pred"] = pred
                new_item["flag"] = flag
                outputs.append(new_item)
        json.dump(outputs, open(out_path, "w"), indent=2)
        rows.append((dataset, len(outputs), correct, correct / len(outputs) if outputs else 0.0))

    with open(strategy_dir / "summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "n", "correct", "accuracy"])
        writer.writerows(rows)

    del model, tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="/nas_data/xueyue.yang/hf_models/Llama-2-7b-hf")
    parser.add_argument("--out_dir", default="/nas_data/xueyue.yang/ohora_runs/ohora_numeric_analysis/llama2_init_only_probe_200_20260502")
    parser.add_argument("--strategies", default="classic,mix_alpha0p3,mix_alpha0p5,mix_alpha0p7,relative")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for strategy in [x for x in args.strategies.split(",") if x]:
        evaluate_strategy(args, strategy)


if __name__ == "__main__":
    main()
