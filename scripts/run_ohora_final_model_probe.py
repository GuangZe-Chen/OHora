#!/usr/bin/env python3
import argparse
import csv
import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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

MODELS = {
    "alpha0_old_classic": Path(
        "/nas_data/xueyue.yang/ohora_runs/requested_l2_l3_commonsense_bs2_lr7e4_20260425_113527/"
        "llama2_7b_lr7e-4_bs2_wrr0.01_e3/sft_lora_model"
    ),
    "alpha0p3_mix": Path(
        "/nas_data/xueyue.yang/ohora_runs/l2_relative_mix_alpha_sweep_20260428_queued/"
        "llama2_7b_lr7e-4_bs2_ga32_relative_scores_mix_alpha0p3_wrr0.01_e3/sft_lora_model"
    ),
    "alpha0p5_mix": Path(
        "/nas_data/xueyue.yang/ohora_runs/init_ablation_bs2_lr7e4_20260426_173600/"
        "llama2_7b_lr7e-4_bs2_ga32_relative_scores_mix_wrr0.01_e3/sft_lora_model"
    ),
    "alpha0p7_mix": Path(
        "/nas_data/xueyue.yang/ohora_runs/l2_relative_mix_alpha_sweep_20260428_queued/"
        "llama2_7b_lr7e-4_bs2_ga32_relative_scores_mix_alpha0p7_wrr0.01_e3/sft_lora_model"
    ),
    "alpha1_relative": Path(
        "/nas_data/xueyue.yang/ohora_runs/init_ablation_bs2_lr7e4_20260426_173600/"
        "llama2_7b_lr7e-4_bs2_ga32_relative_scores_wrr0.01_e3/sft_lora_model"
    ),
}


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


def load_data(dataset, max_samples):
    path = Path("/data/xueyue.yang/OHORA/ohora/datasets/test") / dataset / "test.json"
    data = json.load(open(path))
    return data[:max_samples]


def score_one(model, tokenizer, device, dataset, instruction):
    prompt = generate_prompt(instruction)
    prompt += " the correct answer is"
    prompt_inputs = tokenizer(prompt, return_tensors="pt")
    prompt_ids = prompt_inputs["input_ids"].to(device)
    prompt_mask = prompt_inputs["attention_mask"].to(device)
    best_label = None
    best_score = None
    for label in candidate_labels(dataset):
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
    return best_label or ""


def evaluate_model(args, name, model_dir):
    device = torch.device(args.device)
    out_dir = Path(args.out_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.pad_token_id or 0

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    rows = []
    for dataset in DATASETS:
        out_path = out_dir / f"{dataset}.json"
        if out_path.exists() and args.resume:
            data = json.load(open(out_path))
            n = len(data)
            c = sum(1 for item in data if item.get("flag"))
            rows.append((dataset, n, c, c / n if n else 0.0))
            continue

        data = load_data(dataset, args.max_samples)
        outputs = []
        correct = 0
        for item in data:
            pred = score_one(model, tokenizer, device, dataset, item["instruction"])
            flag = pred == item.get("answer")
            correct += int(flag)
            new_item = dict(item)
            new_item["pred"] = pred
            new_item["flag"] = flag
            outputs.append(new_item)
        json.dump(outputs, open(out_path, "w"), indent=2)
        rows.append((dataset, len(outputs), correct, correct / len(outputs) if outputs else 0.0))

    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "n", "correct", "accuracy"])
        writer.writerows(rows)

    total_n = sum(row[1] for row in rows)
    total_c = sum(row[2] for row in rows)
    macro = sum(row[3] for row in rows) / len(rows)
    print(f"{name}: macro={macro:.6f} weighted={total_c / total_n:.6f} correct={total_c}/{total_n}")

    del model, tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        default="/nas_data/xueyue.yang/ohora_runs/ohora_numeric_analysis/llama2_final_model_probe_200_20260502",
    )
    parser.add_argument("--models", default="alpha0_old_classic,alpha0p3_mix,alpha0p5_mix,alpha0p7_mix,alpha1_relative")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for name in [x for x in args.models.split(",") if x]:
        evaluate_model(args, name, MODELS[name])


if __name__ == "__main__":
    main()
