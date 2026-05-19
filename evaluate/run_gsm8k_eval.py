#!/usr/bin/env python
import argparse
import json
import pathlib
import re

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT_TEMPLATE = "Solve the following grade-school math problem.\n\nQuestion: {question}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--dataset_arrow",
        default="/data/xueyue.yang/.cache/huggingface/datasets/gsm8k/main/0.0.0/ea77aa1bb8875fdf/gsm8k-test.arrow",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def extract_final_number(text: str) -> float:
    matches = re.findall(r"-?\d[\d,]*\.?\d*", text.replace("$", ""))
    if not matches:
        return float("inf")
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return float("inf")


def extract_gold_answer(text: str) -> float:
    match = re.search(r"####\s*(-?\d[\d,]*\.?\d*)", text)
    if match:
        return float(match.group(1).replace(",", ""))
    return extract_final_number(text)


def batched(items, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def main():
    args = parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "gsm8k_predictions.json"
    summary_path = output_dir / "summary.csv"

    dataset = Dataset.from_file(args.dataset_arrow)
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    rows = []
    correct = 0
    batch_size = max(1, args.batch_size)

    examples = list(dataset)
    for chunk in batched(examples, batch_size):
        prompts = [PROMPT_TEMPLATE.format(question=item["question"]) for item in chunk]
        enc = tokenizer(prompts, padding=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated = outputs[:, input_ids.shape[1] :]
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)

        for item, pred_text in zip(chunk, texts):
            gold = extract_gold_answer(item["answer"])
            pred = extract_final_number(pred_text)
            flag = abs(gold - pred) <= 1e-3
            correct += int(flag)
            rows.append(
                {
                    "question": item["question"],
                    "answer": item["answer"],
                    "gold": gold,
                    "pred_text": pred_text,
                    "pred": pred,
                    "flag": flag,
                }
            )

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=True, indent=2)

    total = len(rows)
    acc = correct / total if total else 0.0
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("dataset,n,correct,accuracy\n")
        f.write(f"gsm8k,{total},{correct},{acc:.6f}\n")

    print(f"gsm8k accuracy={acc:.6f} correct={correct}/{total}")


if __name__ == "__main__":
    main()
