import copy
import json
import os
import re
import sys
import argparse

import torch

sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer


def resolve_device(args):
    if args.device != "auto":
        return torch.device(args.device)

    if torch.cuda.is_available():
        if args.gpu_id is not None:
            return torch.device(f"cuda:{args.gpu_id}")
        return torch.device("cuda")

    try:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass

    return torch.device("cpu")


def candidate_labels(dataset: str):
    mapping = {
        "boolq": ["true", "false"],
        "piqa": ["solution1", "solution2"],
        "social_i_qa": ["answer1", "answer2", "answer3"],
        "hellaswag": ["ending1", "ending2", "ending3", "ending4"],
        "winogrande": ["option1", "option2"],
        "ARC-Challenge": ["answer1", "answer2", "answer3", "answer4"],
        "ARC-Easy": ["answer1", "answer2", "answer3", "answer4"],
        "openbookqa": ["answer1", "answer2", "answer3", "answer4"],
    }
    return mapping.get(dataset, [])


def candidate_target_texts(dataset: str):
    labels = candidate_labels(dataset)
    if not labels:
        return []
    return [f"the correct answer is {label}" for label in labels]


def answer_prefix_text(dataset: str):
    if candidate_labels(dataset):
        return "the correct answer is"
    return ""


def main(
        load_8bit: bool = False,
        base_model: str = "",
        lora_weights: str = "tloen/alpaca-lora-7b",
        share_gradio: bool = False,
):
    args = parse_args()
    device = resolve_device(args)

    print(args)

    def evaluate(
            instructions,
            input=None,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            num_beams=4,
            max_new_tokens=32,
            **kwargs,
    ):
        prompts = [generate_prompt(instruction, input) for instruction in instructions]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        generation_kwargs = dict(
            num_beams=num_beams,
            do_sample=False,
            **kwargs,
        )
        if generation_kwargs.get("do_sample"):
            generation_kwargs.update(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        generation_config = GenerationConfig(**generation_kwargs)
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        sequences = generation_output.sequences
        prompt_lengths = attention_mask.sum(dim=1).tolist()
        generated_only = []
        for seq, prompt_len in zip(sequences, prompt_lengths):
            generated_only.append(seq[int(prompt_len):])
        outputs = tokenizer.batch_decode(generated_only, skip_special_tokens=True)
        outputs = [_response_only(o) for o in outputs]
        print(outputs)
        return outputs

    def score_label_predictions(instructions):
        prompts = [generate_prompt(instruction) for instruction in instructions]
        if args.eval_mode == "score_targets":
            candidates = candidate_target_texts(args.dataset)
            prefix_text = ""
        elif args.eval_mode == "score_label_after_prefix":
            candidates = candidate_labels(args.dataset)
            prefix_text = answer_prefix_text(args.dataset)
        else:
            candidates = candidate_labels(args.dataset)
            prefix_text = ""
        if not candidates:
            raise ValueError(f"{args.eval_mode} mode is not supported for dataset: {args.dataset}")

        predictions = []
        for prompt in prompts:
            if prefix_text:
                prompt = prompt + (" " if not prompt.endswith((" ", "\n")) else "") + prefix_text
            prompt_inputs = tokenizer(prompt, return_tensors="pt")
            prompt_ids = prompt_inputs["input_ids"].to(device)
            prompt_mask = prompt_inputs["attention_mask"].to(device)
            best_label = None
            best_score = None

            for candidate in candidates:
                candidate_text = candidate if candidate.startswith(" ") else f" {candidate}"
                label_ids = tokenizer(candidate_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
                input_ids = torch.cat([prompt_ids, label_ids], dim=1)
                attention_mask = torch.cat(
                    [prompt_mask, torch.ones_like(label_ids, device=device)],
                    dim=1,
                )
                labels_masked = input_ids.clone()
                labels_masked[:, :prompt_ids.shape[1]] = -100

                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels_masked,
                    )

                label_len = label_ids.shape[1]
                # HuggingFace causal LM loss is the mean over valid label tokens.
                score = -(outputs.loss.item() * label_len)
                if best_score is None or score > best_score:
                    best_score = score
                    best_label = candidate

            if args.eval_mode == "score_targets" and best_label:
                prefix = "the correct answer is "
                lowered = best_label.lower()
                if lowered.startswith(prefix):
                    best_label = best_label[len(prefix):]
            predictions.append(best_label or "")

        print(predictions)
        return predictions

    save_dir = args.output_dir
    create_dir(save_dir)
    save_file = os.path.join(save_dir, f'{args.model}-ohora-{args.dataset}.json')

    dataset = load_data(args)
    batches = create_batch(dataset, args.batch_size)
    tokenizer, model = load_model(args, device)
    total = len(batches)
    correct = 0
    current = 0
    output_data = []
    pbar = tqdm(total=total)
    for idx, batch in enumerate(batches):
        current += len(batch)
        instructions = [data.get('instruction') for data in batch]

        if args.eval_mode in {"score_labels", "score_targets", "score_label_after_prefix"}:
            outputs = score_label_predictions(instructions)
        else:
            outputs = evaluate(instructions)

        for data, output in zip(batch, outputs):
            label = data.get('answer')
            flag = False
            predict = extract_answer(args, output)
            if label == predict:
                correct += 1
                flag = True
            new_data = copy.deepcopy(data)
            new_data['output_pred'] = output
            new_data['pred'] = predict
            new_data['flag'] = flag
            output_data.append(new_data)
            '''
            print(data["instruction"])
            print(output)
            print('prediction:', predict)
            print('label:', label)
            '''
        print('---------------')
        print(f'\rtest:{idx + 1}/{total} | accuracy {correct}  {correct / current}')
        print('---------------')
        with open(save_file, 'w+') as f:
            json.dump(output_data, f, indent=4)
        pbar.update(1)
    pbar.close()
    print('\n')
    print('test finished')


def create_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)
    return


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Input:
                {input}

                ### Response:
                """  # noqa: E501
    else:
        instruct = "Below is an instruction that describes a task. Write a response that appropriately completes the request. \n### Instruction: \n" + instruction
        instruction = instruct+"\n ### Response:\n"
        return instruction


def load_data(args) -> list:
    """
    read data from dataset file
    Args:
        args:

    Returns:

    """
    file_path = f'/data/xueyue.yang/OHORA/ohora/datasets/test/{args.dataset}/test.json'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    json_data = json.load(open(file_path, 'r'))
    if args.max_samples is not None:
        json_data = json_data[:args.max_samples]
    return json_data

def create_batch(dataset, batch_size):
    batches = []
    num_batch = len(dataset)//batch_size if len(dataset) % batch_size == 0 else len(dataset)//batch_size + 1
    for i in range(num_batch):
        batch = dataset[i*batch_size: min((i+1)*batch_size, len(dataset))]
        batches.append(batch)
    return batches


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Challenge", "ARC-Easy", "openbookqa"],
                        default='winogrande',required=False)
    parser.add_argument('--model', default='llama-7B',choices=['LLaMA-7B', "LLaMA-13B",'BLOOM-7B', 'GPT-j-6B'], required=False)
    parser.add_argument('--adapter', default='',choices=['LoRA', 'AdapterP', 'AdapterH', 'Parallel'],
                        required=False)
    parser.add_argument('--base_model', default='full_l2_7b_cs_lr5e4_wr001_pd1_e3_fresh/20260420_220834/sft_lora_model', required=False)
    #parser.add_argument('--lora_weights', default='',required=False)
    parser.add_argument('--batch_size', default='4',type=int, required=False)
    parser.add_argument('--gpu_id', default=0, type=int, required=False)
    parser.add_argument('--device', default='auto', required=False)
    parser.add_argument('--output_dir', default=os.path.join(os.environ.get('OHORA_OUTPUT_ROOT', '/nas_data/xueyue.yang/guangze/ohora_runs'), 'commonsense_eval'), required=False)
    parser.add_argument('--max_samples', default=None, type=int, required=False)
    parser.add_argument('--eval_mode', default='generate', choices=['generate', 'score_labels', 'score_targets', 'score_label_after_prefix'], required=False)
    #parser.add_argument('--load_8bit', action='store_true', required=False)

    return parser.parse_args()


def load_model(args, device) -> tuple:
    """
    load tuned model
    Args:
        args:

    Returns:
        tuple(tokenizer, model)
    """
    base_model = args.base_model
    if not base_model:
        raise ValueError(f'can not find base model name by the value: {args.model}')
    '''
    lora_weights = args.lora_weights
    if not lora_weights:
        raise ValueError(f'can not find lora weight, the value is: {lora_weights}')
    load_8bit = args.load_8bit
    '''
    if "LLaMA" in args.model:
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    if device.type == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            #load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        ) # fix zwq
        '''
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map={"":0}
        )
        '''
    elif device.type == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(base_model).to(device)
        '''
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )
        '''
        # unwind broken decapoda-research config
        #model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        #model.config.bos_token_id = 1
        #model.config.eos_token_id = 2

        '''
        if not load_8bit:
            model.half()  # seems to fix bugs for some users.
        '''

        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

    return tokenizer, model


def load_instruction(args) -> str:
    instruction = ''
    if not instruction:
        raise ValueError('instruct not initialized')
    return instruction


def _first_match(pattern: str, text: str):
    return re.search(pattern, text, flags=re.IGNORECASE)


def _response_only(text: str) -> str:
    marker = "### Response:"
    if marker in text:
        text = text.split(marker, 1)[1]
    text = text.strip()
    text = re.sub(r"^(response\s*:|:)\s*", "", text, flags=re.IGNORECASE)
    for stop_marker in [
        "\nBelow is an instruction that describes a task",
        "\n### Instruction:",
        "\n### Input:",
        "\n### Response:",
    ]:
        if stop_marker in text:
            text = text.split(stop_marker, 1)[0]
    return text.strip()


def _clean_lines(text: str):
    text = _response_only(text)
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("below is an instruction"):
            continue
        if lower.startswith("### instruction"):
            continue
        if lower.startswith("### input"):
            continue
        if lower.startswith("### response"):
            continue
        if lower.startswith("answer format:"):
            continue
        lines.append(line)
    return lines


def _candidate_regions(text: str):
    text = _response_only(text)
    lines = _clean_lines(text)
    regions = []
    if lines:
        regions.append(lines[0])
        regions.append("\n".join(lines[:2]))
    regions.append(text)
    return regions


def _extract_indexed_label(text: str, prefix: str, max_index: int):
    # First valid match: return the earliest prefix+index mention in the output text.
    pat = rf"\b{re.escape(prefix)}\s*[:\-]?\s*([1-{max_index}])\b"
    m = _first_match(pat, text)
    if not m:
        return ""
    return f"{prefix}{m.group(1)}"


def _exact_label_line(lines, valid_labels):
    valid = {label.lower(): label for label in valid_labels}
    for line in lines:
        candidate = re.sub(r"[\s\.\!\?\,;:'\"`\(\)\[\]\{\}_-]+$", "", line.strip()).lower()
        if candidate in valid:
            return valid[candidate]
    return ""


def _prefixed_label_line(lines, prefixes, valid_labels):
    valid = {label.lower(): label for label in valid_labels}
    pat = r"^(?:%s)\s*[:\-]?\s*(%s)\s*$" % (
        "|".join(re.escape(prefix) for prefix in prefixes),
        "|".join(re.escape(label) for label in valid),
    )
    for line in lines:
        m = re.match(pat, line.strip(), flags=re.IGNORECASE)
        if not m:
            continue
        return valid[m.group(1).lower()]
    return ""


def _unique_label_in_text(text: str, valid_labels):
    found = []
    lowered = text.lower()
    for label in valid_labels:
        if re.search(rf"\b{re.escape(label.lower())}\b", lowered):
            found.append(label)
    if len(found) == 1:
        return found[0]
    return ""


def extract_answer(args, sentence: str) -> float:
    dataset = args.dataset
    sentence_ = sentence.strip()
    lines = _clean_lines(sentence_)
    regions = _candidate_regions(sentence_)
    if dataset == 'boolq':
        pred = _exact_label_line(lines, ['true', 'false', 'yes', 'no'])
        if pred:
            return 'true' if pred.lower() in ['true', 'yes'] else 'false'
        pred = _prefixed_label_line(lines, ['answer', 'response'], ['true', 'false', 'yes', 'no'])
        if pred:
            return 'true' if pred.lower() in ['true', 'yes'] else 'false'
        for region in regions:
            pred = _unique_label_in_text(region, ['true', 'false', 'yes', 'no'])
            if pred.lower() in ['true', 'yes']:
                return 'true'
            if pred.lower() in ['false', 'no']:
                return 'false'
        return ""
    elif dataset == 'piqa':
        pred = _exact_label_line(lines, ['solution1', 'solution2'])
        if pred:
            return pred
        pred = _prefixed_label_line(lines, ['answer', 'response'], ['solution1', 'solution2'])
        if pred:
            return pred
        for region in regions:
            pred = _unique_label_in_text(region, ['solution1', 'solution2'])
            if pred:
                return pred
        return ""
    elif dataset in ['social_i_qa', 'ARC-Challenge', 'ARC-Easy', 'openbookqa']:
        valid_labels = ['answer1', 'answer2', 'answer3', 'answer4', 'answer5']
        pred = _exact_label_line(lines, valid_labels)
        if pred:
            return pred
        pred = _prefixed_label_line(lines, ['answer', 'response'], valid_labels)
        if pred:
            return pred
        for region in regions:
            pred = _unique_label_in_text(region, valid_labels)
            if pred:
                return pred
        return ""
    elif dataset == 'hellaswag':
        valid_labels = ['ending1', 'ending2', 'ending3', 'ending4']
        pred = _exact_label_line(lines, valid_labels)
        if pred:
            return pred
        pred = _prefixed_label_line(lines, ['answer', 'response'], valid_labels)
        if pred:
            return pred
        for region in regions:
            pred = _unique_label_in_text(region, valid_labels)
            if pred:
                return pred
        return ""
    elif dataset == 'winogrande':
        valid_labels = ['option1', 'option2']
        pred = _exact_label_line(lines, valid_labels)
        if pred:
            return pred
        pred = _prefixed_label_line(lines, ['answer', 'response'], valid_labels)
        if pred:
            return pred
        for region in regions:
            pred = _unique_label_in_text(region, valid_labels)
            if pred:
                return pred
        return ""
    return ""


if __name__ == "__main__":
    main()
