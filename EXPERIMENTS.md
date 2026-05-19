# Experiment Inventory

This file documents the experiment families currently present in this repository and how they relate to the selection-component runs.

## Selection-component 24-run Suite

The original launcher is:

```text
scripts/launch_selection_24run_0505.sh
```

It launches 3 blocks for each seed `42, 43, 44`:

| Block | Model | Task | Methods | Count |
| --- | --- | --- | --- | ---: |
| `l2_commonsense_seed{seed}` | LLaMA-2-7B | commonsense instruction tuning + 8-task commonsense eval | `classic`, `relative_scores`, `relative_scores_mix`, `classic_mix_rerank` | 12 |
| `l3_commonsense_seed{seed}` | LLaMA-3-8B | commonsense instruction tuning + 8-task commonsense eval | `classic`, `classic_mix_rerank` | 6 |
| `l2_gsm8k_seed{seed}` | LLaMA-2-7B | GSM8K fine-tuning + GSM8K eval | `classic`, `classic_mix_rerank` | 6 |

Total: `12 + 6 + 6 = 24` runs.

The current GitHub-ready reproducible scripts cover the LLaMA-2 commonsense 12-run suite:

```text
scripts/repro_l2_commonsense_one.sh
scripts/repro_l2_commonsense_12run.sh
scripts/summarize_l2_commonsense_selection.py
```

The older 24-run scripts are also included, but still preserve some machine-specific defaults from the original server:

```text
scripts/run_selection_commonsense_block_0505.sh
scripts/run_selection_gsm8k_block_0505.sh
scripts/launch_selection_24run_0505.sh
```

For a new machine, prefer the `repro_*` scripts where available, or use the older scripts as references and replace local paths with environment variables.

## OHoRA Initialization Methods

The initialization logic lives in:

```text
peft/tuners/lora/layer.py
```

The main methods are:

- `classic`: original OHoRA QR-diagonal selection.
- `relative_scores`: selects by `abs(diag(R)) / column_norm`.
- `relative_scores_mix`: selects by `column_norm^(1-alpha) * relative_score^alpha`.
- `classic_mix_rerank`: selects a high-energy classic candidate pool, then reranks with the mixed score.

The CLI mapping lives in:

```text
fine-tuning_commonse.py
fine-tuning.py
```

Use:

```bash
--use_ohora True
--ohora_init_method classic|relative_scores|relative_scores_mix|classic_mix_rerank
--ohora_mix_alpha 0.5
```

## Task Families in the Codebase

### Commonsense Reasoning

Training:

```text
fine-tuning_commonse.py
build_dataset_reasoning.py
datasets/commonsense_reasoning/commonsense_170k.json
```

Evaluation:

```text
evaluate/run_commonsense_evaluate.py
datasets/test/{boolq,piqa,social_i_qa,hellaswag,winogrande,ARC-Challenge,ARC-Easy,openbookqa}/test.json
```

### GSM8K / Math

Training:

```text
fine-tuning.py
datasets/task_ft/gsm8k/gsm8k_train.json
scripts/run_selection_gsm8k_block_0505.sh
```

Evaluation:

```text
evaluate/run_gsm8k_eval.py
evaluate/run_math_evaluate.py
```

The `selection_component_24run_20260505` suite uses GSM8K for the LLaMA-2 `classic` vs `classic_mix_rerank` comparison across seeds 42/43/44.

### Code / HumanEval-style Fine-tuning

Training data included:

```text
datasets/task_ft/humaneval/humaneval_train.json
```

Training entry:

```text
fine-tuning_code.py
fine-tuning_code.sh
```

### MT-Bench-style Fine-tuning

Training data included:

```text
datasets/task_ft/mt_bench/mt_bench_train.json
```

Use the generic fine-tuning entry as a reference:

```text
fine-tuning.py
```

### MMLU

The local training file exists on the server as:

```text
datasets/task_ft/mmlu/mmlu_train.json
```

It is not tracked in GitHub because it is about 177 MB, which exceeds GitHub's 100 MB hard file limit for regular git blobs. Use Git LFS or external storage if this dataset needs to be versioned.

## Data Tracking Policy

Tracked in GitHub:

- Commonsense training/evaluation data.
- GSM8K training data.
- HumanEval-style training data.
- MT-Bench-style training data.

Not tracked:

- Model weights.
- Checkpoints and merged `sft_lora_model` outputs.
- Logs and eval JSON outputs.
- HF caches.
- MMLU training data unless Git LFS is enabled.

## Reproducibility Notes

- Clean from-scratch runs are preferred for OHoRA experiments.
- Avoid resuming interrupted OHoRA runs for paper-quality numbers unless the checkpoint/merge behavior has been validated.
- Keep model checkpoints external and pass paths with `MODEL_PATH` or the corresponding script variables.
