# OHoRA Reproducible Experiments

This repository contains the OHoRA training/evaluation code used for the LLaMA-2 commonsense initialization-selection experiments.

The main reproducible experiment is:

- model: LLaMA-2-7B HF checkpoint
- task: commonsense instruction tuning
- seeds: `42 43 44`
- initialization methods: `classic`, `relative_scores`, `relative_scores_mix`, `classic_mix_rerank`
- evaluation: BoolQ, PIQA, SocialIQA, HellaSwag, WinoGrande, ARC-Challenge, ARC-Easy, OpenBookQA

Model weights, checkpoints, logs, caches, and run outputs are intentionally not tracked by git. The commonsense training/evaluation JSON files are tracked so the LLaMA-2 commonsense experiment can be reproduced from this repository plus a local model checkpoint.

## Environment

Create an environment with Python 3.10 and install dependencies:

```bash
pip install -r requirements.txt
```

The experiments were run with:

- `torch==2.4.0+cu118`
- `transformers==4.46.3`
- `peft==0.13.0`
- `accelerate==0.27.2`

## Data Layout

The repository includes the commonsense training and evaluation data under:

```text
datasets/
  commonsense_reasoning/
    commonsense_170k.json
  test/
    boolq/test.json
    piqa/test.json
    social_i_qa/test.json
    hellaswag/test.json
    winogrande/test.json
    ARC-Challenge/test.json
    ARC-Easy/test.json
    openbookqa/test.json
```

If your data lives elsewhere, pass `DATASET_DIR` and keep `datasets/test` available for evaluation, or adapt `evaluate/run_commonsense_evaluate.py`.

## Single Run

Set the model path and output root:

```bash
export MODEL_PATH=/path/to/Llama-2-7b-hf
export RUN_ROOT=/path/to/ohora_runs/l2_commonsense_selection
```

Run one experiment from scratch:

```bash
bash scripts/repro_l2_commonsense_one.sh 42 relative_scores
```

Supported methods:

```text
classic
relative_scores
relative_scores_mix
classic_mix_rerank
```

The script refuses to resume by default. This is intentional: OHoRA checkpoints in the interrupted runs were not reliable enough for clean reproducibility. Use a fresh `OUT_DIR` or set `FORCE_OVERWRITE=1` to delete and rerun an existing output directory.

## Full 12-run Sweep

Launch all 12 LLaMA-2 commonsense runs:

```bash
export MODEL_PATH=/path/to/Llama-2-7b-hf
export RUN_ROOT=/path/to/ohora_runs/l2_commonsense_selection
export GPU_CANDIDATES="0 1 2 3 4 5 6 7"

bash scripts/repro_l2_commonsense_12run.sh
```

The launcher creates one tmux session per run and uses simple GPU lock directories under `${RUN_ROOT}/gpu_locks`.

## Summarize Results

After runs finish:

```bash
python scripts/summarize_l2_commonsense_selection.py \
  --run-root "${RUN_ROOT}" \
  --output "${RUN_ROOT}/l2_commonsense_summary.csv"
```

This prints per-run metrics, method-level means/stds, and deltas against `classic`.

## Methods

The OHoRA initialization variants are implemented in:

```text
peft/tuners/lora/layer.py
```

Relevant methods:

- `ohora_init`: classic QR diagonal selection
- `ohora_init_relative_scores`: `abs(diag(R)) / column_norm`
- `ohora_init_relative_scores_mix`: `column_norm^(1-alpha) * relative_score^alpha`
- `ohora_init_classic_mix_rerank`: classic candidate pool followed by mixed-score reranking

The CLI mapping is in:

```text
fine-tuning_commonse.py
```

Use:

```bash
--use_ohora True
--ohora_init_method {classic,relative_scores,relative_scores_mix,classic_mix_rerank}
--ohora_mix_alpha 0.5
```

## Notes

- Use clean-from-scratch runs for paper-quality results.
- Do not compare resumed OHoRA runs unless the checkpointing behavior has been validated for the exact merge/residual workflow.
- Keep `paper_repro_mode=True` to preserve the tokenizer padding behavior used in these experiments.
