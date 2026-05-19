#!/usr/bin/env bash

# Minimal runtime environment template for OHoRA experiments.
# Source this file in a shell before running training or evaluation.

export HF_HOME=/nas_data/xueyue.yang/guangze/hf_cache
export HUGGINGFACE_HUB_CACHE=/nas_data/xueyue.yang/guangze/hf_cache/hub
export TRANSFORMERS_CACHE=/nas_data/xueyue.yang/guangze/hf_cache/transformers
export OHORA_OUTPUT_ROOT=/nas_data/xueyue.yang/guangze/ohora_runs
