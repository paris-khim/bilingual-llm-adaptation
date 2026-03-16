#!/bin/bash
# Distributed Training Launch Script
# Usage: bash train_distributed.sh 8 (for 8 GPUs)

NUM_GPUS=$1

torchrun --nproc_per_node=$NUM_GPUS \
    adapt_llama.py \
    --model_name_or_path "meta-llama/Meta-Llama-3-70B" \
    --dataset_name "bilingual_pref_data.json" \
    --output_dir "./outputs/llama-3-bilingual-dpo" \
    --deepspeed ds_config_zero3.json \
    --bf16 True \
    --use_flash_attention_2 True
