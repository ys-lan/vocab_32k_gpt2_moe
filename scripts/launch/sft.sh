#!/usr/bin/env bash
# Stage 2 - Supervised fine-tuning on instruction data.
# Input:  ckpt/vocab_32k_gpt2_moe (set via train.ckpt in the config)
# Output: ckpt/vocab_32k_gpt2_moe_instruction
set -euo pipefail

# export NCCL_P2P_LEVEL=NVL
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

accelerate launch \
  --config_file configs/accelerate_configs/ds_stage2.yaml \
  train.py \
  --train_config configs/instruct_config.yaml \
  --model_config configs/model_configs/vocab_32k_gpt2_moe.json
