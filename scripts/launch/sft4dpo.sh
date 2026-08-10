#!/usr/bin/env bash
# Stage 3 - Extra SFT pass on the DPO prompt distribution, so that the policy and the
# preference data agree before alignment starts.
# Input:  ckpt/vocab_32k_gpt2_moe_instruction/checkpoint_epoch4
# Output: ckpt/vocab_32k_gpt2_moe_sft4dpo
set -euo pipefail

# export NCCL_P2P_LEVEL=NVL
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5,6,7}"

accelerate launch \
  --config_file configs/accelerate_configs/ds_stage2.yaml \
  train.py \
  --train_config configs/dpo_instruct_config.yaml \
  --model_config configs/model_configs/vocab_32k_gpt2_moe.json
