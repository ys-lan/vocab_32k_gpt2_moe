#!/usr/bin/env bash
# Stage 4 - Direct Preference Optimization on paired preference data.
# Input:  ckpt/vocab_32k_gpt2_moe_sft4dpo/checkpoint_epoch6
# Output: ckpt/vocab_32k_gpt2_moe_dpo
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5,6,7}"

accelerate launch \
  --config_file configs/accelerate_configs/ds_stage2.yaml \
  dpo.py
