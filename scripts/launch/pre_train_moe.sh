#!/usr/bin/env bash
# Stage 1 - Pretrain the MoE base model from scratch.
# Output: ckpt/vocab_32k_gpt2_moe
set -euo pipefail

# Uncomment when NCCL needs tuning on a specific interconnect.
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_NTHREADS=16
# export NCCL_IB_TIMEOUT=3600
# export TORCH_NCCL_BLOCKING_WAIT=1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,4,5,6,7}"

accelerate launch \
  --config_file configs/accelerate_configs/ds_stage2.yaml \
  train.py \
  --train_config configs/pretrain_config.yaml \
  --model_config configs/model_configs/vocab_32k_gpt2_moe.json
