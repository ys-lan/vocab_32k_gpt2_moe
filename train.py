"""Unified pretraining / supervised fine-tuning entry point.

Adapted from Open-Llama/train_lm.py.

Usage:
    accelerate launch --config_file configs/accelerate_configs/ds_stage2.yaml \
        train.py \
        --train_config configs/pretrain_config.yaml \
        --model_config configs/model_configs/vocab_32k_gpt2_moe.json
"""

import logging

import yaml
from absl import app, flags
from accelerate import Accelerator
from datasets.distributed import split_dataset_by_node
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader

from dataset.dataset import construct_dataset
from models.configuration_vocab_32k_gpt2_moe import Vocab32kGPT2MoeConfig
from models.modeling_vocab_32k_gpt2_moe import Vocab32kGPT2MoeForCausalLM
from models.tokenization_vocab_32k_gpt2 import Vocab32kGPT2Tokenizer
from trainer import Trainer

FLAGS = flags.FLAGS
flags.DEFINE_string("train_config", None, "Training config path")
flags.DEFINE_string("model_config", None, "Model config path")


def main(argv):
    with open(FLAGS.train_config, "r", encoding="utf-8") as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)
    data_config = config["data"]

    accelerator = Accelerator(
        gradient_accumulation_steps=config["train"].get("gradient_accumulation_steps", 1)
    )
    tokenizer = Vocab32kGPT2Tokenizer(vocab_file=data_config["tokenizer_model_path"], legacy=False)
    if data_config.get("split_by_shard", False):
        train_dataset = construct_dataset(
            data_config, tokenizer, world_size=accelerator.num_processes
        )
    else:
        train_dataset = construct_dataset(data_config, tokenizer)
    train_dataset = split_dataset_by_node(
        train_dataset,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["train_batch_size"],
        num_workers=config["train"]["train_num_workers_4_dataloader"],
        prefetch_factor=config["train"].get("prefetch_factor", 2),
        pin_memory=True,
    )
    vocab_size = tokenizer.vocab_size

    model_config = Vocab32kGPT2MoeConfig.from_pretrained(FLAGS.model_config)
    model_config.vocab_size = vocab_size
    model_config.pad_token_id = tokenizer.pad_id
    if config["train"]["ckpt"] is not None:
        raw_model = Vocab32kGPT2MoeForCausalLM.from_pretrained(
            config["train"]["ckpt"], config=model_config
        )
        logging.warning("Loaded ckpt from: {}".format(config["train"]["ckpt"]))
    else:
        raw_model = Vocab32kGPT2MoeForCausalLM(config=model_config)

    total_params = sum(param.numel() for param in raw_model.parameters())
    logging.warning("#parameters: {}".format(total_params))

    if config["train"].get("use_lora", False):
        # Gradient checkpointing needs embedding outputs to require grad, see
        # https://github.com/huggingface/transformers/issues/23170
        if hasattr(raw_model, "enable_input_require_grads"):
            raw_model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            raw_model.get_input_embeddings().register_forward_hook(
                make_inputs_require_grad
            )
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj"],
            inference_mode=False,
            r=1,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        raw_model = get_peft_model(raw_model, peft_config)
        raw_model.print_trainable_parameters()
    if config["train"].get("gradient_checkpointing_enable", False):
        raw_model.gradient_checkpointing_enable()

    trainer = Trainer(config, raw_model, train_loader, tokenizer, accelerator)
    trainer.train()


if __name__ == "__main__":
    app.run(main)
