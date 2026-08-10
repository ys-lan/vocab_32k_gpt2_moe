"""Direct Preference Optimization on top of an SFT checkpoint, driven by `trl.DPOTrainer`.

Usage:
    accelerate launch --config_file configs/accelerate_configs/ds_stage2.yaml dpo.py
"""

import os
import random
from dataclasses import dataclass, field
from glob import glob
from typing import Dict, Optional

import torch
from datasets import Dataset, load_dataset
from transformers import HfArgumentParser, TrainingArguments, set_seed
from trl import DPOTrainer

from models.configuration_vocab_32k_gpt2_moe import Vocab32kGPT2MoeConfig
from models.modeling_vocab_32k_gpt2_moe import Vocab32kGPT2MoeForCausalLM
from models.tokenization_vocab_32k_gpt2 import Vocab32kGPT2Tokenizer


@dataclass
class ScriptArguments:
    """Command-line arguments for the DPO training script."""

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    model_name_or_path: Optional[str] = field(
        default="ckpt/vocab_32k_gpt2_moe_sft4dpo/checkpoint_epoch6",
        metadata={"help": "the location of the SFT model name or path"},
    )
    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=500, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=4, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=5, metadata={"help": "the number of gradient accumulation steps"}
    )

    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "whether to use gradient checkpointing"}
    )

    gradient_checkpointing_use_reentrant: Optional[bool] = field(
        default=False, metadata={"help": "whether to use reentrant for gradient checkpointing"}
    )

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=50000, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=10000, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=10000, metadata={"help": "the evaluation frequency"})

    output_dir: Optional[str] = field(default="./ckpt/vocab_32k_gpt2_moe_dpo/", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "whether to load the model in 4bit"})
    model_dtype: Optional[str] = field(
        default="float", metadata={"help": "model_dtype[float16, bfloat16, float] for loading."}
    )

    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default="tensorboard",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    seed: Optional[int] = field(
        default=42, metadata={"help": "Random seed that will be set at the beginning of training."}
    )

def return_prompt_and_responses(samples) -> Dict[str, str]:
    return {
        "prompt": [
            "### Instruction: " + question + "\n\n### System: \n"
            for question in samples["question"]
        ],
        "chosen": samples["response_j"],  # rated better than k
        "rejected": samples["response_k"],  # rated worse than j
    }


def get_dataset_paired(
    data_patterns,
    sanity_check: bool = False,
    num_proc=24,
) -> Dataset:
    """Load paired preference data from local JSONL files and convert it to the format `DPOTrainer` expects.

    Every input record must contain:
    {
        'question': str,
        'response_j': str,  # the preferred response
        'response_k': str,  # the rejected response
    }

    Prompts are rendered with the same instruction template as SFT:
      "### Instruction: " + <question> + "\n\n### System: \n"
    """
    all_data_files = []
    for name, pattern in data_patterns.items():
        data_files = glob(pattern)
        assert len(data_files) > 0
        all_data_files.extend(data_files)
    random.shuffle(all_data_files)

    dataset = load_dataset("json", data_files=all_data_files, split='train', streaming=False)
    original_columns = dataset.column_names

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    set_seed(script_args.seed)

    # 1. Load the SFT model that DPO starts from
    torch_dtype = torch.float
    if script_args.model_dtype == "float16":
        torch_dtype = torch.float16
    elif script_args.model_dtype == "bfloat16":
        torch_dtype = torch.bfloat16

    tokenizer = Vocab32kGPT2Tokenizer(vocab_file="configs/tokenizer_models/vocab_32k_gpt2_moe.model", legacy=False)
    # DPOTrainer pads chosen/rejected pairs with the eos token, matching the convention used by the TRL examples.
    tokenizer.pad_token = tokenizer.eos_token
    model_config = Vocab32kGPT2MoeConfig.from_pretrained("configs/model_configs/vocab_32k_gpt2_moe.json")
    model_config.vocab_size = tokenizer.vocab_size
    model_config.pad_token_id = tokenizer.pad_id
    model = Vocab32kGPT2MoeForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        config=model_config,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
    )
    model.config.use_cache = False

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    # 2. Load the paired preference dataset. New sources must first be converted to the
    #    question / response_j / response_k schema documented on `get_dataset_paired`.
    data_patterns = {
        "mix_dpo_dataset": "data/DPO/mix_dpo_data.jsonl",
    }

    train_dataset = get_dataset_paired(data_patterns=data_patterns, sanity_check=script_args.sanity_check)

    # 3. Optional held-out split, disabled by default:
    # eval_dataset = get_dataset_paired({"eval": "data/DPO/eval_dpo_data.jsonl"}, sanity_check=True)
    # eval_dataset = eval_dataset.filter(
    #     lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length
    #     and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length
    # )

    # 4. Initialize the training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        max_steps=script_args.max_steps,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        evaluation_strategy="no",
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name="vocab_32k_gpt2_moe_dpo",
        gradient_checkpointing_kwargs=dict(use_reentrant=script_args.gradient_checkpointing_use_reentrant),
        seed=script_args.seed,
    )

    # 5. Optional LoRA adapters. Import `LoraConfig` from `peft` and pass `peft_config=peft_config` to `DPOTrainer`.
    # peft_config = LoraConfig(
    #     r=script_args.lora_r,
    #     lora_alpha=script_args.lora_alpha,
    #     lora_dropout=script_args.lora_dropout,
    #     target_modules=[
    #         "q_proj",
    #         "v_proj",
    #         "k_proj",
    #         "out_proj",
    #         "fc_in",
    #         "fc_out",
    #         "wte",
    #     ],
    #     bias="none",
    #     task_type="CAUSAL_LM",
    # )

    # 6. Initialize the DPO trainer. `ref_model=None` makes TRL keep a frozen copy of the policy as reference.
    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
        dataset_num_proc=8
    )

    # 7. Train
    dpo_trainer.train()
    dpo_trainer.save_model(script_args.output_dir)

    # 8. Save the aligned policy
    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)
