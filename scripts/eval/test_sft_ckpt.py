"""Qualitative smoke test for an instruction-tuned (SFT or DPO) checkpoint.

Run from the repository root so that the relative config paths resolve:
    python scripts/eval/test_sft_ckpt.py
"""

import logging

from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

from dataset.validation import val_set_sft
from models.configuration_vocab_32k_gpt2_moe import Vocab32kGPT2MoeConfig
from models.modeling_vocab_32k_gpt2_moe import Vocab32kGPT2MoeForCausalLM
from models.tokenization_vocab_32k_gpt2 import Vocab32kGPT2Tokenizer

MODEL_CONFIG = "configs/model_configs/vocab_32k_gpt2_moe.json"
TOKENIZER_MODEL = "configs/tokenizer_models/vocab_32k_gpt2_moe.model"
CHECKPOINT = "ckpt/vocab_32k_gpt2_moe_instruction/checkpoint_epoch4"

tokenizer = Vocab32kGPT2Tokenizer(TOKENIZER_MODEL, legacy=False)


def load_zero_checkpoint(checkpoint="ckpt/vocab_32k_gpt2_moe_instruction/"):
    """Rebuild fp32 weights from a sharded DeepSpeed ZeRO checkpoint directory."""
    model_config = Vocab32kGPT2MoeConfig.from_pretrained(MODEL_CONFIG)
    model = Vocab32kGPT2MoeForCausalLM(config=model_config)
    state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint)
    model = model.cpu()
    model.load_state_dict(state_dict)
    logging.warning("loading complete")
    model.eval()
    model = model.half().cuda()
    logging.warning("ready")
    return model


def load_pretrained(checkpoint=CHECKPOINT):
    """Load a consolidated checkpoint saved with `save_pretrained`."""
    model_config = Vocab32kGPT2MoeConfig.from_pretrained(MODEL_CONFIG)
    model = Vocab32kGPT2MoeForCausalLM.from_pretrained(checkpoint, config=model_config)
    logging.warning("loading complete")
    model.eval()
    model = model.half().cuda()
    logging.warning("ready")
    return model


model = load_pretrained()

for data in val_set_sft:
    raw_inputs = data
    inputs = tokenizer(
        raw_inputs,
        return_tensors="pt",
        add_special_tokens=False,
        return_attention_mask=False,
    )
    input_length = inputs["input_ids"].shape[1]
    for k, v in inputs.items():
        inputs[k] = v.cuda()
    pred = model.generate(
        **inputs, max_new_tokens=256, do_sample=True, repetition_penalty=2.0
    )
    pred = pred[0, input_length:]
    pred = tokenizer.decode(pred.cpu(), skip_special_tokens=True)
    print(raw_inputs, '\n', pred, '\n')
